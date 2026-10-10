# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise native metadata through the real adapter/caller/loss boundary."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nemo_rl.algorithms.loss.interfaces import LossInputType
from nemo_rl.algorithms.loss.loss_input import prepare_loss_input
from nemo_rl.algorithms.loss.wrapper import wrap_loss_fn_with_input_preparation
from nemo_rl.algorithms.x_token import loss_utils
from nemo_rl.algorithms.x_token.dense_teacher import DenseTeacherIPC
from nemo_rl.algorithms.x_token.loss_utils import (
    NativeStudentContext,
    XTokenLossInput,
    prepare_xtoken_cross_tokenizer_loss_input,
)
from nemo_rl.algorithms.x_token.sparse_teacher import SparseTeacherIPC
from tests.unit.algorithms.x_token.test_native_dense_reader import dense_payload
from tests.unit.distributed.test_native_sparse_reader import _payload as sparse_payload


def _data(modes):
    ids = torch.arange(24).reshape(3, 8) % 5
    data = {
        "input_ids": ids,
        "token_mask": torch.ones(3, 8),
        "kd_token_mask": torch.ones(3, 8),
        "sample_mask": torch.tensor([1, 1, 0]),
    }
    data["token_mask"][0, 2] = 0
    data["kd_token_mask"][1, 6] = 0
    for i, mode in enumerate(modes):
        if mode == "sparse":
            data[f"teacher_{i}_sparse_logits_ipc"] = sparse_payload()[0].samples
        elif mode == "legacy_sparse":
            data[f"teacher_{i}_sparse_logits_ipc"] = [
                {"teacher_shards": [{"transport": "sparse_topk"}]}
            ] * 3
        else:
            data[f"teacher_{i}_full_logits_ipc"] = dense_payload(offset=i * 1000)[
                0
            ].samples
        if mode != "same":
            data[f"teacher_{i}_input_ids"] = ids.flip(1)
            chunks = torch.tensor([[0, 0, -1, 2, 2, 3, 3, 3]]).expand(3, -1)
            data[f"alignment_{i}_student_chunk_id"] = chunks
            data[f"alignment_{i}_teacher_chunk_id"] = chunks.flip(1)
            data[f"alignment_{i}_pair_valid"] = torch.tensor(
                [[True, False, True, True]]
            ).expand(3, -1)
            data[f"alignment_{i}_pair_is_correct"] = data[f"alignment_{i}_pair_valid"]
    return data


def _forbid_reconstruction(monkeypatch):
    for name in (
        "cp_load_balanced_to_contiguous",
        "rebuild_teacher_full_logits_from_ipc",
        "rebuild_teacher_sparse_logits_from_ipc",
        "allgather_cp_contiguous_tensor",
    ):
        monkeypatch.setattr(
            loss_utils,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"Native route called {_name}"
            ),
        )
    monkeypatch.setattr(
        torch.cuda,
        "current_device",
        lambda: pytest.fail("CPU native adapter consulted CUDA"),
    )


@pytest.mark.parametrize("modes", [["sparse"], ["same", "same"], ["sparse", "same"]])
@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 0), (2, 1), (4, 2)])
def test_native_adapter_uses_global_next_targets_without_relayout(
    monkeypatch, modes, cp_size, cp_rank
):
    _forbid_reconstruction(monkeypatch)
    group = object()
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: cp_size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: cp_rank)
    monkeypatch.setattr(loss_utils, "loss_replica_group", lambda group: None)
    data = _data(modes)
    logits = torch.randn(3, 8 // cp_size, 6, requires_grad=True)
    prepared = prepare_xtoken_cross_tokenizer_loss_input(
        logits,
        data,
        teacher_is_cross_tokenizer=[mode != "same" for mode in modes],
        context_parallel_group=group,
        native_cp_enabled=True,
        native_sparse_enabled=True,
        native_same_tokenizer_enabled=True,
        student_vocab_size=6,
    )
    assert isinstance(prepared, XTokenLossInput)
    assert prepared.student_logits_contig is None
    native = prepared.native_student
    assert isinstance(native, NativeStudentContext) and native.logits is logits
    if cp_size == 1:
        expected = torch.arange(8)
    else:
        length = 8 // (2 * cp_size)
        expected = torch.tensor(
            list(range(cp_rank * length, (cp_rank + 1) * length))
            + list(
                range(
                    (2 * cp_size - cp_rank - 1) * length,
                    (2 * cp_size - cp_rank) * length,
                )
            )
        )
    torch.testing.assert_close(native.global_positions, expected)
    next_positions = (expected + 1).clamp(max=7)
    torch.testing.assert_close(
        native.next_token_ids, data["input_ids"][:, next_positions]
    )
    torch.testing.assert_close(
        native.next_token_mask, data["token_mask"][:, next_positions] * (expected < 7)
    )
    torch.testing.assert_close(
        native.next_kd_token_mask,
        data["kd_token_mask"][:, next_positions] * (expected < 7),
    )
    assert native.input_ids is data["input_ids"]
    for index, mode in enumerate(modes):
        if mode == "sparse":
            assert isinstance(prepared.native_sparse_teachers[index], SparseTeacherIPC)
            alignment = prepared.aligns_by_idx[index]
            torch.testing.assert_close(
                alignment.student_spans[0],
                torch.tensor([[0, 2], [0, 0], [3, 5], [5, 8]]),
            )
            torch.testing.assert_close(
                alignment.teacher_input_ids, data[f"teacher_{index}_input_ids"]
            )
            assert alignment.num_chunks.tolist() == [4, 4, 4]
        else:
            assert isinstance(prepared.native_dense_teachers[index], DenseTeacherIPC)


@pytest.mark.parametrize("legacy_mode", ["legacy_sparse", "dense"])
def test_mixed_legacy_consumer_requests_one_shared_compatibility_view(
    monkeypatch, legacy_mode
):
    modes = ["same", "sparse", legacy_mode, legacy_mode]
    data = _data(modes)
    logits = torch.randn(3, 8, 6, requires_grad=True)
    relay = Mock(side_effect=lambda tensor, **kwargs: tensor + 0.0)
    monkeypatch.setattr(loss_utils, "cp_load_balanced_to_contiguous", relay)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    dense = Mock(return_value=(torch.ones(3, 8, 6), 3))
    sparse = Mock(
        return_value=(
            torch.ones(3, 8, 2),
            torch.zeros(3, 8, 2, dtype=torch.int32),
            torch.ones(3, 8),
            None,
        )
    )
    monkeypatch.setattr(loss_utils, "rebuild_teacher_full_logits_from_ipc", dense)
    monkeypatch.setattr(loss_utils, "rebuild_teacher_sparse_logits_from_ipc", sparse)
    prepared = prepare_xtoken_cross_tokenizer_loss_input(
        logits,
        data,
        teacher_is_cross_tokenizer=[False, True, True, True],
        native_cp_enabled=True,
        native_sparse_enabled=True,
        native_same_tokenizer_enabled=True,
        student_vocab_size=6,
    )
    relay.assert_called_once_with(logits, cp_group=None)
    assert prepared.native_student.logits is logits
    assert set(prepared.native_dense_teachers) == {0}
    assert set(prepared.native_sparse_teachers) == {1}
    assert (dense.call_count, sparse.call_count) == (
        (2, 0) if legacy_mode == "dense" else (0, 2)
    )
    prepared.student_logits_contig.sum().backward()
    torch.testing.assert_close(logits.grad, torch.ones_like(logits))


def test_prepare_loss_input_threads_native_metadata_into_loss(monkeypatch):
    _forbid_reconstruction(monkeypatch)
    data = _data(["sparse", "same"])
    logits = torch.randn(3, 8, 6, requires_grad=True)
    received = {}

    class Consumer:
        input_type = LossInputType.DISTILLATION_CROSS_TOKENIZER
        teacher_is_cross_tokenizer = [True, False]
        student_vocab_size = 6

        def __call__(self, **kwargs):
            received.update(kwargs)
            return kwargs["native_student"].logits.sum(), {}

    loss, _ = wrap_loss_fn_with_input_preparation(
        logits,
        data,
        torch.tensor(3),
        torch.tensor(11),
        Consumer(),
        partial(
            prepare_loss_input,
            native_cp_enabled=True,
            native_sparse_enabled=True,
            native_same_tokenizer_enabled=True,
        ),
    )
    assert received["student_logits_contig"] is None
    assert received["native_student"].logits is logits
    assert set(received["native_sparse_teachers"]) == {0}
    assert set(received["native_dense_teachers"]) == {1}
    assert received["teacher_full_logits_by_idx"] == {}
    assert received["teacher_sparse_logits_by_idx"] == {}
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.ones_like(logits))


@pytest.mark.parametrize(
    "enabled,capability", [(False, False), (True, False), (False, True)]
)
def test_native_sparse_descriptor_rejects_disabled_consumer(enabled, capability):
    with pytest.raises(
        NotImplementedError, match="enabled native MCore sparse loss consumer"
    ):
        prepare_xtoken_cross_tokenizer_loss_input(
            torch.zeros(3, 8, 6),
            _data(["sparse"]),
            teacher_is_cross_tokenizer=[True],
            native_cp_enabled=enabled,
            native_sparse_enabled=capability,
        )


def test_capability_defaults_preserve_legacy_same_tokenizer_caller(monkeypatch):
    logits = torch.randn(3, 8, 6)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    rebuild = Mock(return_value=(torch.ones_like(logits), 0))
    monkeypatch.setattr(loss_utils, "rebuild_teacher_full_logits_from_ipc", rebuild)
    fn = SimpleNamespace(
        input_type=LossInputType.DISTILLATION_CROSS_TOKENIZER,
        teacher_is_cross_tokenizer=[False],
    )
    prepared, _ = prepare_loss_input(
        logits, _data(["same"]), fn, native_cp_enabled=True
    )
    assert "native_student" not in prepared
    assert prepared["student_logits_contig"] is logits
    rebuild.assert_called_once()


@pytest.mark.parametrize("native_peer", [False, True])
def test_legacy_dense_exporter_keeps_compatibility_and_cp_contract(
    monkeypatch, native_peer
):
    """A DTensor teacher's existing torch handle cannot enter native reopening."""
    modes = ["same", "sparse"] if native_peer else ["same"]
    data = _data(modes)
    for sample in data["teacher_0_full_logits_ipc"]:
        for shard in sample["teacher_shards"]:
            shard["payload_ipc"] = (("legacy-torch-reduction",),)
    logits = torch.randn(3, 4, 6, requires_grad=True)
    cp_group, tp_group = object(), object()
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group: 2 if group is cp_group else 1
    )
    monkeypatch.setattr(
        torch.distributed, "get_rank", lambda group: 1 if group is cp_group else 0
    )
    monkeypatch.setattr(loss_utils, "loss_replica_group", lambda group: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    relay = Mock(side_effect=lambda value, **kwargs: value + 0.0)
    monkeypatch.setattr(loss_utils, "cp_load_balanced_to_contiguous", relay)
    rebuild = Mock(return_value=(torch.ones(3, 4, 6), 3))
    monkeypatch.setattr(loss_utils, "rebuild_teacher_full_logits_from_ipc", rebuild)
    fn = SimpleNamespace(
        input_type=LossInputType.DISTILLATION_CROSS_TOKENIZER,
        teacher_is_cross_tokenizer=[False, True] if native_peer else [False],
        student_vocab_size=6,
    )
    prepared, _ = prepare_loss_input(
        logits,
        data,
        fn,
        vocab_parallel_group=tp_group,
        context_parallel_group=cp_group,
        native_cp_enabled=True,
        native_sparse_enabled=True,
        native_same_tokenizer_enabled=True,
    )
    relay.assert_called_once_with(logits, cp_group=cp_group)
    rebuild.assert_called_once()
    assert prepared["megatron_cp_normalize"] is True
    assert 0 in prepared["teacher_full_logits_by_idx"]
    assert 0 not in prepared.get("native_dense_teachers", {})
    torch.testing.assert_close(
        prepared["aligns_by_idx"][0].student_input_ids, data["input_ids"][:, 4:]
    )
    if native_peer:
        assert set(prepared["native_sparse_teachers"]) == {1}
        assert prepared["native_student"].logits is logits
    else:
        assert "native_student" not in prepared
    prepared["student_logits_contig"].sum().backward()
    torch.testing.assert_close(logits.grad, torch.ones_like(logits))


@pytest.mark.parametrize(
    "mutation", ["ids", "vocab", "rows", "mask", "samples", "identity"]
)
def test_native_context_rejects_inconsistent_student_or_teacher_metadata(mutation):
    data = _data(["same"])
    logits = torch.zeros(3, 8, 6)
    vocab_size = 6
    if mutation == "ids":
        data["input_ids"][0, -1] = 6
    elif mutation == "vocab":
        vocab_size = 7
    elif mutation == "rows":
        logits = logits[:, :7]
    elif mutation == "mask":
        data["kd_token_mask"] = data["kd_token_mask"][:, :7]
    elif mutation == "samples":
        data["teacher_0_full_logits_ipc"] = data["teacher_0_full_logits_ipc"][:2]
    else:
        data["batch_item_id"] = torch.tensor([11, 22, 33])
    with pytest.raises(ValueError, match="Native"):
        prepare_xtoken_cross_tokenizer_loss_input(
            logits,
            data,
            teacher_is_cross_tokenizer=[False],
            native_cp_enabled=True,
            native_same_tokenizer_enabled=True,
            student_vocab_size=vocab_size,
        )
