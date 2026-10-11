# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Legacy DTensor sparse transport preserves values with reusable ownership."""

from contextlib import nullcontext
from dataclasses import replace
from itertools import count
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from nemo_rl.algorithms.x_token import loss_utils
from nemo_rl.algorithms.x_token.loss_utils import rebuild_teacher_sparse_logits_from_ipc
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.workers import dtensor_policy_worker_v2 as worker_module
from nemo_rl.utils import reusable_cuda_ipc as ipc

_IDENTITIES = count(1)


def _descriptor(tensor):
    identity = next(_IDENTITIES)
    return ipc.ReusableCudaIPCDescriptor(
        identity.to_bytes(64, "little"),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.numel() * tensor.element_size(),
        bytes([1]) * 16,
        123,
        "00000000-0000-0000-0000-000000000001",
    )


def _reader_fixture(*, membership=True, offset=0):
    backing, per_cp = {}, []
    for cp in range(2):
        logits = torch.arange(3 * 5 * 4).reshape(3, 5, 4).float() + offset + cp * 1000
        values = [logits, logits.to(torch.int32), logits[:, :, 0].clone()]
        if membership:
            values.append((logits[:, :, 0].to(torch.int32) % 3 == 0).to(torch.int32))
        descriptors = [_descriptor(value) for value in values]
        backing.update(zip(descriptors, values))
        per_cp.append((values, descriptors))
    records = []
    for sample in range(2):
        shards = []
        for cp in (1, 0):
            descriptors = per_cp[cp][1]
            shard = {
                "topk_logits_ipc": descriptors[0],
                "topk_indices_ipc": descriptors[1],
                "log_z_ipc": descriptors[2],
                "ipc_sample_index": sample,
                "topk_shape": (3, 2),
                "global_seq_start": cp * 3,
                "full_seq_len": 6,
            }
            if membership:
                shard["gt_in_topk_ipc"] = descriptors[3]
            shards.append(shard)
        records.append({"teacher_shards": shards})
    expected = []
    for field in range(4 if membership else 3):
        expected.append(
            torch.cat(
                [
                    values[field][:2, :3, :2] if field < 2 else values[field][:2, :3]
                    for values, _ in per_cp
                ],
                dim=1,
            )
        )
    expected = (
        expected[0],
        expected[1],
        expected[2],
        expected[3].bool() if membership else None,
    )
    return records, backing, expected


def _assert_values(actual, expected):
    for value, reference in zip(actual, expected):
        if reference is None:
            assert value is None
        else:
            torch.testing.assert_close(value.cpu(), reference.cpu())


@pytest.mark.parametrize("membership", [False, True])
def test_reader_slices_capacity_and_scopes_mappings_to_each_teacher_call(
    monkeypatch, membership
):
    first, backing, expected = _reader_fixture(membership=membership)
    second, other_backing, other_expected = _reader_fixture(
        membership=membership, offset=17
    )
    backing.update(other_backing)
    calls = []

    def open_handle(handle, device):
        calls.append(handle)
        return backing[handle]

    monkeypatch.setattr(loss_utils, "open_reusable_cuda_ipc", open_handle)
    _assert_values(rebuild_teacher_sparse_logits_from_ipc(first, device=0), expected)
    per_teacher = 8 if membership else 6
    assert len(calls) == per_teacher  # Shared by two samples, independent CP/fields.
    _assert_values(
        rebuild_teacher_sparse_logits_from_ipc(second, device=0), other_expected
    )
    assert len(calls) == 2 * per_teacher
    for descriptor, value in backing.items():
        if descriptor.dtype == torch.float32:
            value.add_(11)
    refreshed = tuple(
        value + 11 if index in (0, 2) else value for index, value in enumerate(expected)
    )
    _assert_values(rebuild_teacher_sparse_logits_from_ipc(first, device=0), refreshed)
    assert len(calls) == 3 * per_teacher


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_index",
        "negative_index",
        "batch_overflow",
        "width_overflow",
        "wrong_rank",
        "wrong_score_dtype",
        "wrong_bool_score",
    ],
)
def test_reader_rejects_bad_reusable_slice_metadata_before_open(monkeypatch, invalid):
    records, _, _ = _reader_fixture()
    shard = records[0]["teacher_shards"][1]  # CP0 is read first.
    if invalid == "missing_index":
        del shard["ipc_sample_index"]
    elif invalid == "negative_index":
        shard["ipc_sample_index"] = -1
    elif invalid == "batch_overflow":
        shard["ipc_sample_index"] = 3
    elif invalid == "width_overflow":
        shard["topk_shape"] = (3, 5)
    elif invalid == "wrong_rank":
        shard["topk_logits_ipc"] = replace(shard["topk_logits_ipc"], shape=(3, 20))
    elif invalid == "wrong_score_dtype":
        shard["topk_logits_ipc"] = replace(shard["topk_logits_ipc"], dtype=torch.int32)
    else:
        # Use the first-read score field to ensure validation happens before open.
        shard["topk_logits_ipc"] = replace(
            shard["topk_logits_ipc"], dtype=torch.bool, allocation_nbytes=60
        )
    monkeypatch.setattr(
        loss_utils,
        "open_reusable_cuda_ipc",
        lambda *_: pytest.fail("invalid slice opened"),
    )
    with pytest.raises(ValueError, match="Reusable legacy sparse IPC"):
        rebuild_teacher_sparse_logits_from_ipc(records, device=0)


def _producer(monkeypatch, *, device, tp_rank=0, actual_ipc=False):
    worker = SimpleNamespace(
        cfg={"logprob_batch_size": 2},
        cp_size=1,
        tp_mesh=SimpleNamespace(get_local_rank=lambda: tp_rank),
        cp_mesh=SimpleNamespace(get_local_rank=lambda: 0),
        dp_mesh=SimpleNamespace(get_local_rank=lambda: 0),
        device_mesh=None,
        tokenizer=SimpleNamespace(pad_token_id=0),
        model=SimpleNamespace(eval=lambda: None),
        sampling_params=None,
        allow_flash_attn_args=False,
        _autocast_context=nullcontext,
        _teacher_sparse_ipc_buffer=None,
        _teacher_ipc_storage=None,
        _teacher_ipc_handle=None,
    )
    calls, backing = [], {}
    state = {"offset": 0}
    monkeypatch.setattr(BatchedDataDict, "to", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(worker_module, "check_sequence_dim", lambda data: (1, 4))

    def iterator(data, cfg, mbs, *args, **kwargs):
        batches = [
            SimpleNamespace(
                data_dict={"input_ids": data["input_ids"][start : start + mbs]},
                processed_inputs=None,
            )
            for start in range(0, data.size, mbs)
        ]
        return iter(batches), len(batches)

    def forward(*, processed_mb, **kwargs):
        calls.append("forward")
        ids = processed_mb.data_dict["input_ids"][:, 0].to(device)
        values = (
            ids[:, None, None] * 100
            + torch.arange(12, device=device).reshape(1, 4, 3)
            + state["offset"]
        )
        return (
            (
                values.float(),
                values.to(torch.int32),
                values[:, :, 0].float(),
                (values[:, :, 0] % 2).bool(),
            ),
            {},
            None,
        )

    monkeypatch.setattr(worker_module, "get_microbatch_iterator", iterator)
    monkeypatch.setattr(
        worker_module,
        "prepare_model_forward",
        lambda *args, **kwargs: SimpleNamespace(model_context_factory=nullcontext),
    )
    monkeypatch.setattr(worker_module, "forward_with_post_processing_fn", forward)
    allocate, export, synchronize = (
        ipc.allocate_reusable_cuda_tensor,
        ipc.get_reusable_cuda_ipc_handle,
        torch.cuda.synchronize,
    )

    def allocation(shape, *, dtype, device):
        calls.append("allocate")
        return (
            allocate(shape, dtype=dtype, device=device)
            if actual_ipc
            else torch.empty(shape, dtype=dtype, device=device)
        )

    def descriptor(value):
        calls.append("export")
        result = export(value) if actual_ipc else _descriptor(value)
        backing[result] = value
        return result

    def sync():
        calls.append("sync")
        if actual_ipc:
            synchronize()

    monkeypatch.setattr(worker_module, "allocate_reusable_cuda_tensor", allocation)
    monkeypatch.setattr(worker_module, "get_reusable_cuda_ipc_handle", descriptor)
    monkeypatch.setattr(torch.cuda, "synchronize", sync)

    def run(batch_size):
        data = BatchedDataDict(
            {"input_ids": torch.arange(1, batch_size + 1)[:, None].expand(-1, 4)}
        )
        result = worker_module.DTensorPolicyWorkerV2Impl.get_topk_logits_ipc(
            worker,
            data,
            k=3,
            temperature=1.0,
            vocab_size=2048,
            support_mode="row_topk",
            gt_filter_topk=3,
        )
        return result

    return worker, run, calls, backing, state


def test_nonpublisher_completes_forwards_without_allocating_or_exporting(monkeypatch):
    worker, run, calls, _, _ = _producer(
        monkeypatch, device=torch.device("cpu"), tp_rank=1
    )
    monkeypatch.setattr(
        torch,
        "cat",
        lambda *args, **kwargs: pytest.fail("nonpublisher concatenated outputs"),
    )
    result = run(3)
    assert calls == ["forward", "forward"]
    assert result == {
        "per_sample_handles": [],
        "dp_rank": 0,
        "sparse_ipc_publisher": False,
    }
    assert worker._teacher_sparse_ipc_buffer is None


def test_publisher_reuses_grows_and_crops_whole_owned_slabs(monkeypatch):
    worker, run, calls, backing, state = _producer(
        monkeypatch, device=torch.device("cpu")
    )
    monkeypatch.setattr(
        loss_utils, "open_reusable_cuda_ipc", lambda handle, device: backing[handle]
    )
    first = run(3)
    owners = list(worker._teacher_sparse_ipc_buffer)
    assert first["sparse_ipc_publisher"]
    assert all("transport" not in record for record in first["per_sample_handles"])
    assert calls.index("sync") > max(
        i for i, call in enumerate(calls) if call == "forward"
    )
    actual = rebuild_teacher_sparse_logits_from_ipc(
        first["per_sample_handles"], device=0
    )
    torch.testing.assert_close(actual[0][:, 0, 0], torch.tensor([100.0, 200.0, 300.0]))
    state["offset"] = 20
    calls.clear()
    second = run(2)
    assert all(a is b for a, b in zip(owners, worker._teacher_sparse_ipc_buffer))
    assert "allocate" not in calls and calls.count("sync") == 1
    actual = rebuild_teacher_sparse_logits_from_ipc(
        second["per_sample_handles"], device=0
    )
    assert actual[0].shape == (2, 4, 3)
    torch.testing.assert_close(actual[0][:, 0, 0], torch.tensor([120.0, 220.0]))
    run(4)
    assert all(a is not b for a, b in zip(owners, worker._teacher_sparse_ipc_buffer))


def _consume(rank, records, expected):
    torch.cuda.set_device(1)
    values = rebuild_teacher_sparse_logits_from_ipc(records, device=1)
    _assert_values(
        values, tuple(torch.tensor(value, dtype=dtype) for value, dtype in expected)
    )
    del values
    torch.cuda.synchronize()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_cuda_real_worker_descriptors_support_fanout_reuse_growth_and_release(
    monkeypatch,
):
    torch.cuda.set_device(0)
    worker, run, _, backing, state = _producer(
        monkeypatch, device=torch.device("cuda", 0), actual_ipc=True
    )
    # Fixture bookkeeping must not extend raw producer ownership beyond worker release.
    for batch_size, offset in ((2, 0), (1, 17), (3, 31)):
        state["offset"] = offset
        records = run(batch_size)["per_sample_handles"]
        values = rebuild_teacher_sparse_logits_from_ipc(records, device=0)
        expected = [(value.cpu().tolist(), value.dtype) for value in values]
        del values
        mp.spawn(_consume, args=(records, expected), nprocs=2, join=True)
        backing.clear()
    worker_module.DTensorPolicyWorkerV2Impl.release_ipc_buffer(worker)
    assert worker._teacher_sparse_ipc_buffer is None
    worker_module.DTensorPolicyWorkerV2Impl.release_ipc_buffer(worker)
