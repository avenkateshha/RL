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
"""Dense consumer validation, invocation lifetime and frozen teacher scoring."""

import gc
import weakref
from dataclasses import replace
from datetime import timedelta
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.algorithms.x_token.dense_teacher import DenseTeacherIPC
from nemo_rl.algorithms.x_token.loss_utils import LocalizedAlignment
from nemo_rl.distributed.selected_logprobs import cp_native_global_positions
from tests.unit.algorithms.x_token.native_sparse_fixtures import ObjectiveSettings
from tests.unit.algorithms.x_token.test_native_sparse_loss import make_native_student


def _inputs(*, fixture=None, settings=None, cp_rank=0, cp_size=1):
    from tests.unit.algorithms.x_token.test_native_same_tokenizer import (
        make_same_dense_payload,
        make_same_tokenizer_fixture,
        make_same_tokenizer_loss_fn,
    )

    fixture = make_same_tokenizer_fixture() if fixture is None else fixture
    settings = ObjectiveSettings() if settings is None else settings
    fn = make_same_tokenizer_loss_fn(fixture, settings)
    fn._dp_cp_group = None
    positions = cp_native_global_positions(
        fixture.student_logits.shape[1], cp_rank, cp_size
    )
    logits = fixture.student_logits[:, positions].clone().requires_grad_()
    native = make_native_student(fixture, logits, cp_rank=cp_rank, cp_size=cp_size)
    aligns = {
        i: LocalizedAlignment(
            sample_mask=fixture.sample_mask,
            student_input_ids=fixture.student_input_ids,
            student_token_mask=fixture.token_mask,
            student_kd_token_mask=fixture.kd_token_mask,
        )
        for i in range(len(fixture.teachers))
    }
    kwargs = dict(
        logits=logits,
        native_student=native,
        native_sparse_teachers={},
        native_dense_teachers={
            i: make_same_dense_payload(teacher, tp_size=2 - i, cp_size=1 + i)
            for i, teacher in enumerate(fixture.teachers)
        },
        teacher_full_logits_by_idx={},
        teacher_sparse_logits_by_idx={},
        aligns_by_idx=aligns,
        global_valid_chunks_by_idx=None,
    )
    return fixture, fn, kwargs


@pytest.mark.parametrize(
    "invalid", ["real_vocab", "storage_vocab", "samples", "ids", "mask", "duplicate"]
)
def test_native_dense_rejects_inconsistent_teacher_contract(invalid):
    fixture, fn, kwargs = _inputs()
    if invalid == "real_vocab":
        fn.teacher_vocab_sizes[0] -= 1
    elif invalid == "storage_vocab":
        # Consistent exported metadata can still be too narrow for this loss.
        from tests.unit.algorithms.x_token.test_native_same_tokenizer import (
            make_same_dense_payload,
        )

        teacher = replace(
            fixture.teachers[0], logits=fixture.teachers[0].logits[..., :4]
        )
        kwargs["native_dense_teachers"][0] = make_same_dense_payload(teacher)
    elif invalid == "samples":
        kwargs["native_dense_teachers"][0] = DenseTeacherIPC(
            kwargs["native_dense_teachers"][0].samples[:1]
        )
    elif invalid == "ids":
        kwargs["aligns_by_idx"][0].student_input_ids = fixture.student_input_ids[:, :2]
    elif invalid == "mask":
        kwargs["aligns_by_idx"][0].student_kd_token_mask = fixture.kd_token_mask[:, :2]
    else:
        kwargs["teacher_full_logits_by_idx"][0] = fixture.teachers[0].logits
    expected = {
        "real_vocab": "real vocabulary differs",
        "storage_vocab": "storage does not cover",
        "samples": "sample count disagrees",
        "ids": "full global student metadata",
        "mask": "full global KD mask",
        "duplicate": "exactly one legacy or native",
    }
    with pytest.raises(ValueError, match=expected[invalid]):
        fn._prepare_native_loss_context(**kwargs)


def test_true_averaged_logits_rejects_mixed_dense_consumer_protocols():
    fixture, fn, kwargs = _inputs()
    fn.kd_loss_mode = "averaged_logits"
    del kwargs["native_dense_teachers"][1]
    kwargs["teacher_full_logits_by_idx"][1] = fixture.teachers[1].logits
    with pytest.raises(ValueError, match="one native or legacy dense route"):
        fn._prepare_native_loss_context(**kwargs)


def test_native_dense_cache_is_teacher_and_invocation_local(monkeypatch):
    fixture, fn, kwargs = _inputs()
    fn.sum_weights_metric = "entropy"
    context = fn._prepare_native_loss_context(**kwargs)
    readers = [weakref.ref(reader) for reader in context.dense_teachers.values()]
    rows = []
    for i, reader in context.dense_teachers.items():
        spy = Mock(wraps=reader.gather_native_positions)
        monkeypatch.setattr(reader, "gather_native_positions", spy)
        first = fn._native_dense_rows(i, context, kwargs["aligns_by_idx"][i])
        fn._native_teacher_weight_score(
            i, "entropy", context, kwargs["aligns_by_idx"][i]
        )
        assert fn._native_dense_rows(i, context, kwargs["aligns_by_idx"][i]) is first
        spy.assert_called_once()
        assert not first.requires_grad
        torch.testing.assert_close(
            first,
            fixture.teachers[i].logits[
                :, context.student.global_positions, : fixture.student_vocab_size
            ],
        )
        rows.append(weakref.ref(first))
    assert context.dense_rows[0].data_ptr() != context.dense_rows[1].data_ptr()
    replacement = fn._prepare_native_loss_context(**kwargs)
    assert replacement.dense_rows == {}
    assert replacement.dense_teachers[0] is not context.dense_teachers[0]
    later_fixture = replace(
        fixture,
        teachers=tuple(
            replace(teacher, logits=teacher.logits.roll(1, dims=-1))
            for teacher in fixture.teachers
        ),
    )
    _, _, later_inputs = _inputs(fixture=later_fixture)
    later_context = fn._prepare_native_loss_context(**later_inputs)
    later_rows = fn._native_dense_rows(
        0, later_context, later_inputs["aligns_by_idx"][0]
    )
    torch.testing.assert_close(
        later_rows, later_fixture.teachers[0].logits[..., : fixture.student_vocab_size]
    )
    assert not torch.equal(later_rows, context.dense_rows[0])
    monkeypatch.undo()
    del reader, spy, first, context
    gc.collect()
    assert all(reference() is None for reference in readers + rows)


def _score_reference(fixture, teacher, metric):
    logits = teacher.logits[
        :, : fixture.student_input_ids.shape[1], : fixture.student_vocab_size
    ].float()
    if metric == "ce":
        values = -torch.nn.functional.cross_entropy(
            logits[:, :-1].flatten(0, 1),
            fixture.student_input_ids[:, 1:].flatten(),
            reduction="none",
        ).reshape(logits.shape[0], -1)
        mask = fixture.kd_token_mask[:, 1:]
    else:
        probs = logits.softmax(-1)
        values = (
            (probs * (probs + 1e-10).log()).sum(-1)
            if metric == "entropy"
            else probs.max(-1).values
        )
        mask = fixture.kd_token_mask
    mask = mask.float() * fixture.sample_mask.float().unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def _scoring_worker(rank, init_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    from tests.unit.algorithms.x_token.test_native_same_tokenizer import (
        make_same_tokenizer_fixture,
    )

    base = make_same_tokenizer_fixture()
    # Deliberately different CE/KD masks, unequal CP counts, a valid final
    # predictor, and a zero-weight sample expose rank means and shift errors.
    kd_mask = torch.zeros_like(base.kd_token_mask)
    kd_mask[0, [1, 2, 4, 7]] = 1
    kd_mask[1, [2, 3, 5, 6, 7]] = 1
    base = replace(base, kd_token_mask=kd_mask, token_mask=1 - kd_mask)
    one_owner_mask = torch.zeros_like(kd_mask)
    one_owner_mask[:, 1] = 1
    cases = [
        base,
        replace(base, sample_mask=torch.tensor([1.0, 0.0])),
        replace(base, kd_token_mask=torch.zeros_like(kd_mask)),
        replace(base, kd_token_mask=one_owner_mask),
    ]
    try:
        for fixture in cases:
            for metric in ("ce", "entropy", "max_prob"):
                _, fn, kwargs = _inputs(fixture=fixture, cp_rank=rank, cp_size=2)
                fn._dp_cp_group = dist.group.WORLD
                fn.sum_weights_metric = metric
                fn.alpha = 1.7
                context = fn._prepare_native_loss_context(**kwargs)
                expected = torch.stack(
                    [
                        _score_reference(fixture, teacher, metric)
                        for teacher in fixture.teachers
                    ]
                )
                actual = torch.stack(
                    [
                        fn._native_teacher_weight_score(
                            i, metric, context, kwargs["aligns_by_idx"][i]
                        )
                        for i in range(len(fixture.teachers))
                    ]
                )
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=3e-7)
                assert not actual.requires_grad
                weights = fn._compute_dynamic_weights(
                    {
                        "input_ids": fixture.student_input_ids,
                        "sample_mask": fixture.sample_mask,
                    },
                    {},
                    kwargs["aligns_by_idx"],
                    native_context=context,
                )
                torch.testing.assert_close(
                    torch.stack(weights), (1.7 * expected).softmax(0)
                )
                if metric == "ce":
                    fn.sum_weights_metric = None
                    fn.kd_loss_mode = "select_teacher"
                    kd, metrics = fn._select_teacher_kd(
                        None,
                        {"sample_mask": fixture.sample_mask},
                        {},
                        kwargs["aligns_by_idx"],
                        torch.tensor(fixture.global_valid_tokens),
                        teacher_sparse_logits_by_idx={},
                        tp_group=None,
                        cp_group=dist.group.WORLD,
                        native_context=context,
                    )
                    assert metrics["selected_teacher"] == int(expected.argmax())
                    kd.backward()
                    assert torch.isfinite(kwargs["logits"].grad).all()
        if rank == 0:
            print(
                "PASS native dense frozen scoring CP2: 4 masks x 3 metrics, selection and dynamic weights",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def test_native_dense_frozen_scores_match_global_reference_cp2(tmp_path):
    mp.spawn(
        _scoring_worker, args=(str(tmp_path / "scores.init"),), nprocs=2, join=True
    )
