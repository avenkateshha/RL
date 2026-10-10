# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Consumer-contract failures must surface before native loss collectives."""

import gc
import weakref
from dataclasses import replace

import pytest
import torch

from nemo_rl.algorithms.x_token.dense_teacher import DenseTeacherIPC
from nemo_rl.algorithms.x_token.sparse_teacher import SparseTeacherIPC
from tests.unit.algorithms.x_token.native_sparse_fixtures import (
    ObjectiveSettings,
    build_sparse_fixture,
)
from tests.unit.algorithms.x_token.test_native_sparse_loss import (
    make_alignment,
    make_loss_fn,
    make_native_student,
    make_sparse_payload,
)


def _inputs(tmp_path, *, payload_settings=None):
    fixture = build_sparse_fixture()
    settings = ObjectiveSettings()
    fn = make_loss_fn(fixture, settings, tmp_path)
    logits = fixture.student_logits.clone().requires_grad_()
    native = make_native_student(fixture, logits, cp_rank=0, cp_size=1)
    kwargs = dict(
        logits=logits,
        native_student=native,
        native_sparse_teachers={
            i: make_sparse_payload(teacher, payload_settings or settings)
            for i, teacher in enumerate(fixture.teachers)
        },
        native_dense_teachers={},
        teacher_full_logits_by_idx={},
        teacher_sparse_logits_by_idx={},
        aligns_by_idx={
            i: make_alignment(fixture, teacher)
            for i, teacher in enumerate(fixture.teachers)
        },
        global_valid_chunks_by_idx={
            i: torch.tensor(teacher.global_valid_chunks)
            for i, teacher in enumerate(fixture.teachers)
        },
    )
    return fn, kwargs


@pytest.mark.parametrize(
    "change", [{"k": 5}, {"temperature": 0.9}, {"noise_filter_k": 6}]
)
def test_native_consumer_rejects_different_teacher_objective(tmp_path, change):
    fn, kwargs = _inputs(
        tmp_path, payload_settings=replace(ObjectiveSettings(), **change)
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        fn._prepare_native_loss_context(**kwargs)


@pytest.mark.parametrize(
    "invalid", [None, torch.tensor(float("nan")), torch.tensor(-1.0), torch.ones(2)]
)
def test_native_consumer_requires_full_step_chunk_denominator(tmp_path, invalid):
    fn, kwargs = _inputs(tmp_path)
    if invalid is None:
        kwargs["global_valid_chunks_by_idx"] = None
    else:
        kwargs["global_valid_chunks_by_idx"][0] = invalid
    with pytest.raises(ValueError, match="denominator"):
        fn._prepare_native_loss_context(**kwargs)


@pytest.mark.parametrize(
    "invalid",
    [
        "samples",
        "teacher_vocab",
        "student_vocab",
        "duplicate",
        "missing_alignment",
        "teacher_sequence",
        "missing_context",
    ],
)
def test_native_consumer_rejects_inconsistent_payloads(tmp_path, invalid):
    fn, kwargs = _inputs(tmp_path)
    if invalid == "samples":
        kwargs["native_sparse_teachers"][0] = SparseTeacherIPC(
            kwargs["native_sparse_teachers"][0].samples[:1]
        )
    elif invalid == "teacher_vocab":
        fn.teacher_vocab_sizes[0] += 1
    elif invalid == "student_vocab":
        kwargs["native_student"] = replace(kwargs["native_student"], real_vocab_size=3)
    elif invalid == "duplicate":
        kwargs["teacher_full_logits_by_idx"][0] = torch.zeros(2, 8, 20)
    elif invalid == "missing_alignment":
        del kwargs["aligns_by_idx"][0]
    elif invalid == "teacher_sequence":
        kwargs["aligns_by_idx"][0].teacher_input_ids = torch.zeros(
            2, 7, dtype=torch.long
        )
    else:
        kwargs["native_student"] = None
    with pytest.raises(ValueError):
        fn._prepare_native_loss_context(**kwargs)


def test_native_dense_consumer_requires_same_tokenizer(tmp_path):
    fn, kwargs = _inputs(tmp_path)
    del kwargs["native_sparse_teachers"][0]
    kwargs["native_dense_teachers"] = {0: DenseTeacherIPC([])}
    with pytest.raises(ValueError, match="student tokenizer"):
        fn._prepare_native_loss_context(**kwargs)


def test_native_reader_lifetime_is_one_loss_invocation(tmp_path):
    fn, kwargs = _inputs(tmp_path)
    context = fn._prepare_native_loss_context(**kwargs)
    reader = weakref.ref(context.sparse_teachers[0])
    replacement = fn._prepare_native_loss_context(**kwargs)
    assert replacement.sparse_teachers[0] is not reader()
    del context
    gc.collect()
    assert reader() is None


@pytest.mark.parametrize("dynamic", [False, True])
def test_native_two_teacher_bf16_dispatch_uses_fp32_weights(tmp_path, dynamic):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from tests.unit.algorithms.x_token.native_sparse_fixtures import (
        sparse_objective_oracle,
    )

    fixture = build_sparse_fixture()
    settings = replace(ObjectiveSettings(), dynamic_scaling=dynamic)
    fn, inputs = _inputs(tmp_path)
    fn.dynamic_loss_scaling = dynamic
    logits = fixture.student_logits.bfloat16().requires_grad_()
    inputs["logits"] = logits
    inputs["native_student"] = replace(inputs["native_student"], logits=logits)
    data = BatchedDataDict(
        input_ids=fixture.student_input_ids,
        token_mask=fixture.token_mask,
        kd_token_mask=fixture.kd_token_mask,
        sample_mask=fixture.sample_mask,
    )
    for i, teacher in enumerate(fixture.teachers):
        data[f"teacher_{i}_input_ids"] = teacher.input_ids
        data[f"teacher_{i}_token_mask"] = torch.ones_like(teacher.input_ids)
    loss, metrics = fn(
        data,
        torch.tensor(2.0),
        torch.tensor(fixture.global_valid_tokens),
        student_logits_contig=None,
        megatron_cp_normalize=True,
        **inputs,
    )
    loss.backward()
    expected_logits = logits.detach().float().requires_grad_()
    expected = sparse_objective_oracle(fixture, expected_logits, settings)
    expected.loss.backward()
    torch.testing.assert_close(loss, expected.loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        logits.grad.float(), expected_logits.grad, rtol=0.03, atol=3e-4
    )
    for i, teacher in enumerate(fixture.teachers):
        assert metrics[f"weight_t{i}"] == pytest.approx(teacher.weight, abs=3e-8)
