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
"""Native CE/accuracy against a direct full-sequence real-vocabulary oracle."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.algorithms.x_token.loss_utils import _native_student_context
from nemo_rl.algorithms.x_token.native_student import (
    native_next_token_accuracy,
    native_next_token_ce,
)
from nemo_rl.distributed.selected_logprobs import cp_native_global_positions


def _student_cases(tp_size, cp_size, tp_rank, cp_rank, tp_group, cp_group, device):
    torch.manual_seed(338)
    for real_vocab in (7, 3):
        full = torch.randn(2, 8, 8, device=device)
        # Cross-TP ties pick the smaller global token ID. High padded logits
        # must affect neither prediction nor softmax normalization/gradients.
        full[0, 0, 1] = full[0, 0, real_vocab - 1] = 9
        full[..., real_vocab:] = 100
        ids = torch.arange(16, device=device).reshape(2, 8) % real_vocab
        ids[0, 1] = 1
        for case in ("dense", "empty_cp_owner", "empty"):
            mask = torch.tensor(
                [[1, 1, 1, 0, 1, 1, 0, 1], [1, 1, 0, 1, 0, 1, 1, 0]], device=device
            )
            if case == "empty_cp_owner":
                mask[:] = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0], device=device)
            if case == "empty":
                mask.zero_()
            sample_mask = torch.tensor([1.0, 0.5], device=device)
            data = {"input_ids": ids, "token_mask": mask, "sample_mask": sample_mask}
            positions = cp_native_global_positions(8, cp_rank, cp_size, device=device)
            width = 8 // tp_size
            local = full.index_select(1, positions)[
                ..., tp_rank * width : (tp_rank + 1) * width
            ]
            logits = (
                local.transpose(0, 1)
                .contiguous()
                .transpose(0, 1)
                .detach()
                .requires_grad_()
            )
            context = _native_student_context(
                logits,
                data,
                cp_group=cp_group,
                tp_group=tp_group,
                real_vocab_size=real_vocab,
            )
            weights = mask[:, 1:].float() * sample_mask[:, None]
            denominator = weights.sum() + (3 if case != "empty" else 0)
            expected_logits = full.clone().requires_grad_()
            ce_per_row = torch.nn.functional.cross_entropy(
                expected_logits[:, :-1, :real_vocab].flatten(0, 1),
                ids[:, 1:].flatten(),
                reduction="none",
            ).reshape(2, 7)
            expected_ce = (ce_per_row * weights).sum() / denominator.clamp(min=1)
            actual_ce = native_next_token_ce(context, denominator, tp_group=tp_group)
            actual_ce.backward()
            expected_ce.backward()
            torch.testing.assert_close(
                logits.grad,
                expected_logits.grad.index_select(1, positions)[
                    ..., tp_rank * width : (tp_rank + 1) * width
                ],
                rtol=2e-6,
                atol=2e-6,
            )
            complete_ce = actual_ce.detach().clone()
            dist.all_reduce(complete_ce, group=cp_group)
            torch.testing.assert_close(complete_ce, expected_ce, rtol=2e-6, atol=2e-6)
            prediction = full[:, :-1, :real_vocab].argmax(-1)
            expected_accuracy = (
                (prediction == ids[:, 1:]) * weights
            ).sum() / weights.sum().clamp(min=1)
            actual_accuracy = native_next_token_accuracy(
                context, tp_group=tp_group, cp_group=cp_group
            )
            torch.testing.assert_close(actual_accuracy, expected_accuracy)
            assert not actual_accuracy.requires_grad


def _distributed_student(rank, init_file, tp_size, cp_size, backend="gloo"):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=tp_size * cp_size,
        timeout=timedelta(seconds=90),
    )
    try:
        tps = [
            dist.new_group(list(range(cp * tp_size, (cp + 1) * tp_size)))
            for cp in range(cp_size)
        ]
        cps = [
            dist.new_group([tp + cp * tp_size for cp in range(cp_size)])
            for tp in range(tp_size)
        ]
        cp_rank, tp_rank = divmod(rank, tp_size)
        _student_cases(
            tp_size, cp_size, tp_rank, cp_rank, tps[cp_rank], cps[tp_rank], device
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_native_student_cpu_values_accuracy_and_gradients(tmp_path, tp_size, cp_size):
    mp.spawn(
        _distributed_student,
        args=(str(tmp_path / "gloo"), tp_size, cp_size),
        nprocs=tp_size * cp_size,
        join=True,
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="NCCL TP2/CP2 needs four GPUs"
)
@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_native_student_cuda_values_accuracy_and_gradients(tmp_path, tp_size, cp_size):
    mp.spawn(
        _distributed_student,
        args=(str(tmp_path / "nccl"), tp_size, cp_size, "nccl"),
        nprocs=tp_size * cp_size,
        join=True,
    )


@pytest.mark.parametrize("value", [float("nan"), -1.0, [1.0, 2.0]])
def test_native_ce_rejects_invalid_normalizer(value):
    logits = torch.ones(1, 4, 8, requires_grad=True)
    context = _native_student_context(
        logits,
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "token_mask": torch.ones(1, 4),
            "sample_mask": torch.ones(1),
        },
        cp_group=None,
        tp_group=None,
        real_vocab_size=7,
    )
    with pytest.raises(ValueError, match="scalar normalizer"):
        native_next_token_ce(context, torch.tensor(value), tp_group=None)
