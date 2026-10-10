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
"""Same-tokenizer KL on native CP rows and the student's local TP vocabulary.

Teacher rows have already been copied into native student position order. The
common-K objective keeps the established microbatch/CP-wide column maximum;
true averaged logits use full-vocabulary KL. Both paths retain only the original
logits and recompute bounded FP32 row tiles in backward, without gathering a
student vocabulary or sequence.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from nemo_rl.algorithms.x_token.loss_utils import (
    NativeStudentContext,
    select_teacher_topk_indices,
)

if TYPE_CHECKING:
    from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn

_ROW_CHUNK_SIZE = 64


@dataclass(frozen=True)
class _KLGeometry:
    temperature: float
    reverse: bool
    full_vocab: bool
    vocab_start: int
    real_vocab_size: int
    tp_group: dist.ProcessGroup | None


def _reduce(
    value: torch.Tensor, group: dist.ProcessGroup | None, *, maximum: bool = False
) -> torch.Tensor:
    if group is not None and dist.get_world_size(group) > 1:
        dist.all_reduce(
            value, op=dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM, group=group
        )
    return value


def _sharded_logprobs(
    scores: torch.Tensor, group: dist.ProcessGroup | None
) -> torch.Tensor:
    maximum = _reduce(scores.max(-1).values, group, maximum=True)
    centered = scores - maximum[:, None]
    denominator = _reduce(centered.exp().sum(-1), group)
    return centered - denominator.log()[:, None]


def _row_logprobs(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    batch: torch.Tensor,
    sequence: torch.Tensor,
    columns: torch.Tensor,
    geometry: _KLGeometry,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Return student/teacher log probabilities and owned-column geometry."""
    width = logits.shape[-1]
    start = geometry.vocab_start
    if geometry.full_vocab:
        valid_width = max(0, min(width, geometry.real_vocab_size - start))
        student_scores = logits[batch, sequence].float() / geometry.temperature
        student_scores[:, valid_width:] = -torch.inf
        teacher_scores = torch.full_like(student_scores, -torch.inf)
        if valid_width:
            teacher_scores[:, :valid_width] = (
                teacher[batch, sequence, start : start + valid_width].float()
                / geometry.temperature
            )
        student_lp = _sharded_logprobs(student_scores, geometry.tp_group)
        teacher_lp = _sharded_logprobs(teacher_scores, geometry.tp_group)
        # Never subtract -inf padding, including TP ranks beyond the real vocab.
        return student_lp[:, :valid_width], teacher_lp[:, :valid_width], None

    owned = (columns >= start) & (columns < start + width)
    local_columns = (columns - start).clamp(0, width - 1)
    selected = logits[batch[:, None], sequence[:, None], local_columns].float()
    selected.masked_fill_(~owned, 0)
    _reduce(selected, geometry.tp_group)
    student_lp = torch.log_softmax(selected / geometry.temperature, -1)
    teacher_lp = torch.log_softmax(
        teacher[batch[:, None], sequence[:, None], columns].float()
        / geometry.temperature,
        -1,
    )
    return student_lp, teacher_lp, (owned, local_columns)


class _NativeSameTokenizerKL(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        logits: torch.Tensor,
        teacher: torch.Tensor,
        batch: torch.Tensor,
        sequence: torch.Tensor,
        columns: torch.Tensor,
        geometry: _KLGeometry,
    ) -> torch.Tensor:
        output = torch.empty(batch.numel(), device=logits.device, dtype=torch.float32)
        for offset in range(0, batch.numel(), _ROW_CHUNK_SIZE):
            end = offset + _ROW_CHUNK_SIZE
            student_lp, teacher_lp, _ = _row_logprobs(
                logits,
                teacher,
                batch[offset:end],
                sequence[offset:end],
                columns,
                geometry,
            )
            row_loss = (
                student_lp.exp() * (student_lp - teacher_lp)
                if geometry.reverse
                else teacher_lp.exp() * (teacher_lp - student_lp)
            ).sum(-1)
            if geometry.full_vocab:
                _reduce(row_loss, geometry.tp_group)
            output[offset:end] = row_loss
        ctx.save_for_backward(logits, teacher, batch, sequence, columns, output)
        # PyTorch creates this context before forward, outside __init__.
        ctx.geometry = geometry  # pyrefly: ignore[implicitly-defined-attribute]
        return output

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        logits, teacher, batch, sequence, columns, row_loss = ctx.saved_tensors
        geometry = ctx.geometry
        # Contiguous allocation also supports noncontiguous B/S-strided inputs.
        gradient = torch.zeros(logits.shape, dtype=logits.dtype, device=logits.device)
        for offset in range(0, batch.numel(), _ROW_CHUNK_SIZE):
            end = offset + _ROW_CHUNK_SIZE
            b, s = batch[offset:end], sequence[offset:end]
            student_lp, teacher_lp, ownership = _row_logprobs(
                logits, teacher, b, s, columns, geometry
            )
            derivative = (
                student_lp.exp()
                * (student_lp - teacher_lp - row_loss[offset:end, None])
                if geometry.reverse
                else student_lp.exp() - teacher_lp.exp()
            )
            derivative *= grad_output[offset:end, None] / geometry.temperature
            if geometry.full_vocab:
                gradient[b, s, : derivative.shape[-1]] = derivative.to(logits.dtype)
            else:
                assert ownership is not None
                owned, local_columns = ownership
                gradient[b[:, None], s[:, None], local_columns[owned]] = derivative[
                    :, owned
                ].to(logits.dtype)
        return gradient, None, None, None, None, None


def _common_columns(
    teacher_rows: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    cp_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Compute the existing column maximum using bounded masked row tiles."""
    vocab = teacher_rows.shape[-1]
    importance = teacher_rows.new_full((vocab,), -torch.inf)
    batch, sequence = torch.nonzero(mask != 0, as_tuple=True)
    with torch.no_grad():
        for offset in range(0, batch.numel(), _ROW_CHUNK_SIZE):
            end = offset + _ROW_CHUNK_SIZE
            rows = teacher_rows[batch[offset:end], sequence[offset:end]]
            torch.maximum(importance, rows.max(0).values, out=importance)
    return select_teacher_topk_indices(
        importance.reshape(1, 1, vocab), k, cp_group=cp_group
    )


def compute_native_same_tokenizer_kl(
    loss_fn: "CrossTokenizerDistillationLossFn",
    student: NativeStudentContext,
    teacher_rows: torch.Tensor,
    *,
    global_valid_toks: torch.Tensor,
    tp_group: dist.ProcessGroup | None,
    cp_group: dist.ProcessGroup | None,
    full_vocab: bool = False,
) -> torch.Tensor:
    """Return this CP owner's normalized KL, with no CP replication divisor.

    The default selects one common K-column vocabulary from all valid predictor
    rows in the microbatch, including the other CP owners. ``full_vocab`` is used
    after averaging raw teacher logits for the true ``averaged_logits`` objective.
    Teacher gradients are intentionally absent: these are frozen IPC row copies.
    """
    logits = student.logits
    if logits.ndim != 3 or teacher_rows.shape != (
        *logits.shape[:2],
        student.real_vocab_size,
    ):
        raise ValueError(
            "Native same-tokenizer teacher rows must match native B/S and real vocabulary"
        )
    if teacher_rows.device != logits.device:
        raise ValueError("Native teacher rows and student logits must share a device")
    if student.real_vocab_size <= 0 or logits.shape[-1] <= 0:
        raise ValueError("Native same-tokenizer vocabulary sizes must be positive")
    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    if student.real_vocab_size > logits.shape[-1] * tp_size:
        raise ValueError("Native student TP shards do not cover its real vocabulary")
    if global_valid_toks.numel() != 1 or not bool(
        torch.isfinite(global_valid_toks).all() & (global_valid_toks >= 0).all()
    ):
        raise ValueError(
            "Native same-tokenizer KL requires a finite nonnegative scalar normalizer"
        )
    temperature = float(loss_fn.temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Native same-tokenizer temperature must be finite and positive"
        )
    mask = student.next_kd_token_mask.float() * student.sample_mask.float()[:, None]
    if mask.shape != logits.shape[:2]:
        raise ValueError("Native same-tokenizer next-token masks must match native B/S")
    k = min(int(loss_fn.vocab_topk), student.real_vocab_size)
    if k < 0:
        raise ValueError("Native same-tokenizer vocab_topk must be nonnegative")
    full_vocab = full_vocab or k == student.real_vocab_size
    columns = (
        torch.empty(0, device=logits.device, dtype=torch.long)
        if full_vocab
        else _common_columns(teacher_rows, mask, k, cp_group)
    )
    batch, sequence = torch.nonzero(mask != 0, as_tuple=True)
    if not full_vocab and k == 0:
        return logits.sum() * 0
    geometry = _KLGeometry(
        temperature=temperature,
        reverse=loss_fn.reverse_kl,
        full_vocab=full_vocab,
        vocab_start=(dist.get_rank(tp_group) if tp_group is not None else 0)
        * logits.shape[-1],
        real_vocab_size=student.real_vocab_size,
        tp_group=tp_group,
    )
    per_row = _NativeSameTokenizerKL.apply(
        logits, teacher_rows.detach(), batch, sequence, columns, geometry
    )
    return (
        (per_row * mask[batch, sequence]).sum()
        / global_valid_toks.clamp(min=1)
        * temperature**2
    )
