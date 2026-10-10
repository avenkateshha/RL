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

"""Selected vocabulary probabilities on Megatron's original CP row owners.

Adapted from the native sparse primitives at 5acd78e948d2. These helpers never
gather student sequence rows or full vocabulary shards.
"""

import math
from typing import Any

import torch
import torch.distributed as dist

from nemo_rl.distributed.model_utils import group_all_reduce_sum_with_grad_backward_sum


def cp_native_sequence_segments(
    full_seq_len: int, cp_rank: int, cp_size: int
) -> list[tuple[int, int, int]]:
    """Return ``(local_start, global_start, length)`` for native CP rows."""
    if full_seq_len < 0 or cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("Expected nonnegative sequence length and valid CP rank/size")
    if cp_size == 1:
        return [(0, 0, full_seq_len)]
    if full_seq_len % (2 * cp_size):
        raise ValueError(
            "Native CP sequence length must be divisible by 2*cp_size; "
            f"got sequence={full_seq_len}, cp_size={cp_size}"
        )
    block = full_seq_len // (2 * cp_size)
    return [
        (0, cp_rank * block, block),
        (block, (2 * cp_size - cp_rank - 1) * block, block),
    ]


def cp_native_global_positions(
    full_seq_len: int,
    cp_rank: int,
    cp_size: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return the global predictor positions in native Megatron zigzag order."""
    return torch.cat(
        [
            torch.arange(start, start + length, device=device, dtype=torch.long)
            for _, start, length in cp_native_sequence_segments(
                full_seq_len, cp_rank, cp_size
            )
        ]
    )


def _group_size(group: dist.ProcessGroup | None) -> int:
    # None is deliberately local, even when a WORLD process group exists.
    return dist.get_world_size(group) if group is not None else 1


def cp_sum_with_grad(
    value: torch.Tensor, *, cp_group: dist.ProcessGroup | None
) -> torch.Tensor:
    """SUM across CP in both directions, including gradients from nonlocal owners.

    Every CP rank must execute this call and retain its output in the loss
    graph, even when its contribution is multiplied by zero. Unlike the
    identity-backward helper, this routes an owner's prefix gradient to other
    ranks that contributed to a cross-CP mismatch chain.
    """
    return group_all_reduce_sum_with_grad_backward_sum(value, cp_group)


class _SelectedLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        logits: torch.Tensor,
        batch: torch.Tensor,
        sequence: torch.Tensor,
        ids: torch.Tensor,
        vocab_start: int,
        real_vocab_size: int,
        temperature: float,
        tp_group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        squeeze = ids.ndim == 1
        if squeeze:
            ids = ids.unsqueeze(-1)
        rows = logits[batch, sequence].float() / temperature
        local_width = logits.shape[-1]
        valid_width = max(0, min(local_width, real_vocab_size - vocab_start))
        rows[:, valid_width:] = -torch.inf
        valid = (ids >= 0) & (ids < real_vocab_size)
        owned = valid & (ids >= vocab_start) & (ids < vocab_start + valid_width)
        local_ids = (ids - vocab_start).clamp(0, local_width - 1).long()
        selected = torch.where(owned, rows.gather(-1, local_ids), 0.0)
        maximum = rows.amax(-1, keepdim=True)
        if _group_size(tp_group) > 1:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)
        denominator = (rows - maximum).exp().sum(-1, keepdim=True)
        if _group_size(tp_group) > 1:
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=tp_group)
            dist.all_reduce(selected, op=dist.ReduceOp.SUM, group=tp_group)
        log_z = maximum + denominator.log()
        output = torch.where(valid, selected - log_z, -torch.inf)
        # Save only row scalars and request metadata, not an N x V softmax.
        ctx.save_for_backward(logits, batch, sequence, local_ids, valid, owned, log_z)
        # PyTorch creates this context before forward, outside __init__.
        ctx.temperature = temperature  # pyrefly: ignore[implicitly-defined-attribute]
        ctx.valid_width = valid_width  # pyrefly: ignore[implicitly-defined-attribute]
        ctx.squeeze = squeeze  # pyrefly: ignore[implicitly-defined-attribute]
        return output.squeeze(-1) if squeeze else output

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any, grad: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None]:
        logits, batch, sequence, local_ids, valid, owned, log_z = ctx.saved_tensors
        if ctx.squeeze:
            grad = grad.unsqueeze(-1)
        # Invalid support slots are constant -inf; their derivative is zero.
        grad = torch.where(valid, grad.float(), 0.0)
        rows = logits[batch, sequence].float() / ctx.temperature
        rows[:, ctx.valid_width :] = -torch.inf
        gradient = -(rows - log_z).exp() * grad.sum(-1, keepdim=True)
        gradient.scatter_add_(-1, local_ids, torch.where(owned, grad, 0.0))
        gradient /= ctx.temperature
        result = torch.zeros(logits.shape, device=logits.device, dtype=logits.dtype)
        result.view(-1, logits.shape[-1]).index_add_(
            0, batch * logits.shape[1] + sequence, gradient.to(logits.dtype)
        )
        return result, None, None, None, None, None, None, None


def distributed_selected_logprobs(
    vocab_parallel_logits: torch.Tensor,
    batch_indices: torch.Tensor,
    seq_indices: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    real_vocab_size: int,
    temperature: float,
    tp_group: dist.ProcessGroup | None = None,
    row_chunk_size: int | None = None,
) -> torch.Tensor:
    """Compute exact selected log-probs with local-vocabulary softmax gradients.

    Inputs select N rows and N or N-by-K global token IDs. Repeated rows and
    token IDs accumulate gradients. IDs outside the real vocabulary represent
    absent support and return constant -inf. Padded LM-head columns never
    enter normalization or gradients. TP ranks must request identical rows
    and IDs, including the same chunking; each CP group can request its own
    native rows. Temporary vocabulary storage is bounded by row_chunk_size.
    """
    if vocab_parallel_logits.ndim != 3 or not vocab_parallel_logits.is_floating_point():
        raise ValueError("Expected floating logits with shape [B, S, V_local]")
    width = vocab_parallel_logits.shape[-1]
    world = _group_size(tp_group)
    rank = dist.get_rank(tp_group) if tp_group is not None else 0
    if (
        width < 1
        or vocab_start_index != rank * width
        or vocab_end_index != vocab_start_index + width
    ):
        raise ValueError("Vocabulary interval must match the contiguous TP shard")
    if not 0 < real_vocab_size <= width * world:
        raise ValueError("Student TP width does not cover the real vocabulary")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")
    if batch_indices.ndim != 1 or seq_indices.shape != batch_indices.shape:
        raise ValueError("batch_indices and seq_indices must have matching [N] shapes")
    if token_ids.ndim not in (1, 2) or token_ids.shape[0] != batch_indices.shape[0]:
        raise ValueError("token_ids must have shape [N] or [N, K]")
    for indices in (batch_indices, seq_indices, token_ids):
        if indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("Selected row and token indices must have integer dtype")
        if indices.device != vocab_parallel_logits.device:
            raise ValueError("Selected indices must be on the logits device")
    if bool(
        ((batch_indices < 0) | (batch_indices >= vocab_parallel_logits.shape[0])).any()
    ):
        raise ValueError("Selected batch index is out of range")
    if bool(
        ((seq_indices < 0) | (seq_indices >= vocab_parallel_logits.shape[1])).any()
    ):
        raise ValueError("Selected sequence index is out of range")
    n_rows = batch_indices.numel()
    chunk_size = (
        min(n_rows, 64)
        if row_chunk_size is None or row_chunk_size <= 0
        else row_chunk_size
    )
    # Run once for an empty set to preserve a zero, gradient-connected result.
    outputs = [
        _SelectedLogprobs.apply(
            vocab_parallel_logits,
            batch_indices[start : start + chunk_size],
            seq_indices[start : start + chunk_size],
            token_ids[start : start + chunk_size],
            vocab_start_index,
            real_vocab_size,
            temperature,
            tp_group,
        )
        for start in range(0, max(n_rows, 1), max(chunk_size, 1))
    ]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)
