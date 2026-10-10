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

"""Native sparse teacher scoring without gathering full vocabulary shards."""

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class SparseTopKOutput:
    """Natural K support and independent forced-label sidecars for one microbatch."""

    topk_logits: torch.Tensor
    topk_indices: torch.Tensor
    log_z: torch.Tensor
    natural_tail_indices: torch.Tensor
    forced_logits: torch.Tensor
    forced_indices: torch.Tensor
    forced_in_topk: torch.Tensor


def _canonical_topk(
    values: torch.Tensor, ids: torch.Tensor, k: int, *, id_ceiling: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select descending scores, resolving exact ties by ascending global ID.

    FP32 bits and a bounded ID tie-break form a monotonic signed-int64 key.
    Unlike an epsilon perturbation, this preserves adjacent finite FP32 scores
    and BF16/FP16 ties exactly. Adapted from pinned source 5acd78e948d2.
    """
    if not 0 <= id_ceiling < (1 << 31):
        raise ValueError("Sparse vocabulary IDs must fit signed int32")
    scores, ids = torch.broadcast_tensors(values.float(), ids.long())
    if bool(torch.isnan(scores).any()):
        raise ValueError("Sparse top-k does not accept NaN logits")
    scores = torch.where(scores.eq(0), torch.zeros_like(scores), scores).contiguous()
    bits = torch.bitwise_and(scores.view(torch.int32).long(), 0xFFFFFFFF)
    ordered = torch.where(
        torch.bitwise_and(bits, 0x80000000) != 0,
        torch.bitwise_and(torch.bitwise_not(bits), 0xFFFFFFFF),
        torch.bitwise_xor(bits, 0x80000000),
    )
    keys = (ordered << max(1, id_ceiling.bit_length())) + id_ceiling - ids
    selected = torch.topk(keys, k, dim=-1, sorted=True).indices
    return scores.gather(-1, selected), ids.gather(-1, selected)


@torch.no_grad()
def distributed_vocab_topk_logz_force(
    vocab_parallel_logits: torch.Tensor,
    force_token_ids: torch.Tensor,
    k: int,
    tp_group: dist.ProcessGroup | None,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    real_vocab_size: int,
    temperature: float,
    noise_filter_k: int = 0,
    chunk_size: int | None = None,
) -> SparseTopKOutput:
    """Export per-predictor natural top-K, exact logZ, and raw forced scores.

    Candidate exchange follows ``model_utils.distributed_vocab_topk``. Native
    transport additionally requires real-vocabulary exclusion, deterministic
    cutoff ties, exact temperature-scaled logZ, and fixed-width candidates on
    TP shards with fewer than K real columns. Natural K is sorted by token ID;
    its score-rank-K token is carried separately for per-request eviction.
    Forced membership is measured on the original noise-filter support (or K)
    and is unaffected by later insertion. Inputs retain native CP row order.
    """
    if vocab_parallel_logits.ndim != 3 or not vocab_parallel_logits.is_floating_point():
        raise ValueError("Expected floating logits with shape [B, S, V_local]")
    if (
        force_token_ids.ndim != 3
        or force_token_ids.shape[:2] != vocab_parallel_logits.shape[:2]
        or force_token_ids.shape[-1] < 1
    ):
        raise ValueError(
            "force_token_ids must have matching shape [B, S, F] with F > 0"
        )
    if force_token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("force_token_ids must have integer dtype")
    if force_token_ids.device != vocab_parallel_logits.device:
        raise ValueError("Forced IDs must be on the logits device")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")
    if k <= 0 or noise_filter_k < 0 or max(k, noise_filter_k) > real_vocab_size:
        raise ValueError(
            "Require positive K, nonnegative noise_filter_k and support <= real_vocab_size"
        )
    if bool(((force_token_ids < -1) | (force_token_ids >= real_vocab_size)).any()):
        raise ValueError("Forced labels must be -1 or real-vocabulary IDs")
    for slot in range(force_token_ids.shape[-1]):
        duplicate = (
            force_token_ids[..., slot : slot + 1] == force_token_ids[..., slot + 1 :]
        ) & (force_token_ids[..., slot : slot + 1] >= 0)
        if bool(duplicate.any()):
            raise ValueError("Forced sidecar contains duplicate valid labels")
    batch, sequence, width = vocab_parallel_logits.shape
    world = dist.get_world_size(tp_group) if tp_group is not None else 1
    rank = dist.get_rank(tp_group) if tp_group is not None else 0
    if (
        width < 1
        or vocab_start_index != rank * width
        or vocab_end_index != vocab_start_index + width
        or real_vocab_size > width * world
    ):
        raise ValueError(
            "Vocabulary interval/TP width does not cover the real vocabulary"
        )
    real_width = max(0, min(width, real_vocab_size - vocab_start_index))
    support_width = max(k, noise_filter_k)
    membership_width = noise_filter_k or k
    device = vocab_parallel_logits.device
    output = SparseTopKOutput(
        topk_logits=torch.empty(
            (batch, sequence, k), device=device, dtype=torch.float32
        ),
        topk_indices=torch.empty(
            (batch, sequence, k), device=device, dtype=torch.int32
        ),
        log_z=torch.empty((batch, sequence), device=device, dtype=torch.float32),
        natural_tail_indices=torch.empty(
            (batch, sequence, 1), device=device, dtype=torch.int32
        ),
        forced_logits=torch.empty(
            force_token_ids.shape, device=device, dtype=torch.float32
        ),
        forced_indices=force_token_ids.to(torch.int32),
        forced_in_topk=torch.empty(
            force_token_ids.shape, device=device, dtype=torch.bool
        ),
    )
    tile = max(
        1, min(sequence, 64) if chunk_size is None or chunk_size <= 0 else chunk_size
    )
    for start in range(0, sequence, tile):
        stop = min(sequence, start + tile)
        raw = vocab_parallel_logits[:, start:stop].float()
        scaled = raw[..., :real_width] / temperature
        maximum = (
            scaled.amax(-1)
            if real_width
            else torch.full(raw.shape[:2], -torch.inf, device=device)
        )
        if world > 1:
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)
        denominator = (scaled - maximum.unsqueeze(-1)).exp().sum(-1)
        if world > 1:
            dist.all_reduce(denominator, op=dist.ReduceOp.SUM, group=tp_group)
        output.log_z[:, start:stop] = maximum + denominator.log()

        candidates = torch.full(
            (*raw.shape[:2], support_width), -torch.inf, device=device
        )
        candidate_ids = torch.full(
            candidates.shape, real_vocab_size, device=device, dtype=torch.long
        )
        local_k = min(support_width, real_width)
        if local_k:
            ids = torch.arange(
                vocab_start_index, vocab_start_index + real_width, device=device
            )
            values, ids = _canonical_topk(
                raw[..., :real_width], ids, local_k, id_ceiling=real_vocab_size
            )
            candidates[..., :local_k], candidate_ids[..., :local_k] = values, ids
        if world > 1:
            values_by_rank = [torch.empty_like(candidates) for _ in range(world)]
            ids_by_rank = [torch.empty_like(candidate_ids) for _ in range(world)]
            dist.all_gather(values_by_rank, candidates, group=tp_group)
            dist.all_gather(ids_by_rank, candidate_ids, group=tp_group)
            candidates, candidate_ids = (
                torch.cat(values_by_rank, -1),
                torch.cat(ids_by_rank, -1),
            )
        natural_values, natural_ids = _canonical_topk(
            candidates, candidate_ids, support_width, id_ceiling=real_vocab_size
        )
        if bool((natural_ids >= real_vocab_size).any()):
            raise RuntimeError("Sparse top-k selected a padded vocabulary column")
        force = force_token_ids[:, start:stop].long()
        valid = force >= 0
        output.forced_in_topk[:, start:stop] = valid & (
            natural_ids[..., :membership_width].unsqueeze(-2) == force.unsqueeze(-1)
        ).any(-1)
        owned = (
            valid
            & (force >= vocab_start_index)
            & (force < vocab_start_index + real_width)
        )
        selected = raw.gather(-1, (force - vocab_start_index).clamp(0, width - 1))
        selected = torch.where(owned, selected, 0.0)
        if world > 1:
            dist.all_reduce(selected, op=dist.ReduceOp.SUM, group=tp_group)
        output.forced_logits[:, start:stop] = selected
        output.natural_tail_indices[:, start:stop] = natural_ids[..., k - 1 : k]
        sorted_ids, order = natural_ids[..., :k].sort(-1)
        output.topk_indices[:, start:stop] = sorted_ids
        output.topk_logits[:, start:stop] = natural_values[..., :k].gather(-1, order)
    return output
