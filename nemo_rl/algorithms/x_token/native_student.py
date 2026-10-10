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
"""Student CE and accuracy on native Megatron context-parallel rows."""

import torch
import torch.distributed as dist

from nemo_rl.algorithms.x_token.loss_utils import NativeStudentContext
from nemo_rl.distributed.model_utils import group_all_reduce_sum
from nemo_rl.distributed.selected_logprobs import distributed_selected_logprobs


def native_next_token_ce(
    student: NativeStudentContext,
    global_valid_toks: torch.Tensor,
    *,
    tp_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Return this CP owner's CE contribution without sequence gathers or /CP.

    Only requested target probabilities are retained. The selected-log-prob
    primitive excludes LM-head padding and streams bounded FP32 vocab tiles in
    both forward and backward. An empty local mask keeps a zero gradient path.
    """
    if global_valid_toks.numel() != 1 or not bool(
        torch.isfinite(global_valid_toks).all() & (global_valid_toks >= 0).all()
    ):
        raise ValueError("Native CE requires a finite nonnegative scalar normalizer")
    mask = student.next_token_mask.float() * student.sample_mask.float().unsqueeze(-1)
    batch, sequence = torch.nonzero(mask != 0, as_tuple=True)
    width = student.logits.shape[-1]
    tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
    logprobs = distributed_selected_logprobs(
        student.logits,
        batch,
        sequence,
        student.next_token_ids[batch, sequence],
        vocab_start_index=tp_rank * width,
        vocab_end_index=(tp_rank + 1) * width,
        real_vocab_size=student.real_vocab_size,
        temperature=1.0,
        tp_group=tp_group,
    )
    return -(logprobs * mask[batch, sequence]).sum() / global_valid_toks.clamp(min=1)


@torch.no_grad()
def native_next_token_accuracy(
    student: NativeStudentContext,
    *,
    tp_group: dist.ProcessGroup | None,
    cp_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """CP-complete accuracy with deterministic global-ID ties and no padded IDs."""
    logits = student.logits
    width = logits.shape[-1]
    tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
    vocab_start = tp_rank * width
    valid_width = max(0, min(width, student.real_vocab_size - vocab_start))
    if valid_width:
        local_max, local_ids = logits[..., :valid_width].max(dim=-1)
        local_ids += vocab_start
    else:
        local_max = logits.new_full(logits.shape[:2], -torch.inf)
        local_ids = torch.full(
            logits.shape[:2],
            student.real_vocab_size,
            dtype=torch.long,
            device=logits.device,
        )
    global_max = local_max.clone()
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
    candidates = torch.where(
        local_max == global_max, local_ids, student.real_vocab_size
    )
    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        dist.all_reduce(candidates, op=dist.ReduceOp.MIN, group=tp_group)
    mask = student.next_token_mask.float() * student.sample_mask.float().unsqueeze(-1)
    correct = ((candidates == student.next_token_ids).float() * mask).sum()
    correct, count = group_all_reduce_sum(
        torch.stack((correct, mask.sum())), cp_group
    ).unbind()
    return correct / count.clamp(min=1)
