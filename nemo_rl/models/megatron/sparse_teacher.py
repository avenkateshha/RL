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

"""Persistent streaming storage for a single Megatron teacher's sparse IPC."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from nemo_rl.algorithms.x_token.sparse_teacher import SparseBacking, SparseTeacherShard
from nemo_rl.distributed.selected_logprobs import cp_native_sequence_segments
from nemo_rl.distributed.sparse_topk import SparseTopKOutput
from nemo_rl.utils.reusable_cuda_ipc import allocate_reusable_cuda_tensor


@dataclass(frozen=True)
class SparseTeacherIPCHandles:
    """Seven handles tied to one allocation generation, not one microbatch."""

    topk_logits: SparseBacking
    topk_indices: SparseBacking
    log_z: SparseBacking
    natural_tail_indices: SparseBacking
    forced_logits: SparseBacking
    forced_indices: SparseBacking
    forced_in_topk: SparseBacking


@dataclass
class SparseTeacherStorage:
    """Teacher-owned [1, sample_capacity, local_sequence_capacity, width] fields."""

    topk_logits: torch.Tensor
    topk_indices: torch.Tensor
    log_z: torch.Tensor
    natural_tail_indices: torch.Tensor
    forced_logits: torch.Tensor
    forced_indices: torch.Tensor
    forced_in_topk: torch.Tensor
    handles: SparseTeacherIPCHandles | None = None

    @classmethod
    def allocate(
        cls,
        *,
        batch_size: int,
        local_sequence_length: int,
        k: int,
        device: torch.device,
    ) -> "SparseTeacherStorage":
        """Allocate once before the forward; no whole-step sparse staging list."""
        if min(batch_size, local_sequence_length, k) <= 0:
            raise ValueError("Sparse IPC storage dimensions must be positive")
        shape = (1, batch_size, local_sequence_length)
        allocate = (
            allocate_reusable_cuda_tensor if device.type == "cuda" else torch.empty
        )
        return cls(
            topk_logits=allocate((*shape, k), device=device, dtype=torch.float32),
            topk_indices=allocate((*shape, k), device=device, dtype=torch.int32),
            log_z=allocate((*shape, 1), device=device, dtype=torch.float32),
            natural_tail_indices=allocate(
                (*shape, 1), device=device, dtype=torch.int32
            ),
            forced_logits=allocate((*shape, 2), device=device, dtype=torch.float32),
            forced_indices=allocate((*shape, 2), device=device, dtype=torch.int32),
            forced_in_topk=allocate((*shape, 2), device=device, dtype=torch.bool),
        )

    def can_reuse(
        self,
        *,
        batch_size: int,
        local_sequence_length: int,
        k: int,
        device: torch.device,
    ) -> bool:
        """Allow shrink/reuse while preserving complete stored row capacities."""
        return (
            self.topk_logits.device == device
            and self.topk_logits.shape[1] >= batch_size
            and self.topk_logits.shape[2] >= local_sequence_length
            and self.topk_logits.shape[3] == k
        )

    def write(self, output: SparseTopKOutput, *, sample_offset: int) -> None:
        """Commit one microbatch directly to its producer-owned sample slots."""
        batch_size, sequence, k = output.topk_logits.shape
        if sample_offset < 0 or not self.can_reuse(
            batch_size=sample_offset + batch_size,
            local_sequence_length=sequence,
            k=k,
            device=output.topk_logits.device,
        ):
            raise ValueError("Sparse microbatch exceeds preallocated storage geometry")
        fields = (
            (self.topk_logits, output.topk_logits),
            (self.topk_indices, output.topk_indices),
            (self.log_z, output.log_z.unsqueeze(-1)),
            (self.natural_tail_indices, output.natural_tail_indices),
            (self.forced_logits, output.forced_logits),
            (self.forced_indices, output.forced_indices),
            (self.forced_in_topk, output.forced_in_topk),
        )
        # Validate every field before starting writes, so shape errors cannot
        # publish an internally inconsistent record.
        for destination, source in fields:
            expected = (batch_size, sequence, destination.shape[-1])
            if (
                source.shape != expected
                or source.dtype != destination.dtype
                or source.device != destination.device
            ):
                raise ValueError(
                    "Sparse microbatch field shape/dtype/device disagrees with storage"
                )
        for destination, source in fields:
            destination[0, sample_offset : sample_offset + batch_size, :sequence].copy_(
                source
            )

    def export_handles(
        self, handle_factory: Callable[[torch.Tensor], SparseBacking]
    ) -> SparseTeacherIPCHandles:
        """Reuse handles for the allocation's lifetime; callers synchronize writes."""
        if self.handles is None:
            self.handles = SparseTeacherIPCHandles(
                topk_logits=handle_factory(self.topk_logits),
                topk_indices=handle_factory(self.topk_indices),
                log_z=handle_factory(self.log_z),
                natural_tail_indices=handle_factory(self.natural_tail_indices),
                forced_logits=handle_factory(self.forced_logits),
                forced_indices=handle_factory(self.forced_indices),
                forced_in_topk=handle_factory(self.forced_in_topk),
            )
        return self.handles

    def sample_record(
        self,
        *,
        sample_index: int,
        full_sequence_length: int,
        cp_rank: int,
        cp_size: int,
        real_vocab_size: int,
        temperature: float,
        membership_k: int,
    ) -> dict[str, Any]:
        """Serialize this teacher's explicit storage offset and native segments."""
        if self.handles is None:
            raise RuntimeError(
                "Sparse IPC handles must be created before serialization"
            )
        return SparseTeacherShard(
            k=self.topk_logits.shape[-1],
            temperature=temperature,
            real_vocab_size=real_vocab_size,
            membership_k=membership_k,
            force_width=2,
            full_seq_len=full_sequence_length,
            local_seq_len=full_sequence_length // cp_size,
            buf_idx=0,
            sample_index_in_buf=sample_index,
            sequence_segments=tuple(
                cp_native_sequence_segments(full_sequence_length, cp_rank, cp_size)
            ),
            topk_logits_ipc=self.handles.topk_logits,
            topk_indices_ipc=self.handles.topk_indices,
            log_z_ipc=self.handles.log_z,
            natural_tail_indices_ipc=self.handles.natural_tail_indices,
            forced_logits_ipc=self.handles.forced_logits,
            forced_indices_ipc=self.handles.forced_indices,
            forced_in_topk_ipc=self.handles.forced_in_topk,
        ).to_record()


@dataclass(frozen=True)
class StreamedSparseLogitsMetadata:
    """Tensor-free schedule result after one microbatch has been written."""

    batch_size: int
    sample_offset: int
    full_sequence_length: int
    local_sequence_length: int
    local_vocab_size: int
