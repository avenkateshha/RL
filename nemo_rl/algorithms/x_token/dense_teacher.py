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

"""Invocation-scoped dense teacher reads at native student global positions.

The dense producer keeps its existing rectangular or compact contiguous CP
export. This reader maps requested rows directly into that storage, assembling
only their vocabulary columns. It never reconstructs a teacher sequence.
"""

from dataclasses import dataclass
from typing import Any, cast

import torch

from nemo_rl.utils.reusable_cuda_ipc import (
    ReusableCudaIPCDescriptor,
    is_reusable_cuda_ipc_handle,
    open_reusable_cuda_ipc,
)


@dataclass(frozen=True)
class DenseTeacherIPC:
    """Tensor-free descriptor retaining the established per-sample wire format."""

    samples: list[dict[str, Any]]


def supports_native_dense_reads(payload: DenseTeacherIPC) -> bool:
    """Require reusable handles for repeated native reads; keep old IPC legacy.

    PyTorch's ordinary reduction tuple represents one receiver reference, so it
    cannot safely be reopened by independent student ranks/microbatches. Native
    teacher exports carry controller-owned reusable descriptors instead. CPU
    tensors are accepted solely for storage-geometry fixtures.
    """
    if not payload.samples:
        raise ValueError("Dense teacher IPC payload is empty")
    protocols: set[str] = set()
    for sample in payload.samples:
        records = sample.get("teacher_shards")
        if not isinstance(records, list) or not records:
            raise ValueError(
                "Dense teacher sample must contain nonempty teacher_shards"
            )
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Dense teacher shard records must be dictionaries")
            handle = record.get("payload_ipc")
            if is_reusable_cuda_ipc_handle(handle) or (
                isinstance(handle, torch.Tensor) and handle.device.type == "cpu"
            ):
                protocols.add("reusable")
            elif (
                handle is None
                and record.get("ipc_layout") == "flat_valid_prefix_v1"
                and record.get("stored_seq_len") == 0
            ):
                # A compact empty suffix is logically covered but has no storage.
                continue
            elif isinstance(handle, tuple):
                protocols.add("torch")
            else:
                raise ValueError(
                    "Dense teacher has an unsupported IPC backing protocol"
                )
    if len(protocols) > 1:
        raise ValueError("Dense teacher payload mixes reusable and one-use IPC handles")
    return "torch" not in protocols


@dataclass(frozen=True)
class _DenseShard:
    record: dict[str, Any]
    sequence_start: int
    sequence_length: int
    vocab_start: int
    vocab_end: int
    compact_geometry: tuple[int, int, int, int] | None


class DenseTeacherRowReader:
    """Open teacher-specific backings lazily for one loss invocation.

    The producer owns the storage until the consuming backward/evaluation has
    completed. Readers and opened-storage caches must not survive across calls.
    Tensor-backed records support CPU fixtures with the same geometry checks.
    """

    def __init__(
        self, payload: DenseTeacherIPC, *, device: torch.device | str | int
    ) -> None:
        # Policy imports optional model/Ray dependencies; plain module imports
        # must remain usable without loading that runtime.
        from nemo_rl.models.policy.utils import (
            _validate_dense_ipc_shard_coverage,
            validate_compact_teacher_ipc_handle,
        )

        if not payload.samples:
            raise ValueError("Dense teacher IPC payload is empty")
        if not supports_native_dense_reads(payload):
            raise ValueError(
                "Native dense reads require reusable IPC handles, not one-use PyTorch reductions"
            )
        self.payload = payload
        # torch.device's stub declares __get__, producing false descriptor
        # inference for ordinary instance fields in pyrefly 0.24.
        self._consumer_device: torch.device = (  # pyrefly: ignore[read-only]
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        self._device_id = (
            (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            if self.device.type == "cuda"
            else None
        )
        self._backings: dict[tuple[Any, ...], torch.Tensor] = {}
        self._shards: list[list[_DenseShard]] = []
        expected: tuple[int, int] | None = None
        for sample_index, sample in enumerate(payload.samples):
            records = sample.get("teacher_shards")
            if not isinstance(records, list) or not records:
                raise ValueError(
                    "Dense teacher sample must contain nonempty teacher_shards"
                )
            if any(not isinstance(record, dict) for record in records):
                raise ValueError("Dense teacher shard records must be dictionaries")
            identities = {record.get("batch_item_id") for record in records}
            if len(identities) != 1 or (
                "batch_item_id" in sample and identities != {sample["batch_item_id"]}
            ):
                raise ValueError("Dense teacher shards disagree on sample identity")
            _validate_dense_ipc_shard_coverage(sample_index, records)
            shape = (
                int(records[0]["full_seq_len"]),
                int(records[0]["full_vocab_size"]),
            )
            if expected is not None and shape != expected:
                raise ValueError(
                    "Dense teacher samples disagree on sequence/vocabulary shape"
                )
            expected = shape
            shards = []
            for record in records:
                compact = validate_compact_teacher_ipc_handle(record)
                if compact is None and (
                    "payload_ipc" not in record
                    or int(record.get("buf_idx", -1)) < 0
                    or int(record.get("sample_index_in_buf", -1)) < 0
                ):
                    raise ValueError(
                        "Dense teacher rectangular record has invalid storage slots"
                    )
                shards.append(
                    _DenseShard(
                        record=record,
                        sequence_start=int(record["global_seq_start"]),
                        sequence_length=int(record["actual_shape"][0]),
                        vocab_start=int(record["vocab_start_index"]),
                        vocab_end=int(record["vocab_end_index"]),
                        compact_geometry=compact,
                    )
                )
            self._shards.append(shards)
        assert expected is not None
        self.full_seq_len, self.full_vocab_size = expected

    @property
    def device(self) -> torch.device:
        """Device receiving the requested row copies."""
        return self._consumer_device

    def _open(self, shard: _DenseShard) -> torch.Tensor:
        record = shard.record
        handle = record["payload_ipc"]
        key = (
            ("tensor", id(handle))
            if isinstance(handle, torch.Tensor)
            else ("ipc", handle)
        )
        source = self._backings.get(key)
        if source is None:
            if isinstance(handle, torch.Tensor):
                source = handle.detach()
            else:
                if self._device_id is None:
                    raise ValueError("CUDA IPC backing requires a CUDA consumer device")
                source = open_reusable_cuda_ipc(
                    cast(ReusableCudaIPCDescriptor, handle), self._device_id
                ).detach()
            self._backings[key] = source
        if not source.is_floating_point():
            raise TypeError("Dense teacher logits backing must have floating dtype")
        if "dtype" in record and source.dtype != record["dtype"]:
            raise ValueError("Dense teacher backing dtype disagrees with metadata")
        if shard.compact_geometry is not None:
            if source.ndim != 2 or tuple(source.shape) != tuple(
                record["storage_shape"]
            ):
                raise ValueError(
                    "Compact dense teacher backing disagrees with storage_shape"
                )
            if not source.is_contiguous():
                raise ValueError("Compact dense teacher backing must be contiguous")
            offset, stored_length, _, _ = shard.compact_geometry
            return source.narrow(0, offset, stored_length)
        buf_idx = int(record["buf_idx"])
        sample_idx = int(record["sample_index_in_buf"])
        width = shard.vocab_end - shard.vocab_start
        if (
            source.ndim != 4
            or buf_idx >= source.shape[0]
            or sample_idx >= source.shape[1]
            or shard.sequence_length > source.shape[2]
            or width > source.shape[3]
        ):
            raise ValueError("Dense teacher rectangular slot/shape exceeds its backing")
        return source[buf_idx, sample_idx, : shard.sequence_length, :width]

    def gather_rows(
        self,
        batch_indices: torch.Tensor,
        global_positions: torch.Tensor,
        *,
        vocab_start: int = 0,
        vocab_end: int | None = None,
    ) -> torch.Tensor:
        """Copy requested ``[N, V]`` rows, preserving request order/repetitions.

        ``vocab_start/end`` optionally limit copies to real vocabulary columns.
        Rectangular storage strides are honored by indexing; compact omitted
        suffixes are known zeros and do not require an opened backing.
        """
        for index in (batch_indices, global_positions):
            if index.ndim != 1 or index.shape != batch_indices.shape:
                raise ValueError("Dense teacher requests must have matching [N] shapes")
            if index.dtype not in (torch.int32, torch.int64):
                raise TypeError("Dense teacher row indices must have integer dtype")
        vocab_end = self.full_vocab_size if vocab_end is None else vocab_end
        if not 0 <= vocab_start < vocab_end <= self.full_vocab_size:
            raise ValueError(
                "Dense teacher requested vocabulary interval is out of range"
            )
        batches = batch_indices.to(device=self.device, dtype=torch.long)
        positions = global_positions.to(device=self.device, dtype=torch.long)
        if bool(((batches < 0) | (batches >= len(self._shards))).any()) or bool(
            ((positions < 0) | (positions >= self.full_seq_len)).any()
        ):
            raise ValueError("Dense teacher requested sample/position is out of range")
        result = torch.zeros(
            (batches.numel(), vocab_end - vocab_start),
            device=self.device,
            dtype=torch.float32,
        )
        for sample_index, shards in enumerate(self._shards):
            for shard in shards:
                first = max(vocab_start, shard.vocab_start)
                last = min(vocab_end, shard.vocab_end)
                if first >= last:
                    continue
                stored_length = (
                    shard.sequence_length
                    if shard.compact_geometry is None
                    else shard.compact_geometry[1]
                )
                selected = torch.where(
                    (batches == sample_index)
                    & (positions >= shard.sequence_start)
                    & (positions < shard.sequence_start + stored_length)
                )[0]
                if selected.numel() == 0:
                    continue
                source = self._open(shard)
                rows = (positions.index_select(0, selected) - shard.sequence_start).to(
                    source.device
                )
                # Bound transient remote copies independently of microbatch size.
                for begin in range(0, selected.numel(), 64):
                    end = begin + 64
                    values = source[
                        :, first - shard.vocab_start : last - shard.vocab_start
                    ].index_select(0, rows[begin:end])
                    result[
                        selected[begin:end], first - vocab_start : last - vocab_start
                    ] = values.to(device=self.device, dtype=torch.float32)
        return result

    def gather_native_positions(
        self,
        global_positions: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        vocab_end: int | None = None,
    ) -> torch.Tensor:
        """Read native predictor rows, allowing only inactive student padding.

        Student/teacher padded lengths may differ. Any requested predictor past
        the teacher axis must be masked out; an active missing row is an error.
        """
        if global_positions.ndim != 1 or global_positions.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "Native positions must be a one-dimensional integer tensor"
            )
        shape = (len(self._shards), global_positions.numel())
        if tuple(active_mask.shape) != shape:
            raise ValueError(
                "Native dense active mask must match sample/position shape"
            )
        positions = global_positions.to(device=self.device, dtype=torch.long)
        if bool((positions < 0).any()):
            raise ValueError("Native dense positions cannot be negative")
        covered = positions < self.full_seq_len
        if bool((active_mask.to(self.device).bool() & ~covered.unsqueeze(0)).any()):
            raise ValueError("Active native predictor is outside the teacher sequence")
        width = self.full_vocab_size if vocab_end is None else vocab_end
        if not 0 < width <= self.full_vocab_size:
            raise ValueError("Native dense vocabulary width is out of range")
        result = torch.zeros((*shape, width), device=self.device, dtype=torch.float32)
        batches = (
            torch.arange(shape[0], device=self.device)
            .unsqueeze(1)
            .expand(-1, int(covered.sum()))
        )
        rows = positions[covered].unsqueeze(0).expand(shape[0], -1)
        result[:, covered] = self.gather_rows(
            batches.reshape(-1), rows.reshape(-1), vocab_end=width
        ).reshape(shape[0], -1, width)
        return result
