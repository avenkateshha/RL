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

"""Typed native sparse IPC records and invocation-scoped requested-row reads.

The wire retains the existing outer per-sample dictionaries for data slicing.
Only native ``sparse_topk_v2`` records enter this reader; the existing DTensor
sparse reconstruction remains a separate protocol. Adapted from 5acd78e948d2.
"""

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

import torch

from nemo_rl.distributed.selected_logprobs import cp_native_global_positions

SparseBacking = torch.Tensor | tuple[Any, ...]


@dataclass(frozen=True)
class SparseTeacherShard:
    """A sample's storage slot in one native teacher CP shard.

    Each backing has shape [buffer_slots, samples_per_slot, local_sequence,
    field_width]. The seven independent handles remain owned by the producer
    until student backward or evaluation completes.
    """

    k: int
    temperature: float
    real_vocab_size: int
    membership_k: int
    force_width: int
    full_seq_len: int
    local_seq_len: int
    buf_idx: int
    sample_index_in_buf: int
    sequence_segments: tuple[tuple[int, int, int], ...]
    topk_logits_ipc: SparseBacking
    topk_indices_ipc: SparseBacking
    log_z_ipc: SparseBacking
    natural_tail_indices_ipc: SparseBacking
    forced_logits_ipc: SparseBacking
    forced_indices_ipc: SparseBacking
    forced_in_topk_ipc: SparseBacking

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "SparseTeacherShard":
        """Parse the explicit serialization boundary, rejecting malformed shapes."""
        if record.get("transport") != "sparse_topk_v2":
            raise ValueError("Native sparse teacher reader requires sparse_topk_v2")
        try:
            shape = tuple(record["topk_shape"])
            forced_shape = tuple(record["forced_shape"])
            scalar_shape = tuple(record["scalar_shape"])
            shard = cls(
                k=int(record["k"]),
                temperature=float(record["temperature"]),
                real_vocab_size=int(record["real_vocab_size"]),
                membership_k=int(record["forced_in_topk_k"]),
                force_width=int(record["force_width"]),
                full_seq_len=int(record["full_seq_len"]),
                local_seq_len=int(shape[0]),
                buf_idx=int(record["buf_idx"]),
                sample_index_in_buf=int(record["sample_index_in_buf"]),
                sequence_segments=tuple(
                    tuple(segment) for segment in record["sequence_segments"]
                ),
                topk_logits_ipc=record["topk_logits_ipc"],
                topk_indices_ipc=record["topk_indices_ipc"],
                log_z_ipc=record["log_z_ipc"],
                natural_tail_indices_ipc=record["natural_tail_indices_ipc"],
                forced_logits_ipc=record["forced_logits_ipc"],
                forced_indices_ipc=record["forced_indices_ipc"],
                forced_in_topk_ipc=record["forced_in_topk_ipc"],
            )
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("Malformed native sparse teacher record") from error
        if shape != (shard.local_seq_len, shard.k) or forced_shape != (
            shard.local_seq_len,
            shard.force_width,
        ):
            raise ValueError("Sparse teacher backing shapes disagree with metadata")
        if scalar_shape != (shard.local_seq_len,):
            raise ValueError(
                "Sparse teacher scalar_shape disagrees with local sequence"
            )
        if (
            not 0 < shard.k <= shard.real_vocab_size
            or not 0 < shard.membership_k <= shard.real_vocab_size
        ):
            raise ValueError(
                "Sparse teacher K/membership support exceeds real vocabulary"
            )
        if not math.isfinite(shard.temperature) or shard.temperature <= 0:
            raise ValueError("Sparse teacher temperature must be finite and positive")
        if (
            shard.force_width != 2
            or min(shard.local_seq_len, shard.full_seq_len) <= 0
            or min(shard.buf_idx, shard.sample_index_in_buf) < 0
        ):
            raise ValueError(
                "Invalid sparse teacher sequence/slot or two-label sidecar width"
            )
        return shard

    def to_record(self) -> dict[str, Any]:
        """Serialize explicitly without copying backing tensors or CUDA handles."""
        return {
            "transport": "sparse_topk_v2",
            "k": self.k,
            "temperature": self.temperature,
            "real_vocab_size": self.real_vocab_size,
            "forced_in_topk_k": self.membership_k,
            "force_width": self.force_width,
            "full_seq_len": self.full_seq_len,
            "buf_idx": self.buf_idx,
            "sample_index_in_buf": self.sample_index_in_buf,
            "topk_shape": (self.local_seq_len, self.k),
            "scalar_shape": (self.local_seq_len,),
            "forced_shape": (self.local_seq_len, self.force_width),
            "sequence_segments": list(self.sequence_segments),
            "topk_logits_ipc": self.topk_logits_ipc,
            "topk_indices_ipc": self.topk_indices_ipc,
            "log_z_ipc": self.log_z_ipc,
            "natural_tail_indices_ipc": self.natural_tail_indices_ipc,
            "forced_logits_ipc": self.forced_logits_ipc,
            "forced_indices_ipc": self.forced_indices_ipc,
            "forced_in_topk_ipc": self.forced_in_topk_ipc,
        }


@dataclass(frozen=True)
class SparseTeacherIPC:
    """One teacher's microbatch-sliced, outer per-sample IPC wire records."""

    samples: list[dict[str, Any]]


@dataclass(frozen=True)
class _Route:
    global_start: int
    global_end: int
    local_start: int
    shard: SparseTeacherShard


class SparseTeacherRowReader:
    """Route requested rows without reconstructing a complete teacher sequence.

    Construct one instance per teacher per loss invocation. This object's
    cache is never shared with another teacher or invocation. Returned rows
    are copies, and may remain held by autograd after this reader is dropped;
    neither event authorizes release of producer buffers before consumption.
    """

    def __init__(
        self, payload: SparseTeacherIPC, *, device: torch.device | int
    ) -> None:
        if not isinstance(payload, SparseTeacherIPC):
            raise TypeError("Native sparse teacher payload must be SparseTeacherIPC")
        if not payload.samples:
            raise ValueError("Sparse teacher payload contains no samples")
        self.payload = payload
        # Torch's device stub declares __get__, causing erroneous descriptor
        # inference for this ordinary instance field in pyrefly 0.24.
        self._consumer_device: torch.device = (  # pyrefly: ignore[read-only]
            torch.device("cuda", device) if isinstance(device, int) else device
        )
        self._device_id = (
            (
                torch.cuda.current_device()
                if self._consumer_device.index is None
                else self._consumer_device.index
            )
            if self._consumer_device.type == "cuda"
            else None
        )
        self._backing_tensors: dict[tuple[Any, ...], torch.Tensor] = {}
        self._routes: list[list[_Route]] = []
        self._starts: list[list[int]] = []
        expected: tuple[int, float, int, int, int, int] | None = None
        for entry in payload.samples:
            records = entry.get("teacher_shards", [entry])
            if not isinstance(records, (list, tuple)) or not records:
                raise ValueError("Sparse teacher sample contains no valid shards")
            routes: list[_Route] = []
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("Sparse teacher shard record must be a dictionary")
                shard = SparseTeacherShard.from_record(record)
                metadata = (
                    shard.k,
                    shard.temperature,
                    shard.real_vocab_size,
                    shard.membership_k,
                    shard.force_width,
                    shard.full_seq_len,
                )
                if expected is None:
                    expected = metadata
                elif metadata != expected:
                    raise ValueError(
                        "Sparse teacher shards disagree on K/temperature/vocabulary/membership/sequence"
                    )
                local_intervals = []
                for segment in shard.sequence_segments:
                    if len(segment) != 3 or any(
                        not isinstance(value, int) for value in segment
                    ):
                        raise ValueError(
                            "Sparse teacher sequence segments must be integer triples"
                        )
                    local, global_start, length = segment
                    if (
                        min(local, global_start) < 0
                        or length <= 0
                        or local + length > shard.local_seq_len
                        or global_start + length > shard.full_seq_len
                    ):
                        raise ValueError(
                            "Sparse teacher segment exceeds declared sequence coverage"
                        )
                    local_intervals.append((local, local + length))
                    routes.append(
                        _Route(global_start, global_start + length, local, shard)
                    )
                self._check_cover(sorted(local_intervals), shard.local_seq_len)
            routes.sort(key=lambda route: route.global_start)
            assert expected is not None
            self._check_cover(
                [(route.global_start, route.global_end) for route in routes],
                expected[-1],
            )
            self._routes.append(routes)
            self._starts.append([route.global_start for route in routes])
        assert expected is not None
        (
            self.k,
            self.temperature,
            self.real_vocab_size,
            self.membership_k,
            self.force_width,
            self.full_seq_len,
        ) = expected
        self.transport = "sparse_topk_v2"

    @property
    def device(self) -> torch.device:
        """Device receiving the requested row copies."""
        return self._consumer_device

    @staticmethod
    def _check_cover(intervals: list[tuple[int, int]], length: int) -> None:
        cursor = 0
        for start, end in intervals:
            if start != cursor:
                raise ValueError(
                    "Sparse teacher segments must cover each row exactly once"
                )
            cursor = end
        if cursor != length:
            raise ValueError(
                "Sparse teacher segments must cover the complete declared sequence"
            )

    def validate_contract(
        self,
        *,
        k: int,
        temperature: float,
        real_vocab_size: int,
        membership_k: int,
        sample_count: int,
    ) -> None:
        """Reject exporting/consuming a different objective or sample grouping."""
        actual = (
            self.k,
            self.temperature,
            self.real_vocab_size,
            self.membership_k,
            len(self.payload.samples),
        )
        if actual != (k, temperature, real_vocab_size, membership_k, sample_count):
            raise ValueError(
                f"Sparse teacher K/temperature/vocabulary/membership/sample contract mismatch: got {actual}"
            )

    def _backing(
        self, value: SparseBacking, *, field: str, shard: SparseTeacherShard, width: int
    ) -> torch.Tensor:
        key = (
            (field, "tensor", id(value))
            if isinstance(value, torch.Tensor)
            else (field, "ipc", value)
        )
        tensor = self._backing_tensors.get(key)
        if tensor is None:
            if isinstance(value, torch.Tensor):
                # Keep remote tensor-backed fixtures on their source device;
                # only requested rows, never the whole backing, are copied.
                tensor = value.detach()
            else:
                if self._device_id is None:
                    raise ValueError("CUDA IPC backing requires a CUDA consumer device")
                # Policy imports Ray; pure tensor readers must not load it.
                from nemo_rl.models.policy.utils import rebuild_cuda_tensor_from_ipc

                tensor = rebuild_cuda_tensor_from_ipc(value, self._device_id).detach()
            self._backing_tensors[key] = tensor
        if (
            tensor.ndim != 4
            or tensor.shape[0] <= shard.buf_idx
            or tensor.shape[1] <= shard.sample_index_in_buf
            or tensor.shape[2] < shard.local_seq_len
            or tensor.shape[3] != width
        ):
            raise ValueError(
                f"Sparse teacher {field} backing does not cover its declared slot/shape"
            )
        if field.endswith("indices"):
            valid_dtype = tensor.dtype in (torch.int32, torch.int64)
        elif field == "forced_in_topk":
            valid_dtype = tensor.dtype == torch.bool
        else:
            valid_dtype = tensor.is_floating_point()
        if not valid_dtype:
            raise TypeError(
                f"Sparse teacher {field} backing has invalid dtype {tensor.dtype}"
            )
        return tensor

    def gather_rows(
        self,
        batch_indices: torch.Tensor,
        global_positions: torch.Tensor,
        required_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialize an independent natural-K-or-K-minus-one-plus-label support."""
        for indices in (batch_indices, global_positions, required_token_ids):
            if indices.ndim != 1 or indices.shape != batch_indices.shape:
                raise ValueError("Sparse row requests must have matching [N] shapes")
            if indices.dtype not in (torch.int32, torch.int64):
                raise TypeError("Sparse row request indices must be integers")
        required = required_token_ids.to(device=self._consumer_device, dtype=torch.long)
        if bool(((required < 0) | (required >= self.real_vocab_size)).any()):
            raise ValueError("Required tokens must be real-vocabulary IDs")
        count = batch_indices.numel()
        values = torch.empty(
            (count, self.k), device=self._consumer_device, dtype=torch.float32
        )
        ids = torch.empty(
            (count, self.k), device=self._consumer_device, dtype=torch.int32
        )
        log_z = torch.empty((count,), device=self._consumer_device, dtype=torch.float32)
        tails = torch.empty((count,), device=self._consumer_device, dtype=torch.int32)
        forced_values = torch.empty(
            (count, self.force_width), device=self._consumer_device, dtype=torch.float32
        )
        forced_ids = torch.empty(
            (count, self.force_width), device=self._consumer_device, dtype=torch.int32
        )
        forced_membership = torch.empty(
            (count, self.force_width), device=self._consumer_device, dtype=torch.bool
        )
        grouped: dict[int, tuple[SparseTeacherShard, list[int], list[int]]] = {}
        for destination, (batch, position) in enumerate(
            zip(
                batch_indices.cpu().tolist(),
                global_positions.cpu().tolist(),
                strict=True,
            )
        ):
            if (
                not 0 <= batch < len(self._routes)
                or not 0 <= position < self.full_seq_len
            ):
                raise ValueError(
                    "Sparse teacher requested sample/position is out of range"
                )
            route = self._routes[batch][bisect_right(self._starts[batch], position) - 1]
            group = grouped.setdefault(id(route.shard), (route.shard, [], []))
            group[1].append(destination)
            group[2].append(route.local_start + position - route.global_start)
        for shard, destinations, local_rows in grouped.values():
            destination_indices = torch.tensor(
                destinations, device=self._consumer_device
            )
            fields = (
                ("topk_logits", shard.topk_logits_ipc, values, self.k),
                ("topk_indices", shard.topk_indices_ipc, ids, self.k),
                ("log_z", shard.log_z_ipc, log_z, 1),
                ("natural_tail_indices", shard.natural_tail_indices_ipc, tails, 1),
                (
                    "forced_logits",
                    shard.forced_logits_ipc,
                    forced_values,
                    self.force_width,
                ),
                (
                    "forced_indices",
                    shard.forced_indices_ipc,
                    forced_ids,
                    self.force_width,
                ),
                (
                    "forced_in_topk",
                    shard.forced_in_topk_ipc,
                    forced_membership,
                    self.force_width,
                ),
            )
            for field, backing, destination_tensor, width in fields:
                source = self._backing(backing, field=field, shard=shard, width=width)
                row_indices = torch.tensor(local_rows, device=source.device)
                selected = source[
                    shard.buf_idx, shard.sample_index_in_buf
                ].index_select(0, row_indices)
                if destination_tensor.ndim == 1:
                    selected = selected[:, 0]
                destination_tensor.index_copy_(
                    0,
                    destination_indices,
                    selected.to(
                        device=self._consumer_device, dtype=destination_tensor.dtype
                    ),
                )
        if bool(((ids < 0) | (ids >= self.real_vocab_size)).any()) or bool(
            (ids[:, 1:] <= ids[:, :-1]).any()
        ):
            raise ValueError(
                "Sparse teacher natural IDs must be valid and strictly sorted"
            )
        if bool(
            ((forced_ids < -1) | (forced_ids >= self.real_vocab_size)).any()
        ) or bool(
            ((forced_ids[:, 0] == forced_ids[:, 1]) & (forced_ids[:, 0] >= 0)).any()
        ):
            raise ValueError(
                "Sparse teacher forced sidecar contains invalid or duplicate labels"
            )
        matches = forced_ids == required.unsqueeze(-1)
        if bool((matches.sum(-1) != 1).any()):
            raise ValueError(
                "Required label must appear exactly once in the forced sidecar"
            )
        requested_values = torch.where(matches, forced_values, 0.0).sum(-1)
        membership = (matches & forced_membership).any(-1)
        natural_matches = ids == required.unsqueeze(-1)
        natural_values = torch.where(natural_matches, values, 0.0).sum(-1)
        if bool((natural_matches.any(-1) & (natural_values != requested_values)).any()):
            raise ValueError(
                "Natural and forced copies disagree on a requested raw logit"
            )
        tail_matches = ids == tails.unsqueeze(-1)
        if bool((tail_matches.sum(-1) != 1).any()):
            raise ValueError("Natural cutoff must occur exactly once in natural K")
        insert = ~natural_matches.any(-1)
        rows = insert.nonzero().flatten()
        slots = tail_matches.long().argmax(-1)[rows]
        ids[rows, slots] = required[rows].to(ids.dtype)
        values[rows, slots] = requested_values[rows]
        ids, order = ids.sort(-1)
        return values.gather(-1, order), ids, log_z, membership


def build_native_force_token_ids(
    teacher_input_ids: torch.Tensor,
    student_spans: torch.Tensor,
    teacher_spans: torch.Tensor,
    pair_valid: torch.Tensor,
    *,
    kl_chunk_shift: bool,
    teacher_real_vocab_size: int,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build both realized labels for conditional-shift collisions at a row.

    HEAD's derived absent spans are [0, 0); raw (-1, -1) sentinels are also
    accepted. Empty sides have no loss requests and leave unused slots at -1.
    """
    if (
        teacher_input_ids.ndim != 2
        or teacher_input_ids.dtype not in (torch.int32, torch.int64)
        or teacher_real_vocab_size <= 0
        or bool(
            (
                (teacher_input_ids < 0) | (teacher_input_ids >= teacher_real_vocab_size)
            ).any()
        )
    ):
        raise ValueError(
            "Teacher input IDs must be real-vocabulary integers with shape [B, T]"
        )
    if (
        student_spans.ndim != 3
        or student_spans.shape[-1] != 2
        or teacher_spans.shape != student_spans.shape
        or pair_valid.shape != student_spans.shape[:2]
        or student_spans.shape[0] != teacher_input_ids.shape[0]
    ):
        raise ValueError("Expected matching [B, C, 2] spans and [B, C] pair_valid")
    batch_size, sequence_length = teacher_input_ids.shape
    if sample_mask is not None and sample_mask.shape != (batch_size,):
        raise ValueError("sample_mask must have shape [B]")
    forced = teacher_input_ids.new_full((*teacher_input_ids.shape, 2), -1)
    for batch in range(batch_size):
        if sample_mask is not None and not bool(sample_mask[batch]):
            continue
        for chunk in range(pair_valid.shape[1]):
            if not bool(pair_valid[batch, chunk]):
                continue
            student_start, student_end = student_spans[batch, chunk].tolist()
            teacher_start, teacher_end = teacher_spans[batch, chunk].tolist()
            for start, end in (
                (student_start, student_end),
                (teacher_start, teacher_end),
            ):
                if (start, end) not in ((0, 0), (-1, -1)) and not 0 <= start < end:
                    raise ValueError(
                        "Malformed alignment span for forced teacher labels"
                    )
            if teacher_end > sequence_length:
                raise ValueError("Forced-label teacher span exceeds teacher sequence")
            if student_end <= student_start or teacher_end <= teacher_start:
                continue
            shift = int(kl_chunk_shift and student_start > 0 and teacher_start > 0)
            for label_position in range(teacher_start, teacher_end):
                predictor = label_position - shift
                label = teacher_input_ids[batch, label_position]
                row = forced[batch, predictor]
                if bool((row == label).any()):
                    continue
                free = (row == -1).nonzero().flatten()
                if not free.numel():
                    raise ValueError(
                        "Alignment requires more than two labels at a teacher predictor"
                    )
                row[free[0]] = label
    return forced


def shard_native_teacher_force_ids(
    force_token_ids: torch.Tensor,
    *,
    cp_rank: int,
    cp_size: int,
    local_sequence_length: int,
) -> torch.Tensor:
    """Shard [B,T_teacher,2] once using the producing teacher's CP geometry."""
    if force_token_ids.ndim != 3 or force_token_ids.shape[-1] != 2:
        raise ValueError("Native teacher forced labels require shape [B, T_teacher, 2]")
    positions = cp_native_global_positions(
        force_token_ids.shape[1], cp_rank, cp_size, device=force_token_ids.device
    )
    if positions.numel() != local_sequence_length:
        raise ValueError(
            "Native forced-label rows do not match the teacher output rows"
        )
    return force_token_ids.index_select(1, positions)
