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

"""Owner-local v6 prefix-support loss on native MCore TP/CP logits.

The objective follows the pinned sparse v6 implementation: common-token and
bidirectional prefix supports retain global teacher K, a forced realized pair,
and REST. M-to-N chunks retain the historical realized-chain objective. Only
fixed realized student prefixes cross CP; selected final-token probabilities
and teacher IPC row reads stay on the final student predictor's owner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from nemo_rl.algorithms.x_token.loss_utils import (
    LocalizedAlignment,
    NativeStudentContext,
)
from nemo_rl.algorithms.x_token.sparse_teacher import SparseTeacherRowReader
from nemo_rl.distributed.selected_logprobs import (
    cp_sum_with_grad,
    distributed_selected_logprobs,
)

if TYPE_CHECKING:
    from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn

_ROW_BATCH_SIZE = 64


@dataclass(frozen=True)
class _Chunk:
    batch: int
    student_positions: tuple[int, ...]
    student_labels: tuple[int, ...]
    teacher_positions: tuple[int, ...]
    teacher_labels: tuple[int, ...]


def _alignment_chunks(
    student: NativeStudentContext, align: LocalizedAlignment, *, shift: bool
) -> tuple[list[_Chunk], list[_Chunk]]:
    if (
        align.student_spans is None
        or align.teacher_spans is None
        or align.teacher_input_ids is None
        or align.pair_valid is None
    ):
        raise ValueError("Native sparse loss requires full alignment spans and IDs")
    student_spans, teacher_spans = align.student_spans, align.teacher_spans
    batch_size = student.logits.shape[0]
    if (
        student_spans.ndim != 3
        or student_spans.shape[-1] != 2
        or teacher_spans.shape != student_spans.shape
        or align.pair_valid.shape != student_spans.shape[:2]
        or student_spans.shape[0] != batch_size
        or align.sample_mask.shape != (batch_size,)
        or align.teacher_input_ids.ndim != 2
        or align.teacher_input_ids.shape[0] != batch_size
    ):
        raise ValueError("Native sparse alignment geometry disagrees with the student")
    if not bool(((align.sample_mask == 0) | (align.sample_mask == 1)).all()):
        raise ValueError("Native sparse loss requires a binary sample mask")
    s_spans = student_spans.cpu().tolist()
    t_spans = teacher_spans.cpu().tolist()
    valid = align.pair_valid.cpu().tolist()
    samples = align.sample_mask.cpu().tolist()
    s_ids = student.input_ids.cpu().tolist()
    t_ids = align.teacher_input_ids.cpu().tolist()
    common, mismatch = [], []
    for batch in range(batch_size):
        if not samples[batch]:
            continue
        for index, active in enumerate(valid[batch]):
            if not active:
                continue
            s_start, s_end = s_spans[batch][index]
            t_start, t_end = t_spans[batch][index]
            for start, end, length in (
                (s_start, s_end, student.full_seq_len),
                (t_start, t_end, align.teacher_input_ids.shape[1]),
            ):
                if (start, end) not in ((0, 0), (-1, -1)) and not (
                    0 <= start < end <= length
                ):
                    raise ValueError("Native sparse alignment contains an invalid span")
            if s_end <= s_start or t_end <= t_start:
                continue
            offset = int(shift and s_start > 0 and t_start > 0)
            chunk = _Chunk(
                batch,
                tuple(range(s_start - offset, s_end - offset)),
                tuple(s_ids[batch][s_start:s_end]),
                tuple(range(t_start - offset, t_end - offset)),
                tuple(t_ids[batch][t_start:t_end]),
            )
            target = common if s_end - s_start == t_end - t_start == 1 else mismatch
            target.append(chunk)
    return common, mismatch


def compute_native_sparse_teacher_loss(
    loss_fn: CrossTokenizerDistillationLossFn,
    teacher_idx: int,
    student: NativeStudentContext,
    align: LocalizedAlignment,
    teacher_reader: SparseTeacherRowReader,
    *,
    global_valid_chunks: torch.Tensor,
    tp_group: dist.ProcessGroup | None,
    cp_group: dist.ProcessGroup | None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Return an owner-local loss and ``kl_loss``; other diagnostics are CP-complete.

    ``global_valid_chunks`` is the explicit full-step DP denominator, retained
    across student microbatches. One unconditional differentiable CP SUM has
    both forward-SUM and backward-SUM semantics, even on empty owners/teachers.
    Readers and requested rows belong exclusively to this teacher invocation.
    """
    # Import at call time: loss_functions dispatches here and owns the established
    # generalized-JSD numerical contract shared with the retained legacy path.
    from nemo_rl.algorithms.loss.loss_functions import _generalized_jsd

    logits, device, cfg = student.logits, student.logits.device, loss_fn.cfg
    temperature = float(loss_fn.temperature)
    k = int(cfg["teacher_topk_ipc_k"])
    noise_k = int(cfg.get("prefix_bidir_v3_noise_filter_topk", 0) or 0)
    common_kind = str(cfg.get("prefix_bidir_v3_loss_fn", "kl") or "kl")
    mismatch_kind = str(cfg.get("prefix_bidir_v3_last_pos_loss_fn") or common_kind)
    beta = float(cfg.get("prefix_bidir_v3_jsd_beta", 0.5))
    multiplier_value = cfg.get("prefix_bidir_v3_mismatch_loss_beta")
    if multiplier_value is None:
        multiplier_value = cfg.get("prefix_bidir_v3_mismatch_loss_scale")
    multiplier = float(1.0 if multiplier_value is None else multiplier_value)
    if (
        common_kind not in ("kl", "jsd")
        or mismatch_kind not in ("kl", "jsd")
        or not math.isfinite(multiplier)
        or multiplier < 0
        or not 0 <= beta <= 1
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Native sparse loss requires valid KL/JSD configuration")
    if (
        global_valid_chunks.numel() != 1
        or not bool(torch.isfinite(global_valid_chunks).all())
        or float(global_valid_chunks.item()) < 0
    ):
        raise ValueError(
            "Native sparse full-step chunk denominator must be finite and nonnegative"
        )
    if (
        logits.ndim != 3
        or student.input_ids.shape != (logits.shape[0], student.full_seq_len)
        or student.global_positions.shape != (logits.shape[1],)
        or student.real_vocab_size != loss_fn.student_vocab_size
    ):
        raise ValueError("Native sparse student geometry or real vocabulary disagrees")
    teacher_vocab = int(loss_fn.teacher_vocab_sizes[teacher_idx])
    teacher_reader.validate_contract(
        k=k,
        temperature=temperature,
        real_vocab_size=teacher_vocab,
        membership_k=noise_k or k,
        sample_count=logits.shape[0],
    )
    if align.teacher_input_ids is None:
        raise ValueError("Native sparse loss requires full teacher input IDs")
    if teacher_reader.full_seq_len != align.teacher_input_ids.shape[1]:
        raise ValueError("Native sparse reader and teacher sequence lengths disagree")
    for ids, vocab in (
        (student.input_ids, student.real_vocab_size),
        (align.teacher_input_ids, teacher_vocab),
    ):
        if ids.dtype not in (torch.int32, torch.int64) or bool(
            ((ids < 0) | (ids >= vocab)).any()
        ):
            raise ValueError("Native sparse input IDs must be in the real vocabulary")
    positions = student.global_positions.cpu().tolist()
    if len(set(positions)) != len(positions) or any(
        position < 0 or position >= student.full_seq_len for position in positions
    ):
        raise ValueError(
            "Native sparse student global positions must be unique and in range"
        )
    local_of_global = {position: index for index, position in enumerate(positions)}
    tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    vocab_start = tp_rank * logits.shape[-1]
    vocab_end = vocab_start + logits.shape[-1]
    if tp_size * logits.shape[-1] < student.real_vocab_size:
        raise ValueError(
            "Native sparse TP shards are narrower than the real vocabulary"
        )
    common, mismatches = _alignment_chunks(
        student, align, shift=bool(cfg.get("kl_chunk_shift", False))
    )
    common = [
        chunk for chunk in common if chunk.student_positions[-1] in local_of_global
    ]

    def longs(values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.long, device=device)

    def selected(batch, sequence, tokens) -> torch.Tensor:
        return distributed_selected_logprobs(
            logits,
            longs(batch),
            longs(sequence),
            longs(tokens),
            vocab_start_index=vocab_start,
            vocab_end_index=vocab_end,
            real_vocab_size=student.real_vocab_size,
            temperature=temperature,
            tp_group=tp_group,
            row_chunk_size=_ROW_BATCH_SIZE,
        ).float()

    def partition_sum(s_logp, t_logp, support, valid_rows, kind):
        s_logp = s_logp.masked_fill(~support, -1.0e30)
        t_logp = t_logp.masked_fill(~support, -1.0e30)
        s_partition = loss_fn._append_rest_bucket_logp(s_logp)
        t_partition = loss_fn._append_rest_bucket_logp(t_logp)
        if kind == "jsd":
            elements = _generalized_jsd(s_partition, t_partition, beta)
        else:
            source, target = (
                (t_partition, s_partition)
                if loss_fn.reverse_kl
                else (s_partition, t_partition)
            )
            elements = torch.nn.functional.kl_div(
                source, target, reduction="none", log_target=True
            )
        keep = torch.cat((support, torch.ones_like(support[:, :1])), dim=-1)
        total = ((elements * keep).sum(-1) * valid_rows).sum()
        matches = int(
            ((s_logp.argmax(-1) == t_logp.argmax(-1)) & valid_rows).sum().item()
        )
        return total, matches

    def support_tensors(student_rows, teacher_slots):
        width = max(1, max(map(len, student_rows)))
        s_ids = torch.zeros((len(student_rows), width), dtype=torch.long, device=device)
        slots = torch.zeros_like(s_ids)
        mask = torch.zeros_like(s_ids, dtype=torch.bool)
        for row, (ids, indices) in enumerate(
            zip(student_rows, teacher_slots, strict=True)
        ):
            s_ids[row, : len(ids)] = longs(ids)
            slots[row, : len(ids)] = longs(indices)
            mask[row, : len(ids)] = True
        if width > k:
            raise RuntimeError("Native sparse prefix support exceeded global K")
        return s_ids, slots, mask

    zero = logits.sum(dtype=torch.float32) * 0.0
    common_sum = zero
    common_count = common_matches = filtered_common = 0
    if common:
        common_s, common_t = loss_fn._get_common_indices_v3(device, teacher_idx)
        mapping = loss_fn._get_v3_teacher_to_common_student(
            device, teacher_idx, teacher_vocab, common_s, common_t
        )
        for start in range(0, len(common), _ROW_BATCH_SIZE):
            batch = common[start : start + _ROW_BATCH_SIZE]
            raw, ids, log_z, natural = teacher_reader.gather_rows(
                longs([c.batch for c in batch]),
                longs([c.teacher_positions[0] for c in batch]),
                longs([c.teacher_labels[0] for c in batch]),
            )
            realized_s = longs([chunk.student_labels[0] for chunk in batch])
            realized_t = longs([chunk.teacher_labels[0] for chunk in batch])
            realized_slot = (ids == realized_t[:, None]).long().argmax(-1)
            # The reader already guarantees exactly K distinct IDs including
            # the requested label. Removing that column leaves at most K-1
            # alternatives, so common support needs no additional top-k or CPU
            # candidate loop. Keep alternatives in ID order and append the
            # realized pair last, matching the reference's top-1 tie ordering.
            alternatives = torch.arange(k - 1, device=device)[None, :]
            alternatives = alternatives + (alternatives >= realized_slot[:, None])
            slots = torch.cat((alternatives, realized_slot[:, None]), dim=-1)
            mapped = mapping[ids.long()].gather(-1, slots)
            support = (mapped >= 0) & (mapped != realized_s[:, None])
            support[:, -1] = mapped[:, -1] == realized_s
            s_ids = mapped.clamp_min(0)
            s_logp = selected(
                [c.batch for c in batch],
                [local_of_global[c.student_positions[0]] for c in batch],
                s_ids,
            )
            t_logp = raw.gather(-1, slots) / temperature - log_z[:, None]
            valid = support.any(-1) & (natural if noise_k else torch.ones_like(natural))
            value, matches = partition_sum(s_logp, t_logp, support, valid, common_kind)
            common_sum = common_sum + value
            filtered = int((~natural).sum().item()) if noise_k else 0
            common_count += len(batch) - filtered
            filtered_common += filtered
            common_matches += matches

    # Every rank participates exactly once, even with no mismatches or local
    # prefix rows. Backward SUM routes each final owner's derivative to every
    # CP rank that supplied part of the fixed realized prefix.
    prefix_batch, prefix_sequence, prefix_labels, prefix_chunks = [], [], [], []
    for index, chunk in enumerate(mismatches):
        for position, label in zip(
            chunk.student_positions[:-1], chunk.student_labels[:-1], strict=True
        ):
            if position in local_of_global:
                prefix_batch.append(chunk.batch)
                prefix_sequence.append(local_of_global[position])
                prefix_labels.append(label)
                prefix_chunks.append(index)
    local_prefix = zero + torch.zeros(max(1, len(mismatches)), device=device)
    if prefix_batch:
        local_prefix = local_prefix.index_add(
            0,
            longs(prefix_chunks),
            selected(prefix_batch, prefix_sequence, prefix_labels),
        )
    shared_prefix = cp_sum_with_grad(local_prefix, cp_group=cp_group)
    mismatch_sum = shared_prefix.sum() * 0.0
    mismatch_count = mismatch_matches = filtered_mismatch = 0
    owners = [
        index
        for index, chunk in enumerate(mismatches)
        if chunk.student_positions[-1] in local_of_global
    ]
    if owners:
        prefix_index = loss_fn._ensure_bidir_prefix_support_index(device, teacher_idx)
        for start in range(0, len(owners), _ROW_BATCH_SIZE):
            indices = owners[start : start + _ROW_BATCH_SIZE]
            batch = [mismatches[index] for index in indices]
            raw, ids, log_z, natural = teacher_reader.gather_rows(
                longs([c.batch for c in batch]),
                longs([c.teacher_positions[-1] for c in batch]),
                longs([c.teacher_labels[-1] for c in batch]),
            )
            raw_cpu, ids_cpu = raw.cpu().tolist(), ids.cpu().tolist()
            s_rows, t_slots = [], []
            for row, chunk in enumerate(batch):
                m, n = len(chunk.student_labels), len(chunk.teacher_labels)
                realized = chunk.student_labels[-1], chunk.teacher_labels[-1]
                if m == 1 and n > 1:
                    pairs = prefix_index["forward"].get(
                        (n, chunk.teacher_labels[:-1]), ()
                    )
                    s_list, t_list = loss_fn._unique_bidir_pairs_cpu(
                        pairs,
                        swap=False,
                        assume_unique=bool(prefix_index.get("_prededuped", False)),
                    )
                elif m > 1 and n == 1:
                    pairs = prefix_index["reverse"].get(
                        (m, chunk.student_labels[:-1]), ()
                    )
                    s_list, t_list = loss_fn._unique_bidir_pairs_cpu(
                        pairs,
                        swap=True,
                        assume_unique=bool(prefix_index.get("_prededuped", False)),
                    )
                else:
                    s_list, t_list = [realized[0]], [realized[1]]
                candidates = list(zip(s_list, t_list, strict=True))
                if realized not in candidates:
                    candidates.append(realized)
                if any(
                    not 0 <= s < student.real_vocab_size or not 0 <= t < teacher_vocab
                    for s, t in candidates
                ):
                    raise ValueError(
                        "Native sparse prefix support contains a non-real token ID"
                    )
                slot_by_id = {token: slot for slot, token in enumerate(ids_cpu[row])}
                alternatives = [
                    (s, slot_by_id[t])
                    for s, t in candidates
                    if t in slot_by_id and (s, t) != realized
                ]
                alternatives.sort(
                    key=lambda pair: (-raw_cpu[row][pair[1]], ids_cpu[row][pair[1]])
                )
                kept = [*alternatives[: k - 1], (realized[0], slot_by_id[realized[1]])]
                s_rows.append([s for s, _ in kept])
                t_slots.append([slot for _, slot in kept])
            s_ids, slots, support = support_tensors(s_rows, t_slots)
            s_logp = (
                selected(
                    [c.batch for c in batch],
                    [local_of_global[c.student_positions[-1]] for c in batch],
                    s_ids,
                )
                + shared_prefix[longs(indices), None]
            )
            teacher_prefix = torch.zeros(len(batch), device=device)
            noise_ok = natural.clone()
            requests = [
                (row, chunk.batch, position, label)
                for row, chunk in enumerate(batch)
                for position, label in zip(
                    chunk.teacher_positions[:-1], chunk.teacher_labels[:-1], strict=True
                )
            ]
            for offset in range(0, len(requests), _ROW_BATCH_SIZE):
                subset = requests[offset : offset + _ROW_BATCH_SIZE]
                rows, batches, positions, labels = zip(*subset, strict=True)
                prefix_raw, prefix_ids, prefix_z, prefix_natural = (
                    teacher_reader.gather_rows(
                        longs(batches), longs(positions), longs(labels)
                    )
                )
                matching = prefix_ids == longs(labels)[:, None]
                values = (
                    prefix_raw.masked_fill(~matching, 0).sum(-1) / temperature
                    - prefix_z
                )
                teacher_prefix.index_add_(0, longs(rows), values)
                if noise_k:
                    for row in set(rows):
                        noise_ok[row] &= prefix_natural[longs(rows) == row].all()
            t_logp = (
                raw.gather(-1, slots) / temperature
                - log_z[:, None]
                + teacher_prefix[:, None]
            )
            valid = support.any(-1) & (
                noise_ok if noise_k else torch.ones_like(noise_ok)
            )
            value, matches = partition_sum(
                s_logp, t_logp, support, valid, mismatch_kind
            )
            mismatch_sum = mismatch_sum + value
            mismatch_count += int(valid.sum().item())
            mismatch_matches += matches
            filtered_mismatch += int((~noise_ok).sum().item()) if noise_k else 0

    numerator = common_sum + mismatch_sum * multiplier
    count = global_valid_chunks.to(device=device, dtype=torch.float32).reshape(())
    # Keep the prefix collective in the zero-denominator backward graph too.
    denominator = torch.where(count > 0, count, torch.ones_like(count))
    final_loss = numerator * (temperature**2) / denominator * (count > 0)
    stats = torch.tensor(
        [
            float(common_sum.detach()),
            float(mismatch_sum.detach()),
            common_count,
            mismatch_count,
            common_matches,
            mismatch_matches,
            filtered_common,
            filtered_mismatch,
        ],
        device=device,
        dtype=torch.float64,
    )
    if cp_group is not None and dist.get_world_size(cp_group) > 1:
        dist.all_reduce(stats, group=cp_group)
    cs, ms, cc, mc, cm, mm, fc, fm = stats.tolist()
    mismatch_mean = ms / max(mc, 1)
    return final_loss, {
        "kl_loss": float(final_loss.detach()),
        "kl_common_per_chunk": cs / max(cc, 1),
        "kl_partition_last_per_chunk": mismatch_mean,
        "kl_mismatch_combined_per_chunk": mismatch_mean,
        "kl_mismatch_scaled_per_chunk": mismatch_mean * multiplier,
        "num_common_chunks": int(cc),
        "num_mismatch_chunks": int(mc),
        "num_noise_filtered_common_chunks": int(fc),
        "num_noise_filtered_mismatch_chunks": int(fm),
        "top1_acc_per_chunk": (cm + mm) / max(cc + mc, 1),
        "sparse_topk_k": k,
        "prefix_bidir_v3_mismatch_loss_multiplier": multiplier,
        "prefix_bidir_v3_noise_filter_topk": noise_k,
        "native_sparse_v6": 1,
    }
