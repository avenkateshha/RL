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
"""Collator that tokenizes the student once, then tokenizes+aligns each teacher.

The collator runs inside DataLoader worker processes. It does:

1. Tokenizes the student input once; this tokenization is shared by all
   teachers. In ``mode="text"``, tokenizes raw text without a chat template.
   In ``mode="chat"``, renders the student's chat template and identifies
   assistant content and end-of-turn tokens for the loss mask.
2. For each *cross-tokenizer* teacher, tokenizes with that teacher's
   tokenizer and aligns with its :class:`TokenAligner`. Text mode aligns
   the full text; chat mode renders the teacher's own chat template and
   aligns assistant messages independently. Teacher scoring masks also
   select assistant content and end-of-turn tokens in chat mode. Dense-padded alignment and
   teacher inputs are emitted under ``alignment_{i}_*`` / ``teacher_{i}_*``.
3. *Same-tokenizer* teachers (``aligners[i] is None``) reuse the student
   tokenization and skip projection/alignment tensors. Chat mode also records
   each side's selected semantic regions for lockstep packing.
4. Returns a :class:`BatchedDataDict` with the keys :class:`Policy.train`
   expects (``input_ids``, ``input_lengths``, ``token_mask``,
   ``sample_mask``) plus per-teacher tensors and alignment tensors.

Loss-side projection-matrix work happens inside the loss fn; nothing related
to KL/CE math runs here.
"""

from __future__ import annotations

from copy import copy
from dataclasses import fields as dataclass_fields
from functools import partial
from typing import Any, List, Literal, Optional, cast

import torch
from pydantic import BaseModel, PositiveInt
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.x_token.token_aligner import TokenAligner
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.native_chat import RenderedChatDocument, _render_and_tokenize_chat
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class CrossTokenizerCollatorConfig(BaseModel, extra="allow"):
    """Shared batching options for cross-tokenizer distillation.

    Attributes:
        mode: ``"text"`` tokenizes raw text; ``"chat"`` renders each model's
            chat template and supervises assistant turns.
        include_thinking_in_loss: Include reasoning text and its closing marker
            in assistant supervision, excluding the opening scaffold. Separate
            ``reasoning_content`` fields require native thinking alignment;
            ordinary chat supports inline ``<think>`` content.
        native_thinking_alignment: Align reasoning, close, answer/tool and EOT
            independently on supported Qwen/Nano ChatML templates.
            Requires chat mode and thinking loss.
        kd_alignment_regions: Optional subset of reasoning, close, answer,
            and eot regions. Requires native thinking alignment.
        num_packed_rows: Source rows per logical sample. Must remain one;
            controller-level sequence packing combines complete logical samples.
    """

    mode: Literal["text", "chat"] = "text"
    include_thinking_in_loss: bool = False
    native_thinking_alignment: bool = False
    kd_alignment_regions: (
        list[Literal["reasoning", "close", "answer", "eot"]] | None
    ) = None
    num_packed_rows: PositiveInt = 1


class CrossTokenizerCollator:
    """Tokenize the student once, tokenize+align each teacher, return a flat batch.

    Supports N teachers in raw-text or chat mode. Student inputs are tokenized
    once and shared; each cross-tokenizer teacher uses its own tokenizer and
    :class:`TokenAligner`, emitting teacher-indexed keys
    (``teacher_{i}_*`` and ``alignment_{i}_*``). A *same-tokenizer* teacher
    (``aligners[i] is None``) emits nothing extra — its forward reuses the
    student tokenization, so projection and alignment are skipped entirely.
    Chat mode applies each model's chat template, masks loss and teacher
    scoring to assistant content and end-of-turn tokens, and aligns assistant
    messages separately.

    Args:
        config: Typed batching options, including text/chat mode and the
            thinking-region and packing settings.
        student_tokenizer: HF tokenizer matching the student model.
        teacher_tokenizers: Per-teacher HF tokenizers. May be ``None`` for a
            same-tokenizer teacher (its tokenization is the student's).
        aligners: Per-teacher :class:`TokenAligner`. ``None`` marks a
            same-tokenizer teacher (no projection / no alignment).
        ctx_length_student: Hard tokenization length cap on the student
            side. Text mode pads to this cap; chat mode pads to the batch's
            longest tokenization. Sequence-length divisors may add padding.
        ctx_length_teachers: Per-teacher tokenization length caps.
        drop_first_assistant_chunk_kl_by_teacher: Per-teacher flags controlling
            whether chat alignment drops the first content pair in each
            assistant message. The list retains a slot for same-tokenizer
            teachers so its indices match ``aligners``.
        make_seq_div_by_student: Round student sequence length up to a
            multiple of this value (typically TP * CP * 2 for DTensor V2).
        make_seq_div_by_teachers: Per-teacher sequence-length divisors.
        require_routed_experts: Validate and retain rollout routes for replay.
        student_chat_template_kwargs: Explicit student template controls.
        teacher_chat_template_kwargs: Explicit template controls for each teacher.
    """

    def __init__(
        self,
        *,
        config: CrossTokenizerCollatorConfig,
        student_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizers: List[Optional[PreTrainedTokenizerBase]],
        aligners: List[Optional[TokenAligner]],
        ctx_length_student: int,
        ctx_length_teachers: List[int],
        drop_first_assistant_chunk_kl_by_teacher: List[bool],
        make_seq_div_by_student: int = 1,
        make_seq_div_by_teachers: Optional[List[int]] = None,
        require_routed_experts: bool = False,
        student_chat_template_kwargs: Optional[dict[str, Any]] = None,
        teacher_chat_template_kwargs: Optional[List[dict[str, Any]]] = None,
    ) -> None:
        n = len(aligners)
        assert len(teacher_tokenizers) == n and len(ctx_length_teachers) == n, (
            "teacher_tokenizers, aligners, and ctx_length_teachers must all "
            f"have length == num_teachers ({n})."
        )
        if len(drop_first_assistant_chunk_kl_by_teacher) != n:
            raise ValueError(
                "drop_first_assistant_chunk_kl_by_teacher must have length "
                f"== num_teachers ({n}), got "
                f"{len(drop_first_assistant_chunk_kl_by_teacher)}."
            )
        if make_seq_div_by_teachers is None:
            make_seq_div_by_teachers = [1] * n
        assert len(make_seq_div_by_teachers) == n
        if config.mode == "chat":
            if not student_tokenizer.is_fast:
                raise ValueError("mode='chat' requires a fast student tokenizer")
            for i, aligner in enumerate(aligners):
                if aligner is None:
                    continue
                if not getattr(student_tokenizer, "is_fast", False) or not getattr(
                    teacher_tokenizers[i], "is_fast", False
                ):
                    raise ValueError(
                        "mode='chat' requires fast student/teacher tokenizers for "
                        "return_offsets_mapping=True."
                    )
        if config.native_thinking_alignment and (
            config.mode != "chat" or not config.include_thinking_in_loss
        ):
            raise ValueError(
                "native_thinking_alignment requires mode='chat' and "
                "include_thinking_in_loss=true."
            )
        if (
            config.kd_alignment_regions is not None
            and not config.native_thinking_alignment
        ):
            raise ValueError(
                "kd_alignment_regions requires native_thinking_alignment=true."
            )
        if config.kd_alignment_regions == []:
            raise ValueError(
                "kd_alignment_regions must be a non-empty subset of reasoning, close, answer, eot"
            )
        if config.num_packed_rows != 1:
            raise ValueError(
                "xToken lockstep packing keeps one source row as one logical "
                "sample; collator.num_packed_rows must remain 1."
            )
        if teacher_chat_template_kwargs is None:
            teacher_chat_template_kwargs = [{} for _ in range(n)]
        if len(teacher_chat_template_kwargs) != n:
            raise ValueError(
                "teacher_chat_template_kwargs must have one entry per teacher"
            )
        self.require_routed_experts = require_routed_experts
        student_tokenizer = self._with_template_kwargs(
            student_tokenizer, student_chat_template_kwargs or {}, side_id="student"
        )
        teacher_tokenizers = [
            self._with_template_kwargs(tokenizer, kwargs, side_id=f"teacher_{i}")
            if tokenizer is not None
            else None
            for i, (tokenizer, kwargs) in enumerate(
                zip(teacher_tokenizers, teacher_chat_template_kwargs, strict=True)
            )
        ]
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizers = teacher_tokenizers
        self.aligners = aligners
        self.ctx_length_student = ctx_length_student
        self.ctx_length_teachers = ctx_length_teachers
        self.make_seq_div_by_student = make_seq_div_by_student
        self.make_seq_div_by_teachers = make_seq_div_by_teachers
        self.mode = config.mode
        self.drop_first_assistant_chunk_kl_by_teacher = list(
            drop_first_assistant_chunk_kl_by_teacher
        )
        self.include_thinking_in_loss = config.include_thinking_in_loss
        self.native_thinking_alignment = config.native_thinking_alignment
        self.kd_alignment_regions = (
            frozenset(config.kd_alignment_regions)
            if config.kd_alignment_regions is not None
            else None
        )
        # Downstream consumers assume real tokens occupy the leading
        # positions: ``input_lengths = attention_mask.sum(-1)`` plus the
        # ``[:length]`` slices in the policy forward and the token-chunk
        # alignment all treat ``input_ids[:, :length]`` as the content.
        # Pin right-padding rather than trust each tokenizer's default
        # (some tokenizer configs default to left-padding, which would
        # silently misalign without changing the lengths).
        self.student_tokenizer.padding_side = "right"
        if self.student_tokenizer.pad_token_id is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        # Same pinning for each cross-tokenizer teacher tokenizer.
        for i, tok in enumerate(self.teacher_tokenizers):
            if self.aligners[i] is None or tok is None:
                continue
            tok.padding_side = "right"
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token

    def __call__(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        if self.mode == "chat":
            return self._call_chat(batch)
        return self._call_text(batch)

    def _call_text(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        # kd_data_processor carries the raw text as a single assistant
        # message; the collator tokenizes that content for the student and
        # each cross-tokenizer teacher.
        texts = [datum["message_log"][0]["content"] for datum in batch]
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("CrossTokenizerCollator text mode requires string content")
        sample_ids = self._required_sample_ids(batch)
        student_input_ids, student_attention_mask, student_offsets = (
            self._tokenize_batch(
                texts,
                self.student_tokenizer,
                self.ctx_length_student,
                self.make_seq_div_by_student,
                sample_ids=sample_ids,
                side_id="student",
            )
        )

        sample_mask = torch.tensor(
            [datum["loss_multiplier"] for datum in batch], dtype=torch.float32
        )
        idx = [datum["idx"] for datum in batch]
        student_input_lengths = student_attention_mask.sum(dim=-1).long()

        out: dict[str, Any] = {
            # Student-side keys map onto Policy.train's expected names. A
            # single student tokenization is shared across all teachers.
            "input_ids": student_input_ids,
            "input_lengths": student_input_lengths,
            "token_mask": student_attention_mask.long(),
            # Plain-text KD and CE share the same target region. Chat mode
            # overrides this with assistant content plus explicit EOT targets.
            "kd_token_mask": student_attention_mask.long(),
            "sample_mask": sample_mask,
            "idx": idx,
            "sample_id": sample_ids,
        }

        for i, aligner in enumerate(self.aligners):
            if aligner is None:
                # Same-tokenizer teacher: no re-tokenization, no projection,
                # no alignment. Its forward reuses the student tokenization,
                # but its independent context and padding constraints still
                # apply to every reused row.
                self._validate_reused_teacher_context(
                    student_input_lengths.tolist(),
                    sample_ids=sample_ids,
                    ctx_length=self.ctx_length_teachers[i],
                    make_seq_div_by=self.make_seq_div_by_teachers[i],
                    side_id=f"teacher_{i}",
                    length_description="exact token length",
                )
                continue
            teacher_input_ids, teacher_attention_mask, teacher_offsets = (
                self._tokenize_batch(
                    texts,
                    self.teacher_tokenizers[i],
                    self.ctx_length_teachers[i],
                    self.make_seq_div_by_teachers[i],
                    sample_ids=sample_ids,
                    side_id=f"teacher_{i}",
                )
            )
            alignment = aligner.align(
                student_input_ids,
                teacher_input_ids,
                student_offsets=student_offsets,
                teacher_offsets=teacher_offsets,
                student_attention_mask=student_attention_mask,
                teacher_attention_mask=teacher_attention_mask,
            )
            # Teacher-side keys travel with the batch for the teacher forward.
            out[f"teacher_{i}_input_ids"] = teacher_input_ids
            out[f"teacher_{i}_input_lengths"] = teacher_attention_mask.sum(
                dim=-1
            ).long()
            out[f"teacher_{i}_token_mask"] = teacher_attention_mask.long()
            # Alignment payload, dense-padded so DTensor V2 can shard on dim 0.
            # Keys are driven off AlignmentBatch fields so they can't drift
            # from `alignment_from_flat_batch(data, prefix=f"alignment_{i}_")`.
            for f in dataclass_fields(alignment):
                out[f"alignment_{i}_{f.name}"] = getattr(alignment, f.name)

        self._add_router_replay_metadata(
            out,
            batch,
            student_input_ids=student_input_ids,
            student_input_lengths=student_input_lengths,
        )
        return BatchedDataDict(out)

    def _render_document(
        self,
        tokenizer: PreTrainedTokenizerBase,
        datum: DatumSpec,
        ctx_length: int,
        side: str,
    ) -> RenderedChatDocument:
        try:
            document = _render_and_tokenize_chat(
                tokenizer,
                datum["message_log"],
                ctx_length,
                tools=datum.get("tools"),
                message_loss_mask=datum.get("message_loss_mask"),
                include_thinking_in_loss=self.include_thinking_in_loss,
                native_thinking_alignment=self.native_thinking_alignment,
                skip_overlength=True,
            )
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"{side}, sample idx={datum['idx']}: sample_id={datum.get('sample_id')!r}: {error}"
            ) from error
        return document

    def _call_chat(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        """Render complete conversations and align selected assistant regions."""
        sample_ids = self._required_sample_ids(batch)
        student_docs = [
            self._render_document(
                self.student_tokenizer, datum, self.ctx_length_student, "student"
            )
            for datum in batch
        ]
        self._validate_reused_teacher_context(
            [len(doc.input_ids) for doc in student_docs],
            sample_ids=sample_ids,
            ctx_length=self.ctx_length_student,
            make_seq_div_by=self.make_seq_div_by_student,
            side_id="student",
            length_description="exact post-template length",
        )
        (
            student_input_ids,
            student_attention_mask,
            student_offsets,
            student_asst_mask,
        ) = self._pad_chat_batch(
            [doc.input_ids for doc in student_docs],
            [doc.offsets for doc in student_docs],
            [doc.assistant_mask for doc in student_docs],
            self.student_tokenizer.pad_token_id,
            self.make_seq_div_by_student,
        )
        out: dict[str, Any] = {
            "input_ids": student_input_ids,
            "input_lengths": student_attention_mask.sum(dim=-1).long(),
            "token_mask": (student_attention_mask * student_asst_mask).long(),
            "kd_token_mask": self._pad_chat_masks(
                [self._document_kd_mask(doc) for doc in student_docs],
                max_len=student_input_ids.shape[1],
            ),
            "sample_id": sample_ids,
            "student_semantic_regions": [
                self._document_token_regions(doc) for doc in student_docs
            ],
            "sample_mask": torch.tensor(
                [datum["loss_multiplier"] for datum in batch], dtype=torch.float32
            ),
            "idx": [datum["idx"] for datum in batch],
        }
        for i, aligner in enumerate(self.aligners):
            if aligner is None:
                self._validate_reused_teacher_context(
                    [len(doc.input_ids) for doc in student_docs],
                    sample_ids=sample_ids,
                    ctx_length=self.ctx_length_teachers[i],
                    make_seq_div_by=self.make_seq_div_by_teachers[i],
                    side_id=f"teacher_{i}",
                    length_description="exact post-template length",
                )
                out[f"teacher_{i}_semantic_regions"] = list(
                    out["student_semantic_regions"]
                )
                continue
            tokenizer = self.teacher_tokenizers[i]
            teacher_docs = [
                self._render_document(
                    tokenizer, datum, self.ctx_length_teachers[i], f"teacher {i}"
                )
                for datum in batch
            ]
            self._validate_reused_teacher_context(
                [len(doc.input_ids) for doc in teacher_docs],
                sample_ids=sample_ids,
                ctx_length=self.ctx_length_teachers[i],
                make_seq_div_by=self.make_seq_div_by_teachers[i],
                side_id=f"teacher_{i}",
                length_description="exact post-template length",
            )
            for sample, (student_doc, teacher_doc) in enumerate(
                zip(student_docs, teacher_docs)
            ):
                if student_doc.source_turn_indices != teacher_doc.source_turn_indices:
                    raise ValueError(
                        f"sample {sample}, teacher {i}: selected assistant turn identities differ"
                    )
            (
                teacher_input_ids,
                teacher_attention_mask,
                teacher_offsets,
                teacher_asst_mask,
            ) = self._pad_chat_batch(
                [doc.input_ids for doc in teacher_docs],
                [doc.offsets for doc in teacher_docs],
                [doc.assistant_mask for doc in teacher_docs],
                tokenizer.pad_token_id,
                self.make_seq_div_by_teachers[i],
            )
            alignment = aligner.align_chat(
                student_input_ids,
                teacher_input_ids,
                student_offsets=student_offsets,
                teacher_offsets=teacher_offsets,
                student_asst_char_spans=[doc.assistant_spans for doc in student_docs],
                teacher_asst_char_spans=[doc.assistant_spans for doc in teacher_docs],
                student_attention_mask=student_attention_mask,
                teacher_attention_mask=teacher_attention_mask,
                student_asst_mask=student_asst_mask,
                teacher_asst_mask=teacher_asst_mask,
                student_eot_indices=[doc.eot_indices for doc in student_docs],
                teacher_eot_indices=[doc.eot_indices for doc in teacher_docs],
                drop_first_content_pair=self.drop_first_assistant_chunk_kl_by_teacher[
                    i
                ],
                student_alignment_regions=[
                    doc.alignment_regions for doc in student_docs
                ]
                if self.native_thinking_alignment
                else None,
                teacher_alignment_regions=[
                    doc.alignment_regions for doc in teacher_docs
                ]
                if self.native_thinking_alignment
                else None,
                student_answer_parts=[doc.answer_parts for doc in student_docs]
                if self.native_thinking_alignment
                else None,
                teacher_answer_parts=[doc.answer_parts for doc in teacher_docs]
                if self.native_thinking_alignment
                else None,
                student_rendered_texts=[doc.rendered_text for doc in student_docs]
                if self.native_thinking_alignment
                else None,
                teacher_rendered_texts=[doc.rendered_text for doc in teacher_docs]
                if self.native_thinking_alignment
                else None,
                included_regions=self.kd_alignment_regions,
            )
            out[f"teacher_{i}_input_ids"] = teacher_input_ids
            out[f"teacher_{i}_input_lengths"] = teacher_attention_mask.sum(
                dim=-1
            ).long()
            out[f"teacher_{i}_token_mask"] = (
                teacher_attention_mask * teacher_asst_mask
            ).long()
            out[f"teacher_{i}_semantic_regions"] = [
                self._document_token_regions(doc) for doc in teacher_docs
            ]
            for field in dataclass_fields(alignment):
                out[f"alignment_{i}_{field.name}"] = getattr(alignment, field.name)
        self._add_router_replay_metadata(
            out,
            batch,
            student_input_ids=student_input_ids,
            student_input_lengths=out["input_lengths"],
        )
        return BatchedDataDict(out)

    @staticmethod
    def _render_and_tokenize_chat(
        tokenizer: PreTrainedTokenizerBase,
        messages: List[dict],
        ctx_length: int,
    ) -> tuple[
        List[int], List[tuple[int, int]], List[int], List[tuple[int, int]], List[int]
    ]:
        """Compatibility helper for inspecting retained truncation boundaries.

        Production chat collation uses complete documents and rejects overflow.
        """
        doc = _render_and_tokenize_chat(
            tokenizer,
            messages,
            ctx_length,
            include_thinking_in_loss=False,
            native_thinking_alignment=False,
            skip_overlength=False,
        )
        return (
            doc.input_ids,
            doc.offsets,
            doc.assistant_mask,
            doc.assistant_spans,
            doc.eot_indices,
        )

    @staticmethod
    def _pad_chat_batch(
        ids_list: List[List[int]],
        off_list: List[List[tuple[int, int]]],
        mask_list: List[List[int]],
        pad_token_id: int,
        make_seq_div_by: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Right-pad per-sample chat tokenizations into dense ``[B, T]`` tensors.

        Pads to the batch-max length rounded up to ``make_seq_div_by``. Padding
        positions get ``attention_mask=0``, ``offset=(0, 0)``,
        ``assistant_mask=0``.
        """
        b = len(ids_list)
        max_len = max((len(ids) for ids in ids_list), default=0)
        if make_seq_div_by > 1 and max_len % make_seq_div_by:
            max_len += make_seq_div_by - (max_len % make_seq_div_by)
        max_len = max(max_len, make_seq_div_by, 1)

        input_ids = torch.full((b, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        offsets = torch.zeros((b, max_len, 2), dtype=torch.long)
        assistant_mask = torch.zeros((b, max_len), dtype=torch.long)
        for j, (ids, off, mask) in enumerate(zip(ids_list, off_list, mask_list)):
            n = len(ids)
            input_ids[j, :n] = torch.tensor(ids, dtype=torch.long)
            attention_mask[j, :n] = 1
            if n:
                offsets[j, :n] = torch.tensor(off, dtype=torch.long)
                assistant_mask[j, :n] = torch.tensor(mask, dtype=torch.long)
        return input_ids, attention_mask, offsets, assistant_mask

    @staticmethod
    def _tokenize_batch(
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        ctx_length: int,
        make_seq_div_by: int,
        *,
        sample_ids: Optional[List[str | int]] = None,
        side_id: str = "model",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a batch and pad to a multiple of ``make_seq_div_by``.

        Also returns per-token character offsets (``offset_mapping``), which
        :meth:`TokenAligner.align` needs to align the student and teacher
        tokenizations of the same source text. This requires a *fast* HF
        tokenizer; special and padding positions carry ``(0, 0)``.
        """
        if sample_ids is None:
            sample_ids = list(range(len(texts)))
        if len(sample_ids) != len(texts):
            raise ValueError("sample_ids and texts must have the same length")

        ids_list: list[list[int]] = []
        offsets_list: list[list[tuple[int, int]]] = []
        for text, sample_id in zip(texts, sample_ids, strict=True):
            encoded = tokenizer(
                text,
                truncation=False,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            ids = list(encoded["input_ids"])
            offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
            effective_len = (
                (len(ids) + make_seq_div_by - 1) // make_seq_div_by
            ) * make_seq_div_by
            if len(ids) > ctx_length or effective_len > ctx_length:
                raise ValueError(
                    f"xToken sample_id={sample_id!r} exceeds {side_id} context: "
                    f"exact token length {len(ids)}, effective length "
                    f"{effective_len}, capacity {ctx_length}; truncation is "
                    "forbidden."
                )
            ids_list.append(ids)
            offsets_list.append(offsets)

        b = len(ids_list)
        max_len = max((len(ids) for ids in ids_list), default=0)
        max_len = max(max_len, 1)
        if make_seq_div_by > 1:
            max_len = (
                (max_len + make_seq_div_by - 1) // make_seq_div_by
            ) * make_seq_div_by
        input_ids = torch.full((b, max_len), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        offset_mapping = torch.zeros((b, max_len, 2), dtype=torch.long)
        for batch_index, (ids, offsets) in enumerate(
            zip(ids_list, offsets_list, strict=True)
        ):
            length = len(ids)
            if length:
                input_ids[batch_index, :length] = torch.tensor(ids, dtype=torch.long)
                attention_mask[batch_index, :length] = 1
                offset_mapping[batch_index, :length] = torch.tensor(
                    offsets, dtype=torch.long
                )
        return input_ids, attention_mask, offset_mapping

    def _add_router_replay_metadata(
        self,
        out: dict[str, Any],
        batch: List[DatumSpec],
        *,
        student_input_ids: torch.Tensor,
        student_input_lengths: torch.Tensor,
    ) -> None:
        """Validate and batch rollout-recorded routes for the student forward.

        Router replay is meaningful only when the recorded routes describe the
        exact token sequence that the student will train on.  xToken normally
        re-tokenizes source text/chat in the collator, so compare the persisted
        per-message ``token_ids`` with that result before carrying the matching
        ``routed_experts`` tensor forward.  This turns stale tokenizer/template
        settings into an actionable input error instead of silently replaying
        routes at the wrong token positions.
        """
        if not self.require_routed_experts:
            return

        padded_seq_len = student_input_ids.shape[1]
        routed_rows: list[torch.Tensor] = []
        route_shape: Optional[tuple[int, int]] = None
        route_dtype: Optional[torch.dtype] = None

        for row_index, datum in enumerate(batch):
            sample_id = datum["sample_id"]
            token_parts: list[torch.Tensor] = []
            route_parts: list[torch.Tensor] = []
            message_log = cast(List[dict[str, Any]], datum["message_log"])
            for turn_index, message in enumerate(message_log):
                token_ids_value = message.get("token_ids")
                routed_experts_value = message.get("routed_experts")
                if token_ids_value is None:
                    raise RuntimeError(
                        "policy.router_replay.enabled=true requires rollout-recorded "
                        "token_ids and routed_experts on every message; "
                        f"sample_id={sample_id!r}, turn_index={turn_index} is "
                        "missing metadata. Use a vLLM router-replay rollout as "
                        "the xToken data source."
                    )

                token_ids = torch.as_tensor(token_ids_value, dtype=torch.long)
                if token_ids.dim() != 1:
                    raise ValueError(
                        "router-replay token_ids must have shape [tokens]; "
                        f"sample_id={sample_id!r}, turn_index={turn_index}, "
                        f"got {tuple(token_ids.shape)}."
                    )
                # Some message serializers erase the trailing dimensions of an
                # empty [0, L, K] tensor. Empty turns contribute no positions,
                # so they can be skipped without weakening route coverage.
                if token_ids.numel() == 0:
                    continue
                if routed_experts_value is None:
                    raise RuntimeError(
                        "policy.router_replay.enabled=true requires rollout-recorded "
                        "token_ids and routed_experts on every non-empty message; "
                        f"sample_id={sample_id!r}, turn_index={turn_index} is "
                        "missing routed_experts. Use a vLLM router-replay "
                        "rollout as the xToken data source."
                    )
                routed_experts = torch.as_tensor(routed_experts_value)
                if routed_experts.dim() != 3:
                    raise ValueError(
                        "router-replay routed_experts must have shape "
                        "[tokens, layers, topk]; "
                        f"sample_id={sample_id!r}, turn_index={turn_index}, "
                        f"got {tuple(routed_experts.shape)}."
                    )
                if routed_experts.shape[0] != token_ids.shape[0]:
                    raise ValueError(
                        "router-replay token_ids and routed_experts token axes "
                        f"differ for sample_id={sample_id!r}, "
                        f"turn_index={turn_index}: {token_ids.shape[0]} != "
                        f"{routed_experts.shape[0]}."
                    )

                current_shape = (
                    int(routed_experts.shape[1]),
                    int(routed_experts.shape[2]),
                )
                if route_shape is None:
                    route_shape = current_shape
                    route_dtype = routed_experts.dtype
                elif (
                    current_shape != route_shape or routed_experts.dtype != route_dtype
                ):
                    raise ValueError(
                        "router-replay routed_experts must use one [layers, topk] "
                        "shape and dtype across the batch; "
                        f"expected {route_shape}/{route_dtype}, got "
                        f"{current_shape}/{routed_experts.dtype} for "
                        f"sample_id={sample_id!r}, turn_index={turn_index}."
                    )
                token_parts.append(token_ids.detach().cpu())
                route_parts.append(routed_experts.detach().cpu())

            if not token_parts or route_shape is None or route_dtype is None:
                raise RuntimeError(
                    "policy.router_replay.enabled=true requires non-empty "
                    f"rollout route metadata; sample_id={sample_id!r} has none."
                )

            recorded_token_ids = torch.cat(token_parts, dim=0)
            recorded_routes = torch.cat(route_parts, dim=0)
            student_length = int(student_input_lengths[row_index].item())
            expected_token_ids = student_input_ids[row_index, :student_length].cpu()
            if not torch.equal(recorded_token_ids, expected_token_ids):
                raise RuntimeError(
                    "Cannot replay routed experts because rollout-recorded "
                    "student token_ids do not match the xToken student "
                    f"tokenization for sample_id={sample_id!r} "
                    f"(recorded={recorded_token_ids.shape[0]} tokens, "
                    f"retokenized={student_length}). Keep the rollout and "
                    "training tokenizer/chat-template settings identical."
                )

            pad_len = padded_seq_len - recorded_routes.shape[0]
            if pad_len < 0:
                raise RuntimeError(
                    "router-replay route metadata is longer than the student "
                    f"batch row for sample_id={sample_id!r}: "
                    f"{recorded_routes.shape[0]} > {padded_seq_len}."
                )
            if pad_len:
                recorded_routes = torch.nn.functional.pad(
                    recorded_routes, (0, 0, 0, 0, 0, pad_len), value=0
                )
            routed_rows.append(recorded_routes)

        out["routed_experts"] = torch.stack(routed_rows, dim=0)

    @staticmethod
    def _required_sample_ids(batch: List[DatumSpec]) -> list[object]:
        """Return durable source IDs; post-transform indices are not identity."""
        sample_ids: list[object] = []
        for row_index, datum in enumerate(batch):
            sample_id = datum.get("sample_id")
            if sample_id is None or not str(sample_id).strip():
                raise ValueError(
                    "CrossTokenizerCollator requires a durable, non-empty "
                    "sample_id on every row; positional idx cannot survive "
                    f"filtering/splitting (batch row {row_index})."
                )
            sample_ids.append(sample_id)
        return sample_ids

    @staticmethod
    def _validate_reused_teacher_context(
        raw_lengths: List[int],
        *,
        sample_ids: List[object],
        ctx_length: int,
        make_seq_div_by: int,
        side_id: str,
        length_description: str,
    ) -> None:
        """Enforce one same-tokenizer teacher's independent context limit."""
        if len(raw_lengths) != len(sample_ids):
            raise ValueError("raw_lengths and sample_ids must have the same length")
        for raw_length, sample_id in zip(raw_lengths, sample_ids, strict=True):
            effective_len = (
                (raw_length + make_seq_div_by - 1) // make_seq_div_by
            ) * make_seq_div_by
            if raw_length > ctx_length or effective_len > ctx_length:
                raise ValueError(
                    f"xToken sample_id={sample_id!r} exceeds {side_id} context: "
                    f"{length_description} {raw_length}, effective length "
                    f"{effective_len}, capacity {ctx_length}; truncation is "
                    "forbidden."
                )

    @staticmethod
    def _pad_chat_masks(masks: List[List[int]], *, max_len: int) -> torch.Tensor:
        """Right-pad semantic masks to an already materialized sequence width."""
        padded = torch.zeros((len(masks), max_len), dtype=torch.long)
        for row_index, mask in enumerate(masks):
            if len(mask) > max_len:
                raise ValueError(
                    f"semantic mask length {len(mask)} exceeds padded width {max_len}"
                )
            if mask:
                padded[row_index, : len(mask)] = torch.tensor(mask, dtype=torch.long)
        return padded

    @staticmethod
    def _validate_template_kwargs(
        kwargs: dict[str, Any], *, side_id: str
    ) -> dict[str, Any]:
        reserved = {"tokenize", "tools"}.intersection(kwargs)
        if reserved:
            raise ValueError(
                f"{side_id} chat template kwargs may not override controller-owned "
                f"keys {sorted(reserved)!r}."
            )
        return dict(kwargs)

    @staticmethod
    def _with_template_kwargs(
        tokenizer: PreTrainedTokenizerBase,
        kwargs: dict[str, Any],
        *,
        side_id: str,
    ) -> PreTrainedTokenizerBase:
        """Bind side-specific controls without changing the caller's tokenizer."""
        kwargs = CrossTokenizerCollator._validate_template_kwargs(
            kwargs, side_id=side_id
        )
        if kwargs.get("add_generation_prompt", False):
            raise ValueError(f"{side_id} training requires add_generation_prompt=false")
        if not kwargs:
            return tokenizer
        configured = copy(tokenizer)
        configured.apply_chat_template = partial(
            tokenizer.apply_chat_template, **kwargs
        )
        return configured

    def _document_token_regions(
        self, document: RenderedChatDocument
    ) -> tuple[tuple[int, str, str, int, int], ...]:
        """Record selected logical turns in each tokenizer's own token coordinates."""
        regions: list[tuple[int, str, str, int, int]] = []
        for ordinal, (turn, span, eot) in enumerate(
            zip(
                document.source_turn_indices,
                document.assistant_spans,
                document.eot_indices,
                strict=True,
            )
        ):
            if self.native_thinking_alignment:
                native_regions = document.alignment_regions[ordinal]
                char_regions = [
                    (name, value)
                    for name in ("reasoning", "close", "answer")
                    if (value := native_regions.get(name)) is not None
                    and (
                        self.kd_alignment_regions is None
                        or name in self.kd_alignment_regions
                    )
                ]
            else:
                char_regions = [("content", span)]
            for name, (start, end) in char_regions:
                indices = [
                    index
                    for index, (left, right) in enumerate(document.offsets)
                    if start <= left < right <= end and document.assistant_mask[index]
                ]
                if indices:
                    regions.append(
                        (turn, "assistant", name, indices[0], indices[-1] + 1)
                    )
            if eot >= 0 and (
                self.kd_alignment_regions is None or "eot" in self.kd_alignment_regions
            ):
                regions.append((turn, "assistant", "eot", eot, eot + 1))
        return tuple(regions)

    def _document_kd_mask(self, document: RenderedChatDocument) -> list[int]:
        """Limit same-tokenizer KD to selected semantic regions without changing CE."""
        if not self.native_thinking_alignment or self.kd_alignment_regions is None:
            return list(document.assistant_mask)
        mask = [0] * len(document.input_ids)
        for _, _, _, start, end in self._document_token_regions(document):
            for index in range(start, end):
                mask[index] = document.assistant_mask[index]
        return mask
