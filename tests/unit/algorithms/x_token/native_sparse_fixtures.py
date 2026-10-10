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

"""Small independent selected-support/REST oracle for native sparse xToken.

No production loss or transport helper is imported here. Full dense logits are
used only to obtain exact selected probabilities; KL/JSD is evaluated on the
sparse candidate support plus its complement, never over the full vocabulary.
The explicit full-step denominators must survive splitting this B2 fixture into
two B1 microbatches. Run this module directly for oracle self-checks.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import torch

Span = tuple[int, int]
TokenChain = tuple[int, ...]


@dataclass(frozen=True)
class Chunk:
    name: str
    student: Span
    teacher: Span
    valid: bool = True

    def predictors(self, shift: bool) -> tuple[range, range]:
        offset = int(shift and self.student[0] > 0 and self.teacher[0] > 0)
        return (
            range(self.student[0] - offset, self.student[1] - offset),
            range(self.teacher[0] - offset, self.teacher[1] - offset),
        )


@dataclass(frozen=True)
class TeacherFixture:
    name: str
    logits: torch.Tensor
    input_ids: torch.Tensor
    chunks: tuple[tuple[Chunk, ...], ...]
    forward: tuple[TokenChain, ...]
    reverse: tuple[TokenChain, ...]
    weight: float
    global_valid_chunks: float
    real_vocab_size: int
    microbatch_size: int

    def slice_samples(self, start: int, end: int) -> "TeacherFixture":
        return replace(
            self,
            logits=self.logits[start:end],
            input_ids=self.input_ids[start:end],
            chunks=self.chunks[start:end],
        )

    def alignment_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([[c.student for c in row] for row in self.chunks]),
            torch.tensor([[c.teacher for c in row] for row in self.chunks]),
            torch.tensor([[c.valid for c in row] for row in self.chunks]),
        )

    def without_valid_chunks(self) -> "TeacherFixture":
        """Keep a teacher in the ordered loss loop with no eligible chunks."""
        return replace(
            self,
            chunks=tuple(
                tuple(replace(chunk, valid=False) for chunk in row)
                for row in self.chunks
            ),
        )


@dataclass(frozen=True)
class SparseFixture:
    student_logits: torch.Tensor
    student_input_ids: torch.Tensor
    token_mask: torch.Tensor
    kd_token_mask: torch.Tensor
    sample_mask: torch.Tensor
    teachers: tuple[TeacherFixture, ...]
    student_vocab_size: int = 16
    global_valid_tokens: float = 31.0

    def slice_samples(self, start: int, end: int) -> "SparseFixture":
        return replace(
            self,
            student_logits=self.student_logits[start:end],
            student_input_ids=self.student_input_ids[start:end],
            token_mask=self.token_mask[start:end],
            kd_token_mask=self.kd_token_mask[start:end],
            sample_mask=self.sample_mask[start:end],
            teachers=tuple(t.slice_samples(start, end) for t in self.teachers),
        )


@dataclass(frozen=True)
class ObjectiveSettings:
    k: int = 4
    temperature: float = 1.7
    shift: bool = True
    common_loss: Literal["kl", "jsd"] = "jsd"
    mismatch_loss: Literal["kl", "jsd"] = "kl"
    jsd_beta: float = 0.3
    reverse_kl: bool = False
    mismatch_multiplier: float = 1.4
    noise_filter_k: int = 0
    normalize_teacher_by_vocab: bool = False
    dynamic_scaling: bool = False
    ce_weight: float = 0.4
    kd_weight: float = 0.8


@dataclass(frozen=True)
class ChunkResult:
    sample: int
    name: str
    student_predictors: tuple[int, ...]
    teacher_predictors: tuple[int, ...]
    support: tuple[tuple[int, int], ...]
    loss: torch.Tensor
    filtered: bool


@dataclass(frozen=True)
class TeacherResult:
    loss: torch.Tensor
    chunks: tuple[ChunkResult, ...]


@dataclass(frozen=True)
class ObjectiveResult:
    loss: torch.Tensor
    ce: torch.Tensor
    kd: torch.Tensor
    ratio: torch.Tensor
    teachers: tuple[TeacherResult, ...]


def _tables(
    teacher_index: int,
) -> tuple[tuple[TokenChain, ...], tuple[TokenChain, ...]]:
    teacher_vocab = 16 if teacher_index == 0 else 20
    forward: list[TokenChain] = [()] * 16
    reverse: list[TokenChain] = [()] * teacher_vocab
    for student_id in range(10):
        teacher_id = student_id if teacher_index == 0 else student_id + 5
        forward[student_id] = (teacher_id,)
        reverse[teacher_id] = (student_id,)
    teacher_prefix = (8,) if teacher_index == 0 else (15, 16)
    for student_id, final in zip(
        (10, 11, 12), ((3, 4, 6) if teacher_index == 0 else (2, 3, 4))
    ):
        forward[student_id] = (*teacher_prefix, final)
    for teacher_id, final in zip(
        ((10, 11, 12) if teacher_index == 0 else (17, 18, 19)), (3, 4, 6)
    ):
        reverse[teacher_id] = (8, final)
    return tuple(forward), tuple(reverse)


def build_sparse_fixture(
    *, generation: int = 0, dtype: torch.dtype = torch.float32
) -> SparseFixture:
    """Two distinguishable teachers and all four alignment shapes at B2/S8.

    Student logits have four extra padded vocabulary columns with very large
    scores. Each teacher also has four such columns. Correct normalization must
    ignore them. Generation changes logits while preserving sample identity and
    alignment, supporting consecutive-buffer tests without random side effects.
    """
    generator = torch.Generator().manual_seed(90210 + generation)
    student = torch.randn(2, 8, 20, generator=generator, dtype=dtype)
    student[..., 16:] = 30.0 + generation
    student_ids = torch.tensor([[0, 10, 8, 3, 6, 7, 9, 4], [1, 2, 11, 8, 4, 6, 9, 5]])
    student_spans = [
        ((0, 1), (1, 2), (2, 4), (6, 8), (0, 0), (4, 5)),
        ((0, 1), (1, 2), (2, 3), (3, 5), (6, 8), (5, 6)),
    ]
    spans_a = [
        ((0, 1), (1, 3), (3, 4), (5, 8), (4, 5), (4, 5)),
        ((0, 1), (1, 2), (2, 4), (4, 5), (6, 8), (0, 0)),
    ]
    spans_b = [
        ((0, 1), (1, 4), (5, 6), (8, 11), (7, 8), (6, 7)),
        ((0, 0), (2, 3), (3, 6), (7, 8), (9, 12), (0, 0)),
    ]
    names = [
        (
            "origin_common",
            "one_to_many",
            "many_to_one_cross_cp",
            "many_to_many_cross_cp",
            "absent_student",
            "masked_pair",
        ),
        (
            "origin_common_or_absent",
            "common",
            "one_to_many",
            "many_to_one",
            "many_to_many_cross_cp",
            "absent_teacher",
        ),
    ]
    ids_a = torch.tensor([[0, 8, 3, 10, 5, 9, 5, 4], [1, 2, 8, 4, 11, 6, 9, 5]])
    ids_b = torch.tensor(
        [
            [5, 15, 16, 2, 11, 17, 0, 1, 14, 9, 4, 0],
            [11, 12, 7, 15, 16, 3, 2, 18, 1, 14, 8, 9],
        ]
    )
    teachers = []
    for teacher_index, (ids, teacher_spans) in enumerate(
        ((ids_a, spans_a), (ids_b, spans_b))
    ):
        real_vocab = 16 if teacher_index == 0 else 20
        logits = (
            torch.randn(
                2, ids.shape[1], real_vocab + 4, generator=generator, dtype=dtype
            )
            * 0.15
            - 2.0
        )
        logits[..., real_vocab:] = 40.0 + generation
        # The cutoff tie is exact, including in BF16. Lower token ID wins.
        for token_id, score in (
            (1, 3.0),
            (2, 2.5),
            (4, 1.0),
            (real_vocab - 1, 1.0),
            (real_vocab - 2, 1.0),
        ):
            logits[..., token_id] = score + generation * 0.03
        chunks = tuple(
            tuple(
                Chunk(
                    names[b][j],
                    student_spans[b][j],
                    teacher_spans[b][j],
                    not (b == 0 and j == 5),
                )
                for j in range(6)
            )
            for b in range(2)
        )
        # Rich reverse-prefix support in addition to the forced realized label.
        for b, chunk_index in ((0, 2), (1, 3)):
            position = chunks[b][chunk_index].predictors(True)[1][-1]
            alternatives = (11, 12) if teacher_index == 0 else (18, 19)
            logits[b, position, list(alternatives)] = torch.tensor(
                [4.0, 3.5], dtype=dtype
            )
        # Force the origin and first mismatch prefix labels outside natural K.
        for required in (int(ids[0, 0]), int(ids[0, 1])):
            logits[0, 0, required] = -4.0
        forward, reverse = _tables(teacher_index)
        teachers.append(
            TeacherFixture(
                name=("teacher_a" if teacher_index == 0 else "teacher_b"),
                logits=logits,
                input_ids=ids,
                chunks=chunks,
                forward=forward,
                reverse=reverse,
                weight=(0.3 if teacher_index == 0 else 0.7),
                global_valid_chunks=(13.0 if teacher_index == 0 else 17.0),
                real_vocab_size=real_vocab,
                microbatch_size=(1 if teacher_index == 0 else 2),
            )
        )
    token_mask = torch.tensor(
        [[0, 1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 0, 1, 1]], dtype=dtype
    )
    kd_mask = token_mask.clone()
    kd_mask[:, 5] = 0
    return SparseFixture(
        student,
        student_ids,
        token_mask,
        kd_mask,
        torch.ones(2, dtype=dtype),
        tuple(teachers),
    )


def stable_natural_ids(
    raw: torch.Tensor, *, real_vocab_size: int, k: int
) -> tuple[int, ...]:
    """Independent deterministic score-descending/ID-ascending support."""
    return tuple(sorted(range(real_vocab_size), key=lambda i: (-float(raw[i]), i))[:k])


def requested_support(
    raw: torch.Tensor, required: int, *, real_vocab_size: int, k: int
) -> tuple[int, ...]:
    natural = stable_natural_ids(raw, real_vocab_size=real_vocab_size, k=k)
    chosen = natural if required in natural else (*natural[: k - 1], required)
    return tuple(sorted(chosen))


def forced_label_sidecar(
    teacher: TeacherFixture, *, shift: bool = True
) -> torch.Tensor:
    """Two-slot label oracle in the teacher's full predictor-position order."""
    labels = torch.full((*teacher.input_ids.shape, 2), -1, dtype=torch.long)
    for b, chunks in enumerate(teacher.chunks):
        for chunk in chunks:
            if (
                not chunk.valid
                or chunk.student[0] == chunk.student[1]
                or chunk.teacher[0] == chunk.teacher[1]
            ):
                continue
            _, predictions = chunk.predictors(shift)
            for predictor, label_position in zip(predictions, range(*chunk.teacher)):
                required = int(teacher.input_ids[b, label_position])
                if required in labels[b, predictor]:
                    continue
                free = (labels[b, predictor] == -1).nonzero().flatten()
                if free.numel() == 0:
                    raise AssertionError(
                        "Fixture needs more than two forced labels on one teacher row"
                    )
                labels[b, predictor, free[0]] = required
    return labels


def write_tables(teacher: TeacherFixture, directory: Path) -> tuple[Path, Path]:
    """Write this teacher's forward/reverse tables in the production file schema."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, chains in (("forward", teacher.forward), ("reverse", teacher.reverse)):
        width = max(map(len, chains))
        subtoks = torch.full((len(chains), width), -1, dtype=torch.long)
        for row, chain in enumerate(chains):
            subtoks[row, : len(chain)] = torch.tensor(chain, dtype=torch.long)
        path = directory / f"{teacher.name}_{name}.pt"
        torch.save(
            {"subtoks": subtoks, "lengths": torch.tensor(list(map(len, chains)))}, path
        )
        paths.append(path)
    return paths[0], paths[1]


def _candidate_pairs(
    teacher: TeacherFixture,
    student_labels: tuple[int, ...],
    teacher_labels: tuple[int, ...],
) -> list[tuple[int, int]]:
    realized = (student_labels[-1], teacher_labels[-1])
    pairs: list[tuple[int, int]] = []
    if len(student_labels) == 1 and len(teacher_labels) > 1:
        pairs = [
            (s, chain[-1])
            for s, chain in enumerate(teacher.forward)
            if len(chain) == len(teacher_labels) and chain[:-1] == teacher_labels[:-1]
        ]
    elif len(student_labels) > 1 and len(teacher_labels) == 1:
        pairs = [
            (chain[-1], t)
            for t, chain in enumerate(teacher.reverse)
            if len(chain) == len(student_labels) and chain[:-1] == student_labels[:-1]
        ]
    # Historical M-to-N stays realized-chain ALM, with its binary REST bucket.
    deduped = []
    seen_student, seen_teacher = set(), set()
    for s, t in pairs:
        if s not in seen_student and t not in seen_teacher:
            deduped.append((s, t))
            seen_student.add(s)
            seen_teacher.add(t)
    if realized not in deduped:
        deduped.append(realized)
    return deduped


def partition_divergence(
    student_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    *,
    kind: Literal["kl", "jsd"],
    reverse: bool,
    beta: float,
) -> torch.Tensor:
    """Compute divergence only on selected probabilities and a REST bucket."""

    def partition(logp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probability = logp.exp()
        rest = (1 - probability.sum()).clamp_min(1e-12)
        return torch.cat((probability, rest[None])), torch.cat((logp, rest.log()[None]))

    student_p, student_lp = partition(student_logp)
    teacher_p, teacher_lp = partition(teacher_logp)
    if kind == "kl" or beta in (0.0, 1.0):
        use_reverse = reverse if kind == "kl" else beta == 1.0
        return (
            (student_p * (student_lp - teacher_lp)).sum()
            if use_reverse
            else (teacher_p * (teacher_lp - student_lp)).sum()
        )
    mixture_lp = ((1 - beta) * student_p + beta * teacher_p).log()
    return (
        beta * teacher_p * (teacher_lp - mixture_lp)
        + (1 - beta) * student_p * (student_lp - mixture_lp)
    ).sum()


def teacher_sparse_oracle(
    fixture: SparseFixture,
    teacher: TeacherFixture,
    student_logits: torch.Tensor,
    settings: ObjectiveSettings = ObjectiveSettings(),
) -> TeacherResult:
    student_lp = (
        student_logits[..., : fixture.student_vocab_size] / settings.temperature
    ).log_softmax(-1)
    teacher_lp = (
        teacher.logits[..., : teacher.real_vocab_size] / settings.temperature
    ).log_softmax(-1)
    common_map: dict[int, int] = {}
    for student_id, chain in enumerate(teacher.forward):
        if len(chain) == 1:
            common_map.setdefault(chain[0], student_id)
    total = student_logits.sum() * 0
    results = []
    for b, chunks in enumerate(teacher.chunks):
        if not bool(fixture.sample_mask[b]):
            continue
        for chunk in chunks:
            if (
                not chunk.valid
                or chunk.student[0] == chunk.student[1]
                or chunk.teacher[0] == chunk.teacher[1]
            ):
                continue
            student_predictions, teacher_predictions = chunk.predictors(settings.shift)
            student_labels = tuple(
                int(x) for x in fixture.student_input_ids[b, slice(*chunk.student)]
            )
            teacher_labels = tuple(
                int(x) for x in teacher.input_ids[b, slice(*chunk.teacher)]
            )
            raw = teacher.logits[b, teacher_predictions[-1]]
            support = requested_support(
                raw,
                teacher_labels[-1],
                real_vocab_size=teacher.real_vocab_size,
                k=settings.k,
            )
            common = len(student_labels) == len(teacher_labels) == 1
            if common:
                alternatives = [
                    (common_map[t], t)
                    for t in support
                    if t in common_map
                    and common_map[t] != student_labels[-1]
                    and t != teacher_labels[-1]
                ]
                alternatives.sort(key=lambda pair: (-float(raw[pair[1]]), pair[1]))
                pairs = alternatives[: settings.k - 1]
                if common_map.get(teacher_labels[-1]) == student_labels[-1]:
                    pairs.append((student_labels[-1], teacher_labels[-1]))
            else:
                candidates = _candidate_pairs(teacher, student_labels, teacher_labels)
                pairs = [pair for pair in candidates if pair[1] in support]
                realized = (student_labels[-1], teacher_labels[-1])
                alternatives = [pair for pair in pairs if pair != realized]
                alternatives.sort(key=lambda pair: (-float(raw[pair[1]]), pair[1]))
                pairs = [*alternatives[: settings.k - 1], realized]
            filtered = settings.noise_filter_k > 0 and any(
                label
                not in stable_natural_ids(
                    teacher.logits[b, predictor],
                    real_vocab_size=teacher.real_vocab_size,
                    k=settings.noise_filter_k,
                )
                for predictor, label in zip(teacher_predictions, teacher_labels)
            )
            if pairs and not filtered:
                s_prefix = sum(
                    (
                        student_lp[b, p, label]
                        for p, label in zip(
                            student_predictions[:-1], student_labels[:-1]
                        )
                    ),
                    student_logits.new_zeros(()),
                )
                t_prefix = sum(
                    (
                        teacher_lp[b, p, label]
                        for p, label in zip(
                            teacher_predictions[:-1], teacher_labels[:-1]
                        )
                    ),
                    teacher.logits.new_zeros(()),
                )
                s_selected = torch.stack(
                    [
                        s_prefix + student_lp[b, student_predictions[-1], s]
                        for s, _ in pairs
                    ]
                )
                t_selected = torch.stack(
                    [
                        t_prefix + teacher_lp[b, teacher_predictions[-1], t]
                        for _, t in pairs
                    ]
                )
                value = partition_divergence(
                    s_selected,
                    t_selected,
                    kind=(settings.common_loss if common else settings.mismatch_loss),
                    reverse=settings.reverse_kl,
                    beta=settings.jsd_beta,
                )
                if not common:
                    value = value * settings.mismatch_multiplier
                total = total + value
            else:
                value = student_logits.sum() * 0
            results.append(
                ChunkResult(
                    b,
                    chunk.name,
                    tuple(student_predictions),
                    tuple(teacher_predictions),
                    tuple(pairs),
                    value,
                    filtered,
                )
            )
    return TeacherResult(
        total * settings.temperature**2 / teacher.global_valid_chunks, tuple(results)
    )


def sparse_objective_oracle(
    fixture: SparseFixture,
    student_logits: torch.Tensor,
    settings: ObjectiveSettings = ObjectiveSettings(),
) -> ObjectiveResult:
    teachers = tuple(
        teacher_sparse_oracle(fixture, teacher, student_logits, settings)
        for teacher in fixture.teachers
    )
    kd = student_logits.sum() * 0
    smallest_vocab = min(teacher.real_vocab_size for teacher in fixture.teachers)
    for teacher, result in zip(fixture.teachers, teachers, strict=True):
        vocabulary_scale = (
            (
                torch.log(student_logits.new_tensor(float(teacher.real_vocab_size)))
                / torch.log(student_logits.new_tensor(float(smallest_vocab)))
            )
            if settings.normalize_teacher_by_vocab
            else 1.0
        )
        kd = kd + teacher.weight * vocabulary_scale * result.loss
    student_lp = student_logits[..., : fixture.student_vocab_size].log_softmax(-1)
    target_lp = (
        student_lp[:, :-1]
        .gather(-1, fixture.student_input_ids[:, 1:, None])
        .squeeze(-1)
    )
    label_mask = fixture.token_mask[:, 1:] * fixture.sample_mask[:, None]
    ce = -(target_lp * label_mask).sum() / fixture.global_valid_tokens
    ratio = (
        torch.where(
            kd.detach().abs() > 0,
            ce.detach().abs() / kd.detach().abs(),
            torch.ones_like(kd),
        )
        if settings.dynamic_scaling
        else torch.ones_like(kd)
    )
    loss = (
        ce + ratio * kd
        if settings.dynamic_scaling
        else settings.ce_weight * ce + settings.kd_weight * kd
    )
    return ObjectiveResult(loss, ce, kd, ratio, teachers)


def run_oracle_self_checks() -> None:
    """Finite differences, teacher/sample partitioning, and explicit edge cases."""
    fixture = build_sparse_fixture(dtype=torch.float64)
    logits = fixture.student_logits.clone().requires_grad_()
    result = sparse_objective_oracle(fixture, logits)
    gradient = torch.autograd.grad(result.loss, logits)[0]
    assert bool(torch.isfinite(gradient).all()) and bool(torch.isfinite(result.loss))
    assert torch.count_nonzero(gradient[..., fixture.student_vocab_size :]) == 0
    for coordinate in ((0, 1, 8), (0, 5, 9), (0, 6, 4), (1, 3, 8), (1, 6, 5)):
        plus, minus = logits.detach().clone(), logits.detach().clone()
        plus[coordinate] += 1e-5
        minus[coordinate] -= 1e-5
        derivative = (
            sparse_objective_oracle(fixture, plus).loss
            - sparse_objective_oracle(fixture, minus).loss
        ) / 2e-5
        torch.testing.assert_close(
            derivative, gradient[coordinate], rtol=1e-5, atol=1e-8
        )
    permuted = replace(fixture, teachers=tuple(reversed(fixture.teachers)))
    torch.testing.assert_close(
        sparse_objective_oracle(permuted, logits).loss, result.loss
    )
    pieces = [
        sparse_objective_oracle(fixture.slice_samples(b, b + 1), logits[b : b + 1]).loss
        for b in range(2)
    ]
    torch.testing.assert_close(sum(pieces), result.loss)
    split_gradient = torch.autograd.grad(sum(pieces), logits)[0]
    torch.testing.assert_close(split_gradient, gradient)
    for teacher, teacher_result in zip(fixture.teachers, result.teachers, strict=True):
        sidecar = forced_label_sidecar(teacher)
        assert bool((sidecar[0, 0] >= 0).all())
        assert int(sidecar[0, 0, 0]) != int(sidecar[0, 0, 1])
        assert all(len(record.support) <= 4 for record in teacher_result.chunks)
        assert all(
            len(record.support) == 1
            for record in teacher_result.chunks
            if "many_to_many" in record.name
        )
        assert any(
            len(record.support) >= 2
            for record in teacher_result.chunks
            if "many_to_one" in record.name
        )
        assert any(
            record.student_predictors == (1, 2) for record in teacher_result.chunks
        )
        natural = stable_natural_ids(
            teacher.logits[0, 0], real_vocab_size=teacher.real_vocab_size, k=4
        )
        assert all(int(token) not in natural for token in sidecar[0, 0])
        assert not any(
            "absent" in record.name and len(record.student_predictors) == 0
            for record in teacher_result.chunks
        )
    empty = replace(fixture, sample_mask=torch.zeros_like(fixture.sample_mask))
    zero = sparse_objective_oracle(
        empty, logits, ObjectiveSettings(dynamic_scaling=True)
    )
    assert zero.loss == 0 and zero.ratio == 1
    torch.testing.assert_close(
        torch.autograd.grad(zero.loss, logits)[0], torch.zeros_like(logits)
    )
    zero_weight = replace(
        fixture,
        teachers=(replace(fixture.teachers[0], weight=0.0), fixture.teachers[1]),
    )
    weighted = sparse_objective_oracle(zero_weight, logits)
    torch.testing.assert_close(weighted.kd, weighted.teachers[1].loss * 0.7)
    empty_teacher = replace(
        fixture,
        teachers=(fixture.teachers[0].without_valid_chunks(), fixture.teachers[1]),
    )
    empty_result = sparse_objective_oracle(empty_teacher, logits)
    assert empty_result.teachers[0].loss == 0
    torch.testing.assert_close(empty_result.kd, empty_result.teachers[1].loss * 0.7)
    dynamic = sparse_objective_oracle(
        fixture, logits, ObjectiveSettings(dynamic_scaling=True)
    )
    assert not dynamic.ratio.requires_grad
    torch.testing.assert_close(dynamic.loss, 2 * dynamic.ce)
    # Dynamic gradients use one detached microbatch-wide ratio, rather than
    # differentiating through CE/KD or silently choosing per-sample ratios.
    dynamic_gradient = torch.autograd.grad(dynamic.loss, logits)[0]
    fixed_terms = sparse_objective_oracle(fixture, logits)
    frozen_ratio_gradient = torch.autograd.grad(
        fixed_terms.ce + dynamic.ratio * fixed_terms.kd, logits
    )[0]
    torch.testing.assert_close(dynamic_gradient, frozen_ratio_gradient)
    filtered = sparse_objective_oracle(
        fixture, logits, ObjectiveSettings(noise_filter_k=6)
    )
    assert any(
        chunk.filtered for teacher in filtered.teachers for chunk in teacher.chunks
    )
    assert not torch.equal(
        build_sparse_fixture(generation=1).teachers[0].logits,
        fixture.teachers[0].logits.float(),
    )
    print(
        f"PASS sparse oracle loss={float(result.loss.detach()):.10f} "
        f"teacher_losses={[float(t.loss.detach()) for t in result.teachers]}"
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    run_oracle_self_checks()
