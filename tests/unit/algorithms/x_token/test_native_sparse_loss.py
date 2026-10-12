"""Native sparse v6 loss/gradient equivalence against an independent oracle."""

from dataclasses import fields, replace
from datetime import timedelta
from itertools import product
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.algorithms.x_token.loss_utils import (
    LocalizedAlignment,
    NativeStudentContext,
)
from nemo_rl.algorithms.x_token.native_sparse_loss import (
    compute_native_sparse_teacher_loss,
)
from nemo_rl.algorithms.x_token.sparse_teacher import (
    SparseTeacherIPC,
    SparseTeacherRowReader,
    SparseTeacherShard,
)
from nemo_rl.distributed.selected_logprobs import (
    cp_native_global_positions,
    cp_native_sequence_segments,
)
from nemo_rl.distributed.sparse_topk import distributed_vocab_topk_logz_force
from tests.unit.algorithms.x_token.native_sparse_fixtures import (
    ObjectiveSettings,
    SparseFixture,
    build_sparse_fixture,
    forced_label_sidecar,
    teacher_sparse_oracle,
    write_tables,
)


def make_loss_fn(fixture, settings, table_dir, *, loss_overrides=None):
    from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn

    tables = [write_tables(teacher, table_dir) for teacher in fixture.teachers]
    return CrossTokenizerDistillationLossFn(
        {
            "temperature": settings.temperature,
            "vocab_topk": settings.k,
            "reverse_kl": settings.reverse_kl,
            "kl_loss_weight": settings.kd_weight,
            "ce_loss_scale": settings.ce_weight,
            "dynamic_loss_scaling": settings.dynamic_scaling,
            "kd_loss_mode": "sum",
            "normalize_teacher_by_vocab": settings.normalize_teacher_by_vocab,
            "alpha": 1.0,
            "sum_weights_metric": None,
            "student_vocab_size": fixture.student_vocab_size,
            "teacher_vocab_sizes": [t.real_vocab_size for t in fixture.teachers],
            "projection_matrix_paths": [None] * len(fixture.teachers),
            "teacher_is_cross_tokenizer": [True] * len(fixture.teachers),
            "teacher_weights": [t.weight for t in fixture.teachers],
            "pseudo_target_paths": [str(fwd) for fwd, _ in tables],
            "reverse_pseudo_target_paths": [str(rev) for _, rev in tables],
            "common_indices_from_subtoks": True,
            "teacher_topk_ipc_k": settings.k,
            "teacher_topk_ipc_keep_realized": True,
            "kl_chunk_shift": settings.shift,
            "prefix_bidir_v3_loss_fn": settings.common_loss,
            "prefix_bidir_v3_last_pos_loss_fn": settings.mismatch_loss,
            "prefix_bidir_v3_jsd_beta": settings.jsd_beta,
            "prefix_bidir_v3_mismatch_pos0_alpha": 0.0,
            "prefix_bidir_v3_mismatch_loss_beta": settings.mismatch_multiplier,
            "prefix_bidir_v3_noise_filter_topk": settings.noise_filter_k,
            **(
                {
                    "prefix_bidir_v3_realized_only_mismatch": (
                        settings.realized_only_mismatch
                    )
                }
                if settings.realized_only_mismatch is not None
                else {}
            ),
            **(loss_overrides or {}),
        }
    )


def make_sparse_payload(teacher, settings, *, cp_size=1, device=torch.device("cpu")):
    """Seven real scored fields, with independent teacher CP/slot geometry."""
    device = torch.device(device)
    batch_size, sequence, vocabulary = teacher.logits.shape
    mbs = teacher.microbatch_size
    slot_count = (batch_size + mbs - 1) // mbs
    samples = [
        {"teacher_shards": [], "batch_item_id": batch} for batch in range(batch_size)
    ]
    forced = forced_label_sidecar(teacher, shift=settings.shift).to(device)
    for cp_rank in range(cp_size):
        positions = cp_native_global_positions(
            sequence, cp_rank, cp_size, device=device
        )
        scored = distributed_vocab_topk_logz_force(
            teacher.logits.to(device).index_select(1, positions),
            forced.index_select(1, positions),
            settings.k,
            None,
            vocab_start_index=0,
            vocab_end_index=vocabulary,
            real_vocab_size=teacher.real_vocab_size,
            temperature=settings.temperature,
            noise_filter_k=settings.noise_filter_k,
            chunk_size=2,
        )
        buffers = {}
        for field in fields(scored):
            value = getattr(scored, field.name)
            if value.ndim == 2:
                value = value.unsqueeze(-1)
            buffer = torch.zeros(
                (slot_count * mbs, *value.shape[1:]), dtype=value.dtype, device=device
            )
            buffer[:batch_size] = value
            buffers[field.name + "_ipc"] = buffer.reshape(
                slot_count, mbs, *value.shape[1:]
            )
        for batch in range(batch_size):
            shard = SparseTeacherShard(
                k=settings.k,
                temperature=settings.temperature,
                real_vocab_size=teacher.real_vocab_size,
                membership_k=settings.noise_filter_k or settings.k,
                force_width=2,
                full_seq_len=sequence,
                local_seq_len=len(positions),
                buf_idx=batch // mbs,
                sample_index_in_buf=batch % mbs,
                sequence_segments=tuple(
                    cp_native_sequence_segments(sequence, cp_rank, cp_size)
                ),
                **buffers,
            ).to_record()
            shard["batch_item_id"] = batch
            samples[batch]["teacher_shards"].append(shard)
    return SparseTeacherIPC(samples)


def make_alignment(fixture, teacher, *, device=torch.device("cpu")):
    student_spans, teacher_spans, pair_valid = teacher.alignment_tensors()
    return LocalizedAlignment(
        sample_mask=fixture.sample_mask.to(device),
        pair_valid=pair_valid.to(device),
        student_input_ids=fixture.student_input_ids.to(device),
        student_token_mask=fixture.token_mask.to(device),
        student_kd_token_mask=fixture.kd_token_mask.to(device),
        teacher_input_ids=teacher.input_ids.to(device),
        student_spans=student_spans.to(device),
        teacher_spans=teacher_spans.to(device),
        num_chunks=pair_valid.sum(-1).to(device),
    )


def make_native_student(fixture, logits, *, cp_rank=0, cp_size=1):
    """Logits are already native CP-local; IDs/masks remain the full sample."""
    device = logits.device
    sequence = fixture.student_input_ids.shape[1]
    positions = cp_native_global_positions(sequence, cp_rank, cp_size, device=device)
    next_positions = (positions + 1).clamp_max(sequence - 1)
    active = (positions + 1 < sequence).to(logits.dtype)
    input_ids = fixture.student_input_ids.to(device)
    return NativeStudentContext(
        logits=logits,
        global_positions=positions,
        input_ids=input_ids,
        next_token_ids=input_ids.index_select(1, next_positions),
        next_token_mask=fixture.token_mask.to(device).index_select(1, next_positions)
        * active,
        next_kd_token_mask=fixture.kd_token_mask.to(device).index_select(
            1, next_positions
        )
        * active,
        sample_mask=fixture.sample_mask.to(device),
        full_seq_len=sequence,
        real_vocab_size=fixture.student_vocab_size,
    )


def make_parallel_groups(tp_size, cp_size):
    """Return TP then CP groups for rank = tp_rank * cp_size + cp_rank."""
    rank = dist.get_rank()
    tp_group = cp_group = None
    for cp in range(cp_size):
        ranks = [tp * cp_size + cp for tp in range(tp_size)]
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group
    for tp in range(tp_size):
        ranks = [tp * cp_size + cp for cp in range(cp_size)]
        group = dist.new_group(ranks)
        if rank in ranks:
            cp_group = group
    return tp_group, cp_group


def _assert_case(
    fixture,
    settings,
    table_dir,
    *,
    tp_group=None,
    cp_group=None,
    tp_rank=0,
    tp_size=1,
    cp_rank=0,
    cp_size=1,
    device=torch.device("cpu"),
    zero_count=False,
    loss_fn=None,
):
    if loss_fn is None:
        loss_fn = make_loss_fn(fixture, settings, table_dir)
    full = fixture.student_logits.to(device)
    positions = cp_native_global_positions(
        full.shape[1], cp_rank, cp_size, device=device
    )
    width = full.shape[-1] // tp_size
    local = (
        full.index_select(1, positions)[..., tp_rank * width : (tp_rank + 1) * width]
        .clone()
        .requires_grad_()
    )
    reference = full.float().clone().requires_grad_()
    combined, expected = local.sum() * 0, reference.sum() * 0
    for index, teacher in enumerate(fixture.teachers):
        teacher_device = replace(teacher, logits=teacher.logits.to(device).float())
        payload = make_sparse_payload(teacher, settings, cp_size=2, device=device)
        reader = SparseTeacherRowReader(payload, device=device)
        actual, metrics = compute_native_sparse_teacher_loss(
            loss_fn,
            index,
            make_native_student(fixture, local, cp_rank=cp_rank, cp_size=cp_size),
            make_alignment(fixture, teacher, device=device),
            reader,
            global_valid_chunks=torch.tensor(
                0.0 if zero_count else teacher.global_valid_chunks, device=device
            ),
            tp_group=tp_group,
            cp_group=cp_group,
        )
        oracle = teacher_sparse_oracle(
            fixture, teacher_device, reference, settings
        ).loss
        if zero_count:
            oracle = oracle * 0
        reported = actual.detach().clone()
        if cp_group is not None:
            dist.all_reduce(reported, group=cp_group)
        torch.testing.assert_close(reported, oracle, rtol=2e-5, atol=2e-6)
        assert metrics["kl_loss"] == float(actual.detach())
        assert metrics["native_sparse_v6"] == 1
        combined = combined + teacher.weight * actual
        expected = expected + teacher.weight * oracle
    actual_gradient = torch.autograd.grad(combined, local)[0]
    expected_gradient = torch.autograd.grad(expected, reference)[0]
    wanted = expected_gradient.index_select(1, positions)[
        ..., tp_rank * width : (tp_rank + 1) * width
    ]
    torch.testing.assert_close(
        actual_gradient.float(),
        wanted,
        rtol=8e-3 if local.dtype == torch.bfloat16 else 3e-5,
        atol=5e-5 if local.dtype == torch.bfloat16 else 3e-6,
    )
    padded_start = max(0, fixture.student_vocab_size - tp_rank * width)
    assert actual_gradient[..., padded_start:].count_nonzero() == 0
    return combined.detach(), actual_gradient


@pytest.mark.parametrize(
    "settings",
    [
        ObjectiveSettings(),
        ObjectiveSettings(common_loss="kl", reverse_kl=True),
        ObjectiveSettings(mismatch_loss="jsd", jsd_beta=0.5),
        ObjectiveSettings(noise_filter_k=6),
        ObjectiveSettings(noise_filter_k=1),
        ObjectiveSettings(k=1),
        ObjectiveSettings(shift=False),
        ObjectiveSettings(realized_only_mismatch=False),
        ObjectiveSettings(realized_only_mismatch=True),
        ObjectiveSettings(
            realized_only_mismatch=True, common_loss="kl", reverse_kl=True
        ),
        ObjectiveSettings(realized_only_mismatch=True, mismatch_loss="jsd"),
        ObjectiveSettings(realized_only_mismatch=True, noise_filter_k=6),
        ObjectiveSettings(realized_only_mismatch=True, noise_filter_k=1),
        ObjectiveSettings(realized_only_mismatch=True, shift=False),
    ],
)
def test_native_sparse_two_teachers_match_selected_support_oracle(tmp_path, settings):
    _assert_case(build_sparse_fixture(), settings, tmp_path)


@pytest.mark.parametrize(
    "case", ["masked", "empty_teacher", "zero_weight", "zero_multiplier"]
)
@pytest.mark.parametrize("realized_only", [False, True])
def test_native_sparse_empty_and_zero_terms_keep_correct_gradients(
    tmp_path, case, realized_only
):
    fixture = build_sparse_fixture()
    settings = ObjectiveSettings(realized_only_mismatch=realized_only)
    if case == "masked":
        fixture = replace(fixture, sample_mask=torch.zeros_like(fixture.sample_mask))
    elif case == "empty_teacher":
        fixture = replace(
            fixture,
            teachers=(fixture.teachers[0].without_valid_chunks(), fixture.teachers[1]),
        )
    elif case == "zero_weight":
        fixture = replace(
            fixture,
            teachers=(replace(fixture.teachers[0], weight=0.0), fixture.teachers[1]),
        )
    else:
        settings = replace(settings, mismatch_multiplier=0.0)
    _assert_case(fixture, settings, tmp_path)


@pytest.mark.parametrize("realized_only", [False, True])
def test_native_sparse_reordered_microbatches_keep_full_step_denominators(
    tmp_path, realized_only
):
    fixture = build_sparse_fixture()
    settings = ObjectiveSettings(realized_only_mismatch=realized_only)
    for batch in (1, 0):
        _assert_case(fixture.slice_samples(batch, batch + 1), settings, tmp_path)


@pytest.mark.parametrize("denominator", [0.0, -1.0, float("inf"), float("nan")])
@pytest.mark.parametrize("realized_only", [False, True])
def test_native_sparse_explicit_normalizer_validation(
    tmp_path, denominator, realized_only
):
    fixture = build_sparse_fixture()
    settings = ObjectiveSettings(realized_only_mismatch=realized_only)
    loss_fn = make_loss_fn(fixture, settings, tmp_path)
    logits = fixture.student_logits.clone().requires_grad_()
    teacher = fixture.teachers[0]
    args = (
        loss_fn,
        0,
        make_native_student(fixture, logits),
        make_alignment(fixture, teacher),
        SparseTeacherRowReader(
            make_sparse_payload(teacher, settings), device=torch.device("cpu")
        ),
    )
    kwargs = dict(
        global_valid_chunks=torch.tensor(denominator), tp_group=None, cp_group=None
    )
    if denominator != 0:
        with pytest.raises(ValueError, match="denominator"):
            compute_native_sparse_teacher_loss(*args, **kwargs)
    else:
        loss, _ = compute_native_sparse_teacher_loss(*args, **kwargs)
        assert loss.item() == 0
        assert torch.autograd.grad(loss, logits)[0].count_nonzero() == 0


def _only_shape(fixture: SparseFixture, shape: str) -> SparseFixture:
    def matches(student: tuple[int, int], teacher: tuple[int, int]) -> bool:
        student_size, teacher_size = student[1] - student[0], teacher[1] - teacher[0]
        return {
            "common": student_size == teacher_size == 1,
            "one_to_many": student_size == 1 and teacher_size > 1,
            "many_to_one": student_size > 1 and teacher_size == 1,
            "many_to_many": student_size > 1 and teacher_size > 1,
        }[shape]

    return replace(
        fixture,
        teachers=tuple(
            replace(
                teacher,
                chunks=tuple(
                    tuple(
                        replace(
                            chunk,
                            valid=chunk.valid and matches(chunk.student, chunk.teacher),
                        )
                        for chunk in row
                    )
                    for row in teacher.chunks
                ),
            )
            for teacher in fixture.teachers
        ),
    )


@pytest.mark.parametrize(
    "shape", ["common", "one_to_many", "many_to_one", "many_to_many"]
)
def test_realized_only_changes_only_mismatch_alternatives(
    tmp_path: Path, shape: str
) -> None:
    fixture = _only_shape(build_sparse_fixture(), shape)
    settings = ObjectiveSettings(k=4, realized_only_mismatch=False)
    old_oracle = teacher_sparse_oracle(
        fixture, fixture.teachers[0], fixture.student_logits, settings
    )
    enabled = replace(settings, realized_only_mismatch=True)
    new_oracle = teacher_sparse_oracle(
        fixture, fixture.teachers[0], fixture.student_logits, enabled
    )
    if shape != "many_to_many":
        assert any(len(chunk.support) > 1 for chunk in old_oracle.chunks)
    if shape != "common":
        assert all(len(chunk.support) == 1 for chunk in new_oracle.chunks)
    old_loss, old_grad = _assert_case(fixture, settings, tmp_path)
    new_loss, new_grad = _assert_case(fixture, enabled, tmp_path)
    if shape in ("common", "many_to_many"):
        torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
        torch.testing.assert_close(new_grad, old_grad, rtol=0, atol=0)
    else:
        assert not torch.isclose(new_loss, old_loss)
        assert not torch.allclose(new_grad, old_grad)


def test_realized_only_k1_and_absent_flag_are_backward_compatible(
    tmp_path: Path,
) -> None:
    fixture = build_sparse_fixture()
    for first, second in (
        (ObjectiveSettings(), ObjectiveSettings(realized_only_mismatch=False)),
        (ObjectiveSettings(k=1), ObjectiveSettings(k=1, realized_only_mismatch=True)),
    ):
        old_loss, old_grad = _assert_case(fixture, first, tmp_path)
        new_loss, new_grad = _assert_case(fixture, second, tmp_path)
        torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
        torch.testing.assert_close(new_grad, old_grad, rtol=0, atol=0)


def realized_artifact_overrides(
    fixture: SparseFixture, directory: Path, *, common_from_subtoks: bool
) -> dict[str, object]:
    """Create only the common-map artifacts used by realized-only mode."""
    directory.mkdir(parents=True, exist_ok=True)
    forward_paths, projection_paths = [], []
    for teacher in fixture.teachers:
        if common_from_subtoks:
            forward, reverse = write_tables(teacher, directory)
            reverse.unlink()
            forward_paths.append(str(forward))
            projection_paths.append(None)
        else:
            indices = torch.zeros(fixture.student_vocab_size, 1, dtype=torch.long)
            likelihoods = torch.zeros(fixture.student_vocab_size, 1)
            for student_id, chain in enumerate(teacher.forward):
                if len(chain) == 1:
                    indices[student_id, 0] = chain[0]
                    likelihoods[student_id, 0] = 1
            projection = directory / f"{teacher.name}_projection.pt"
            torch.save({"indices": indices, "likelihoods": likelihoods}, projection)
            forward_paths.append(None)
            projection_paths.append(str(projection))
    return {
        "common_indices_from_subtoks": common_from_subtoks,
        "projection_matrix_paths": projection_paths,
        "pseudo_target_paths": forward_paths,
        "reverse_pseudo_target_paths": [None] * len(fixture.teachers),
    }


@pytest.mark.parametrize("common_from_subtoks", [False, True])
def test_realized_only_never_accesses_prefix_support_and_allows_artifact_omissions(
    tmp_path: Path, common_from_subtoks: bool
) -> None:
    fixture = build_sparse_fixture()
    settings = ObjectiveSettings(realized_only_mismatch=True)
    artifacts = realized_artifact_overrides(
        fixture, tmp_path / "artifacts", common_from_subtoks=common_from_subtoks
    )
    loss_fn = make_loss_fn(
        fixture, settings, tmp_path / "unused", loss_overrides=artifacts
    )
    with patch.object(
        loss_fn,
        "_ensure_bidir_prefix_support_index",
        side_effect=AssertionError("Prefix support must not be accessed"),
    ):
        _assert_case(fixture, settings, tmp_path, loss_fn=loss_fn)


@pytest.mark.parametrize("transport", ["dense", "sparse"])
def test_realized_only_rejected_at_legacy_loss_boundary(
    tmp_path: Path, transport: str
) -> None:
    fixture = build_sparse_fixture()
    settings = ObjectiveSettings(realized_only_mismatch=True)
    loss_fn = make_loss_fn(fixture, settings, tmp_path)
    teacher = fixture.teachers[0]
    with pytest.raises(ValueError, match="realized_only_mismatch"):
        loss_fn._compute_prefix_bidir_partition_kl_v3(
            0,
            fixture.student_logits,
            teacher.logits if transport == "dense" else None,
            make_alignment(fixture, teacher),
            teacher_sparse_payload=None
            if transport == "dense"
            else (None, None, None, None),
            teacher_vocab_size=teacher.real_vocab_size,
            global_valid_chunks=torch.tensor(teacher.global_valid_chunks),
        )


@pytest.mark.parametrize("invalid", ["false", "true", 0, 1, None])
def test_realized_only_constructor_rejects_nonboolean_values(
    tmp_path: Path, invalid: object
) -> None:
    with pytest.raises(ValueError, match="realized_only_mismatch"):
        make_loss_fn(
            build_sparse_fixture(),
            ObjectiveSettings(),
            tmp_path,
            loss_overrides={"prefix_bidir_v3_realized_only_mismatch": invalid},
        )


@pytest.mark.parametrize("unsupported", ["k_zero", "no_cross_teacher"])
def test_realized_only_constructor_rejects_unsupported_routes(
    tmp_path: Path, unsupported: str
) -> None:
    overrides = (
        {"teacher_topk_ipc_k": 0}
        if unsupported == "k_zero"
        else {
            "teacher_is_cross_tokenizer": [False, False],
            "pseudo_target_paths": [None, None],
            "reverse_pseudo_target_paths": [None, None],
        }
    )
    with pytest.raises(ValueError, match="realized_only_mismatch"):
        make_loss_fn(
            build_sparse_fixture(),
            ObjectiveSettings(realized_only_mismatch=True),
            tmp_path,
            loss_overrides=overrides,
        )


@pytest.mark.parametrize("transport", ["dense", "sparse"])
def test_absent_and_false_preserve_legacy_loss_and_gradients(
    tmp_path: Path, transport: str
) -> None:
    fixture, settings = build_sparse_fixture(), ObjectiveSettings()
    teacher = fixture.teachers[0]
    raw = teacher.logits[..., : teacher.real_vocab_size]
    values, ids = raw.topk(settings.k, dim=-1)
    payload = (values, ids.int(), (raw / settings.temperature).logsumexp(-1), None)
    results = []
    for flag in (None, False):
        loss_fn = make_loss_fn(
            fixture, replace(settings, realized_only_mismatch=flag), tmp_path
        )
        logits = fixture.student_logits.clone().requires_grad_()
        loss, _ = loss_fn._compute_prefix_bidir_partition_kl_v3(
            0,
            logits,
            raw if transport == "dense" else None,
            make_alignment(fixture, teacher),
            teacher_sparse_payload=payload if transport == "sparse" else None,
            teacher_vocab_size=teacher.real_vocab_size,
            global_valid_chunks=torch.tensor(teacher.global_valid_chunks),
        )
        results.append((loss.detach(), torch.autograd.grad(loss, logits)[0]))
    for old, explicit_false in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(old, explicit_false, rtol=0, atol=0)


def _grid_process(rank, tp_size, cp_size, init_path, table_dir, cuda):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=tp_size * cp_size,
        timeout=timedelta(seconds=120),
    )
    try:
        tp_rank, cp_rank = divmod(rank, cp_size)
        tp_group, cp_group = make_parallel_groups(tp_size, cp_size)
        fixture = build_sparse_fixture()
        for case, realized_only in product(
            (
                "normal",
                "masked",
                "empty_teacher",
                "zero_weight",
                "filtered",
                "bf16",
                "prefix_only",
                "zero_denominator",
            ),
            (False, True),
        ):
            current = fixture
            settings = ObjectiveSettings(realized_only_mismatch=realized_only)
            if case == "masked":
                current = replace(
                    fixture, sample_mask=torch.zeros_like(fixture.sample_mask)
                )
            elif case == "empty_teacher":
                current = replace(
                    fixture,
                    teachers=(
                        fixture.teachers[0].without_valid_chunks(),
                        fixture.teachers[1],
                    ),
                )
            elif case == "zero_weight":
                current = replace(
                    fixture,
                    teachers=(
                        replace(fixture.teachers[0], weight=0.0),
                        fixture.teachers[1],
                    ),
                )
            elif case == "filtered":
                settings = replace(settings, noise_filter_k=1)
            elif case == "bf16":
                current = replace(
                    fixture,
                    student_logits=fixture.student_logits.to(torch.bfloat16),
                    teachers=tuple(
                        replace(teacher, logits=teacher.logits.to(torch.bfloat16))
                        for teacher in fixture.teachers
                    ),
                )
            elif case == "prefix_only":
                current = replace(
                    fixture,
                    teachers=tuple(
                        replace(
                            teacher,
                            chunks=tuple(
                                tuple(
                                    replace(chunk, valid=(batch == 0 and index == 2))
                                    for index, chunk in enumerate(chunks)
                                )
                                for batch, chunks in enumerate(teacher.chunks)
                            ),
                        )
                        for teacher in fixture.teachers
                    ),
                )
            _assert_case(
                current,
                settings,
                table_dir / f"rank{rank}",
                tp_group=tp_group,
                cp_group=cp_group,
                tp_rank=tp_rank,
                tp_size=tp_size,
                cp_rank=cp_rank,
                cp_size=cp_size,
                device=device,
                zero_count=case == "zero_denominator",
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_native_sparse_cpu_tp_cp_loss_and_gradient_grids(tmp_path, tp_size, cp_size):
    mp.spawn(
        _grid_process,
        args=(tp_size, cp_size, str(tmp_path / "init"), tmp_path / "tables", False),
        nprocs=tp_size * cp_size,
        join=True,
    )


@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="Native sparse NCCL grids need four GPUs"
)
def test_native_sparse_cuda_tp_cp_loss_and_gradient_grids(tmp_path, tp_size, cp_size):
    mp.spawn(
        _grid_process,
        args=(tp_size, cp_size, str(tmp_path / "init"), tmp_path / "tables", True),
        nprocs=tp_size * cp_size,
        join=True,
    )
