"""Native same-tokenizer objective, dispatcher, and corrected baseline parity."""

import tempfile
from contextlib import ExitStack
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.algorithms.x_token.dense_teacher import DenseTeacherIPC
from nemo_rl.algorithms.x_token.loss_utils import LocalizedAlignment
from nemo_rl.algorithms.x_token.native_same_tokenizer import (
    compute_native_same_tokenizer_kl,
)
from tests.unit.algorithms.x_token.native_sparse_fixtures import (
    ObjectiveResult,
    ObjectiveSettings,
    TeacherResult,
    build_sparse_fixture,
)
from tests.unit.algorithms.x_token.test_native_sparse_loss import (
    make_native_student,
    make_parallel_groups,
)


def make_same_tokenizer_fixture():
    """Distinct padded dense teachers; shared IDs, masks and explicit full counts."""
    fixture = build_sparse_fixture()
    generator = torch.Generator().manual_seed(4421)
    teachers = []
    for index, teacher in enumerate(fixture.teachers):
        sequence = 8 if index == 0 else 12
        logits = torch.randn(2, sequence, 20, generator=generator) * 1.3
        # Each native CP owner has a distinct important common-K column.
        logits[0, 1, 3 + index] = 8
        logits[0, 5, 10 + index] = 9
        # Final / masked predictor maxima and padded vocab must not select K.
        logits[:, 7:, 14] = 100
        logits[..., 16:] = 200
        input_ids = torch.zeros(2, sequence, dtype=torch.long)
        input_ids[:, :8] = fixture.student_input_ids
        teachers.append(
            replace(
                teacher,
                logits=logits,
                input_ids=input_ids,
                real_vocab_size=fixture.student_vocab_size,
                forward=(),
                reverse=(),
            )
        )
    return replace(fixture, teachers=tuple(teachers))


def make_same_tokenizer_loss_fn(fixture, settings, *, mode="sum"):
    from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn

    count = len(fixture.teachers)
    return CrossTokenizerDistillationLossFn(
        {
            "temperature": settings.temperature,
            "vocab_topk": settings.k,
            "reverse_kl": settings.reverse_kl,
            "kl_loss_weight": settings.kd_weight,
            "ce_loss_scale": settings.ce_weight,
            "dynamic_loss_scaling": settings.dynamic_scaling,
            "kd_loss_mode": mode,
            "normalize_teacher_by_vocab": settings.normalize_teacher_by_vocab,
            "alpha": 1.0,
            "sum_weights_metric": None,
            "student_vocab_size": fixture.student_vocab_size,
            "teacher_vocab_sizes": [t.real_vocab_size for t in fixture.teachers],
            "projection_matrix_paths": [None] * count,
            "teacher_is_cross_tokenizer": [False] * count,
            "teacher_weights": [t.weight for t in fixture.teachers],
            "pseudo_target_paths": [None] * count,
            "reverse_pseudo_target_paths": [None] * count,
            "teacher_topk_ipc_k": 0,
        }
    )


def make_same_dense_payload(teacher, *, tp_size=1, cp_size=1):
    """Existing rectangular dense wire layout with unequal producer MB slots.

    CPU-backed descriptors are intentional geometry fixtures even when the
    student test runs on CUDA. Production CUDA transport is covered separately
    by the reusable owner/fanout and actual MCore integration tests.
    """
    logits = teacher.logits.detach().cpu().float()
    batch, sequence, vocab = logits.shape
    assert sequence % cp_size == vocab % tp_size == 0
    mbs = teacher.microbatch_size
    slots = (batch + mbs - 1) // mbs
    samples = [{"teacher_shards": [], "batch_item_id": b} for b in range(batch)]
    local_seq, local_vocab = sequence // cp_size, vocab // tp_size
    for cp_rank in range(cp_size):
        start_seq = cp_rank * local_seq
        for tp_rank in range(tp_size):
            start_vocab = tp_rank * local_vocab
            buffer = torch.zeros(slots * mbs, local_seq, local_vocab)
            buffer[:batch] = logits[
                :,
                start_seq : start_seq + local_seq,
                start_vocab : start_vocab + local_vocab,
            ]
            buffer = buffer.reshape(slots, mbs, local_seq, local_vocab)
            for b in range(batch):
                samples[b]["teacher_shards"].append(
                    {
                        "payload_ipc": buffer,
                        "buf_idx": b // mbs,
                        "sample_index_in_buf": b % mbs,
                        "batch_item_id": b,
                        "tp_rank": tp_rank,
                        "tp_size": tp_size,
                        "cp_rank": cp_rank,
                        "cp_size": cp_size,
                        "global_seq_start": start_seq,
                        "actual_shape": (local_seq, local_vocab),
                        "full_seq_len": sequence,
                        "full_vocab_size": vocab,
                        "vocab_start_index": start_vocab,
                        "vocab_end_index": start_vocab + local_vocab,
                    }
                )
    return DenseTeacherIPC(samples)


def _teacher_same_oracle(fixture, logits, teacher_rows, settings, *, full_vocab=False):
    mask = fixture.kd_token_mask[:, 1:] * fixture.sample_mask[:, None]
    teacher = teacher_rows[
        :, : logits.shape[1] - 1, : fixture.student_vocab_size
    ].float()
    student = logits[:, :-1, : fixture.student_vocab_size].float()
    if not full_vocab:
        k = min(settings.k, fixture.student_vocab_size)
        valid_rows = teacher[mask.bool()]
        columns = (
            valid_rows.max(0).values.topk(k).indices.sort().values
            if valid_rows.shape[0]
            else torch.arange(k, device=logits.device)
        )
        teacher, student = teacher[..., columns], student[..., columns]
    student_lp = (student / settings.temperature).log_softmax(-1)
    teacher_lp = (teacher / settings.temperature).log_softmax(-1)
    per_row = (
        student_lp.exp() * (student_lp - teacher_lp)
        if settings.reverse_kl
        else teacher_lp.exp() * (teacher_lp - student_lp)
    ).sum(-1)
    return (
        (per_row * mask).sum() * settings.temperature**2 / fixture.global_valid_tokens
    )


def same_tokenizer_oracle(fixture, student_logits, settings, *, mode="sum"):
    """Independent raw dense oracle with the same microbatch common-K objective."""
    teachers = tuple(
        TeacherResult(
            _teacher_same_oracle(fixture, student_logits, t.logits, settings), ()
        )
        for t in fixture.teachers
    )
    if mode == "averaged_logits":
        rows = sum(
            t.logits[:, : student_logits.shape[1], : fixture.student_vocab_size].float()
            * t.weight
            for t in fixture.teachers
        ) / sum(t.weight for t in fixture.teachers)
        kd = _teacher_same_oracle(
            fixture, student_logits, rows, settings, full_vocab=True
        )
    else:
        kd = sum(
            t.weight * result.loss for t, result in zip(fixture.teachers, teachers)
        )
    logprobs = student_logits[..., : fixture.student_vocab_size].float().log_softmax(-1)
    target_lp = (
        logprobs[:, :-1].gather(-1, fixture.student_input_ids[:, 1:, None]).squeeze(-1)
    )
    mask = fixture.token_mask[:, 1:] * fixture.sample_mask[:, None]
    ce = -(target_lp * mask).sum() / fixture.global_valid_tokens
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


@pytest.mark.parametrize(
    "k,reverse,full_vocab",
    [
        (4, False, False),
        (4, True, False),
        (16, False, False),
        (16, True, False),
        (1, False, False),
        (0, False, False),
        (4, False, True),
        (4, True, True),
    ],
)
def test_native_same_tokenizer_direct(k, reverse, full_vocab):
    torch.set_num_threads(1)
    fixture = make_same_tokenizer_fixture()
    settings = ObjectiveSettings(k=k, reverse_kl=reverse)
    loss_fn = make_same_tokenizer_loss_fn(fixture, settings)
    # A strided B/S tensor exercises backward allocation and direct indexing.
    storage = torch.empty(4, 16, 20)
    storage[::2, ::2] = fixture.student_logits
    logits = storage[::2, ::2].detach().requires_grad_()
    teacher = fixture.teachers[0].logits[:, :8, :16].clone().requires_grad_()
    actual = compute_native_same_tokenizer_kl(
        loss_fn,
        make_native_student(fixture, logits),
        teacher,
        global_valid_toks=torch.tensor(fixture.global_valid_tokens),
        tp_group=None,
        cp_group=None,
        full_vocab=full_vocab,
    )
    reference = fixture.student_logits.clone().requires_grad_()
    expected = _teacher_same_oracle(
        fixture, reference, teacher.detach(), settings, full_vocab=full_vocab
    )
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, reference.grad, atol=2e-7, rtol=2e-5)
    assert teacher.grad is None
    assert torch.count_nonzero(logits.grad[..., 16:]) == 0


@pytest.mark.parametrize("count", [-1.0, float("nan"), float("inf")])
def test_native_same_tokenizer_invalid_normalizer(count):
    fixture = make_same_tokenizer_fixture()
    with pytest.raises(ValueError, match="normalizer"):
        compute_native_same_tokenizer_kl(
            make_same_tokenizer_loss_fn(fixture, ObjectiveSettings()),
            make_native_student(fixture, fixture.student_logits),
            fixture.teachers[0].logits[..., :16],
            global_valid_toks=torch.tensor(count),
            tp_group=None,
            cp_group=None,
        )


def test_native_same_tokenizer_bounded_tiles():
    fixture = make_same_tokenizer_fixture()
    repeat = 40
    fixture = replace(
        fixture,
        student_logits=fixture.student_logits.repeat(repeat, 1, 1),
        student_input_ids=fixture.student_input_ids.repeat(repeat, 1),
        token_mask=fixture.token_mask.repeat(repeat, 1),
        kd_token_mask=fixture.kd_token_mask.repeat(repeat, 1),
        sample_mask=fixture.sample_mask.repeat(repeat),
        teachers=tuple(
            replace(t, logits=t.logits.repeat(repeat, 1, 1)) for t in fixture.teachers
        ),
    )
    from nemo_rl.algorithms.x_token import native_same_tokenizer as implementation

    original = implementation._row_logprobs
    observed = []

    def check_rows(logits, teacher, batch, *args):
        observed.append(batch.numel())
        assert batch.numel() <= 64
        return original(logits, teacher, batch, *args)

    for full_vocab in (False, True):
        settings = ObjectiveSettings(reverse_kl=True)
        logits = fixture.student_logits.clone().requires_grad_()
        with patch.object(implementation, "_row_logprobs", side_effect=check_rows):
            actual = compute_native_same_tokenizer_kl(
                make_same_tokenizer_loss_fn(fixture, settings),
                make_native_student(fixture, logits),
                fixture.teachers[0].logits[..., :16],
                global_valid_toks=torch.tensor(fixture.global_valid_tokens),
                tp_group=None,
                cp_group=None,
                full_vocab=full_vocab,
            )
            actual.backward()
        reference = fixture.student_logits.clone().requires_grad_()
        expected = _teacher_same_oracle(
            fixture,
            reference,
            fixture.teachers[0].logits,
            settings,
            full_vocab=full_vocab,
        )
        expected.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(logits.grad, reference.grad, rtol=2e-5, atol=2e-7)
    assert len(observed) > 8 and max(observed) == 64


def _same_data(fixture, payloads):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    data = BatchedDataDict(
        input_ids=fixture.student_input_ids,
        token_mask=fixture.token_mask,
        kd_token_mask=fixture.kd_token_mask,
        sample_mask=fixture.sample_mask,
    )
    for i, payload in enumerate(payloads):
        data[f"teacher_{i}_full_logits_ipc"] = payload.samples
    return data


def _variants():
    fixture = make_same_tokenizer_fixture()
    settings = ObjectiveSettings()
    yield "common_k", fixture, settings, "sum"
    yield "dynamic", fixture, replace(settings, dynamic_scaling=True), "sum"
    yield "reverse", fixture, replace(settings, reverse_kl=True), "sum"
    yield "full", fixture, replace(settings, k=16), "sum"
    yield "average", fixture, settings, "averaged_logits"
    yield (
        "average_dynamic",
        fixture,
        replace(settings, dynamic_scaling=True, reverse_kl=True),
        "averaged_logits",
    )
    yield (
        "all_masked",
        replace(fixture, sample_mask=torch.zeros_like(fixture.sample_mask)),
        replace(settings, dynamic_scaling=True),
        "sum",
    )
    # One CP owner has no active KD rows, while it still participates in common K.
    mask = torch.zeros_like(fixture.kd_token_mask)
    mask[:, 3:5] = 1
    yield "empty_owner", replace(fixture, kd_token_mask=mask), settings, "sum"
    yield (
        "zero_weight",
        replace(
            fixture,
            teachers=(replace(fixture.teachers[0], weight=0.0), fixture.teachers[1]),
        ),
        settings,
        "sum",
    )
    yield (
        "reverse_order",
        replace(fixture, teachers=fixture.teachers[::-1]),
        settings,
        "sum",
    )
    yield (
        "bf16",
        replace(fixture, student_logits=fixture.student_logits.to(torch.bfloat16)),
        settings,
        "sum",
    )
    yield (
        "average_bf16",
        replace(fixture, student_logits=fixture.student_logits.to(torch.bfloat16)),
        settings,
        "averaged_logits",
    )
    # With the original padded width20 and realV8, TP rank1 owns only padding.
    tiny_vocab = replace(
        fixture,
        student_vocab_size=8,
        student_input_ids=fixture.student_input_ids % 8,
        teachers=tuple(
            replace(t, real_vocab_size=8, input_ids=t.input_ids % 8)
            for t in fixture.teachers
        ),
    )
    yield "padding_only_tp", tiny_vocab, settings, "averaged_logits"


def _to_device(fixture, device):
    return replace(
        fixture,
        student_logits=fixture.student_logits.to(device),
        student_input_ids=fixture.student_input_ids.to(device),
        token_mask=fixture.token_mask.to(device),
        kd_token_mask=fixture.kd_token_mask.to(device),
        sample_mask=fixture.sample_mask.to(device),
        teachers=tuple(
            replace(t, logits=t.logits.to(device), input_ids=t.input_ids.to(device))
            for t in fixture.teachers
        ),
    )


class _CPUReferenceGather(torch.autograd.Function):
    """The pinned MCore concat/split gather, with CPU-compatible allocation.

    MCore mappings.py::_GatherFromModelParallelRegion gathers in forward and
    only splits in backward. Its CUDA-only allocator prevents using that
    unchanged reference on Gloo; CUDA tests use the actual MCore implementation.
    """

    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        tensors = [torch.empty_like(value) for _ in range(dist.get_world_size(group))]
        dist.all_gather(tensors, value, group=group)
        return torch.cat(tensors, -1)

    @staticmethod
    def backward(ctx, gradient):
        return gradient.chunk(dist.get_world_size(ctx.group), -1)[
            dist.get_rank(ctx.group)
        ].contiguous(), None


def _check_corrected_baseline(
    fixture, settings, mode, *, tp_group, cp_group, tp_rank, tp_size, cp_rank, cp_size
):
    """Contiguous CP baseline with padded vocabulary removed before full KL."""
    loss_fn = make_same_tokenizer_loss_fn(fixture, settings, mode=mode)
    sequence = fixture.student_logits.shape[1]
    local_sequence = sequence // cp_size
    positions = slice(cp_rank * local_sequence, (cp_rank + 1) * local_sequence)
    # The pre-native full-vocab helper included padded LM-head columns. Removing
    # them before TP slicing is the real-vocabulary corrected dense reference.
    width = fixture.student_vocab_size // tp_size
    vocab = slice(tp_rank * width, (tp_rank + 1) * width)
    logits = (
        fixture.student_logits.float()[:, positions, : fixture.student_vocab_size][
            ..., vocab
        ]
        .clone()
        .requires_grad_()
    )
    align = LocalizedAlignment(
        sample_mask=fixture.sample_mask,
        student_input_ids=fixture.student_input_ids[:, positions],
        student_token_mask=fixture.token_mask[:, positions],
        student_kd_token_mask=fixture.kd_token_mask[:, positions],
    )
    count = torch.tensor(fixture.global_valid_tokens, device=logits.device)
    if mode == "averaged_logits":
        teacher_rows = sum(
            t.weight * t.logits[:, :sequence, : fixture.student_vocab_size].float()
            for t in fixture.teachers
        ) / sum(t.weight for t in fixture.teachers)
        with ExitStack() as stack:
            if logits.device.type == "cpu" and tp_size > 1:
                stack.enter_context(
                    patch(
                        "megatron.core.tensor_parallel.gather_from_tensor_model_parallel_region",
                        side_effect=lambda value, group: _CPUReferenceGather.apply(
                            value, group
                        ),
                    )
                )
            baseline = loss_fn._direct_full_vocab_kl(
                logits,
                teacher_rows[:, positions],
                align,
                count,
                tp_group=tp_group,
                cp_group=cp_group,
            )
    else:
        baseline = sum(
            t.weight
            * loss_fn._direct_topk_kl(
                logits,
                t.logits[:, positions, : fixture.student_vocab_size],
                align,
                count,
                tp_group=tp_group,
                cp_group=cp_group,
            )
            for t in fixture.teachers
        )
    baseline.backward()
    reference = fixture.student_logits.float().clone().requires_grad_()
    expected = same_tokenizer_oracle(fixture, reference, settings, mode=mode).kd
    expected.backward()
    complete = baseline.detach().clone()
    dist.all_reduce(complete, group=cp_group)
    torch.testing.assert_close(complete, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        logits.grad, reference.grad[:, positions, vocab], rtol=3e-5, atol=2e-6
    )


def _run_same_grid(rank, tp_size, cp_size, rendezvous, backend):
    from nemo_rl.algorithms.loss.loss_input import prepare_loss_input
    from nemo_rl.distributed.selected_logprobs import cp_native_global_positions
    from tests.unit.algorithms.x_token.test_native_sparse_dispatch import (
        _accuracy,
        _no_legacy_paths,
    )

    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=tp_size * cp_size,
        timeout=timedelta(seconds=240),
    )
    try:
        tp_rank, cp_rank = divmod(rank, cp_size)
        tp_group, cp_group = make_parallel_groups(tp_size, cp_size)
        checked = 0
        for name, original, settings, mode in _variants():
            fixture = _to_device(original, device)
            loss_fn = make_same_tokenizer_loss_fn(fixture, settings, mode=mode)
            for microbatch_size in (2, 1):
                for start in range(0, 2, microbatch_size):
                    part = fixture.slice_samples(start, start + microbatch_size)
                    positions = cp_native_global_positions(
                        8, cp_rank, cp_size, device=device
                    )
                    width = part.student_logits.shape[-1] // tp_size
                    vocab = slice(tp_rank * width, (tp_rank + 1) * width)
                    logits = (
                        part.student_logits[:, positions, vocab]
                        .clone()
                        .requires_grad_()
                    )
                    # Differing teacher/student TP/CP factorizations keep the
                    # same two/four process placement envelope.
                    payloads = [
                        make_same_dense_payload(t, tp_size=cp_size, cp_size=tp_size)
                        for t in part.teachers
                    ]
                    data = _same_data(part, payloads)
                    with ExitStack() as guards:
                        _no_legacy_paths(guards)
                        inputs, _ = prepare_loss_input(
                            logits,
                            data,
                            loss_fn,
                            vocab_parallel_group=tp_group,
                            context_parallel_group=cp_group,
                            native_cp_enabled=True,
                            native_same_tokenizer_enabled=True,
                        )
                        assert inputs["student_logits_contig"] is None
                        assert inputs["native_student"].logits is logits
                        count = torch.tensor(part.global_valid_tokens, device=device)
                        actual, metrics = loss_fn(
                            data,
                            torch.tensor(float(part.sample_mask.sum()), device=device),
                            count,
                            global_valid_kd_toks=count,
                            **inputs,
                        )
                        actual.backward()
                    reference = part.student_logits.float().clone().requires_grad_()
                    expected = same_tokenizer_oracle(
                        part, reference, settings, mode=mode
                    )
                    expected.loss.backward()
                    complete = actual.detach().clone()
                    dist.all_reduce(complete, group=cp_group)
                    torch.testing.assert_close(
                        complete, expected.loss, rtol=5e-5, atol=5e-6
                    )
                    gradient = reference.grad[:, positions, vocab]
                    if logits.dtype == torch.bfloat16:
                        norm = gradient.norm().clamp_min(1e-12)
                        assert (logits.grad.float() - gradient).norm() / norm < 0.02
                    else:
                        torch.testing.assert_close(
                            logits.grad, gradient, rtol=5e-5, atol=5e-6
                        )
                    for key, value in (
                        ("loss", expected.loss),
                        ("ce_loss", expected.ce),
                        ("kl_loss", expected.kd),
                        ("kl_loss_scale", expected.ratio),
                        ("accuracy", _accuracy(part)),
                    ):
                        assert metrics[key] == pytest.approx(
                            float(value.detach()), rel=5e-5, abs=5e-6
                        ), (name, key)
                    if mode == "sum":
                        for i, result in enumerate(expected.teachers):
                            assert metrics[f"kl_loss_t{i}"] == pytest.approx(
                                float(result.loss.detach()), rel=5e-5, abs=5e-6
                            )
                    _check_corrected_baseline(
                        part,
                        settings,
                        mode,
                        tp_group=tp_group,
                        cp_group=cp_group,
                        tp_rank=tp_rank,
                        tp_size=tp_size,
                        cp_rank=cp_rank,
                        cp_size=cp_size,
                    )
                    checked += 1
        if rank == 0:
            print(
                f"PASS native same-tokenizer {backend} TP{tp_size}/CP{cp_size}: {checked} dispatcher + corrected contiguous baseline comparisons",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_native_same_tokenizer_cpu_grid(tp_size, cp_size):
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _run_same_grid,
            args=(tp_size, cp_size, str(Path(tmp) / "init"), "gloo"),
            nprocs=tp_size * cp_size,
            join=True,
        )


@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_native_same_tokenizer_cuda_grid(tp_size, cp_size):
    if torch.cuda.device_count() < tp_size * cp_size:
        pytest.skip("CUDA/NCCL grid requires enough visible GPUs")
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _run_same_grid,
            args=(tp_size, cp_size, str(Path(tmp) / "init"), "nccl"),
            nprocs=tp_size * cp_size,
            join=True,
        )
