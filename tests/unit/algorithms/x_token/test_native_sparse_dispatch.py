"""Independent two-teacher checks through the native xToken loss dispatcher.

These tests preserve full-step normalizers while splitting student microbatches.
Only fixed scaling is required to be invariant to that split. Dynamic scaling
is checked against the oracle with the identical microbatch grouping.
"""

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

from tests.unit.algorithms.x_token.native_sparse_fixtures import (
    ObjectiveSettings,
    build_sparse_fixture,
    sparse_objective_oracle,
)


def _variants():
    fixture = build_sparse_fixture()
    settings = ObjectiveSettings()
    yield "fixed", fixture, settings
    yield "dynamic", fixture, replace(settings, dynamic_scaling=True)
    yield "vocab_scale", fixture, replace(settings, normalize_teacher_by_vocab=True)
    yield "reverse_order", replace(fixture, teachers=fixture.teachers[::-1]), settings
    yield (
        "zero_weight",
        replace(
            fixture,
            teachers=(replace(fixture.teachers[0], weight=0.0), fixture.teachers[1]),
        ),
        settings,
    )
    yield (
        "empty_teacher",
        replace(
            fixture,
            teachers=(fixture.teachers[0].without_valid_chunks(), fixture.teachers[1]),
        ),
        settings,
    )
    yield (
        "all_masked",
        replace(fixture, sample_mask=torch.zeros_like(fixture.sample_mask)),
        replace(settings, dynamic_scaling=True),
    )
    yield (
        "all_zero_weights",
        replace(
            fixture,
            teachers=tuple(replace(t, weight=0.0) for t in fixture.teachers),
        ),
        replace(settings, dynamic_scaling=True),
    )
    yield "noise", fixture, replace(settings, noise_filter_k=6)
    yield "next_generation", build_sparse_fixture(generation=1), settings
    yield "averaged_fallback", fixture, settings
    yield (
        "bf16_student",
        replace(fixture, student_logits=fixture.student_logits.to(torch.bfloat16)),
        settings,
    )


def _batch_data(fixture, payloads):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    data = BatchedDataDict(
        input_ids=fixture.student_input_ids,
        token_mask=fixture.token_mask,
        kd_token_mask=fixture.kd_token_mask,
        sample_mask=fixture.sample_mask,
    )
    for i, teacher in enumerate(fixture.teachers):
        student_chunks = torch.full_like(fixture.student_input_ids, -1)
        teacher_chunks = torch.full_like(teacher.input_ids, -1)
        for b, chunks in enumerate(teacher.chunks):
            for j, chunk in enumerate(chunks):
                student_chunks[b, slice(*chunk.student)] = j
                teacher_chunks[b, slice(*chunk.teacher)] = j
        _, _, valid = teacher.alignment_tensors()
        data[f"teacher_{i}_input_ids"] = teacher.input_ids
        data[f"teacher_{i}_token_mask"] = torch.ones_like(teacher.input_ids)
        data[f"teacher_{i}_sparse_logits_ipc"] = payloads[i].samples
        data[f"alignment_{i}_student_chunk_id"] = student_chunks
        data[f"alignment_{i}_teacher_chunk_id"] = teacher_chunks
        data[f"alignment_{i}_pair_valid"] = valid
        data[f"alignment_{i}_pair_is_correct"] = valid
    return data


def _no_legacy_paths(stack):
    # Guard preparation, objective and metrics; all tensors must stay on the
    # native student owner, including labels crossing its two segment edges.
    from nemo_rl.algorithms.loss import loss_functions
    from nemo_rl.algorithms.x_token import loss_utils

    for module, names in (
        (
            loss_utils,
            (
                "cp_load_balanced_to_contiguous",
                "allgather_cp_contiguous_tensor",
                "rebuild_teacher_full_logits_from_ipc",
                "rebuild_teacher_sparse_logits_from_ipc",
            ),
        ),
        (
            loss_functions,
            (
                "allgather_cp_contiguous_tensor",
                "student_next_token_ce",
                "get_next_token_logprobs_from_logits",
                "next_token_accuracy",
            ),
        ),
    ):
        for name in names:
            stack.enter_context(
                patch.object(
                    module,
                    name,
                    side_effect=AssertionError(f"Native dispatcher called {name}"),
                )
            )


def _accuracy(fixture):
    mask = fixture.token_mask[:, 1:] * fixture.sample_mask[:, None]
    predicted = fixture.student_logits[:, :-1, : fixture.student_vocab_size].argmax(-1)
    correct = (predicted == fixture.student_input_ids[:, 1:]) * mask
    return correct.sum() / mask.sum().clamp_min(1)


def _run_dispatch(rank, tp_size, cp_size, rendezvous, table_root):
    from nemo_rl.algorithms.loss.loss_input import prepare_loss_input
    from tests.unit.algorithms.x_token.test_native_sparse_loss import (
        make_alignment,
        make_loss_fn,
        make_native_student,
        make_parallel_groups,
        make_sparse_payload,
    )

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=tp_size * cp_size,
        timeout=timedelta(seconds=180),
    )
    try:
        tp_rank, cp_rank = divmod(rank, cp_size)
        tp_group, cp_group = make_parallel_groups(tp_size, cp_size)
        checked = 0
        for name, fixture, settings in _variants():
            vocab_width = fixture.student_logits.shape[-1] // tp_size
            vocab_slice = slice(tp_rank * vocab_width, (tp_rank + 1) * vocab_width)
            table_dir = Path(table_root) / str(rank) / name
            table_dir.mkdir(parents=True)
            loss_fn = make_loss_fn(fixture, settings, table_dir)
            if name == "averaged_fallback":
                loss_fn.kd_loss_mode = "averaged_logits"
                loss_fn.cfg["kd_loss_mode"] = "averaged_logits"
            partition_results = []
            for microbatch_size in (2, 1):
                full_gradient = torch.zeros_like(fixture.student_logits)
                reported_loss = 0.0
                for start in range(0, fixture.student_logits.shape[0], microbatch_size):
                    stop = start + microbatch_size
                    part = fixture.slice_samples(start, stop)
                    # Construct local logits from the explicit native mapping,
                    # independently of the adapter's implementation.
                    if cp_size == 1:
                        positions = torch.arange(part.student_logits.shape[1])
                    else:
                        segment = part.student_logits.shape[1] // (2 * cp_size)
                        positions = torch.cat(
                            (
                                torch.arange(
                                    cp_rank * segment, (cp_rank + 1) * segment
                                ),
                                torch.arange(
                                    (2 * cp_size - cp_rank - 1) * segment,
                                    (2 * cp_size - cp_rank) * segment,
                                ),
                            )
                        )
                    local = part.student_logits[:, positions, vocab_slice].clone()
                    local.requires_grad_()
                    native = make_native_student(
                        part, local, cp_rank=cp_rank, cp_size=cp_size
                    )
                    payloads = {
                        i: make_sparse_payload(teacher, settings, cp_size=2)
                        for i, teacher in enumerate(part.teachers)
                    }
                    aligns = {
                        i: make_alignment(part, teacher)
                        for i, teacher in enumerate(part.teachers)
                    }
                    data = _batch_data(part, payloads)
                    with ExitStack() as guards:
                        _no_legacy_paths(guards)
                        # Exercise the complete public adapter/caller path on
                        # ordinary alignments as well as the direct typed loss
                        # boundary used for exact collision fixture spans.
                        if name == "fixed":
                            inputs, _ = prepare_loss_input(
                                local,
                                data,
                                loss_fn,
                                vocab_parallel_group=tp_group,
                                context_parallel_group=cp_group,
                                native_cp_enabled=True,
                                native_sparse_enabled=True,
                            )
                            assert inputs["student_logits_contig"] is None
                            assert inputs["native_student"].logits is local
                        else:
                            inputs = dict(
                                logits=local,
                                student_logits_contig=None,
                                teacher_full_logits_by_idx={},
                                teacher_sparse_logits_by_idx={},
                                aligns_by_idx=aligns,
                                native_student=native,
                                native_sparse_teachers=payloads,
                                tp_group=tp_group,
                                cp_group=cp_group,
                                megatron_cp_normalize=True,
                            )
                        loss, metrics = loss_fn(
                            data,
                            torch.tensor(float(part.sample_mask.sum())),
                            torch.tensor(part.global_valid_tokens),
                            global_valid_kd_toks=torch.tensor(part.global_valid_tokens),
                            global_valid_chunks_by_idx={
                                i: torch.tensor(t.global_valid_chunks)
                                for i, t in enumerate(part.teachers)
                            },
                            **inputs,
                        )
                        loss.backward()
                    oracle_logits = part.student_logits.float().clone().requires_grad_()
                    expected = sparse_objective_oracle(part, oracle_logits, settings)
                    expected.loss.backward()
                    cp_loss = loss.detach().clone()
                    dist.all_reduce(cp_loss, group=cp_group)
                    torch.testing.assert_close(
                        cp_loss, expected.loss, rtol=1e-4, atol=1e-5
                    )
                    reference_gradient = oracle_logits.grad[:, positions, vocab_slice]
                    if local.dtype == torch.bfloat16:
                        actual_gradient = local.grad.float()
                        norm = reference_gradient.norm().clamp_min(1e-12)
                        assert (
                            actual_gradient - reference_gradient
                        ).norm() / norm <= 0.02
                        assert (actual_gradient.norm() / norm - 1).abs() <= 0.02
                    else:
                        torch.testing.assert_close(
                            local.grad,
                            reference_gradient,
                            rtol=1e-4,
                            atol=1e-5,
                        )
                    for key, expected_value in (
                        ("loss", expected.loss),
                        ("ce_loss", expected.ce),
                        ("kl_loss", expected.kd),
                        ("kl_loss_scale", expected.ratio),
                        ("accuracy", _accuracy(part)),
                    ):
                        assert metrics[key] == pytest.approx(
                            float(expected_value.detach()), rel=1e-4, abs=1e-5
                        ), (name, key)
                    for i, teacher_result in enumerate(expected.teachers):
                        assert metrics[f"kl_loss_t{i}"] == pytest.approx(
                            float(teacher_result.loss.detach()), rel=1e-4, abs=1e-5
                        )
                    full_gradient[start:stop, positions, vocab_slice] = local.grad
                    reported_loss += metrics["loss"]
                    checked += 1
                partition_results.append((reported_loss, full_gradient))
            if not settings.dynamic_scaling:
                assert partition_results[0][0] == pytest.approx(
                    partition_results[1][0], rel=1e-4, abs=1e-5
                )
                torch.testing.assert_close(
                    partition_results[0][1],
                    partition_results[1][1],
                    rtol=1e-4,
                    atol=1e-5,
                )
        if rank == 0:
            print(
                f"PASS native sparse dispatcher TP{tp_size}/CP{cp_size}: {checked} combined loss/gradient comparisons",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_native_sparse_dispatch(tp_size, cp_size):
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _run_dispatch,
            args=(tp_size, cp_size, str(Path(tmp) / "rendezvous"), tmp),
            nprocs=tp_size * cp_size,
            join=True,
        )


if __name__ == "__main__":
    for tp_size, cp_size in ((1, 1), (2, 1), (1, 2), (2, 2)):
        test_native_sparse_dispatch(tp_size, cp_size)
