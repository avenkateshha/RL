"""Native MCore sparse + reconstructed DTensor sparse share one student.

The legacy reference deliberately executes the preserved CP1 v6 implementation
on the identical frozen sparse rows. Its transport/support semantics are not
substituted with the new native forced-sidecar objective.
"""

import tempfile
from dataclasses import replace
from datetime import timedelta
from itertools import product
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.unit.algorithms.x_token.native_sparse_fixtures import (
    ObjectiveSettings,
    build_sparse_fixture,
    teacher_sparse_oracle,
)


def _legacy_payload(teacher, settings):
    raw = teacher.logits[..., : teacher.real_vocab_size].float()
    # Old DTensor payloads carry a per-position support and exact logZ. This
    # controlled fixture keeps that support fixed in all TP/CP configurations.
    ids = raw.argsort(dim=-1, descending=True, stable=True)[..., : settings.k]
    return (
        raw.gather(-1, ids),
        ids.to(torch.int32),
        (raw / settings.temperature).logsumexp(-1),
        None,
    )


def _run_mixed(rank, tp_size, cp_size, rendezvous, table_root):
    from megatron.core.tensor_parallel import mappings

    from nemo_rl.algorithms.x_token.loss_utils import cp_load_balanced_to_contiguous
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
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
    original_gather = mappings._gather_along_last_dim

    def cpu_gather(tensor, group):
        with patch("torch.cuda.current_device", return_value=tensor.device):
            return original_gather(tensor, group)

    try:
        with patch.object(mappings, "_gather_along_last_dim", side_effect=cpu_gather):
            tp_rank, cp_rank = divmod(rank, cp_size)
            tp_group, cp_group = make_parallel_groups(tp_size, cp_size)
            fixture = build_sparse_fixture()
            fixture = replace(
                fixture,
                student_logits=fixture.student_logits[
                    ..., : fixture.student_vocab_size
                ].clone(),
            )
            width = fixture.student_vocab_size // tp_size
            v_slice = slice(tp_rank * width, (tp_rank + 1) * width)
            checked = 0
            for reverse_order in (False, True):
                ordered = (
                    replace(fixture, teachers=fixture.teachers[::-1])
                    if reverse_order
                    else fixture
                )
                native_index, legacy_index = (1, 0) if reverse_order else (0, 1)
                for mode, dynamic in product(("sum", "averaged_logits"), (False, True)):
                    settings = ObjectiveSettings(dynamic_scaling=dynamic)
                    table_dir = (
                        Path(table_root)
                        / str(rank)
                        / f"{reverse_order}-{dynamic}-{mode}"
                    )
                    table_dir.mkdir(parents=True)
                    fn = make_loss_fn(ordered, settings, table_dir)
                    fn.kd_loss_mode = mode
                    fn.cfg["kd_loss_mode"] = mode
                    for microbatch_size in (2, 1):
                        for start in range(0, 2, microbatch_size):
                            part = ordered.slice_samples(start, start + microbatch_size)
                            segment = 8 // (2 * cp_size)
                            positions = (
                                torch.arange(8)
                                if cp_size == 1
                                else torch.cat(
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
                            )
                            local = (
                                part.student_logits[:, positions, v_slice]
                                .clone()
                                .requires_grad_()
                            )
                            native = make_native_student(
                                part, local, cp_rank=cp_rank, cp_size=cp_size
                            )
                            contiguous = cp_load_balanced_to_contiguous(
                                local, cp_group=cp_group
                            )
                            s_slice = slice(
                                cp_rank * (8 // cp_size), (cp_rank + 1) * (8 // cp_size)
                            )
                            native_teacher = part.teachers[native_index]
                            legacy_teacher = part.teachers[legacy_index]
                            legacy_rows = _legacy_payload(legacy_teacher, settings)
                            t_width = legacy_teacher.input_ids.shape[1] // cp_size
                            t_slice = slice(cp_rank * t_width, (cp_rank + 1) * t_width)
                            # Legacy reconstruction replicates the full sparse
                            # sequence on every student CP rank. Its alignment
                            # input IDs alone retain contiguous local windows.
                            legacy_align = replace(
                                make_alignment(part, legacy_teacher),
                                student_input_ids=part.student_input_ids[:, s_slice],
                                teacher_input_ids=legacy_teacher.input_ids[:, t_slice],
                                student_token_mask=part.token_mask[:, s_slice],
                                student_kd_token_mask=part.kd_token_mask[:, s_slice],
                            )
                            data = BatchedDataDict(
                                input_ids=part.student_input_ids,
                                token_mask=part.token_mask,
                                kd_token_mask=part.kd_token_mask,
                                sample_mask=part.sample_mask,
                            )
                            for i, teacher in enumerate(part.teachers):
                                data[f"teacher_{i}_input_ids"] = teacher.input_ids
                                data[f"teacher_{i}_token_mask"] = torch.ones_like(
                                    teacher.input_ids
                                )
                            loss, metrics = fn(
                                data,
                                torch.tensor(float(part.sample_mask.sum())),
                                torch.tensor(part.global_valid_tokens),
                                logits=local,
                                student_logits_contig=contiguous,
                                teacher_full_logits_by_idx={},
                                teacher_sparse_logits_by_idx={
                                    legacy_index: legacy_rows
                                },
                                aligns_by_idx={
                                    native_index: make_alignment(part, native_teacher),
                                    legacy_index: legacy_align,
                                },
                                native_student=native,
                                native_sparse_teachers={
                                    native_index: make_sparse_payload(
                                        native_teacher, settings, cp_size=2
                                    )
                                },
                                global_valid_chunks_by_idx={
                                    i: torch.tensor(t.global_valid_chunks)
                                    for i, t in enumerate(part.teachers)
                                },
                                global_valid_kd_toks=torch.tensor(
                                    part.global_valid_tokens
                                ),
                                tp_group=tp_group,
                                cp_group=cp_group,
                                megatron_cp_normalize=True,
                            )
                            loss.backward()
                            reference = part.student_logits.clone().requires_grad_()
                            native_kd = teacher_sparse_oracle(
                                part, native_teacher, reference, settings
                            ).loss
                            legacy_kd, _ = fn._compute_prefix_bidir_partition_kl_v3(
                                legacy_index,
                                reference,
                                None,
                                make_alignment(part, legacy_teacher),
                                teacher_sparse_payload=legacy_rows,
                                teacher_vocab_size=legacy_teacher.real_vocab_size,
                                global_valid_chunks=torch.tensor(
                                    legacy_teacher.global_valid_chunks
                                ),
                                tp_group=None,
                                cp_group=None,
                            )
                            ce = torch.nn.functional.cross_entropy(
                                reference[:, :-1].flatten(0, 1),
                                part.student_input_ids[:, 1:].flatten(),
                                reduction="none",
                            )
                            ce = (
                                ce.reshape_as(part.token_mask[:, 1:])
                                * part.token_mask[:, 1:]
                                * part.sample_mask[:, None]
                            ).sum() / part.global_valid_tokens
                            kd = (
                                native_teacher.weight * native_kd
                                + legacy_teacher.weight * legacy_kd
                            )
                            ratio = (
                                ce.detach().abs() / kd.detach().abs()
                                if dynamic
                                else kd.new_tensor(1.0)
                            )
                            expected = (
                                ce + ratio * kd
                                if dynamic
                                else settings.ce_weight * ce + settings.kd_weight * kd
                            )
                            expected.backward()
                            reported = loss.detach().clone()
                            dist.all_reduce(reported, group=cp_group)
                            torch.testing.assert_close(
                                reported, expected.detach(), rtol=1e-4, atol=1e-5
                            )
                            torch.testing.assert_close(
                                local.grad,
                                reference.grad[:, positions, v_slice],
                                rtol=1e-4,
                                atol=1e-5,
                            )
                            for key, value in (
                                ("ce_loss", ce),
                                ("kl_loss", kd),
                                ("kl_loss_scale", ratio),
                                (f"kl_loss_t{native_index}", native_kd),
                                (f"kl_loss_t{legacy_index}", legacy_kd),
                            ):
                                assert metrics[key] == pytest.approx(
                                    float(value.detach()), rel=1e-4, abs=1e-5
                                ), key
                            checked += 1
            if rank == 0:
                print(
                    f"PASS mixed native/legacy sparse TP{tp_size}/CP{cp_size}: {checked} loss/gradient comparisons",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_native_sparse_legacy_mix(tp_size, cp_size):
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _run_mixed,
            args=(tp_size, cp_size, str(Path(tmp) / "rendezvous"), tmp),
            nprocs=tp_size * cp_size,
            join=True,
        )


if __name__ == "__main__":
    for tp_size, cp_size in ((1, 1), (2, 1), (1, 2), (2, 2)):
        test_native_sparse_legacy_mix(tp_size, cp_size)
