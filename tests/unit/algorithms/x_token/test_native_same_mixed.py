"""Combined native dense KL with native sparse or retained legacy dense v6.

Each microbatch uses the same common-K support as its independent reference;
changing microbatch size may change that support and is not an invariance claim.
"""

import math
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
    teacher_sparse_oracle,
)


def _cases():
    settings = ObjectiveSettings()
    yield "fixed", settings, "sum"
    yield "dynamic", replace(settings, dynamic_scaling=True), "sum"
    yield "reverse_kl", replace(settings, reverse_kl=True), "sum"
    yield "full_vocab", replace(settings, k=16), "sum"
    yield "vocab_scale", replace(settings, normalize_teacher_by_vocab=True), "sum"
    yield "fallback", settings, "averaged_logits"
    yield "zero_weight", settings, "sum"
    yield "empty_cross", settings, "sum"
    yield "masked", replace(settings, dynamic_scaling=True), "sum"


def _run_mixed(rank, tp_size, cp_size, legacy_dense, rendezvous, table_root):
    from megatron.core.tensor_parallel import mappings

    from nemo_rl.algorithms.x_token import loss_utils
    from nemo_rl.algorithms.loss.loss_input import prepare_loss_input
    from nemo_rl.distributed.selected_logprobs import cp_native_global_positions
    from tests.unit.algorithms.x_token.test_native_same_tokenizer import (
        make_same_dense_payload,
        make_same_tokenizer_fixture,
        same_tokenizer_oracle,
    )
    from tests.unit.algorithms.x_token.test_native_sparse_dispatch import (
        _batch_data,
        _no_legacy_paths,
    )
    from tests.unit.algorithms.x_token.test_native_sparse_loss import (
        make_alignment,
        make_loss_fn,
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
            base = build_sparse_fixture()
            same = make_same_tokenizer_fixture().teachers[0]
            # Retain valid table files for the mixed constructor; the same
            # teacher's routing flag ensures they are never consulted.
            same = replace(
                same, forward=base.teachers[0].forward, reverse=base.teachers[0].reverse
            )
            base = replace(base, teachers=(same, base.teachers[1]))
            if legacy_dense:
                # Preserve legacy v6's established unpadded vocabulary input.
                base = replace(
                    base, student_logits=base.student_logits[..., :16].clone()
                )
            width = base.student_logits.shape[-1] // tp_size
            v_slice = slice(tp_rank * width, (tp_rank + 1) * width)
            checked = 0
            for reverse_order in (False, True):
                same_index, cross_index = (1, 0) if reverse_order else (0, 1)
                for name, settings, mode in _cases():
                    fixture = base
                    if name == "zero_weight":
                        fixture = replace(
                            fixture,
                            teachers=(replace(same, weight=0), fixture.teachers[1]),
                        )
                    elif name == "empty_cross":
                        fixture = replace(
                            fixture,
                            teachers=(same, fixture.teachers[1].without_valid_chunks()),
                        )
                    elif name == "masked":
                        fixture = replace(
                            fixture, sample_mask=torch.zeros_like(fixture.sample_mask)
                        )
                    if reverse_order:
                        fixture = replace(fixture, teachers=fixture.teachers[::-1])
                    fn = make_loss_fn(
                        fixture,
                        settings,
                        Path(table_root) / str(rank) / f"{reverse_order}-{name}",
                    )
                    fn.teacher_is_cross_tokenizer[same_index] = False
                    fn.kd_loss_mode = mode
                    fn.cfg["kd_loss_mode"] = mode
                    if legacy_dense:
                        fn.cfg["teacher_topk_ipc_k"] = 0
                    for microbatch_size in (2, 1):
                        for start in range(0, 2, microbatch_size):
                            part = fixture.slice_samples(start, start + microbatch_size)
                            positions = cp_native_global_positions(
                                8, cp_rank, cp_size, device="cpu"
                            )
                            local = (
                                part.student_logits[:, positions, v_slice]
                                .clone()
                                .requires_grad_()
                            )
                            same_teacher, cross_teacher = (
                                part.teachers[same_index],
                                part.teachers[cross_index],
                            )
                            payloads = {
                                i: make_same_dense_payload(t, tp_size=2, cp_size=2)
                                if i == same_index or legacy_dense
                                else make_sparse_payload(t, settings, cp_size=2)
                                for i, t in enumerate(part.teachers)
                            }
                            data = _batch_data(part, payloads)
                            data[f"teacher_{same_index}_full_logits_ipc"] = data.pop(
                                f"teacher_{same_index}_sparse_logits_ipc"
                            )
                            t_width = cross_teacher.input_ids.shape[1] // cp_size
                            t_slice = slice(cp_rank * t_width, (cp_rank + 1) * t_width)
                            s_width = 8 // cp_size
                            s_slice = slice(cp_rank * s_width, (cp_rank + 1) * s_width)
                            if legacy_dense:
                                data[f"teacher_{cross_index}_full_logits_ipc"] = (
                                    data.pop(f"teacher_{cross_index}_sparse_logits_ipc")
                                )
                            with ExitStack() as guards:
                                if legacy_dense:
                                    # The retained adapter's device selection is
                                    # CUDA-only; keep its tensor/CP routing real
                                    # while running this numerical fixture on CPU.
                                    guards.enter_context(
                                        patch(
                                            "torch.cuda.current_device",
                                            return_value=local.device,
                                        )
                                    )
                                    relay = guards.enter_context(
                                        patch.object(
                                            loss_utils,
                                            "cp_load_balanced_to_contiguous",
                                            wraps=loss_utils.cp_load_balanced_to_contiguous,
                                        )
                                    )
                                    rebuild = guards.enter_context(
                                        patch.object(
                                            loss_utils,
                                            "rebuild_teacher_full_logits_from_ipc",
                                            return_value=(
                                                cross_teacher.logits[
                                                    :,
                                                    t_slice,
                                                    : cross_teacher.real_vocab_size,
                                                ].float(),
                                                0,
                                            ),
                                        )
                                    )
                                else:
                                    _no_legacy_paths(guards)
                                inputs, _ = prepare_loss_input(
                                    local,
                                    data,
                                    fn,
                                    vocab_parallel_group=tp_group,
                                    context_parallel_group=cp_group,
                                    native_cp_enabled=True,
                                    native_sparse_enabled=True,
                                    native_same_tokenizer_enabled=True,
                                )
                                assert set(inputs["native_dense_teachers"]) == {
                                    same_index
                                }
                                if legacy_dense:
                                    assert relay.call_count == 1
                                    assert rebuild.call_count == 1
                                    assert inputs["student_logits_contig"] is not None
                                else:
                                    assert inputs["student_logits_contig"] is None
                                # Explicit spans retain the collision fixture;
                                # adapter-derived spans are checked separately.
                                alignment = make_alignment(part, cross_teacher)
                                if legacy_dense:
                                    alignment = replace(
                                        alignment,
                                        student_input_ids=part.student_input_ids[
                                            :, s_slice
                                        ],
                                        teacher_input_ids=cross_teacher.input_ids[
                                            :, t_slice
                                        ],
                                        student_token_mask=part.token_mask[:, s_slice],
                                        student_kd_token_mask=part.kd_token_mask[
                                            :, s_slice
                                        ],
                                    )
                                inputs["aligns_by_idx"][cross_index] = alignment
                                kd_denominator = 23.0
                                loss, metrics = fn(
                                    data,
                                    torch.tensor(float(part.sample_mask.sum())),
                                    torch.tensor(part.global_valid_tokens),
                                    global_valid_kd_toks=torch.tensor(kd_denominator),
                                    global_valid_chunks_by_idx={
                                        i: torch.tensor(t.global_valid_chunks)
                                        for i, t in enumerate(part.teachers)
                                    },
                                    **inputs,
                                )
                                loss.backward()
                            reference = (
                                part.student_logits.float().clone().requires_grad_()
                            )
                            same_ref = (
                                same_tokenizer_oracle(
                                    replace(
                                        part,
                                        teachers=(same_teacher,),
                                        global_valid_tokens=kd_denominator,
                                    ),
                                    reference,
                                    settings,
                                )
                                .teachers[0]
                                .loss
                            )
                            if legacy_dense:
                                cross_ref, _ = fn._compute_prefix_bidir_partition_kl_v3(
                                    cross_index,
                                    reference,
                                    cross_teacher.logits[
                                        ..., : cross_teacher.real_vocab_size
                                    ].float(),
                                    make_alignment(part, cross_teacher),
                                    teacher_vocab_size=cross_teacher.real_vocab_size,
                                    global_valid_chunks=torch.tensor(
                                        cross_teacher.global_valid_chunks
                                    ),
                                    tp_group=None,
                                    cp_group=None,
                                )
                            else:
                                cross_ref = teacher_sparse_oracle(
                                    part, cross_teacher, reference, settings
                                ).loss
                            terms = {same_index: same_ref, cross_index: cross_ref}
                            weighted = {
                                i: t.weight
                                * terms[i]
                                * (
                                    math.log(t.real_vocab_size)
                                    / math.log(min(fn.teacher_vocab_sizes))
                                    if settings.normalize_teacher_by_vocab
                                    and mode == "sum"
                                    else 1
                                )
                                for i, t in enumerate(part.teachers)
                            }
                            kd = sum(weighted.values())
                            ce = torch.nn.functional.cross_entropy(
                                reference[:, :-1, :16].flatten(0, 1),
                                part.student_input_ids[:, 1:].flatten(),
                                reduction="none",
                            ).reshape_as(part.token_mask[:, 1:])
                            ce = (
                                ce * part.token_mask[:, 1:] * part.sample_mask[:, None]
                            ).sum() / part.global_valid_tokens
                            ratio = (
                                torch.where(
                                    kd.detach().abs() > 0,
                                    ce.detach().abs() / kd.detach().abs(),
                                    torch.ones_like(kd),
                                )
                                if settings.dynamic_scaling
                                else torch.ones_like(kd)
                            )
                            expected = (
                                ce + ratio * kd
                                if settings.dynamic_scaling
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
                                (f"kl_loss_t{same_index}", same_ref),
                                (f"kl_loss_t{cross_index}", cross_ref),
                                *(
                                    (f"teacher_{i}/weighted_kl", weighted[i])
                                    for i in range(2)
                                ),
                            ):
                                assert metrics[key] == pytest.approx(
                                    float(value.detach()), rel=1e-4, abs=1e-5
                                ), key
                            checked += 1
            if rank == 0:
                print(
                    f"PASS native same + {'legacy dense' if legacy_dense else 'native sparse'} TP{tp_size}/CP{cp_size}: {checked} combined loss/gradient comparisons",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "legacy_dense", [False, True], ids=["native_sparse", "legacy_dense"]
)
@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_native_same_mixed(tp_size, cp_size, legacy_dense):
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _run_mixed,
            args=(tp_size, cp_size, legacy_dense, f"{tmp}/rendezvous", f"{tmp}/tables"),
            nprocs=tp_size * cp_size,
            join=True,
        )
