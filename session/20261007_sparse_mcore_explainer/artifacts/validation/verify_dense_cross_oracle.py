"""CPU checks of R6 retained dense objective capture and term aggregation.

Dense v6 itself is deliberately the retained CP1 implementation in both paths;
this checks dispatcher plumbing, independent CE/weights/ratio and gradients. It
does not establish CUDA IPC, CP scheduling, model or optimizer acceptance.
"""

import json
import tempfile
from dataclasses import replace
from itertools import product
from pathlib import Path

import model_oracle
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from tests.unit.algorithms.x_token.native_sparse_fixtures import (
    ObjectiveSettings,
    build_sparse_fixture,
)
from tests.unit.algorithms.x_token.test_native_same_tokenizer import (
    make_same_dense_payload,
    make_same_tokenizer_fixture,
)
from tests.unit.algorithms.x_token.test_native_sparse_loss import (
    make_alignment,
    make_loss_fn,
    make_native_student,
)


def main():
    torch.set_num_threads(2)
    base = build_sparse_fixture()
    base = replace(base, student_logits=base.student_logits[..., :16].clone())
    same = make_same_tokenizer_fixture().teachers[0]
    same = replace(
        same, forward=base.teachers[0].forward, reverse=base.teachers[0].reverse
    )
    checked = 0
    position_zero_kd = {}
    position_zero_gradients = {}
    with tempfile.TemporaryDirectory() as tmp:
        for routing in ("dense_only", "dense_same", "same_dense"):
            teachers = {
                "dense_only": (base.teachers[1],),
                "dense_same": (base.teachers[1], same),
                "same_dense": (same, base.teachers[1]),
            }[routing]
            cross_indices = tuple(
                i for i, teacher in enumerate(teachers) if teacher is not same
            )
            fixture = replace(base, teachers=teachers)
            for mode, dynamic, reverse, position_zero in product(
                ("sum", "averaged_logits"), (False, True), (False, True), (False, True)
            ):
                settings = replace(
                    ObjectiveSettings(),
                    dynamic_scaling=dynamic,
                    reverse_kl=reverse,
                    normalize_teacher_by_vocab=True,
                )
                fn = make_loss_fn(fixture, settings, Path(tmp) / str(checked))
                fn.cfg["teacher_topk_ipc_k"] = 0
                fn.cfg["prefix_bidir_v3_position_0_kl"] = position_zero
                # Match R6's retained coefficient defaults. The sparse-native
                # fixture's alpha=0 would otherwise disable the term under test.
                fn.cfg.pop("prefix_bidir_v3_mismatch_pos0_alpha")
                fn.cfg.pop("prefix_bidir_v3_mismatch_loss_beta")
                fn.kd_loss_mode = fn.cfg["kd_loss_mode"] = mode
                fn.teacher_is_cross_tokenizer = [
                    i in cross_indices for i in range(len(teachers))
                ]
                fn.cfg["teacher_is_cross_tokenizer"] = fn.teacher_is_cross_tokenizer
                cfg = dict(fn.cfg)
                data = BatchedDataDict(
                    input_ids=fixture.student_input_ids,
                    token_mask=fixture.token_mask,
                    kd_token_mask=fixture.kd_token_mask,
                    sample_mask=fixture.sample_mask,
                )
                for index in cross_indices:
                    data[f"teacher_{index}_token_mask"] = torch.ones_like(
                        teachers[index].input_ids
                    )
                actual_logits = fixture.student_logits.clone().requires_grad_()
                expected_logits = fixture.student_logits.clone().requires_grad_()
                native_dense = {
                    i: make_same_dense_payload(teacher)
                    for i, teacher in enumerate(teachers)
                    if i not in cross_indices
                }
                denominator = 23.0
                actual, metrics = fn(
                    data,
                    torch.tensor(2.0),
                    torch.tensor(fixture.global_valid_tokens),
                    actual_logits,
                    actual_logits,
                    {
                        i: teacher.logits[..., : teacher.real_vocab_size]
                        for i, teacher in enumerate(teachers)
                        if i in cross_indices
                    },
                    {
                        i: make_alignment(fixture, teacher)
                        for i, teacher in enumerate(teachers)
                    },
                    global_valid_kd_toks=torch.tensor(denominator),
                    global_valid_chunks_by_idx={
                        i: torch.tensor(teacher.global_valid_chunks)
                        for i, teacher in enumerate(teachers)
                        if i in cross_indices
                    },
                    native_student=make_native_student(fixture, actual_logits)
                    if native_dense
                    else None,
                    native_dense_teachers=native_dense,
                    megatron_cp_normalize=True,
                )
                # sparse_fixture contains only cross-tokenizer entries,
                # while observed logits and indexed config cover all.
                cross_fixture = replace(
                    fixture,
                    teachers=tuple(teachers[i] for i in cross_indices),
                )
                expected, expected_metrics = model_oracle.full_objective(
                    cross_fixture,
                    [teacher.logits for teacher in teachers],
                    data,
                    expected_logits,
                    cfg,
                    denominator,
                    legacy_loss_fn=fn,
                    legacy_dense_teacher_indices=cross_indices,
                )
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                for key, value in expected_metrics.items():
                    torch.testing.assert_close(
                        torch.as_tensor(metrics[key]),
                        value.detach(),
                        rtol=1e-4,
                        atol=1e-5,
                    )
                actual.backward()
                expected.backward()
                torch.testing.assert_close(
                    actual_logits.grad,
                    expected_logits.grad,
                    rtol=1e-4,
                    atol=1e-5,
                )
                position_zero_kd[routing, mode, dynamic, reverse, position_zero] = (
                    float(expected_metrics["kl_loss"].detach())
                )
                position_zero_gradients[
                    routing, mode, dynamic, reverse, position_zero
                ] = expected_logits.grad.clone()
                checked += 1
        position_zero_deltas = []
        for routing, mode, dynamic, reverse in product(
            ("dense_only", "dense_same", "same_dense"),
            ("sum", "averaged_logits"),
            (False, True),
            (False, True),
        ):
            disabled = position_zero_kd[routing, mode, dynamic, reverse, False]
            enabled = position_zero_kd[routing, mode, dynamic, reverse, True]
            assert enabled > disabled, "Legacy position-zero term was not exercised"
            gradient_delta = (
                (
                    position_zero_gradients[routing, mode, dynamic, reverse, True]
                    - position_zero_gradients[routing, mode, dynamic, reverse, False]
                )
                .norm()
                .item()
            )
            assert gradient_delta > 0, "Legacy position-zero gradient was not exercised"
            position_zero_deltas.append(
                {
                    "routing": routing,
                    "mode": mode,
                    "dynamic": dynamic,
                    "reverse": reverse,
                    "kd_difference": enabled - disabled,
                    "gradient_difference_norm": gradient_delta,
                }
            )
        native_cfg = dict(cfg, teacher_topk_ipc_k=4)
        for mode in ("sum", "averaged_logits"):
            native_cfg["kd_loss_mode"] = mode
            try:
                model_oracle.full_objective(
                    cross_fixture,
                    [t.logits for t in teachers],
                    data,
                    expected_logits,
                    native_cfg,
                    denominator,
                )
            except AssertionError as error:
                assert "Native cross-tokenizer oracle" in str(error)
            else:
                raise AssertionError(
                    "Native cross-tokenizer position-zero was accepted"
                )
        for bad_indices in ((0, 0), (-1,), (5,), (0,)):
            try:
                model_oracle.full_objective(
                    cross_fixture,
                    [t.logits for t in teachers],
                    data,
                    expected_logits,
                    cfg,
                    denominator,
                    legacy_loss_fn=fn,
                    legacy_dense_teacher_indices=bad_indices,
                )
            except AssertionError:
                pass
            else:
                raise AssertionError(
                    f"Accepted invalid dense route indices: {bad_indices}"
                )
    print(
        f"PASS retained dense oracle: {checked} scalar/gradient/metric comparisons; 24 nonzero position-zero deltas; two native position-zero guards; four invalid-index guards; CPU only"
    )
    print(json.dumps({"position_zero_deltas": position_zero_deltas}, indent=2))


if __name__ == "__main__":
    main()
