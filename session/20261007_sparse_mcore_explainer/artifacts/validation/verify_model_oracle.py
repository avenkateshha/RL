"""CPU scalar/gradient checks for the real-run same-tokenizer oracle.

Compare its independent dense expressions to actual native production helpers;
no distributed process group, IPC, model forward, or optimizer is simulated.
"""

from dataclasses import replace
from types import SimpleNamespace

import model_oracle
import torch

from nemo_rl.algorithms.x_token.loss_utils import NativeStudentContext
from nemo_rl.algorithms.x_token.native_same_tokenizer import (
    compute_native_same_tokenizer_kl,
)
from nemo_rl.algorithms.x_token.native_student import native_next_token_ce
from tests.unit.algorithms.x_token.native_sparse_fixtures import SparseFixture


def context(fixture, logits):
    ids = fixture.student_input_ids
    return NativeStudentContext(
        logits=logits,
        global_positions=torch.arange(ids.shape[1]),
        input_ids=ids,
        next_token_ids=torch.cat((ids[:, 1:], ids[:, -1:]), dim=1),
        next_token_mask=torch.cat(
            (fixture.token_mask[:, 1:], torch.zeros_like(fixture.token_mask[:, :1])),
            dim=1,
        ),
        next_kd_token_mask=torch.cat(
            (
                fixture.kd_token_mask[:, 1:],
                torch.zeros_like(fixture.kd_token_mask[:, :1]),
            ),
            dim=1,
        ),
        sample_mask=fixture.sample_mask,
        full_seq_len=ids.shape[1],
        real_vocab_size=fixture.student_vocab_size,
    )


def main():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(1081)
    fixture = SparseFixture(
        student_logits=torch.randn(2, 6, 10, generator=generator),
        student_input_ids=torch.tensor([[1, 4, 2, 6, 7, 3], [0, 2, 1, 6, 5, 4]]),
        token_mask=torch.tensor([[0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0]]),
        kd_token_mask=torch.tensor([[0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 0]]),
        sample_mask=torch.ones(2),
        teachers=(),
        student_vocab_size=8,
        global_valid_tokens=23.0,
    )
    teachers = [torch.randn(2, 6, 10, generator=generator) for _ in range(2)]
    for teacher in teachers:
        teacher[..., 8:] = 1000  # Padded vocabulary must never enter KL.
        teacher[:, -1, 7] = 1000  # Invalid predictors must not select common K.
    comparisons = 0
    for mode in ("sum", "averaged_logits"):
        for k in (0, 3, 8):
            for dynamic in (False, True):
                for reverse in (False, True):
                    for empty in (False, True):
                        case = (
                            replace(
                                fixture,
                                sample_mask=torch.zeros(2),
                                global_valid_tokens=0,
                            )
                            if empty
                            else fixture
                        )
                        cfg = {
                            "student_vocab_size": 8,
                            "teacher_vocab_sizes": [8, 8],
                            "teacher_is_cross_tokenizer": [False, False],
                            "teacher_weights": [0.3, 0.7],
                            "vocab_topk": k,
                            "temperature": 1.7,
                            "reverse_kl": reverse,
                            "teacher_topk_ipc_k": 0,
                            "kl_chunk_shift": True,
                            "prefix_bidir_v3_loss_fn": "kl",
                            "prefix_bidir_v3_jsd_beta": 0.3,
                            "prefix_bidir_v3_noise_filter_topk": 0,
                            "kd_loss_mode": mode,
                            "dynamic_loss_scaling": dynamic,
                            "ce_loss_scale": 0.4,
                            "kl_loss_weight": 0.8,
                        }
                        data = {
                            "input_ids": case.student_input_ids,
                            "token_mask": case.token_mask,
                            "kd_token_mask": case.kd_token_mask,
                            "sample_mask": case.sample_mask,
                        }
                        actual_logits = case.student_logits.clone().requires_grad_()
                        reference_logits = case.student_logits.clone().requires_grad_()
                        denominator = 0.0 if empty else 19.0
                        reference, metrics = model_oracle.full_objective(
                            case, teachers, data, reference_logits, cfg, denominator
                        )
                        student = context(case, actual_logits)
                        loss_config = SimpleNamespace(
                            temperature=cfg["temperature"],
                            reverse_kl=reverse,
                            vocab_topk=k,
                        )

                        def kd(rows, full_vocab=False):
                            return compute_native_same_tokenizer_kl(
                                loss_config,
                                student,
                                rows[..., :8],
                                global_valid_toks=torch.tensor(denominator),
                                tp_group=None,
                                cp_group=None,
                                full_vocab=full_vocab,
                            )

                        terms = (
                            [kd(teacher) for teacher in teachers]
                            if mode == "sum"
                            else []
                        )
                        weighted = (
                            [0.3 * terms[0], 0.7 * terms[1]]
                            if terms
                            else [
                                weight * kd(0.3 * teachers[0] + 0.7 * teachers[1], True)
                                for weight in (0.3, 0.7)
                            ]
                        )
                        total_kd = sum(weighted)
                        ce = native_next_token_ce(
                            student,
                            torch.tensor(case.global_valid_tokens),
                            tp_group=None,
                        )
                        ratio = (
                            ce.detach().abs() / total_kd.detach().abs()
                            if dynamic and total_kd.detach().abs() > 0
                            else torch.ones_like(total_kd)
                        )
                        actual = (
                            ce + ratio * total_kd
                            if dynamic
                            else 0.4 * ce + 0.8 * total_kd
                        )
                        torch.testing.assert_close(
                            actual, reference, atol=2e-6, rtol=2e-5
                        )
                        for index, value in enumerate(weighted):
                            torch.testing.assert_close(
                                metrics[f"teacher_{index}/weighted_kl"], value
                            )
                        assert ("kl_loss_t0" in metrics) == (mode == "sum")
                        torch.testing.assert_close(metrics["kl_loss"], total_kd)
                        torch.testing.assert_close(metrics["kl_loss_scale"], ratio)
                        actual.backward()
                        reference.backward()
                        torch.testing.assert_close(
                            actual_logits.grad,
                            reference_logits.grad,
                            atol=2e-6,
                            rtol=2e-5,
                        )
                        if mode == "sum" and k == 0:
                            assert total_kd == 0 and metrics["kl_loss"] == 0
                        if empty:
                            assert (
                                actual == 0
                                and torch.count_nonzero(actual_logits.grad) == 0
                            )
                        comparisons += 1
    print(
        f"PASS real-run same oracle: {comparisons} native scalar/gradient/metric comparisons"
    )


if __name__ == "__main__":
    main()
