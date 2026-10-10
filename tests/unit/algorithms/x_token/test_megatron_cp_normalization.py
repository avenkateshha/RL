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

"""MCore per-term CP normalization, using real loss and relayout autograd.

The CPU fixture redirects MCore's CUDA-specific gather allocation to CPU;
its collective, log-softmax and backward implementations are unchanged.
Run directly to avoid the repository's Ray session fixture on CPU-only hosts:
``uv run --no-sync python -m tests.unit.algorithms.x_token.test_megatron_cp_normalization``.
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _config(*, mode: str, topk: int, dynamic: bool, reverse: bool) -> dict[str, Any]:
    return {
        "temperature": 1.7,
        "vocab_topk": topk,
        "reverse_kl": reverse,
        "kl_loss_weight": 0.8,
        "ce_loss_scale": 0.4,
        "dynamic_loss_scaling": dynamic,
        "kd_loss_mode": mode,
        "normalize_teacher_by_vocab": False,
        "alpha": 1.0,
        "sum_weights_metric": None,
        "student_vocab_size": 12,
        "teacher_vocab_sizes": [12, 12],
        "projection_matrix_paths": [None, None],
        "teacher_is_cross_tokenizer": [False, False],
        "teacher_weights": [0.3, 0.7],
    }


def _reference(
    logits: torch.Tensor,
    teachers: list[torch.Tensor],
    ids: torch.Tensor,
    mask: torch.Tensor,
    sample_mask: torch.Tensor,
    denom: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    label_mask = mask[:, 1:] * sample_mask[:, None]
    ce = torch.nn.functional.cross_entropy(
        logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten(), reduction="none"
    ).reshape_as(label_mask)
    ce = (ce * label_mask).sum() / denom
    valid = torch.cat((label_mask, torch.zeros_like(label_mask[:, :1])), dim=1).bool()
    temperature = cfg["temperature"]

    def kl(teacher: torch.Tensor, k: int) -> torch.Tensor:
        importance = (
            teacher.masked_fill(~valid[..., None], -torch.inf)
            .flatten(0, 1)
            .max(0)
            .values
        )
        indices = (
            torch.arange(k)
            if not valid.any()
            else importance.topk(k).indices.sort().values
        )
        student_logp = (logits[..., indices] / temperature).log_softmax(-1)
        teacher_logp = (teacher[..., indices] / temperature).log_softmax(-1)
        input_logp, target_logp = (
            (teacher_logp, student_logp)
            if cfg["reverse_kl"]
            else (student_logp, teacher_logp)
        )
        per_position = torch.nn.functional.kl_div(
            input_logp, target_logp, reduction="none", log_target=True
        ).sum(-1)
        return (per_position * valid).sum() * temperature**2 / denom

    if cfg["kd_loss_mode"] == "averaged_logits":
        teacher = sum(w * t for w, t in zip(cfg["teacher_weights"], teachers)) / sum(
            cfg["teacher_weights"]
        )
        kd = kl(teacher, 12)
    else:
        kd = sum(
            w * kl(t, cfg["vocab_topk"])
            for w, t in zip(cfg["teacher_weights"], teachers)
        )
    scale = (
        torch.where(
            kd.detach().abs() > 0,
            ce.detach().abs() / kd.detach().abs(),
            torch.ones_like(kd),
        )
        if cfg["dynamic_loss_scaling"]
        else torch.ones_like(kd)
    )
    loss = (
        ce + scale * kd
        if cfg["dynamic_loss_scaling"]
        else cfg["ce_loss_scale"] * ce + cfg["kl_loss_weight"] * kd
    )
    return loss, ce, kd, scale


def _run_normalization(
    rank: int, tp_size: int, cp_size: int, rendezvous: str, fixture_paths: list[str]
) -> None:
    # Import production loss lazily so the spawned process owns its groups.
    from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn
    from nemo_rl.algorithms.x_token.loss_utils import LocalizedAlignment
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.model_utils import (
        _get_tokens_on_this_cp_rank,
        allgather_cp_sharded_tensor,
    )
    from megatron.core.tensor_parallel import mappings

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=tp_size * cp_size,
    )
    # MCore's TP gather allocates on current_device rather than input.device.
    # Keep its real collective and backward implementation, but place its
    # output on CPU for this gloo-only numerical regression.
    original_gather = mappings._gather_along_last_dim

    def gather_with_input_device(
        tensor: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        # Scope the allocator override to this collective: Transformer Engine
        # import-time capability checks must still see the actual CUDA device.
        with patch("torch.cuda.current_device", return_value=tensor.device):
            return original_gather(tensor, group)

    cpu_allocator = patch.object(
        mappings, "_gather_along_last_dim", side_effect=gather_with_input_device
    )
    cpu_allocator.start()
    try:
        tp_rank, cp_rank = divmod(rank, cp_size)
        tp_group = cp_group = None
        for c in range(cp_size):
            group = dist.new_group([t * cp_size + c for t in range(tp_size)])
            if c == cp_rank:
                tp_group = group
        for t in range(tp_size):
            group = dist.new_group([t * cp_size + c for c in range(cp_size)])
            if t == tp_rank:
                cp_group = group
        torch.manual_seed(23)
        full = torch.randn(4, 8, 12)
        teachers = [
            torch.randn_like(full) * 1.2,
            torch.randn_like(full) * 0.6 + full * 0.2,
        ]
        ids = torch.randint(0, 12, (4, 8))
        mask = torch.tensor(
            [
                [0, 1, 1, 0, 1, 1, 1, 0],
                [0, 1, 0, 1, 1, 0, 1, 1],
                [0, 1, 1, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.float32,
        )
        sample_mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
        denominator = (mask[:, 1:] * sample_mask[:, None]).sum()
        v_slice = slice(tp_rank * (12 // tp_size), (tp_rank + 1) * (12 // tp_size))
        s_slice = slice(cp_rank * (8 // cp_size), (cp_rank + 1) * (8 // cp_size))

        def shard(value: torch.Tensor) -> torch.Tensor:
            return (
                _get_tokens_on_this_cp_rank(
                    value, cp_rank=cp_rank, cp_size=cp_size, seq_dim=1
                ).contiguous()
                if cp_size > 1
                else value.clone()
            )

        cases = [
            (mode, topk, dynamic, reverse, zero)
            for mode in ("sum", "averaged_logits")
            for topk in (5, 12)
            for dynamic in (False, True)
            for reverse in (False, True)
            for zero in (False, True)
        ]
        for mode, topk, dynamic, reverse, zero in cases:
            cfg = _config(mode=mode, topk=topk, dynamic=dynamic, reverse=reverse)
            if zero:
                cfg["teacher_weights"] = [0.0, 1.0]
            loss_fn = CrossTokenizerDistillationLossFn(cfg)
            for batch_start in (0, 2):
                b = slice(batch_start, batch_start + 2)
                # This gives one microbatch a fully empty KD/CE mask on every
                # rank, and exercises zero-weight teachers without skipping them.
                masks = (
                    torch.zeros_like(mask[b]) if zero and batch_start == 2 else mask[b]
                )
                if zero and batch_start == 0:
                    masks = masks.clone()
                    masks[:, 4:] = 0  # CP1 has no valid contiguous KD predictors.
                ref_logits = full[b].clone().requires_grad_()
                ref, ref_ce, ref_kd, ref_scale = _reference(
                    ref_logits,
                    [t[b] for t in teachers],
                    ids[b],
                    masks,
                    sample_mask[b],
                    denominator,
                    cfg,
                )
                ref.backward()
                local_logits = shard(full[b, :, v_slice]).clone().requires_grad_()
                contig = (
                    allgather_cp_sharded_tensor(local_logits, cp_group, seq_dim=1)[
                        :, s_slice
                    ]
                    if cp_size > 1
                    else local_logits
                )
                align = LocalizedAlignment(
                    sample_mask=sample_mask[b],
                    student_input_ids=ids[b, s_slice],
                    student_token_mask=masks[:, s_slice],
                )
                data = BatchedDataDict(
                    {
                        "input_ids": ids[b],
                        "token_mask": masks,
                        "sample_mask": sample_mask[b],
                    }
                )
                actual, metrics = loss_fn(
                    data,
                    sample_mask[b].sum(),
                    denominator,
                    local_logits,
                    contig,
                    {i: t[b, s_slice] for i, t in enumerate(teachers)},
                    {0: align, 1: align},
                    tp_group=tp_group,
                    cp_group=cp_group,
                    megatron_cp_normalize=True,
                )
                actual.backward()
                torch.testing.assert_close(
                    local_logits.grad,
                    shard(ref_logits.grad[..., v_slice]),
                    rtol=1e-4,
                    atol=1e-5,
                )
                for key, expected in (
                    ("loss", ref),
                    ("ce_loss", ref_ce),
                    ("kl_loss", ref_kd),
                    ("kl_loss_scale", ref_scale),
                ):
                    torch.testing.assert_close(
                        torch.tensor(metrics[key]),
                        expected.detach(),
                        rtol=1e-4,
                        atol=1e-5,
                    )
                if mode == "sum":
                    torch.testing.assert_close(
                        torch.tensor(
                            sum(metrics[f"teacher_{i}/weighted_kl"] for i in range(2))
                        ),
                        ref_kd.detach(),
                        rtol=1e-4,
                        atol=1e-5,
                    )

        # With full support and fixed weights, undoing only the old outer /CP
        # produces exactly CP times the old same-tokenizer KD gradient.
        cfg = _config(mode="sum", topk=12, dynamic=False, reverse=False)
        loss_fn = CrossTokenizerDistillationLossFn(cfg)
        corrected = shard(full[..., v_slice]).clone().requires_grad_()
        contig = (
            allgather_cp_sharded_tensor(corrected, cp_group, seq_dim=1)[:, s_slice]
            if cp_size > 1
            else corrected
        )
        align = LocalizedAlignment(
            sample_mask=sample_mask,
            student_input_ids=ids[:, s_slice],
            student_token_mask=mask[:, s_slice],
        )
        kd = loss_fn._direct_topk_kl(
            contig,
            teachers[0][:, s_slice],
            align,
            denominator,
            tp_group=tp_group,
            cp_group=cp_group,
        )
        old_grad = torch.autograd.grad(kd / cp_size, corrected, retain_graph=True)[0]
        new_grad = torch.autograd.grad(kd, corrected)[0]
        torch.testing.assert_close(new_grad, old_grad * cp_size, rtol=1e-5, atol=1e-6)
        # Retained CE still receives precisely the old /CP correction.
        loss_fn.kl_loss_weight = 0.0
        loss_fn.ce_loss_scale = 1.0
        ce_logits = shard(full[..., v_slice]).clone().requires_grad_()
        ce_contig = (
            allgather_cp_sharded_tensor(ce_logits, cp_group, seq_dim=1)[:, s_slice]
            if cp_size > 1
            else ce_logits
        )
        data = BatchedDataDict(
            {"input_ids": ids, "token_mask": mask, "sample_mask": sample_mask}
        )
        ce_old = (
            loss_fn._compute_ce(
                ce_logits, data, denominator, tp_group=tp_group, cp_group=cp_group
            )
            / cp_size
        )
        ce_old_grad = torch.autograd.grad(ce_old, ce_logits)[0]
        ce_corrected, _ = loss_fn(
            data,
            sample_mask.sum(),
            denominator,
            ce_logits,
            ce_contig,
            {i: t[:, s_slice] for i, t in enumerate(teachers)},
            {0: align, 1: align},
            tp_group=tp_group,
            cp_group=cp_group,
            megatron_cp_normalize=True,
        )
        ce_corrected.backward()
        torch.testing.assert_close(ce_corrected, ce_old, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(ce_logits.grad, ce_old_grad, rtol=1e-5, atol=1e-6)

        _check_legacy_v6(
            tp_group, cp_group, tp_rank, cp_rank, tp_size, cp_size, fixture_paths
        )
    finally:
        cpu_allocator.stop()
        dist.destroy_process_group()


def _check_legacy_v6(
    tp_group: dist.ProcessGroup,
    cp_group: dist.ProcessGroup,
    tp_rank: int,
    cp_rank: int,
    tp_size: int,
    cp_size: int,
    fixture_paths: list[str],
) -> None:
    # These helpers import the optional loss stack only inside each rank.
    from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn
    from nemo_rl.algorithms.x_token.loss_utils import LocalizedAlignment
    from tests.unit.algorithms.x_token.test_v6_sharded_student_memory import (
        _loss_config,
    )

    for path in fixture_paths:
        fx = torch.load(path, weights_only=False)
        fn = CrossTokenizerDistillationLossFn(_loss_config(fx))
        batch_size, seq_len, vocab_size = fx["student_logits"].shape
        s_slice = slice(
            cp_rank * (seq_len // cp_size), (cp_rank + 1) * (seq_len // cp_size)
        )
        v_slice = slice(
            tp_rank * (vocab_size // tp_size), (tp_rank + 1) * (vocab_size // tp_size)
        )

        def alignment(s_ids: torch.Tensor, t_ids: torch.Tensor) -> LocalizedAlignment:
            return LocalizedAlignment(
                sample_mask=torch.ones(batch_size),
                pair_valid=fx["pair_valid"],
                student_input_ids=s_ids,
                teacher_input_ids=t_ids,
                student_spans=fx["s_spans"],
                teacher_spans=fx["t_spans"],
                num_chunks=fx["num_chunks"],
            )

        count = torch.tensor(fx["global_valid_chunks"])
        reference_logits = fx["student_logits"].clone().requires_grad_()
        reference_loss, _ = fn._compute_prefix_bidir_partition_kl_v3(
            0,
            reference_logits,
            fx["teacher_logits"],
            alignment(fx["student_ids"], fx["teacher_ids"]),
            teacher_vocab_size=fx["v_t"],
            tp_group=None,
            cp_group=None,
            global_valid_chunks=count,
        )
        reference_loss.backward()
        local = fx["student_logits"][:, s_slice, v_slice].clone().requires_grad_()
        align = alignment(fx["student_ids"][:, s_slice], fx["teacher_ids"][:, s_slice])
        old, _ = fn._compute_prefix_bidir_partition_kl_v3(
            0,
            local,
            fx["teacher_logits"][:, s_slice],
            align,
            teacher_vocab_size=fx["v_t"],
            tp_group=tp_group,
            cp_group=cp_group,
            global_valid_chunks=count,
        )
        old_grad = torch.autograd.grad(old / cp_size, local)[0]
        corrected, _ = fn._compute_teacher_kd(
            0,
            local,
            {},
            {0: fx["teacher_logits"][:, s_slice]},
            {0: align},
            count,
            teacher_sparse_logits_by_idx={},
            tp_group=tp_group,
            cp_group=cp_group,
            global_valid_chunks_by_idx={0: count},
            legacy_cp_size=cp_size,
        )
        corrected.backward()
        torch.testing.assert_close(corrected, old / cp_size, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(local.grad, old_grad, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            local.grad, reference_logits.grad[:, s_slice, v_slice], rtol=1e-4, atol=1e-4
        )


@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_megatron_cp_normalization(tp_size: int, cp_size: int) -> None:
    from tests.unit.algorithms.x_token.test_v6_loss_parity import _build_case

    with tempfile.TemporaryDirectory() as tmp:
        fixtures = []
        for case in ("common_only", "with_mismatch"):
            case_dir = Path(tmp) / case
            case_dir.mkdir()
            fixture = _build_case(case, str(case_dir))
            fixture_path = case_dir / "fixture.pt"
            torch.save(fixture, fixture_path)
            fixtures.append(str(fixture_path))
        mp.spawn(
            _run_normalization,
            args=(tp_size, cp_size, str(Path(tmp) / "init"), fixtures),
            nprocs=tp_size * cp_size,
            join=True,
        )


@pytest.mark.parametrize(
    "packed,dynamic,pp,has_packed_metadata,eligible",
    [
        (False, False, 1, False, True),
        (False, False, 2, False, False),
        (True, False, 1, False, False),
        (False, True, 1, False, False),
        (False, False, 1, True, False),
    ],
)
def test_xtoken_wrapper_keeps_schedule_compensation(
    packed: bool, dynamic: bool, pp: int, has_packed_metadata: bool, eligible: bool
) -> None:
    from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn
    from nemo_rl.models.megatron import train

    loss_fn = CrossTokenizerDistillationLossFn(
        _config(mode="sum", topk=12, dynamic=False, reverse=False)
    )
    with (
        patch.object(train, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(train, "get_tensor_model_parallel_group"),
        patch.object(train, "get_context_parallel_group"),
        patch.object(train, "get_context_parallel_world_size", return_value=2),
        patch.object(train, "get_pipeline_model_parallel_world_size", return_value=pp),
        patch.object(
            train,
            "wrap_loss_fn_with_input_preparation",
            return_value=(torch.tensor(3.0), {}),
        ) as wrapped_loss,
    ):
        wrapped = train.LossPostProcessor(
            loss_fn,
            {
                "sequence_packing": {"enabled": packed},
                "dynamic_batching": {"enabled": dynamic},
            },
            num_microbatches=4,
        )({}, packed_seq_params=object() if has_packed_metadata else None)
        loss, _ = wrapped(torch.empty(0))
        assert (
            wrapped_loss.call_args.kwargs["prepare_fn"].keywords["native_cp_enabled"]
            is eligible
        )
        capabilities = wrapped_loss.call_args.kwargs["prepare_fn"].keywords
        assert capabilities["native_sparse_enabled"] is True
        assert capabilities.get("native_same_tokenizer_enabled", False) is False
    # MCore applies CP / num_microbatches after this compensation.
    torch.testing.assert_close(loss, torch.tensor(6.0))


@pytest.mark.parametrize("explicit_tp", [False, True])
def test_megatron_loss_input_enables_per_term_cp_normalization(
    explicit_tp: bool,
) -> None:
    from nemo_rl.algorithms.loss import loss_input as module
    from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn
    from nemo_rl.algorithms.x_token.loss_utils import XTokenLossInput

    fn = CrossTokenizerDistillationLossFn(
        _config(mode="sum", topk=12, dynamic=False, reverse=False)
    )
    logits = torch.zeros(1, 4, 12)
    group = object() if explicit_tp else None
    with patch.object(
        module,
        "prepare_xtoken_cross_tokenizer_loss_input",
        return_value=XTokenLossInput(logits, {}, {}, {}, {}, group, None, None),
    ):
        prepared, _ = module.prepare_loss_input(
            logits, {}, fn, vocab_parallel_group=group
        )
    assert prepared["megatron_cp_normalize"] is explicit_tp


if __name__ == "__main__":
    for tp, cp in ((1, 1), (2, 1), (1, 2), (2, 2)):
        test_megatron_cp_normalization(tp, cp)
        print(f"PASS normalization TP={tp} CP={cp}", flush=True)
