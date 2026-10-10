"""Small FP32 CPU/Gloo equivalence tests for native sparse transport math."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
from nemo_rl.distributed.selected_logprobs import (
    cp_native_global_positions,
    cp_native_sequence_segments,
    cp_sum_with_grad,
    distributed_selected_logprobs,
)
from nemo_rl.distributed.sparse_topk import distributed_vocab_topk_logz_force


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_native_segments_match_model_splitting(cp_size):
    tokens = torch.arange(32).reshape(1, 16, 2)
    for rank in range(cp_size):
        positions = cp_native_global_positions(16, rank, cp_size)
        segments = cp_native_sequence_segments(16, rank, cp_size)
        assert positions.tolist() == [
            value
            for _, start, length in segments
            for value in range(start, start + length)
        ]
        actual = tokens.index_select(1, positions)
        expected = (
            tokens
            if cp_size == 1
            else _get_tokens_on_this_cp_rank(tokens, rank, cp_size, seq_dim=1)
        )
        torch.testing.assert_close(actual, expected)


def test_native_partition_validation():
    with pytest.raises(ValueError, match="divisible"):
        cp_native_global_positions(10, 0, 2)
    with pytest.raises(ValueError, match="valid CP"):
        cp_native_global_positions(8, 2, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        cp_native_sequence_segments(-1, 0, 1)


def _selected_case(
    *,
    tp_size=1,
    tp_rank=0,
    cp_size=1,
    cp_rank=0,
    tp_group=None,
    device=torch.device("cpu"),
):
    torch.manual_seed(81)
    full = torch.randn(2, 8, 8).to(device)
    full[..., 7] = 90
    positions = cp_native_global_positions(8, cp_rank, cp_size, device=device)
    local_width = 8 // tp_size
    # Deliberately noncontiguous across B/S, as model output can be.
    base = full.index_select(1, positions)[
        ..., tp_rank * local_width : (tp_rank + 1) * local_width
    ]
    actual_logits = (
        base.transpose(0, 1).contiguous().transpose(0, 1).detach().requires_grad_()
    )
    reference_logits = full.clone().requires_grad_()
    batch = torch.tensor([0, 0, 1, 1], device=device)
    sequence = torch.tensor([0, 0, positions.numel() - 1, 1], device=device)
    ids = torch.tensor([[0, 6], [6, 2], [1, 1], [-1, 7]], device=device)
    weights = torch.tensor(
        [[0.3, -0.7], [1.5, 0.8], [0.4, 1.1], [2.0, -3.0]], device=device
    )
    actual = distributed_selected_logprobs(
        actual_logits,
        batch,
        sequence,
        ids,
        vocab_start_index=tp_rank * local_width,
        vocab_end_index=(tp_rank + 1) * local_width,
        real_vocab_size=7,
        temperature=0.7,
        tp_group=tp_group,
        row_chunk_size=2,
    )
    reference = (
        (reference_logits[batch, positions[sequence], :7] / 0.7)
        .log_softmax(-1)
        .gather(-1, ids.clamp(0, 6))
    )
    reference = torch.where((ids >= 0) & (ids < 7), reference, -torch.inf)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    # Nonzero incoming gradients for invalid IDs must still have zero derivative.
    actual_grad = torch.autograd.grad(actual, actual_logits, weights)[0]
    reference_grad = torch.autograd.grad(reference, reference_logits, weights)[0]
    expected = reference_grad.index_select(1, positions)[
        ..., tp_rank * local_width : (tp_rank + 1) * local_width
    ]
    torch.testing.assert_close(actual_grad, expected, rtol=2e-6, atol=2e-6)
    if tp_rank == tp_size - 1:
        assert torch.count_nonzero(actual_grad[..., -1]) == 0
    empty = torch.empty((0,), dtype=torch.long, device=device)
    zero = distributed_selected_logprobs(
        actual_logits,
        empty,
        empty,
        empty,
        vocab_start_index=tp_rank * local_width,
        vocab_end_index=(tp_rank + 1) * local_width,
        real_vocab_size=7,
        temperature=0.7,
        tp_group=tp_group,
    )
    assert zero.shape == (0,)
    assert torch.count_nonzero(torch.autograd.grad(zero.sum(), actual_logits)[0]) == 0


def test_selected_probabilities_repeated_rows_invalid_ids_and_noncontiguous_gradients():
    _selected_case()


def _topk_case(
    *,
    tp_size=1,
    tp_rank=0,
    cp_size=1,
    cp_rank=0,
    tp_group=None,
    device=torch.device("cpu"),
):
    full = ((torch.arange(2 * 8 * 8).reshape(2, 8, 8) * 13) % 7).to(
        device=device, dtype=torch.bfloat16
    )
    full[..., 7] = 100
    positions = cp_native_global_positions(8, cp_rank, cp_size, device=device)
    local_full = full.index_select(1, positions)
    force = (
        torch.stack((positions % 7, (positions + 3) % 7), -1)
        .unsqueeze(0)
        .expand(2, -1, -1)
        .clone()
    )
    force[1, -1, 1] = -1
    width = 8 // tp_size
    output = distributed_vocab_topk_logz_force(
        local_full[..., tp_rank * width : (tp_rank + 1) * width],
        force,
        5,
        tp_group,
        vocab_start_index=tp_rank * width,
        vocab_end_index=(tp_rank + 1) * width,
        real_vocab_size=7,
        temperature=1.3,
        noise_filter_k=6,
        chunk_size=3,
    )
    torch.testing.assert_close(
        output.log_z,
        (local_full[..., :7].float() / 1.3).logsumexp(-1),
        rtol=1e-6,
        atol=1e-6,
    )
    for batch in range(2):
        for row in range(positions.numel()):
            values = local_full[batch, row, :7].float()
            score_ids = sorted(
                range(7), key=lambda token: (-float(values[token]), token)
            )
            expected_ids = sorted(score_ids[:5])
            assert output.topk_indices[batch, row].tolist() == expected_ids
            assert output.natural_tail_indices[batch, row].item() == score_ids[4]
            torch.testing.assert_close(
                output.topk_logits[batch, row], values[expected_ids]
            )
            assert output.forced_logits[batch, row].tolist() == [
                float(values[label]) if label >= 0 else 0.0
                for label in force[batch, row]
            ]
            assert output.forced_in_topk[batch, row].tolist() == [
                int(label) in score_ids[:6] for label in force[batch, row]
            ]


def test_sparse_topk_real_vocab_bf16_ties_forced_labels_and_noise_membership():
    _topk_case()


def _distributed_worker(rank, init_file, tp_size, cp_size, backend="gloo"):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if backend == "nccl":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=tp_size * cp_size,
        timeout=timedelta(seconds=60),
    )
    try:
        tp_groups = [
            dist.new_group(list(range(cp * tp_size, (cp + 1) * tp_size)))
            for cp in range(cp_size)
        ]
        cp_groups = [
            dist.new_group([tp + cp * tp_size for cp in range(cp_size)])
            for tp in range(tp_size)
        ]
        cp_rank, tp_rank = divmod(rank, tp_size)
        kwargs = dict(
            tp_size=tp_size,
            tp_rank=tp_rank,
            cp_size=cp_size,
            cp_rank=cp_rank,
            tp_group=tp_groups[cp_rank],
            device=device,
        )
        _selected_case(**kwargs)
        _topk_case(**kwargs)
        # Only the final CP owner consumes a mismatch; every producer receives
        # that owner's gradient, including ranks with zero local objective.
        value = torch.tensor(float(cp_rank + 1), device=device, requires_grad=True)
        combined = cp_sum_with_grad(value, cp_group=cp_groups[tp_rank])
        coefficient = 3.0 if cp_rank == cp_size - 1 else 0.0
        (combined * coefficient).backward()
        assert value.grad.item() == 3.0
        assert combined.item() == cp_size * (cp_size + 1) / 2
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_gloo_tp_cp_primitive_values_and_preclip_gradients(tmp_path, tp_size, cp_size):
    mp.spawn(
        _distributed_worker,
        args=(str(tmp_path / "gloo"), tp_size, cp_size),
        nprocs=tp_size * cp_size,
        join=True,
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="NCCL TP2/CP2 requires four GPUs"
)
@pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_cuda_tp_cp_primitive_values_and_preclip_gradients(tmp_path, tp_size, cp_size):
    mp.spawn(
        _distributed_worker,
        args=(str(tmp_path / "nccl"), tp_size, cp_size, "nccl"),
        nprocs=tp_size * cp_size,
        join=True,
    )


@pytest.mark.parametrize(
    "setting", ["k", "noise", "temperature", "forced", "duplicate"]
)
def test_topk_rejects_invalid_contract(setting):
    kwargs = dict(k=2, temperature=1.0, noise_filter_k=0)
    forced = torch.tensor([[[0, -1]]])
    if setting == "k":
        kwargs["k"] = 6
    elif setting == "noise":
        kwargs["noise_filter_k"] = 6
    elif setting == "temperature":
        kwargs["temperature"] = float("nan")
    elif setting == "forced":
        forced[0, 0, 0] = 5
    else:
        forced[0, 0, 1] = 0
    with pytest.raises(ValueError):
        distributed_vocab_topk_logz_force(
            torch.ones(1, 1, 6),
            forced,
            tp_group=None,
            vocab_start_index=0,
            vocab_end_index=6,
            real_vocab_size=5,
            **kwargs,
        )
