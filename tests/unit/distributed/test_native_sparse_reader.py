"""CPU tests for teacher-indexed native sparse records and requested supports."""

import copy

import pytest
import torch

from nemo_rl.algorithms.x_token.sparse_teacher import (
    SparseTeacherIPC,
    SparseTeacherRowReader,
    SparseTeacherShard,
    build_native_force_token_ids,
    shard_native_teacher_force_ids,
)
from nemo_rl.distributed.selected_logprobs import (
    cp_native_global_positions,
    cp_native_sequence_segments,
)
from nemo_rl.distributed.sparse_topk import distributed_vocab_topk_logz_force


def _payload(*, sequence=8, offset=0.0, teacher_mbs=2):
    samples = [{"teacher_shards": []} for _ in range(3)]
    logits = (
        torch.arange(3 * sequence * 6).reshape(3, sequence, 6).float() / 100 + offset
    )
    # Natural rank order 1, 3, 0; requests choose 1 and 4 independently.
    logits += torch.tensor([2.0, 4.0, 0.0, 3.0, -2.0, 100.0])
    force = torch.tensor([1, 4]).expand(3, sequence, 2)
    for cp in range(2):
        positions = cp_native_global_positions(sequence, cp, 2)
        output = distributed_vocab_topk_logz_force(
            logits.index_select(1, positions),
            force.index_select(1, positions),
            2,
            None,
            vocab_start_index=0,
            vocab_end_index=6,
            real_vocab_size=5,
            temperature=1.5,
            noise_filter_k=3,
        )
        fields = [
            output.topk_logits,
            output.topk_indices,
            output.log_z.unsqueeze(-1),
            output.natural_tail_indices,
            output.forced_logits,
            output.forced_indices,
            output.forced_in_topk,
        ]
        backings = []
        for field in fields:
            backing = torch.zeros(
                (
                    (3 + teacher_mbs - 1) // teacher_mbs,
                    teacher_mbs,
                    positions.numel(),
                    field.shape[-1],
                ),
                dtype=field.dtype,
            )
            for sample in range(3):
                backing[sample // teacher_mbs, sample % teacher_mbs] = field[sample]
            backings.append(backing)
        for sample in range(3):
            shard = SparseTeacherShard(
                k=2,
                temperature=1.5,
                real_vocab_size=5,
                membership_k=3,
                force_width=2,
                full_seq_len=sequence,
                local_seq_len=positions.numel(),
                buf_idx=sample // teacher_mbs,
                sample_index_in_buf=sample % teacher_mbs,
                sequence_segments=tuple(cp_native_sequence_segments(sequence, cp, 2)),
                topk_logits_ipc=backings[0],
                topk_indices_ipc=backings[1],
                log_z_ipc=backings[2],
                natural_tail_indices_ipc=backings[3],
                forced_logits_ipc=backings[4],
                forced_indices_ipc=backings[5],
                forced_in_topk_ipc=backings[6],
            )
            samples[sample]["teacher_shards"].append(shard.to_record())
    return SparseTeacherIPC(samples), logits


def test_requested_rows_two_teachers_microbatch_offsets_and_independent_support(
    monkeypatch,
):
    monkeypatch.setattr(
        torch.cuda, "current_device", lambda: pytest.fail("CPU reader consulted CUDA")
    )
    first, first_logits = _payload()
    second, second_logits = _payload(sequence=12, offset=70.0, teacher_mbs=1)
    batches = torch.tensor([2, 0, 2, 1])
    positions = torch.tensor([5, 7, 5, 0])
    required = torch.tensor([1, 4, 4, 1])
    readers = [
        SparseTeacherRowReader(first, device=torch.device("cpu")),
        SparseTeacherRowReader(second, device=torch.device("cpu")),
    ]
    for reader, source in zip(readers, (first_logits, second_logits), strict=True):
        reader.validate_contract(
            k=2, temperature=1.5, real_vocab_size=5, membership_k=3, sample_count=3
        )
        values, ids, log_z, membership = reader.gather_rows(
            batches, positions, required
        )
        assert ids.tolist() == [[1, 3], [1, 4], [1, 4], [1, 3]]
        torch.testing.assert_close(
            values, source[batches, positions].gather(-1, ids.long())
        )
        torch.testing.assert_close(
            log_z, (source[batches, positions, :5] / 1.5).logsumexp(-1)
        )
        assert membership.tolist() == [True, False, False, True]
        cache = dict(reader._backing_tensors)
        reader.gather_rows(batches.flip(0), positions.flip(0), required.flip(0))
        assert cache.keys() == reader._backing_tensors.keys()
        assert all(cache[key] is reader._backing_tensors[key] for key in cache)
    assert readers[0]._backing_tensors is not readers[1]._backing_tensors


def test_reader_new_generation_and_sliced_sample_identity():
    first, _ = _payload(offset=0.0)
    second, _ = _payload(offset=40.0)
    args = (torch.tensor([0]), torch.tensor([6]), torch.tensor([4]))
    old_reader = SparseTeacherRowReader(
        SparseTeacherIPC([first.samples[2]]), device=torch.device("cpu")
    )
    new_reader = SparseTeacherRowReader(
        SparseTeacherIPC([second.samples[2]]), device=torch.device("cpu")
    )
    old = old_reader.gather_rows(*args)[0]
    new = new_reader.gather_rows(*args)[0]
    torch.testing.assert_close(new - old, torch.full_like(old, 40.0))
    assert not old_reader._backing_tensors.keys() & new_reader._backing_tensors.keys()


@pytest.mark.parametrize(
    "field,value",
    [
        ("k", 1),
        ("temperature", 2.0),
        ("real_vocab_size", 6),
        ("membership_k", 2),
        ("sample_count", 2),
    ],
)
def test_reader_consumption_contract(field, value):
    reader = SparseTeacherRowReader(_payload()[0], device=torch.device("cpu"))
    kwargs = dict(
        k=2, temperature=1.5, real_vocab_size=5, membership_k=3, sample_count=3
    )
    kwargs[field] = value
    with pytest.raises(ValueError, match="contract mismatch"):
        reader.validate_contract(**kwargs)


@pytest.mark.parametrize("failure", ["overlap", "gap", "local_gap", "slot", "metadata"])
def test_reader_rejects_broken_routes_and_offsets(failure):
    payload = copy.deepcopy(_payload()[0])
    shards = payload.samples[0]["teacher_shards"]
    if failure == "overlap":
        shards[1]["sequence_segments"] = [(0, 1, 2), (2, 4, 2)]
    elif failure == "gap":
        shards[1]["sequence_segments"] = [(0, 3, 1), (1, 4, 3)]
    elif failure == "local_gap":
        shards[0]["sequence_segments"] = [(0, 0, 2), (3, 6, 1)]
    elif failure == "slot":
        shards[0]["sample_index_in_buf"] = 9
    else:
        shards[1]["temperature"] = 9.0
    with pytest.raises(ValueError):
        reader = SparseTeacherRowReader(payload, device=torch.device("cpu"))
        reader.gather_rows(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))


def test_empty_request_has_no_cuda_or_storage_reads():
    reader = SparseTeacherRowReader(_payload()[0], device=torch.device("cpu"))
    empty = torch.empty(0, dtype=torch.long)
    values, ids, log_z, membership = reader.gather_rows(empty, empty, empty)
    assert values.shape == ids.shape == (0, 2)
    assert log_z.shape == membership.shape == (0,)
    assert not reader._backing_tensors


def test_reader_natural_label_outside_smaller_membership_support():
    payload, logits = _payload()
    for sample in payload.samples:
        for shard in sample["teacher_shards"]:
            shard["forced_in_topk_k"] = 1
            # Shared backings are updated repeatedly to the same values.
            shard["forced_indices_ipc"][..., 1] = 3
            positions = [
                position
                for _, start, length in shard["sequence_segments"]
                for position in range(start, start + length)
            ]
            for batch in range(3):
                shard["forced_logits_ipc"][batch // 2, batch % 2, :, 1] = logits[
                    batch, positions, 3
                ]
            shard["forced_in_topk_ipc"][..., 1] = False
    reader = SparseTeacherRowReader(payload, device=torch.device("cpu"))
    _, ids, _, membership = reader.gather_rows(
        torch.tensor([0]), torch.tensor([6]), torch.tensor([3])
    )
    assert ids.tolist() == [[1, 3]]
    assert membership.tolist() == [False]


def test_reader_requires_advertised_label_and_scalar_shape():
    payload, _ = _payload()
    reader = SparseTeacherRowReader(payload, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="forced sidecar"):
        reader.gather_rows(torch.tensor([0]), torch.tensor([0]), torch.tensor([2]))
    del payload.samples[0]["teacher_shards"][0]["scalar_shape"]
    with pytest.raises(ValueError, match="Malformed"):
        SparseTeacherRowReader(payload, device=torch.device("cpu"))


def test_force_builder_rejects_input_id_outside_teacher_vocabulary():
    with pytest.raises(ValueError, match="real-vocabulary"):
        build_native_force_token_ids(
            torch.tensor([[5]]),
            torch.tensor([[[0, 1]]]),
            torch.tensor([[[0, 1]]]),
            torch.ones(1, 1, dtype=torch.bool),
            kl_chunk_shift=True,
            teacher_real_vocab_size=5,
        )


def test_forced_labels_preserve_origin_collision_absent_spans_and_sample_masks():
    ids = torch.tensor([[4, 5, 6, 7], [9, 10, 11, 12]])
    spans = torch.tensor([[[0, 1], [1, 3], [0, 0], [3, 4]]] * 2)
    teacher_spans = spans.clone()
    teacher_spans[:, 2] = torch.tensor([3, 4])
    output = build_native_force_token_ids(
        ids,
        spans,
        teacher_spans,
        torch.ones(2, 4, dtype=torch.bool),
        kl_chunk_shift=True,
        teacher_real_vocab_size=13,
        sample_mask=torch.tensor([1, 0]),
    )
    assert output[0].tolist() == [[4, 5], [6, -1], [7, -1], [-1, -1]]
    assert output[1].tolist() == [[-1, -1]] * 4
    unshifted = build_native_force_token_ids(
        ids,
        spans,
        teacher_spans,
        torch.ones(2, 4, dtype=torch.bool),
        kl_chunk_shift=False,
        teacher_real_vocab_size=13,
    )
    assert unshifted[0, :, 0].tolist() == ids[0].tolist()
    assert unshifted[0, :, 1].tolist() == [-1] * 4


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_teacher_sidecar_shards_sequence_axis_once(cp_size):
    forced = torch.arange(32).reshape(1, 16, 2)
    forced[:, 3, 1] = -1
    for rank in range(cp_size):
        actual = shard_native_teacher_force_ids(
            forced, cp_rank=rank, cp_size=cp_size, local_sequence_length=16 // cp_size
        )
        positions = cp_native_global_positions(16, rank, cp_size)
        torch.testing.assert_close(actual, forced.index_select(1, positions))
    with pytest.raises(ValueError, match="divisible"):
        shard_native_teacher_force_ids(
            forced[:, :5], cp_rank=0, cp_size=2, local_sequence_length=2
        )
    with pytest.raises(ValueError, match="output rows"):
        shard_native_teacher_force_ids(
            forced, cp_rank=0, cp_size=2, local_sequence_length=4
        )
