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

"""Native-position reads preserve dense teacher storage and sample identity."""

import copy

import pytest
import torch

from nemo_rl.algorithms.x_token.dense_teacher import (
    DenseTeacherIPC,
    DenseTeacherRowReader,
    supports_native_dense_reads,
)


def dense_payload(*, sequence=8, offset=0.0, compact=False, teacher_mbs=2):
    """Identifiable two-TP/two-CP export with independent teacher buffer slots."""
    full = torch.arange(3 * sequence * 8).reshape(3, sequence, 8).float() + offset
    valid = [sequence - 1, sequence // 2 + 1, 0]
    samples = [{"teacher_shards": []} for _ in range(3)]
    for cp in range(2):
        start, length = cp * sequence // 2, sequence // 2
        lengths = [max(0, min(item - start, length)) for item in valid]
        offsets = [1, 1 + lengths[0], 1 + lengths[0] + lengths[1]]
        used = 1 + sum(lengths)
        for tp in range(2):
            if compact:
                backing = torch.zeros(used + 2, 4)
            else:
                # Strided producer views must be indexed with their real strides.
                backing = torch.zeros(
                    (3 + teacher_mbs - 1) // teacher_mbs, teacher_mbs, length, 8
                )[..., ::2]
            for batch in range(3):
                row = full[batch, start : start + length, 4 * tp : 4 * tp + 4]
                record = {
                    "actual_shape": (length, 4),
                    "full_seq_len": sequence,
                    "full_vocab_size": 8,
                    "global_seq_start": start,
                    "tp_rank": tp,
                    "tp_size": 2,
                    "cp_rank": cp,
                    "cp_size": 2,
                    "vocab_start_index": 4 * tp,
                    "vocab_end_index": 4 * tp + 4,
                    "dtype": torch.float32,
                    "payload_ipc": backing,
                }
                if compact:
                    backing[offsets[batch] : offsets[batch] + lengths[batch]] = row[
                        : lengths[batch]
                    ]
                    record.update(
                        ipc_layout="flat_valid_prefix_v1",
                        storage_shape=tuple(backing.shape),
                        storage_capacity_tokens=backing.shape[0],
                        storage_used_tokens=used,
                        storage_token_offset=offsets[batch],
                        stored_seq_len=lengths[batch],
                        valid_seq_len=valid[batch],
                    )
                else:
                    backing[batch // teacher_mbs, batch % teacher_mbs] = row
                    record.update(
                        buf_idx=batch // teacher_mbs,
                        sample_index_in_buf=batch % teacher_mbs,
                    )
                samples[batch]["teacher_shards"].append(record)
    if compact:
        for batch, count in enumerate(valid):
            full[batch, count:] = 0
    return DenseTeacherIPC(samples), full


@pytest.mark.parametrize("compact", [False, True])
def test_native_dense_reads_distinct_teachers_and_microbatch_slots(
    compact, monkeypatch
):
    monkeypatch.setattr(
        torch.cuda, "current_device", lambda: pytest.fail("CPU reader consulted CUDA")
    )
    payload_a, full_a = dense_payload(compact=compact)
    payload_b, full_b = dense_payload(
        sequence=12, offset=900.0, compact=compact, teacher_mbs=1
    )
    a = DenseTeacherRowReader(payload_a, device="cpu")
    b = DenseTeacherRowReader(payload_b, device="cpu")
    batches = torch.tensor([2, 0, 2, 1, 0])
    rows = torch.tensor([1, 7, 1, 4, 2])
    assert not a._backings and not b._backings
    torch.testing.assert_close(a.gather_rows(batches, rows), full_a[batches, rows])
    torch.testing.assert_close(
        b.gather_rows(batches, rows, vocab_start=2, vocab_end=7),
        full_b[batches, rows, 2:7],
    )
    assert a._backings is not b._backings
    before = set(a._backings)
    torch.testing.assert_close(a.gather_rows(batches, rows), full_a[batches, rows])
    assert set(a._backings) == before
    # Student microbatch order differs from the teacher producer's slot order.
    sliced = DenseTeacherRowReader(
        DenseTeacherIPC([payload_a.samples[2], payload_a.samples[0]]), device="cpu"
    )
    torch.testing.assert_close(
        sliced.gather_rows(torch.tensor([1, 0]), torch.tensor([6, 3])),
        full_a[torch.tensor([0, 2]), torch.tensor([6, 3])],
    )
    next_payload, next_full = dense_payload(offset=1700.0, compact=compact)
    next_call = DenseTeacherRowReader(next_payload, device="cpu")
    torch.testing.assert_close(
        next_call.gather_rows(batches, rows), next_full[batches, rows]
    )


@pytest.mark.parametrize("compact", [False, True])
def test_native_dense_positions_padding_and_empty_requests(compact):
    payload, full = dense_payload(compact=compact)
    reader = DenseTeacherRowReader(payload, device="cpu")
    empty = torch.empty(0, dtype=torch.long)
    assert reader.gather_rows(empty, empty).shape == (0, 8)
    assert not reader._backings
    positions = torch.tensor([0, 1, 6, 7, 8, 9])
    mask = torch.ones(3, 6)
    mask[:, 4:] = 0
    actual = reader.gather_native_positions(positions, active_mask=mask, vocab_end=7)
    torch.testing.assert_close(actual[:, :4], full[:, positions[:4], :7])
    assert actual[:, 4:].count_nonzero() == 0
    mask[1, 4] = 1
    with pytest.raises(ValueError, match="Active native predictor"):
        reader.gather_native_positions(positions, active_mask=mask)
    with pytest.raises(ValueError, match="out of range"):
        reader.gather_rows(torch.tensor([0]), torch.tensor([8]))
    with pytest.raises(ValueError, match="vocabulary interval"):
        reader.gather_rows(empty, empty, vocab_end=9)


@pytest.mark.parametrize(
    "mutation",
    ["gap", "overlap", "vocab", "missing", "offset", "shape", "dtype", "identity"],
)
def test_native_dense_rejects_malformed_coverage_and_storage(mutation):
    payload, _ = dense_payload(compact=mutation == "offset")
    samples = copy.deepcopy(payload.samples)
    first = samples[0]["teacher_shards"][0]
    if mutation == "gap":
        first["global_seq_start"] = 1
    elif mutation == "overlap":
        samples[0]["teacher_shards"][2]["global_seq_start"] = 0
    elif mutation == "vocab":
        first["vocab_start_index"] = 1
    elif mutation == "missing":
        samples[0]["teacher_shards"].pop()
    elif mutation == "offset":
        first["storage_token_offset"] = first["storage_used_tokens"]
    elif mutation == "shape":
        first["sample_index_in_buf"] = 9
    elif mutation == "dtype":
        first["dtype"] = torch.bfloat16
    else:
        first["batch_item_id"] = 9
    with pytest.raises(ValueError):
        reader = DenseTeacherRowReader(DenseTeacherIPC(samples), device="cpu")
        reader.gather_rows(torch.tensor([0]), torch.tensor([0]))


def test_compact_empty_suffix_does_not_open_payload(monkeypatch):
    payload, _ = dense_payload(compact=True)
    reader = DenseTeacherRowReader(payload, device="cpu")
    monkeypatch.setattr(
        reader, "_open", lambda shard: pytest.fail("Zero suffix opened backing")
    )
    actual = reader.gather_rows(torch.tensor([2, 2, 0]), torch.tensor([0, 7, 7]))
    assert actual.count_nonzero() == 0


def test_native_dense_rejects_one_use_torch_handles_and_mixed_protocols():
    payload, _ = dense_payload()
    assert supports_native_dense_reads(payload)
    payload.samples[0]["teacher_shards"][0]["payload_ipc"] = (
        ("one-use-torch-handle",),
    )
    with pytest.raises(ValueError, match="mixes reusable and one-use"):
        supports_native_dense_reads(payload)
    for sample in payload.samples:
        for shard in sample["teacher_shards"]:
            shard["payload_ipc"] = (("one-use-torch-handle",),)
    assert not supports_native_dense_reads(payload)
    with pytest.raises(ValueError, match="require reusable IPC handles"):
        DenseTeacherRowReader(payload, device="cpu")


def test_native_dense_all_empty_compact_records_need_no_handle():
    payload, _ = dense_payload(compact=True)
    for sample in payload.samples:
        for shard in sample["teacher_shards"]:
            shard.update(
                payload_ipc=None,
                valid_seq_len=0,
                stored_seq_len=0,
                storage_token_offset=0,
            )
    assert supports_native_dense_reads(payload)
    reader = DenseTeacherRowReader(payload, device="cpu")
    actual = reader.gather_rows(torch.tensor([0, 1, 2]), torch.tensor([0, 4, 7]))
    assert actual.count_nonzero() == 0
    assert not reader._backings
