# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable dense IPC retains legacy row geometry and zero-copy ownership."""

import gc
from dataclasses import replace

import pytest
import torch
import torch.multiprocessing as mp

from nemo_rl.algorithms.x_token import loss_utils
from nemo_rl.models.policy.utils import DENSE_TEACHER_IPC_FLAT_LAYOUT
from nemo_rl.utils import reusable_cuda_ipc as ipc


def _descriptor(tensor, identity=1):
    return ipc.ReusableCudaIPCDescriptor(
        identity.to_bytes(64, "little"),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.numel() * tensor.element_size(),
        b"u" * 16,
        123,
        "00000000-0000-0000-0000-000000000001",
    )


def _entries(
    payload,
    shape,
    *,
    compact,
    batch=2,
    full_seq=8,
    vocab=5,
    stored_seq=None,
    offset=3,
    vocab_start=0,
    full_vocab=None,
):
    stored_seq = full_seq if stored_seq is None else stored_seq
    entries = []
    for sample in range(batch):
        shard = {
            "payload_ipc": payload,
            "buf_idx": 0,
            "sample_index_in_buf": sample,
            "actual_shape": (full_seq, vocab),
            "vocab_start_index": vocab_start,
            "vocab_end_index": vocab_start + vocab,
            "global_seq_start": 0,
            "full_seq_len": full_seq,
            "full_vocab_size": vocab if full_vocab is None else full_vocab,
            "dtype": torch.float32,
        }
        if compact:
            shard.update(
                {
                    "ipc_layout": DENSE_TEACHER_IPC_FLAT_LAYOUT,
                    "storage_shape": shape,
                    "storage_token_offset": offset + sample * stored_seq,
                    "storage_used_tokens": offset + batch * stored_seq,
                    "storage_capacity_tokens": shape[0],
                    "stored_seq_len": stored_seq,
                    "valid_seq_len": stored_seq,
                }
            )
        entries.append({"teacher_shards": [shard]})
    return entries


def _storage(*, compact, device="cpu"):
    shape = (23, 5) if compact else (1, 3, 10, 7)
    storage = torch.arange(
        torch.tensor(shape).prod().item(), dtype=torch.float32, device=device
    ).reshape(shape)
    expected = storage[3:19].reshape(2, 8, 5) if compact else storage[0, :2, :8, :5]
    return storage, expected


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("cp_rank", [0, 1])
def test_raw_zero_copy_keeps_capacity_and_cp_slice(monkeypatch, compact, cp_rank):
    storage, expected = _storage(compact=compact)
    handle = _descriptor(storage)
    calls = []

    def open_raw(payload, device):
        calls.append(payload)
        return storage

    monkeypatch.setattr(loss_utils, "open_reusable_cuda_ipc", open_raw)
    monkeypatch.setattr(
        "nemo_rl.models.policy.utils.rebuild_cuda_tensor_from_ipc",
        lambda *a: pytest.fail("raw payload used Torch counter"),
    )
    entries = _entries(handle, tuple(storage.shape), compact=compact)
    result = loss_utils._try_zero_copy_teacher_logits(
        entries, student_cp_rank=cp_rank, student_cp_size=2, device=0
    )
    assert len(calls) == 1
    expected = expected[:, cp_rank * 4 : (cp_rank + 1) * 4]
    assert result.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    torch.testing.assert_close(result, expected)
    result[0, 0, 0] = -17
    assert expected[0, 0, 0] == -17


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_raw_compact_tp_assembly_preserves_short_prefix_and_zero_suffix(
    monkeypatch, cp_rank
):
    whole = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6)
    backing, shards = {}, []
    for tp_rank in range(2):
        storage = torch.full((10, 3), -99.0)
        storage[2:8].copy_(whole[:6, tp_rank * 3 : (tp_rank + 1) * 3])
        handle = _descriptor(storage, identity=tp_rank + 1)
        backing[handle] = storage
        entry = _entries(
            handle,
            tuple(storage.shape),
            compact=True,
            batch=1,
            vocab=3,
            stored_seq=6,
            offset=2,
            vocab_start=tp_rank * 3,
            full_vocab=6,
        )[0]
        shards.extend(entry["teacher_shards"])
    monkeypatch.setattr(
        loss_utils, "open_reusable_cuda_ipc", lambda payload, device: backing[payload]
    )
    result = loss_utils.assemble_teacher_logits_from_shards(
        shards, student_cp_rank=cp_rank, student_cp_size=2, device="cpu"
    )
    expected = whole.clone()
    expected[6:] = 0
    torch.testing.assert_close(result, expected[cp_rank * 4 : (cp_rank + 1) * 4])


def test_raw_compact_zero_length_row_does_not_open(monkeypatch):
    entries = _entries(None, (1, 5), compact=True, batch=1, stored_seq=0, offset=0)
    monkeypatch.setattr(
        loss_utils,
        "open_reusable_cuda_ipc",
        lambda *a: pytest.fail("empty row opened storage"),
    )
    result, fallback = loss_utils.rebuild_teacher_full_logits_from_ipc(
        entries, cp_group=None, device="cpu"
    )
    assert fallback == 1
    torch.testing.assert_close(result, torch.zeros(1, 8, 5))


@pytest.mark.parametrize(
    "invalid", ["dtype", "rank", "compact_shape", "rectangle_slice"]
)
def test_raw_dense_rejects_invalid_metadata(monkeypatch, invalid):
    compact = invalid == "compact_shape"
    storage, _ = _storage(compact=compact)
    handle = _descriptor(storage)
    if invalid == "dtype":
        handle = replace(handle, dtype=torch.int32)
    if invalid == "rank":
        handle = replace(handle, shape=(storage.numel(),))
    entry = _entries(handle, tuple(storage.shape), compact=compact)[0][
        "teacher_shards"
    ][0]
    if invalid == "compact_shape":
        # Preserve valid metadata, but the actual mapped slab has a different capacity.
        storage = storage[:22]
    if invalid == "rectangle_slice":
        entry["sample_index_in_buf"] = 3
    monkeypatch.setattr(loss_utils, "open_reusable_cuda_ipc", lambda *a: storage)
    with pytest.raises(
        ValueError,
        match="Reusable dense|unexpected storage shape|outside rebuilt storage",
    ):
        loss_utils._rebuild_teacher_ipc_row(entry, device=0)


def _consumer(rank, entries, expected):
    torch.cuda.set_device(1)
    closed = []
    original_close = ipc._CudaArrayOwner.close

    def counted_close(owner):
        pointer = owner.pointer
        original_close(owner)
        if pointer and owner.imported:
            closed.append(pointer)

    ipc._CudaArrayOwner.close = counted_close
    stream = torch.cuda.Stream(device=1)
    for iteration in range(2):
        result, fallback = loss_utils.rebuild_teacher_full_logits_from_ipc(
            entries, cp_group=None, device=1
        )
        assert fallback == 0
        alias = result[:, :, 1:]
        del result
        gc.collect()
        assert len(closed) == iteration
        with torch.cuda.stream(stream):
            copied = alias.clone()
        del alias
        gc.collect()
        assert len(closed) == iteration + 1
        torch.testing.assert_close(copied.cpu(), torch.tensor(expected)[:, :, 1:])


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
@pytest.mark.parametrize("compact", [False, True])
def test_cuda_legacy_dense_zero_copy_alias_retains_imported_owner(compact):
    torch.cuda.set_device(0)
    values, expected = _storage(compact=compact, device="cuda:0")
    storage = ipc.allocate_reusable_cuda_tensor(
        tuple(values.shape), dtype=values.dtype, device=values.device
    )
    storage.copy_(values)
    torch.cuda.synchronize()
    handle = ipc.get_reusable_cuda_ipc_handle(storage)
    entries = _entries(handle, tuple(storage.shape), compact=compact)
    mp.spawn(_consumer, args=(entries, expected.cpu().tolist()), nprocs=2, join=True)
    del storage
    gc.collect()
