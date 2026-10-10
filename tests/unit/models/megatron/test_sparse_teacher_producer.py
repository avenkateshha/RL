"""Small streaming-storage and actual MCore producer-dispatch regressions."""

from dataclasses import fields, replace
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.multiprocessing as mp

from nemo_rl.algorithms.x_token.sparse_teacher import (
    SparseTeacherIPC,
    SparseTeacherRowReader,
)
from nemo_rl.distributed.sparse_topk import distributed_vocab_topk_logz_force
from nemo_rl.models.megatron.sparse_teacher import (
    SparseTeacherStorage,
    StreamedSparseLogitsMetadata,
)


def _output(offset=0.0, *, sequence=4, device=torch.device("cpu")):
    logits = (
        torch.arange(sequence * 6, device=device).reshape(1, sequence, 6).float() / 10
        + offset
    )
    force = torch.tensor([0, 4], device=device).expand(1, sequence, 2)
    return distributed_vocab_topk_logz_force(
        logits,
        force,
        2,
        None,
        vocab_start_index=0,
        vocab_end_index=6,
        real_vocab_size=5,
        temperature=1.5,
    )


def test_storage_streams_microbatches_reuses_handles_and_keeps_teachers_separate():
    first = SparseTeacherStorage.allocate(
        batch_size=2, local_sequence_length=4, k=2, device=torch.device("cpu")
    )
    second = SparseTeacherStorage.allocate(
        batch_size=3, local_sequence_length=6, k=2, device=torch.device("cpu")
    )
    first.write(_output(), sample_offset=0)
    first.write(_output(10.0), sample_offset=1)
    factory = MagicMock(side_effect=lambda tensor: tensor)
    handles = first.export_handles(factory)
    assert factory.call_count == 7
    assert first.export_handles(factory) is handles
    first.write(_output(30.0), sample_offset=0)
    assert first.export_handles(factory) is handles
    assert factory.call_count == 7
    assert second.handles is None
    record = first.sample_record(
        sample_index=1,
        full_sequence_length=4,
        cp_rank=0,
        cp_size=1,
        real_vocab_size=5,
        temperature=1.5,
        membership_k=2,
    )
    reader = SparseTeacherRowReader(
        SparseTeacherIPC([record]), device=torch.device("cpu")
    )
    values, ids, _, _ = reader.gather_rows(
        torch.tensor([0]), torch.tensor([2]), torch.tensor([0])
    )
    assert ids.tolist() == [[0, 4]]
    torch.testing.assert_close(values, torch.tensor([[11.2, 11.6]]))
    assert first.can_reuse(
        batch_size=1, local_sequence_length=2, k=2, device=torch.device("cpu")
    )
    assert not first.can_reuse(
        batch_size=3, local_sequence_length=4, k=2, device=torch.device("cpu")
    )
    assert not first.can_reuse(
        batch_size=2, local_sequence_length=5, k=2, device=torch.device("cpu")
    )
    assert not first.can_reuse(
        batch_size=2, local_sequence_length=4, k=3, device=torch.device("cpu")
    )


def test_storage_validates_capacity_and_all_fields_before_writes():
    storage = SparseTeacherStorage.allocate(
        batch_size=1, local_sequence_length=4, k=2, device=torch.device("cpu")
    )
    with pytest.raises(ValueError, match="geometry"):
        storage.write(_output(), sample_offset=1)
    storage.topk_logits.fill_(-777)
    output = _output()
    invalid = replace(output, forced_indices=output.forced_indices.float())
    with pytest.raises(ValueError, match="shape/dtype/device"):
        storage.write(invalid, sample_offset=0)
    assert bool((storage.topk_logits == -777).all())
    with pytest.raises(RuntimeError, match="handles"):
        storage.sample_record(
            sample_index=0,
            full_sequence_length=4,
            cp_rank=0,
            cp_size=1,
            real_vocab_size=5,
            temperature=1.5,
            membership_k=2,
        )


@pytest.mark.mcore
def test_postprocessor_writes_native_teacher_rows_and_returns_only_metadata(
    monkeypatch,
):
    # Import MCore/telemetry only for tests that need the actor runtime.
    import nemo_rl.models.megatron.train as train
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    monkeypatch.setattr(train, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(train, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(train, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(train, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(
        train,
        "cp_load_balanced_to_contiguous",
        lambda *args, **kwargs: pytest.fail(
            "Native sparse producer performed CP relayout"
        ),
    )
    storage = SparseTeacherStorage.allocate(
        batch_size=2, local_sequence_length=4, k=2, device=torch.device("cpu")
    )
    processor = train.SparseTeacherLogitsPostProcessor(
        k=2,
        temperature=1.5,
        real_vocab_size=5,
        noise_filter_k=0,
        output_storage=storage,
    )
    force = torch.full((1, 8, 2), -1, dtype=torch.long)
    force[0, :, 0] = torch.arange(8) % 5
    for batch in range(2):
        raw = torch.arange(1 * 4 * 6).reshape(1, 4, 6).float() / 10 + batch * 10
        _, result = processor(BatchedDataDict(force_include_token_ids=force), None)(raw)
        metadata = result["sparse_logits"]
        assert isinstance(metadata, StreamedSparseLogitsMetadata)
        assert metadata.sample_offset == batch
        assert all(
            not isinstance(getattr(metadata, field.name), torch.Tensor)
            for field in fields(metadata)
        )
    assert processor.sample_cursor == 2
    assert storage.forced_indices[0, 0, :, 0].tolist() == [0, 1, 1, 2]
    with pytest.raises(ValueError, match="unpacked"):
        processor(BatchedDataDict(force_include_token_ids=force), torch.tensor([0, 8]))


@pytest.mark.mcore
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Actual worker IPC handles require CUDA"
)
def test_worker_streaming_reuse_growth_synchronization_and_release(monkeypatch):
    # Worker integration needs the optional MCore/telemetry actor runtime.
    import nemo_rl.models.megatron.train as train
    import nemo_rl.models.policy.workers.megatron_policy_worker as worker_module
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    worker_type = worker_module.MegatronPolicyWorkerImpl
    device = torch.device("cuda", torch.cuda.current_device())
    worker = SimpleNamespace(
        cfg={
            "sequence_packing": {"enabled": False},
            "dynamic_batching": {"enabled": False},
            "logprob_batch_size": 1,
        },
        delegate_pack_to_model=False,
        delegate_mtp_loss_mask_to_model=False,
        model_slices_context_parallel_inputs=False,
        mtp_enabled=False,
        defer_fp32_logits=False,
        model=MagicMock(),
        mcore_state=SimpleNamespace(straggler_timer=None),
        _teacher_sparse_ipc_storage=None,
        _teacher_ipc_storage=None,
        _teacher_ipc_handles=[],
    )
    for name in (
        "get_context_parallel_rank",
        "get_tensor_model_parallel_rank",
        "get_data_parallel_rank",
    ):
        monkeypatch.setattr(worker_module.parallel_state, name, lambda: 0)
    for name in (
        "get_context_parallel_world_size",
        "get_tensor_model_parallel_world_size",
        "get_pipeline_model_parallel_world_size",
    ):
        monkeypatch.setattr(worker_module.parallel_state, name, lambda: 1)
    monkeypatch.setattr(train, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(train, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(train, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(train, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(worker_module, "get_model_config", lambda model: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *args, **kwargs: 0)
    calls = []
    original_sync = torch.cuda.synchronize
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *args, **kwargs: (calls.append("sync"), original_sync(*args, **kwargs))[
            -1
        ],
    )
    original_handle = worker_module.get_reusable_cuda_ipc_handle
    monkeypatch.setattr(
        worker_module,
        "get_reusable_cuda_ipc_handle",
        lambda tensor: (calls.append("handle"), original_handle(tensor))[-1],
    )
    generation = [0.0]

    def iterator(data, cfg, mbs, **kwargs):
        batches = [
            BatchedDataDict(
                {
                    key: value[index : index + mbs].to(device)
                    for key, value in data.items()
                }
            )
            for index in range(0, data.size, mbs)
        ]
        return (
            iter(batches),
            len(batches),
            mbs,
            data["input_ids"].shape[1],
            data["input_ids"].shape[1],
        )

    def forward(*, data_iterator, post_processing_fn, **kwargs):
        results = []
        for batch in data_iterator:
            calls.append("write")
            raw = (
                torch.arange(
                    batch.size * batch["input_ids"].shape[1] * 6, device=device
                )
                .reshape(batch.size, -1, 6)
                .float()
                / 10
                + generation[0]
            )
            _, metadata = post_processing_fn(batch, None)(raw)
            results.append(metadata)
        return results

    monkeypatch.setattr(worker_module, "get_microbatch_iterator", iterator)
    monkeypatch.setattr(worker_module, "megatron_forward_backward", forward)

    def run(batch_size):
        data = BatchedDataDict(
            input_ids=torch.ones(batch_size, 4, dtype=torch.long),
            force_include_token_ids=torch.tensor([0, 4])
            .expand(batch_size, 4, 2)
            .clone(),
        )
        return worker_type.get_topk_logits_ipc(
            worker, data, k=2, temperature=1.5, vocab_size=5, support_mode="row_topk"
        )

    first = run(2)
    original_storage = worker._teacher_sparse_ipc_storage
    first_handles = original_storage.handles
    saved_values = original_storage.topk_logits.clone()
    assert len(first["per_sample_handles"]) == 2
    assert calls.index("sync") > max(
        index for index, call in enumerate(calls) if call == "write"
    )
    calls.clear()
    generation[0] = 10.0
    run(2)
    assert worker._teacher_sparse_ipc_storage is original_storage
    assert original_storage.handles is first_handles
    torch.testing.assert_close(
        original_storage.topk_logits - saved_values, torch.full_like(saved_values, 10.0)
    )
    assert calls == ["write", "write", "sync"]
    run(3)
    assert worker._teacher_sparse_ipc_storage is not original_storage
    assert worker._teacher_sparse_ipc_storage.handles is not first_handles
    worker_type.release_ipc_buffer(worker)
    worker_type.release_ipc_buffer(worker)
    assert worker._teacher_sparse_ipc_storage is None

    # The opt-in dense producer keeps its existing layout and values, while
    # allocating reusable raw CUDA storage for native same-tokenizer readers.
    from nemo_rl.algorithms.x_token.dense_teacher import (
        DenseTeacherIPC,
        DenseTeacherRowReader,
    )
    from nemo_rl.utils.reusable_cuda_ipc import ReusableCudaIPCDescriptor

    worker._teacher_ipc_reusable = False
    worker._ensure_teacher_ipc_storage = MethodType(
        worker_type._ensure_teacher_ipc_storage, worker
    )
    monkeypatch.setattr(
        worker_module.parallel_state,
        "is_pipeline_last_stage",
        lambda ignore_virtual: True,
    )

    def dense_iterator(data, cfg, mbs, **kwargs):
        batches, count, size, original, padded = iterator(data, cfg, mbs, **kwargs)
        return (
            (SimpleNamespace(data_dict=batch) for batch in batches),
            count,
            size,
            original,
            padded,
        )

    def dense_forward(*, data_iterator, post_processing_fn, **kwargs):
        results = []
        for processed in data_iterator:
            batch = processed.data_dict
            raw = (
                torch.arange(batch.size * 4 * 6, device=device)
                .reshape(batch.size, 4, 6)
                .float()
                + generation[0]
            )
            _, result = post_processing_fn(batch, None)(raw)
            results.append(result)
        return results

    monkeypatch.setattr(worker_module, "get_microbatch_iterator", dense_iterator)
    monkeypatch.setattr(worker_module, "megatron_forward_backward", dense_forward)

    def run_dense(batch_size):
        return worker_type.get_full_logits_ipc(
            worker,
            BatchedDataDict(
                input_ids=torch.ones(batch_size, 4, dtype=torch.long),
                input_lengths=torch.full((batch_size,), 4, dtype=torch.long),
                batch_item_id=torch.arange(100, 100 + batch_size),
            ),
            reusable_ipc=True,
        )["per_sample_handles"]

    dense_records = run_dense(2)
    dense_storage = worker._teacher_ipc_storage
    dense_handle = dense_records[0]["payload_ipc"]
    assert isinstance(dense_handle, ReusableCudaIPCDescriptor)
    assert [record["batch_item_id"] for record in dense_records] == [100, 101]
    assert [record["storage_token_offset"] for record in dense_records] == [0, 4]
    generation[0] = 50.0
    dense_records = run_dense(2)
    assert worker._teacher_ipc_storage is dense_storage
    assert dense_records[0]["payload_ipc"] == dense_handle
    reader = DenseTeacherRowReader(
        DenseTeacherIPC([{"teacher_shards": [record]} for record in dense_records]),
        device=device,
    )
    rows = reader.gather_rows(torch.tensor([1]), torch.tensor([2]))
    torch.testing.assert_close(rows.cpu(), torch.arange(12, 18)[None].float() + 50.0)
    del reader
    grown_records = run_dense(3)
    assert worker._teacher_ipc_storage is not dense_storage
    assert grown_records[0]["payload_ipc"] != dense_handle
    worker_type.release_ipc_buffer(worker)
    assert worker._teacher_ipc_storage is None


@pytest.mark.mcore
def test_non_tp0_postprocessor_scores_without_allocating_persistent_storage(
    monkeypatch,
):
    # This dispatch check needs MCore imports, but no initialized process group.
    import nemo_rl.models.megatron.train as train
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    monkeypatch.setattr(train, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(train, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(train, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(train, "get_tensor_model_parallel_group", lambda: None)
    scoring = MagicMock(return_value=_output())
    monkeypatch.setattr(train, "distributed_vocab_topk_logz_force", scoring)
    processor = train.SparseTeacherLogitsPostProcessor(
        k=2, temperature=1.5, real_vocab_size=5, noise_filter_k=0, output_storage=None
    )
    force = torch.tensor([0, 4]).expand(1, 4, 2)
    _, result = processor(BatchedDataDict(force_include_token_ids=force), None)(
        torch.zeros(1, 4, 3)
    )
    assert scoring.call_count == 1
    assert scoring.call_args.kwargs["vocab_start_index"] == 3
    assert processor.output_storage is None
    assert isinstance(result["sparse_logits"], StreamedSparseLogitsMetadata)


def _peer_reader_process(rank, first, second, generation_offset):
    torch.cuda.set_device(1)
    for payload, position, expected in (
        (first, 2, [[11.2, 11.6]]),
        (second, 4, [[92.4, 92.8]]),
    ):
        reader = SparseTeacherRowReader(payload, device=torch.device("cuda", 1))
        values, ids, _, _ = reader.gather_rows(
            torch.tensor([0]), torch.tensor([position]), torch.tensor([0])
        )
        assert ids.tolist() == [[0, 4]]
        torch.testing.assert_close(
            values.cpu(),
            torch.tensor(expected) + generation_offset,
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.mcore
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Peer CUDA IPC needs two GPUs"
)
def test_two_teachers_cuda_ipc_requested_rows_in_separate_peer_process():
    # Real handle creation requires the actor's policy utilities and runtime.
    from nemo_rl.utils.reusable_cuda_ipc import get_reusable_cuda_ipc_handle

    torch.cuda.set_device(0)
    producers = []
    payloads = []
    for sequence, batch_size, offset in ((4, 2, 0.0), (6, 3, 70.0)):
        storage = SparseTeacherStorage.allocate(
            batch_size=batch_size,
            local_sequence_length=sequence,
            k=2,
            device=torch.device("cuda", 0),
        )
        for batch in range(batch_size):
            storage.write(
                _output(
                    offset + 10 * batch,
                    sequence=sequence,
                    device=torch.device("cuda", 0),
                ),
                sample_offset=batch,
            )
        torch.cuda.synchronize()
        storage.export_handles(get_reusable_cuda_ipc_handle)
        record = storage.sample_record(
            sample_index=batch_size - 1,
            full_sequence_length=sequence,
            cp_rank=0,
            cp_size=1,
            real_vocab_size=5,
            temperature=1.5,
            membership_k=2,
        )
        producers.append(storage)
        payloads.append(SparseTeacherIPC([record]))
    # Both producer allocations coexist until the consumer process exits.
    mp.spawn(_peer_reader_process, args=(*payloads, 0.0), nprocs=1, join=True)
    for storage, offset in zip(producers, (0.0, 70.0), strict=True):
        for batch in range(storage.topk_logits.shape[1]):
            storage.write(
                _output(
                    offset + 10 * batch + 100.0,
                    sequence=storage.topk_logits.shape[2],
                    device=torch.device("cuda", 0),
                ),
                sample_offset=batch,
            )
    torch.cuda.synchronize()
    # Reuse exactly the same handles in a fresh consumer invocation. Storage
    # survives both generations, while each reader opens a private cache.
    mp.spawn(_peer_reader_process, args=(*payloads, 100.0), nprocs=1, join=True)
    assert len(producers) == 2
