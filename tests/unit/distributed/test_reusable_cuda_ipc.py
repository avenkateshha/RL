"""Reusable raw CUDA IPC fanout, view lifetime and asynchronous-copy cleanup."""

import gc
import pickle
import weakref
from dataclasses import replace

import pytest
import torch
import torch.multiprocessing as mp

from nemo_rl.utils.reusable_cuda_ipc import (
    ReusableCudaIPCDescriptor,
    allocate_reusable_cuda_tensor,
    get_reusable_cuda_ipc_handle,
    is_reusable_cuda_ipc_handle,
    open_reusable_cuda_ipc,
)


def _descriptor():
    return ReusableCudaIPCDescriptor(
        memory_handle=b"m" * 64,
        shape=(2, 4),
        dtype=torch.float32,
        allocation_nbytes=32,
        producer_uuid=b"u" * 16,
        producer_pid=1,
        host_boot_id="00112233-4455-6677-8899-aabbccddeeff",
    )


def test_descriptor_is_hashable_pickleable_and_rejects_invalid_geometry():
    descriptor = _descriptor()
    assert pickle.loads(pickle.dumps(descriptor)) == descriptor
    assert len({descriptor, descriptor}) == 1
    assert is_reusable_cuda_ipc_handle(descriptor)
    assert not is_reusable_cuda_ipc_handle(("torch_one_use_counter",))
    for changes in (
        {"shape": (2, 0)},
        {"shape": ("2", 4)},
        {"memory_handle": b"short"},
        {"producer_uuid": bytes(16)},
        {"allocation_nbytes": 16},
        {"dtype": torch.bfloat16},
    ):
        with pytest.raises(ValueError):
            replace(descriptor, **changes)
    with pytest.raises(ValueError, match="CUDA device"):
        allocate_reusable_cuda_tensor(
            (2, 4), dtype=torch.float32, device=torch.device("cpu")
        )
    with pytest.raises(ValueError, match="fp32/int32/bool"):
        allocate_reusable_cuda_tensor(
            (2, 4), dtype=torch.bfloat16, device=torch.device("cuda", 0)
        )


def _fanout_consumer(rank, handle, offset):
    import nemo_rl.utils.reusable_cuda_ipc as module

    device = torch.device("cuda", rank + 1)
    torch.cuda.set_device(device)
    closed = []
    original_close = module._CudaArrayOwner.close

    def counted_close(owner):
        pointer = owner.pointer
        original_close(owner)
        if pointer:
            closed.append((pointer, owner.imported))

    module._CudaArrayOwner.close = counted_close
    stream = torch.cuda.Stream(device=device)
    for iteration in range(3):
        tensor = open_reusable_cuda_ipc(handle, device)
        view = tensor[:, 1:]
        del tensor
        gc.collect()
        assert len(closed) == iteration
        with torch.cuda.stream(stream):
            copied = view.clone()
        # The imported owner must wait for the nondefault stream's copy before
        # CUDA closes its mapping; retaining only a tensor view is sufficient.
        del view
        gc.collect()
        assert len(closed) == iteration + 1
        assert closed[-1][1]
        torch.testing.assert_close(
            copied.cpu(), torch.arange(8).reshape(2, 4)[:, 1:].float() + offset
        )
    # A failed metadata check must close immediately, even while the exception
    # object keeps its traceback and local owner alive.
    with pytest.raises(ValueError, match="UUID disagree") as failure:
        open_reusable_cuda_ipc(replace(handle, producer_uuid=b"x" * 16), device)
    assert failure.value is not None
    assert len(closed) == 4


@pytest.mark.skipif(
    torch.cuda.device_count() < 3, reason="CUDA IPC fanout needs 3 GPUs"
)
def test_reusable_cuda_ipc_fanout_reopen_update_growth_and_lifetime(monkeypatch):
    from cuda.bindings import runtime

    import nemo_rl.utils.reusable_cuda_ipc as module

    torch.cuda.set_device(0)
    released = []
    original_free = runtime.cudaFree

    def tracked_free(pointer):
        result = original_free(pointer)
        released.append(pointer)
        return result

    monkeypatch.setattr(runtime, "cudaFree", tracked_free)
    tensor = allocate_reusable_cuda_tensor(
        (2, 4), dtype=torch.float32, device=torch.device("cuda", 0)
    )
    pointer = tensor.data_ptr()
    tensor.copy_(torch.arange(8).reshape(2, 4))
    handle = get_reusable_cuda_ipc_handle(tensor)
    assert get_reusable_cuda_ipc_handle(tensor) == handle
    local = open_reusable_cuda_ipc(handle, 0)
    assert local.data_ptr() == pointer
    del local
    torch.cuda.synchronize()
    for offset in (0.0, 100.0):
        tensor.copy_(torch.arange(8).reshape(2, 4) + offset)
        torch.cuda.synchronize()
        mp.spawn(_fanout_consumer, args=(handle, offset), nprocs=2, join=True)
        assert released == []
    producer_ref = weakref.ref(tensor)
    del tensor
    gc.collect()
    assert producer_ref() is None
    assert released == [pointer]
    with pytest.raises(ValueError, match="owner is no longer live"):
        open_reusable_cuda_ipc(handle, 0)
    grown = allocate_reusable_cuda_tensor(
        (3, 4), dtype=torch.float32, device=torch.device("cuda", 0)
    )
    assert get_reusable_cuda_ipc_handle(grown).memory_handle != handle.memory_handle
    del grown
    gc.collect()
    assert len(released) == 2
    for target in (torch, module):
        attribute = "as_tensor" if target is torch else "_CudaArrayOwner"
        with monkeypatch.context() as context:

            def fail(*args, **kwargs):
                raise RuntimeError("injected constructor failure")

            context.setattr(target, attribute, fail)
            with pytest.raises(RuntimeError, match="injected constructor failure"):
                allocate_reusable_cuda_tensor(
                    (2, 4), dtype=torch.float32, device=torch.device("cuda", 0)
                )
    assert len(released) == 4
