"""Small read-only counter probe for cached CUDA IPC descriptors."""

import gc
import inspect
import os
import struct

import torch
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from nemo_rl.models.policy.utils import (
    get_handle_from_tensor,
    rebuild_cuda_tensor_from_ipc,
)


def counter(handle):
    args = inspect.signature(rebuild_cuda_tensor).bind(*handle[0]).arguments
    name = args["ref_counter_handle"].decode()
    fd = os.open("/dev/shm/" + name.lstrip("/"), os.O_RDONLY)
    try:
        # RefcountedMapAllocator::data() starts after its 64-byte map header.
        value = os.pread(fd, 8, 64 + args["ref_counter_offset"] * 8)
    finally:
        os.close(fd)
    return struct.unpack("q", value)[0]


def consumer(rank, handle):
    torch.cuda.set_device(1)
    tensor = rebuild_cuda_tensor_from_ipc(handle, 1)
    assert tensor.cpu().tolist() == [1.0, 2.0]
    torch.cuda.synchronize()
    del tensor
    gc.collect()
    torch.cuda.synchronize()


if __name__ == "__main__":
    torch.cuda.set_device(0)
    tensor = torch.tensor([1.0, 2.0], device="cuda")
    handle = get_handle_from_tensor(tensor)
    print("counter after one export:", counter(handle), flush=True)
    for generation in range(2):
        mp.spawn(consumer, args=(handle,), nprocs=1, join=True)
        print("counter after consumer", generation, ":", counter(handle), flush=True)
    del tensor
    gc.collect()
    torch.cuda.ipc_collect()
