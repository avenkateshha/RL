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

"""Reusable, explicitly producer-owned CUDA IPC slabs.

Torch's multiprocessing export creates a one-use reference counter. A descriptor
cannot reuse that counter across fresh readers or fan out to multiple ranks.
These slabs instead use CUDA's refcounted open/close API directly. The controller
must retain the producing tensor until every consumer completes or terminates;
receivers never own or free the producing allocation.

Only allocations from ``allocate_reusable_cuda_tensor`` may be exported. Using
cudaMalloc directly also avoids depending on PyTorch's expandable-segment IPC
format. The existing generic Torch IPC transport is deliberately unchanged.
"""

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import UUID
from weakref import WeakValueDictionary

import torch

_TYPES = {
    torch.float32: ("<f4", 4),
    torch.int32: ("<i4", 4),
    torch.bool: ("|b1", 1),
}
# Weak producer-only ownership registries. They never retain an allocation or
# cache an imported reader. Local consumers resolve an authenticated live owner
# rather than attempting CUDA IPC into the exporting process itself.
_ALLOCATIONS: WeakValueDictionary[int, torch.Tensor] = WeakValueDictionary()
_EXPORTED_ALLOCATIONS: WeakValueDictionary[bytes, torch.Tensor] = WeakValueDictionary()


@lru_cache(maxsize=1)
def _host_boot_id() -> str:
    return str(UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip()))


def _checked(result: tuple[Any, ...], operation: str) -> Any:
    code, *values = result
    if int(code) != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {code}")
    return values[0] if values else None


def _device_index(device: torch.device | int) -> int:
    normalized = torch.device("cuda", device) if isinstance(device, int) else device
    if normalized.type != "cuda":
        raise ValueError("Reusable CUDA IPC requires a CUDA device")
    return torch.cuda.current_device() if normalized.index is None else normalized.index


@lru_cache(maxsize=1)
def _cuda_runtime() -> ModuleType:
    # CUDA bindings ships this compiled extension without a discoverable stub.
    from cuda.bindings import runtime  # pyrefly: ignore[missing-module-attribute]

    return runtime


def _device_uuid(device_index: int) -> bytes:
    runtime = _cuda_runtime()

    properties = _checked(
        runtime.cudaGetDeviceProperties(device_index), "cudaGetDeviceProperties"
    )
    return bytes(properties.uuid.bytes)


@dataclass(frozen=True)
class ReusableCudaIPCDescriptor:
    """Pickleable native CUDA handle, without a one-use Torch refcounter."""

    memory_handle: bytes
    shape: tuple[int, ...]
    dtype: torch.dtype
    allocation_nbytes: int
    producer_uuid: bytes
    producer_pid: int
    host_boot_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.memory_handle, bytes)
            or len(self.memory_handle) != 64
            or not isinstance(self.shape, tuple)
            or not self.shape
            or any(not isinstance(size, int) or size <= 0 for size in self.shape)
            or self.dtype not in _TYPES
            or not isinstance(self.producer_uuid, bytes)
            or len(self.producer_uuid) != 16
            or self.producer_uuid == bytes(16)
            or self.producer_pid <= 0
        ):
            raise ValueError("Malformed reusable CUDA IPC descriptor")
        if self.allocation_nbytes != math.prod(self.shape) * _TYPES[self.dtype][1]:
            raise ValueError("Reusable CUDA IPC shape/dtype/byte count disagree")
        UUID(self.host_boot_id)


def is_reusable_cuda_ipc_handle(value: object) -> bool:
    return isinstance(value, ReusableCudaIPCDescriptor)


def _release_pointer(pointer: int, device_index: int, *, imported: bool) -> None:
    runtime = _cuda_runtime()
    with torch.cuda.device(device_index):
        # cudaIpcCloseMemHandle does not wait for kernels using the mapping.
        # Wait for all receiver streams, not merely its current stream.
        _checked(runtime.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
        operation = runtime.cudaIpcCloseMemHandle if imported else runtime.cudaFree
        _checked(
            operation(pointer), "cudaIpcCloseMemHandle" if imported else "cudaFree"
        )


class _CudaArrayOwner:
    """Retained by the Tensor storage's CUDA-array-interface deleter.

    PyTorch v2.11 tensor_numpy.cpp:480-495 increfs the interface object and
    decrefs it in the from_blob storage deleter. This object holds no tensor,
    avoiding a reference cycle. Synchronization precedes close/free so queued
    indexing and peer copies cannot outlive the mapping.
    """

    pointer: int = 0

    def __init__(
        self,
        pointer: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device_index: int,
        *,
        imported: bool,
    ) -> None:
        self.device_index = device_index
        self.imported = imported
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": _TYPES[dtype][0],
            "data": (pointer, False),
            "strides": None,
            "version": 3,
        }
        # A partial constructor owns nothing. The caller closes the raw pointer
        # if any metadata construction above raises before transfer completes.
        self.pointer = pointer

    def close(self) -> None:
        if not self.pointer:
            return
        _release_pointer(self.pointer, self.device_index, imported=self.imported)
        self.pointer = 0

    def __del__(self) -> None:
        self.close()


def allocate_reusable_cuda_tensor(
    shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Create a contiguous slab whose Torch storage retains its cudaMalloc owner."""
    runtime = _cuda_runtime()

    index = _device_index(device)
    if (
        dtype not in _TYPES
        or not shape
        or any(not isinstance(size, int) or size <= 0 for size in shape)
    ):
        raise ValueError(
            "Reusable CUDA IPC needs positive dimensions and fp32/int32/bool"
        )
    with torch.cuda.device(index):
        pointer = int(
            _checked(
                runtime.cudaMalloc(math.prod(shape) * _TYPES[dtype][1]), "cudaMalloc"
            )
        )
        owner = None
        transferred = False
        try:
            owner = _CudaArrayOwner(pointer, shape, dtype, index, imported=False)
            tensor = torch.as_tensor(owner, device=torch.device("cuda", index))
            if tensor.data_ptr() != pointer:
                raise RuntimeError("Torch copied the external CUDA allocation")
            _ALLOCATIONS[pointer] = tensor
            transferred = True
        finally:
            if not transferred:
                if owner is None:
                    _release_pointer(pointer, index, imported=False)
                else:
                    owner.close()
        return tensor


def get_reusable_cuda_ipc_handle(tensor: torch.Tensor) -> ReusableCudaIPCDescriptor:
    """Export a whole slab allocated by ``allocate_reusable_cuda_tensor``.

    This does not synchronize producer writes. Exporters must synchronize every
    successful publication, including publications that reuse this descriptor.
    CUDA's handle is stable for one allocation and changes after reallocation.
    """
    runtime = _cuda_runtime()

    index = _device_index(tensor.device)
    if (
        not tensor.is_contiguous()
        or tensor.storage_offset() != 0
        or tensor.dtype not in _TYPES
        or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
    ):
        raise ValueError("Reusable CUDA IPC exports require a whole contiguous slab")
    if _ALLOCATIONS.get(tensor.data_ptr()) is not tensor:
        raise ValueError("Reusable CUDA IPC must export an owned raw CUDA allocation")
    with torch.cuda.device(index):
        handle = _checked(
            runtime.cudaIpcGetMemHandle(tensor.data_ptr()), "cudaIpcGetMemHandle"
        )
    descriptor = ReusableCudaIPCDescriptor(
        memory_handle=bytes(handle.reserved),
        shape=tuple(tensor.shape),
        dtype=tensor.dtype,
        allocation_nbytes=tensor.numel() * tensor.element_size(),
        producer_uuid=_device_uuid(index),
        producer_pid=os.getpid(),
        host_boot_id=_host_boot_id(),
    )
    _EXPORTED_ALLOCATIONS[descriptor.memory_handle] = tensor
    return descriptor


def open_reusable_cuda_ipc(
    handle: ReusableCudaIPCDescriptor, device: torch.device | int
) -> torch.Tensor:
    """Open one independent CUDA mapping, retained until the returned tensor dies."""
    runtime = _cuda_runtime()

    if not isinstance(handle, ReusableCudaIPCDescriptor):
        raise TypeError("Native reusable CUDA IPC requires its typed descriptor")
    handle.__post_init__()
    if handle.host_boot_id != _host_boot_id():
        raise ValueError("Reusable CUDA IPC cannot cross hosts")
    index = _device_index(device)
    if handle.producer_pid == os.getpid():
        tensor = _EXPORTED_ALLOCATIONS.get(handle.memory_handle)
        if tensor is None or get_reusable_cuda_ipc_handle(tensor) != handle:
            raise ValueError("Reusable CUDA IPC local producer owner is no longer live")
        # Keep the original device; callers copy requested rows to the consumer.
        return tensor.detach()
    native_handle = runtime.cudaIpcMemHandle_t()
    native_handle.reserved = handle.memory_handle
    with torch.cuda.device(index):
        pointer = int(
            _checked(
                runtime.cudaIpcOpenMemHandle(native_handle, 1), "cudaIpcOpenMemHandle"
            )
        )
        owner = None
        transferred = False
        try:
            owner = _CudaArrayOwner(
                pointer, handle.shape, handle.dtype, index, imported=True
            )
            attributes = _checked(
                runtime.cudaPointerGetAttributes(pointer), "cudaPointerGetAttributes"
            )
            if _device_uuid(attributes.device) != handle.producer_uuid:
                raise ValueError("Reusable CUDA IPC pointer and producer UUID disagree")
            tensor = torch.as_tensor(owner, device=torch.device("cuda", index))
            if tensor.data_ptr() != pointer:
                raise RuntimeError("Torch copied the imported CUDA IPC allocation")
            transferred = True
        finally:
            if not transferred:
                if owner is None:
                    _release_pointer(pointer, index, imported=True)
                else:
                    owner.close()
        return tensor
