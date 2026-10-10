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

"""Small real CUDA dense-reader fanout and repeated-generation regression."""

import gc
from multiprocessing.connection import Connection

import pytest
import torch
import torch.multiprocessing as mp

from nemo_rl.algorithms.x_token.dense_teacher import (
    DenseTeacherIPC,
    DenseTeacherRowReader,
)
from nemo_rl.utils.reusable_cuda_ipc import (
    allocate_reusable_cuda_tensor,
    get_reusable_cuda_ipc_handle,
)
from tests.unit.algorithms.x_token.test_native_dense_reader import dense_payload


def _dense_consumer(device_index: int, connection: Connection) -> None:
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    try:
        for generation in range(2):
            payloads, references = connection.recv()
            for _invocation in range(2):
                for payload, reference in zip(payloads, references, strict=True):
                    # Student sample order/microbatch differs from teacher slots.
                    reader = DenseTeacherRowReader(
                        DenseTeacherIPC([payload.samples[2], payload.samples[0]]),
                        device=device,
                    )
                    batches = torch.tensor([1, 0, 1, 0], device=device)
                    positions = torch.tensor([7, 3, 1, 6], device=device)
                    rows = reader.gather_rows(
                        batches, positions, vocab_start=1, vocab_end=7
                    )
                    expected = torch.tensor(reference, device=device)[
                        torch.tensor([0, 2, 0, 2], device=device), positions, 1:7
                    ]
                    torch.testing.assert_close(rows, expected)
                    assert all(
                        backing.device == device
                        for backing in reader._backings.values()
                    )
                    # Mapping lifetime ends while copied teacher rows survive to
                    # consumer backward, as in the actual distillation loss.
                    del reader
                    gc.collect()
                    student = torch.ones_like(rows, requires_grad=True)
                    (student * rows).sum().backward()
                    torch.testing.assert_close(student.grad, expected)
                    del rows, expected, student
            torch.cuda.synchronize(device)
            connection.send(("consumed", generation))
    finally:
        connection.close()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Dense CUDA IPC fanout needs two GPUs"
)
def test_reusable_dense_two_teachers_fresh_readers_and_generations() -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    payloads = []
    references = []
    compact_flags = []
    owners = []
    for sequence, offset, microbatch, compact in (
        (8, 0.0, 2, False),
        (12, 900.0, 1, True),
    ):
        payload, reference = dense_payload(
            sequence=sequence, offset=offset, teacher_mbs=microbatch, compact=compact
        )
        handles = {}
        for sample in payload.samples:
            for shard in sample["teacher_shards"]:
                source = shard["payload_ipc"]
                if id(source) not in handles:
                    owner = allocate_reusable_cuda_tensor(
                        tuple(source.shape), dtype=source.dtype, device=device
                    )
                    owner.copy_(source)
                    owners.append(owner)
                    handles[id(source)] = get_reusable_cuda_ipc_handle(owner)
                shard["payload_ipc"] = handles[id(source)]
        payloads.append(payload)
        references.append(reference)
        compact_flags.append(compact)
    torch.cuda.synchronize(device)
    context = mp.get_context("spawn")
    consumers = []
    connections = []
    try:
        for index in range(2):
            parent, child = context.Pipe()
            process = context.Process(target=_dense_consumer, args=(index, child))
            process.start()
            child.close()
            consumers.append(process)
            connections.append(parent)
        for generation in range(2):
            if generation:
                for owner in owners:
                    owner.add_(1000.0)
                for reference, compact in zip(references, compact_flags, strict=True):
                    if compact:
                        for sample, valid in enumerate(
                            (reference.shape[1] - 1, reference.shape[1] // 2 + 1, 0)
                        ):
                            reference[sample, :valid] += 1000.0
                    else:
                        reference.add_(1000.0)
                torch.cuda.synchronize(device)
            for connection in connections:
                connection.send(
                    (payloads, [reference.tolist() for reference in references])
                )
            for connection in connections:
                assert connection.poll(120), (
                    "Dense IPC consumer did not acknowledge completion"
                )
                assert connection.recv() == ("consumed", generation)
        for process in consumers:
            process.join(timeout=30)
            assert process.exitcode == 0
    finally:
        # A failed consumer must terminate before raw producer allocations die.
        for process in consumers:
            if process.is_alive():
                process.terminate()
            process.join(timeout=30)
        for connection in connections:
            connection.close()
        owners.clear()
        gc.collect()
