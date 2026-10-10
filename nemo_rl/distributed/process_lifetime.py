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
"""Linux process-exit evidence for consumers of reusable CUDA IPC storage."""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import ray


@dataclass(frozen=True)
class WorkerProcessIdentity:
    node_id: str
    pid: int
    start_ticks: int
    boot_id: str
    pid_namespace: int


@dataclass(frozen=True)
class WorkerProcessSnapshot:
    actor_ids: tuple[str, ...]
    processes: tuple[WorkerProcessIdentity, ...]


def _process_start_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    # comm (field 2) can contain spaces and parentheses. Field 22 is starttime.
    return int(stat.rsplit(")", 1)[1].split()[19])


def capture_worker_process(_actor: object) -> WorkerProcessIdentity:
    """Execute via Ray's built-in __ray_call__ while the actor is healthy."""
    pid = os.getpid()
    start_ticks = _process_start_ticks(pid)
    if start_ticks is None:
        raise RuntimeError("Cannot observe the worker's Linux process identity")
    return WorkerProcessIdentity(
        node_id=ray.get_runtime_context().get_node_id(),
        pid=pid,
        start_ticks=start_ticks,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        pid_namespace=os.stat("/proc/self/ns/pid").st_ino,
    )


def wait_for_processes_exit(
    identities: tuple[WorkerProcessIdentity, ...], timeout: float | None
) -> bool:
    """Observe actual process disappearance, independently of Ray actor state.

    Namespace/boot checks prevent an observer in a different container namespace
    from mistaking an invisible live process for an exited one. A reused PID is
    safe only when the recorded Linux start time also differs. Zombies continue
    to count as present until the OS reaps them.
    """
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    pid_namespace = os.stat("/proc/self/ns/pid").st_ino
    if any(
        identity.boot_id != boot_id or identity.pid_namespace != pid_namespace
        for identity in identities
    ):
        raise RuntimeError(
            "Process observer does not share the worker's boot/PID namespace"
        )
    deadline = None if timeout is None else time.monotonic() + timeout
    remaining = list(identities)
    while remaining:
        remaining = [
            identity
            for identity in remaining
            if _process_start_ticks(identity.pid) == identity.start_ticks
        ]
        if not remaining:
            return True
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


@ray.remote(num_cpus=0, num_gpus=0, max_retries=0)
def observe_worker_processes_exit(  # pragma: no cover
    identities: tuple[WorkerProcessIdentity, ...], timeout: float | None
) -> bool:
    node_id = ray.get_runtime_context().get_node_id()
    if any(identity.node_id != node_id for identity in identities):
        raise RuntimeError("Process observer is not running on the worker's node")
    return wait_for_processes_exit(identities, timeout)
