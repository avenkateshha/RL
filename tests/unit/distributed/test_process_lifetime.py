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
"""Actual OS disappearance, PID reuse, and idempotent Ray shutdown checks."""

import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import ray

from nemo_rl.distributed import process_lifetime
from nemo_rl.distributed.process_lifetime import (
    WorkerProcessIdentity,
    WorkerProcessSnapshot,
    _process_start_ticks,
    wait_for_processes_exit,
)
from nemo_rl.distributed.worker_groups import RayWorkerGroup


def _identity(pid: int) -> WorkerProcessIdentity:
    start_ticks = _process_start_ticks(pid)
    assert start_ticks is not None
    return WorkerProcessIdentity(
        node_id="1" * 56,
        pid=pid,
        start_ticks=start_ticks,
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        pid_namespace=os.stat("/proc/self/ns/pid").st_ino,
    )


def _bare_group() -> RayWorkerGroup:
    group = RayWorkerGroup.__new__(RayWorkerGroup)
    group._workers = []
    group._worker_metadata = []
    group._initializer_pool = {}
    group._termination_snapshot = None
    group._termination_unconfirmed = False
    return group


def _worker(actor_id: str):
    return SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: actor_id))


def test_process_observer_waits_for_actual_subprocess_exit():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = _identity(process.pid)
        assert not wait_for_processes_exit((identity,), timeout=0.01)
        process.kill()
        process.wait(timeout=5)
        assert wait_for_processes_exit((identity,), timeout=1)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_reused_pid_is_not_the_original_consumer():
    identity = _identity(os.getpid())
    assert wait_for_processes_exit(
        (replace(identity, start_ticks=identity.start_ticks + 1),), timeout=0
    )
    assert not wait_for_processes_exit((identity,), timeout=0)


@pytest.mark.parametrize("change", [{"boot_id": "other-boot"}, {"pid_namespace": -1}])
def test_invisible_namespace_is_not_exit_evidence(change):
    with pytest.raises(RuntimeError, match="boot/PID namespace"):
        wait_for_processes_exit((replace(_identity(os.getpid()), **change),), timeout=0)


def test_proc_command_parentheses_do_not_corrupt_starttime(monkeypatch):
    # Field 3 is state; starttime is field 22. comm can itself contain ')'.
    fields = ["S"] + ["0"] * 18 + ["812345"] + ["0"] * 5
    monkeypatch.setattr(
        Path, "read_text", lambda self: "52 (a strange) command) " + " ".join(fields)
    )
    assert _process_start_ticks(52) == 812345


def test_observer_permission_failure_is_not_exit_evidence(monkeypatch):
    identity = _identity(os.getpid())
    monkeypatch.setattr(
        process_lifetime,
        "_process_start_ticks",
        Mock(side_effect=PermissionError("denied")),
    )
    with pytest.raises(PermissionError, match="denied"):
        wait_for_processes_exit((identity,), timeout=0)


def test_missing_snapshot_remains_unconfirmed_after_workers_cleared(monkeypatch):
    group = _bare_group()
    group._workers = [_worker("worker")]
    kill = Mock()
    monkeypatch.setattr(ray, "kill", kill)
    assert not group.shutdown(force=True, wait_for_termination=True)
    assert not group._workers
    assert not group.shutdown(force=True, wait_for_termination=True)
    kill.assert_called_once()


def test_replaced_actor_cannot_use_old_exited_snapshot(monkeypatch):
    group = _bare_group()
    group._workers = [_worker("replacement")]
    group._termination_snapshot = WorkerProcessSnapshot(
        ("old",), (_identity(os.getpid()),)
    )
    monkeypatch.setattr(ray, "kill", Mock())
    assert not group.shutdown(force=True, wait_for_termination=True)
    assert not group.shutdown(force=True, wait_for_termination=True)


def test_unconfirmed_shutdown_retries_process_observer(monkeypatch):
    group = _bare_group()
    group._workers = [_worker("worker")]
    snapshot = WorkerProcessSnapshot(("worker",), (_identity(os.getpid()),))
    group._termination_snapshot = snapshot
    monkeypatch.setattr(ray, "kill", Mock())
    monkeypatch.setattr(ray, "cancel", Mock())
    get = Mock(side_effect=[[False], [True]])
    monkeypatch.setattr(ray, "get", get)
    observer = Mock()
    monkeypatch.setattr(
        "nemo_rl.distributed.worker_groups.observe_worker_processes_exit", observer
    )
    assert not group.shutdown(force=True, wait_for_termination=True)
    assert not group._workers
    assert group._termination_snapshot is snapshot
    assert group.shutdown(force=True, wait_for_termination=True)
    assert group._termination_snapshot is None
    assert group.shutdown(force=True, wait_for_termination=True)
    assert get.call_count == 2
    for invocation in observer.options.call_args_list:
        strategy = invocation.kwargs["scheduling_strategy"]
        assert strategy.node_id == "1" * 56
        assert strategy.soft is False


def test_observer_ray_timeout_preserves_snapshot_for_retry(monkeypatch):
    group = _bare_group()
    group._termination_unconfirmed = True
    snapshot = WorkerProcessSnapshot(("worker",), (_identity(os.getpid()),))
    group._termination_snapshot = snapshot
    monkeypatch.setattr(
        ray, "get", Mock(side_effect=ray.exceptions.GetTimeoutError("pending"))
    )
    monkeypatch.setattr(ray, "cancel", Mock())
    monkeypatch.setattr(
        "nemo_rl.distributed.worker_groups.observe_worker_processes_exit", Mock()
    )
    assert not group.shutdown(force=True, wait_for_termination=True, timeout=0.1)
    assert group._termination_snapshot is snapshot


def test_default_shutdown_still_returns_without_observer(monkeypatch):
    group = _bare_group()
    group._workers = [_worker("worker")]
    monkeypatch.setattr(ray, "kill", Mock())
    observer = Mock(side_effect=AssertionError("default shutdown must not wait"))
    monkeypatch.setattr(group, "_wait_for_worker_processes_exit", observer)
    assert group.shutdown(force=True)
    observer.assert_not_called()


def test_incomplete_capture_raises_before_publishing_snapshot(monkeypatch):
    group = _bare_group()
    worker = _worker("worker")
    worker.__ray_call__ = SimpleNamespace(remote=Mock())
    group._workers = [worker]
    monkeypatch.setattr(ray, "get", Mock(return_value=[]))
    with pytest.raises(RuntimeError, match="Incomplete"):
        group.record_worker_processes()
    assert group._termination_snapshot is None


def test_nested_capture_timeout_keeps_matching_process_snapshot(monkeypatch):
    group = _bare_group()
    worker = _worker("worker")
    worker.__ray_call__ = SimpleNamespace(remote=Mock())
    group._workers = [worker]
    snapshot = WorkerProcessSnapshot(("worker",), (_identity(os.getpid()),))
    group._termination_snapshot = snapshot
    monkeypatch.setattr(
        ray, "get", Mock(side_effect=ray.exceptions.GetTimeoutError("busy"))
    )
    with pytest.raises(ray.exceptions.GetTimeoutError):
        group.record_worker_processes(timeout=0)
    assert group._termination_snapshot is snapshot


def test_default_shutdown_preserves_groups_built_without_constructor(monkeypatch):
    group = RayWorkerGroup.__new__(RayWorkerGroup)
    group._workers = [_worker("worker")]
    group._worker_metadata = []
    monkeypatch.setattr(ray, "kill", Mock())
    assert group.shutdown(force=True)


@ray.remote(num_cpus=0)
class _Consumer:
    def alive(self):
        return os.getpid()


def test_ray_actor_shutdown_confirms_os_exit(monkeypatch):
    """Exercise real __ray_call__, node-affinity observers, kill and OS reaping."""
    assert not ray.is_initialized(), "test owns its isolated Ray runtime"
    # This local test shares the installed environment; do not package the
    # entire repository through Ray's automatic uv-run propagation hook.
    monkeypatch.setattr(
        ray._private.ray_constants, "RAY_ENABLE_UV_RUN_RUNTIME_ENV", False
    )
    runtime_dir = tempfile.TemporaryDirectory(prefix="nrl-ray-")
    ray.init(num_cpus=2, include_dashboard=False, _temp_dir=runtime_dir.name)
    group = _bare_group()
    try:
        group._workers = [_Consumer.remote(), _Consumer.remote()]
        ray.get([worker.alive.remote() for worker in group._workers])
        group.record_worker_processes(timeout=30)
        snapshot = group._termination_snapshot
        assert snapshot is not None and len(snapshot.processes) == 2
        assert all(
            _process_start_ticks(identity.pid) == identity.start_ticks
            for identity in snapshot.processes
        )
        assert group.shutdown(force=True, wait_for_termination=True, timeout=30)
        assert all(
            _process_start_ticks(identity.pid) != identity.start_ticks
            for identity in snapshot.processes
        )
        assert group.shutdown(force=True, wait_for_termination=True, timeout=1)
    finally:
        group.shutdown(force=True)
        ray.shutdown()
        runtime_dir.cleanup()
