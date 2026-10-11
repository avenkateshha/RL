"""Mocked CPU lifecycle audit of the run-owned teardown observer.

This verifies all-policy attempts, failure reporting and method restoration.
Fake worker groups never execute Ray, CUDA, IPC or OS process observations.
"""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

from controller_teardown import ControllerTeardown


def main():
    events = []

    class FakeGroup:
        def __init__(self, name, failure):
            self.name, self.failure = name, failure
            self.workers = [object()]

        def record_worker_processes(self, timeout):
            assert timeout == 30.0
            events.append((self.name, "capture"))
            if self.failure == "capture":
                raise RuntimeError("mock identity failure")

        def shutdown(self, *, wait_for_termination, timeout):
            assert wait_for_termination and timeout == 30.0
            events.append((self.name, "wait"))
            self.workers = []
            if self.failure == "wait":
                raise RuntimeError("mock observer failure")
            return self.failure != "unconfirmed"

    class FakePolicy:
        def __init__(self, config, name_prefix="lm_policy"):
            self.name = name_prefix
            self.failure = config.get("failure")
            self.worker_group = FakeGroup(self.name, self.failure)

        def shutdown(self):
            events.append((self.name, "cleanup"))
            if self.failure == "cleanup":
                raise RuntimeError("mock cleanup failure")
            self.worker_group.workers = []
            return self.failure != "cleanup_false"

    original = FakePolicy.__init__
    module = ModuleType("nemo_rl.models.policy.lm_policy")
    module.Policy = FakePolicy
    with (
        TemporaryDirectory(prefix="xtoken-teardown-") as directory,
        patch.dict(sys.modules, {module.__name__: module}),
    ):
        run = Path(directory)
        for failure in (
            None,
            "capture",
            "cleanup",
            "cleanup_false",
            "wait",
            "unconfirmed",
            "write",
        ):
            events.clear()
            helper = ControllerTeardown(run)
            helper.install()
            FakePolicy({"model_name": "a"}, name_prefix="teacher_0")
            FakePolicy({"model_name": "s", "failure": failure}, name_prefix="student")
            FakePolicy({"model_name": "b"}, name_prefix="teacher_1")
            if failure == "write":
                original_write = helper._write_result
                attempts = []

                def write_once_fails():
                    attempts.append(True)
                    if len(attempts) == 1:
                        raise OSError("mock full filesystem")
                    original_write()

                helper._write_result = write_once_fails
            failed = False
            try:
                helper.verify_and_shutdown()
            except AssertionError:
                failed = True
            assert failed == (failure is not None), failure
            assert FakePolicy.__init__ is original
            assert events == [
                (name, action)
                for name in ("student", "teacher_0", "teacher_1")
                for action in ("capture", "cleanup", "wait")
            ], (failure, events)
            report = json.loads((run / "controller-process-exit.json").read_text())
            assert len(report["policies"]) == 3
            assert report["status"] == ("PASS" if failure is None else "INCOMPLETE")
            student = report["policies"][0]
            assert student["os_identity_captured"] == (failure != "capture")
            assert student["os_process_exit_confirmed"] == (
                failure not in ("capture", "wait", "unconfirmed")
            ), (failure, student)
            # The capture-failure fixture still returns True from shutdown
            # after cleanup empties the group; it must never certify OS exit.
            assert all(
                policy["os_process_exit_confirmed"] for policy in report["policies"][1:]
            )
            if failure == "write":
                assert len(attempts) == 3
                assert "report write: OSError" in report["policies"][0]["errors"][0]
            helper.restore()  # The launcher's finally also restores; idempotent.
            assert FakePolicy.__init__ is original

        empty = ControllerTeardown(run)
        empty.install()
        try:
            empty.verify_and_shutdown()
        except AssertionError as error:
            assert "No actual policies" in str(error)
        else:
            raise AssertionError("Empty policy capture passed")
        assert FakePolicy.__init__ is original
        interrupted = ControllerTeardown(run)
        interrupted.install()
        interrupted.restore()  # Failed controller does not invoke policy teardown.
        assert FakePolicy.__init__ is original
        ControllerTeardown(run).restore()  # No installation is also harmless.
    print(
        "PASS mocked teardown: all3 policies attempted across7 cases, failed identity never certifies exit, write-error continuation, empty/failed-controller restoration; no Ray/OS/CUDA proof"
    )


if __name__ == "__main__":
    main()
