"""Reject numerical-only success when legacy counter or exit evidence is incomplete."""

import copy
import itertools
import json
import tempfile
from pathlib import Path

from summarize_matrix_run import legacy_lifetime_evidence


def main() -> None:
    """Exercise missing coverage, real counter anomalies and unverified exits."""
    valid_counters = {
        "coverage_status": "PASS",
        "expected_worker_count": 8,
        "observed_worker_count": 8,
        "counter_lifetime_status": "NO_COUNTER_ANOMALY_OBSERVED",
    }
    valid_exits = {
        "status": "PASS",
        "policies": [
            {
                "policy": name,
                "workers": 8,
                "os_identity_captured": True,
                "actual_backend_cleanup": True,
                "os_process_exit_confirmed": True,
                "errors": [],
            }
            for name in ("student", "teacher_0", "teacher_1")
        ],
    }
    checked = 0
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        counter_file = run / "legacy-ipc-observations-summary.json"
        exit_file = run / "controller-process-exit.json"
        for counter_case, exit_case in itertools.product(
            (
                "missing",
                "good",
                "coverage",
                "worker_count",
                "negative",
                "positive",
                "unproven",
            ),
            ("missing", "good", "identity", "cleanup", "exit"),
        ):
            counter_file.unlink(missing_ok=True)
            exit_file.unlink(missing_ok=True)
            counters = copy.deepcopy(valid_counters)
            exits = copy.deepcopy(valid_exits)
            if counter_case == "coverage":
                counters["coverage_status"] = "FAIL"
            elif counter_case == "worker_count":
                counters["observed_worker_count"] = 7
            elif counter_case == "negative":
                counters["counter_lifetime_status"] = "NEGATIVE_COUNTERS_CONFIRMED"
            elif counter_case == "positive":
                counters["counter_lifetime_status"] = "OUTSTANDING_COUNTERS_OBSERVED"
            elif counter_case == "unproven":
                counters["counter_lifetime_status"] = "UNPROVEN"
            if exit_case in ("identity", "cleanup", "exit"):
                field = {
                    "identity": "os_identity_captured",
                    "cleanup": "actual_backend_cleanup",
                    "exit": "os_process_exit_confirmed",
                }[exit_case]
                exits["policies"][1][field] = False
            if counter_case != "missing":
                counter_file.write_text(json.dumps(counters))
            if exit_case != "missing":
                exit_file.write_text(json.dumps(exits))
            status = legacy_lifetime_evidence(run)["status"]
            expected = (
                "PASS_NUMERICAL_LEGACY_IPC_LIFETIME_FAILURE"
                if counter_case in ("negative", "positive")
                else "PASS"
                if counter_case == exit_case == "good"
                else "PASS_NUMERICAL_PENDING_LEGACY_LIFETIME_AUDIT"
            )
            assert status == expected, (counter_case, exit_case, status)
            checked += 1
    print(json.dumps({"status": "PASS", "acceptance_cases": checked}))


if __name__ == "__main__":
    main()
