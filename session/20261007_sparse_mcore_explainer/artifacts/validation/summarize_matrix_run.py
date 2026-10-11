"""Aggregate completed small correctness jobs; never infer IPC lifetime from silence."""

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tarfile
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def update_runtime_status(run, runtime, status):
    manifest = run / "source-manifest.json"
    if manifest.exists() and "runtime.json" in read(manifest)["owned_file_sha256"]:
        # Standalone replay launchers freeze this metadata as an input. Its
        # preparation-time status stays intact; results.json is authoritative.
        return
    runtime.update(status=status, result_file="results.json")
    runtime.pop("reason", None)
    write(run / "runtime.json", runtime)


def verify_sources(run):
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("verify_frozen_run.py")),
            str(run),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return read(run / "manual-final-source-verification.json")


def accounting(run, job_id):
    result = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "--noheader",
            "--parsable2",
            "-X",
            "--format=JobIDRaw,State,ExitCode,Elapsed,AllocTRES",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (run / "accounting.txt").write_text(result.stdout)
    rows = [line.split("|") for line in result.stdout.splitlines()]
    row = next(row for row in rows if row[0] == job_id)
    assert row[1:3] == ["COMPLETED", "0:0"], row
    return dict(
        zip(
            ["job_id", "slurm_state", "exit_code", "elapsed", "allocated_resources"],
            row,
        )
    )


def controller_evidence(run, runtime):
    completion = read(run / "controller-completion.json")
    assert completion["status"] == "PASS"
    assert completion["optimizer_steps_recorded"] == [1, 2, 3]
    assert completion["evaluation_steps_recorded"] == [3]
    metrics = [
        json.loads(line)
        for line in (run / "controller-metrics.jsonl").read_text().splitlines()
    ]
    train = sorted(
        [r for r in metrics if r["prefix"] == "train"], key=lambda r: r["step"]
    )
    evaluation = [r for r in metrics if r["prefix"] == "validation"]
    assert [r["step"] for r in train] == [1, 2, 3]
    assert [r["step"] for r in evaluation] == [3]
    for row in train + evaluation:
        for key in ["loss", "ce_loss", "kl_loss"]:
            assert math.isfinite(row["metrics"][key]), (row["step"], key)
        if row["prefix"] == "train":
            assert (
                math.isfinite(row["metrics"]["grad_norm"])
                and row["metrics"]["grad_norm"] > 0
            )
    reports = [read(path) for path in (run / "controller-reference").glob("*.json")]
    assert reports and all(r["status"] == "PASS" for r in reports)
    training = [r for r in reports if r.get("dlogits") is not None]
    world = runtime["tp"] * runtime["cp"] * runtime["dp"]
    assert len(training) == world * runtime["student_microbatches_per_dp_step"] * 3
    assert {r["rank"] for r in training} == set(range(world))
    evaluated = [r for r in reports if r.get("dlogits") is None]
    corpus = read(run / "fixture-manifest.json")["plaintext_corpus"]["records"]
    evaluation_microbatches = corpus // (
        runtime["dp"] * runtime["student_micro_batch_size"]
    )
    assert corpus % (runtime["dp"] * runtime["student_micro_batch_size"]) == 0
    assert len(evaluated) == evaluation_microbatches * world
    assert all(
        sum(r["rank"] == rank for r in evaluated) == evaluation_microbatches
        for rank in range(world)
    )
    native = [
        r
        for r in reports
        if not r["legacy_sparse_teacher_indices"]
        and not r.get("legacy_dense_teacher_indices", [])
    ]
    assert all(
        all(v == 0 for v in r["production_layout_calls"].values()) for r in native
    )
    return {
        "optimizer_steps": completion["optimizer_steps_recorded"],
        "evaluation_steps": completion["evaluation_steps_recorded"],
        "numerical_microbatch_reports": len(reports),
        "training_dlogits_reports": len(training),
        "evaluation_loss_metric_reports": len(evaluated),
        "max_dlogits_relative_l2": max(r["dlogits"]["relative_l2"] for r in training),
        "actual_preclip_grad_norms": [r["metrics"]["grad_norm"] for r in train],
        "actual_step_loss": [r["metrics"]["loss"] for r in train],
        "native_only_no_student_relayout_or_full_reconstruction_reports": len(native),
        "legacy_cross_tokenizer_consumer_reports": len(reports) - len(native),
    }


def legacy_lifetime_evidence(run: Path) -> dict[str, object]:
    """Keep numerical success separate from observed legacy IPC acceptance."""
    counter_path = run / "legacy-ipc-observations-summary.json"
    exit_path = run / "controller-process-exit.json"
    counters = read(counter_path) if counter_path.exists() else None
    exits = read(exit_path) if exit_path.exists() else None
    status = "PASS_NUMERICAL_PENDING_LEGACY_LIFETIME_AUDIT"
    reasons = []
    if counters is None:
        reasons.append("Counter coverage summary is missing")
    else:
        if (
            counters["coverage_status"] != "PASS"
            or counters["expected_worker_count"] <= 0
            or counters["observed_worker_count"] != counters["expected_worker_count"]
        ):
            reasons.append("Counter observation coverage is incomplete")
        lifetime = counters["counter_lifetime_status"]
        if lifetime in (
            "NEGATIVE_COUNTERS_CONFIRMED",
            "OUTSTANDING_COUNTERS_OBSERVED",
        ):
            status = "PASS_NUMERICAL_LEGACY_IPC_LIFETIME_FAILURE"
            reasons.append(lifetime)
        elif lifetime != "NO_COUNTER_ANOMALY_OBSERVED":
            reasons.append("Counter lifetime remains unproven")
    if exits is None:
        reasons.append("Actual worker process-exit evidence is missing")
    else:
        policies = exits.get("policies", [])
        if (
            exits["status"] != "PASS"
            or not policies
            or policies[0]["policy"] != "student"
            or any(
                policy["workers"] <= 0
                or not policy["os_identity_captured"]
                or not policy["actual_backend_cleanup"]
                or not policy["os_process_exit_confirmed"]
                or policy["errors"]
                for policy in policies
            )
        ):
            reasons.append("Actual worker identity, cleanup or OS exit is incomplete")
    if not reasons:
        status = "PASS"
    return {
        "status": status,
        "reasons": reasons,
        "counter_observations": counters,
        "worker_process_exit": exits,
        "scope": "Observed counters and actual process exits; warning-free logs alone never establish acceptance.",
    }


def replay_evidence(run, runtime):
    reports = [read(path) for path in run.glob("replay-rank*.json")]
    world = runtime["tp"] * runtime["cp"] * runtime["dp"]
    assert len(reports) == world and {r["rank"] for r in reports} == set(range(world))
    threshold = runtime["tolerances"]["gpu_bfloat16"]
    for report in reports:
        assert report["status"] == "PASS"
        assert report["range_coverage"].startswith("PASS:")
        assert (
            len(report["microbatches"])
            == report["expected_microbatches"]
            == runtime["student_microbatches_per_dp_step"]
        )
        errors = report["actual_optimizer_preclip_gradients"]
        for key, bound in [
            ("relative_l2", "gradient_relative_l2_max"),
            ("relative_norm", "gradient_norm_relative_error_max"),
        ]:
            assert math.isfinite(errors[key]) and errors[key] <= threshold[bound]
    errors = [r["actual_optimizer_preclip_gradients"] for r in reports]
    return {
        "status": "PASS",
        "run": str(run),
        "ranks": len(reports),
        "microbatches_per_rank": runtime["student_microbatches_per_dp_step"],
        "range_coverage": reports[0]["range_coverage"],
        "covered_parameter_elements_per_tp_rank": sorted(
            {r["covered_parameter_elements_per_tp_rank"] for r in reports}
        ),
        "max_relative_l2": max(e["relative_l2"] for e in errors),
        "max_relative_norm_error": max(e["relative_norm"] for e in errors),
        "actual_norm_including_tp_replicas": errors[0][
            "actual_norm_including_tp_replicas"
        ],
        "reference_norm_including_tp_replicas": errors[0][
            "reference_norm_including_tp_replicas"
        ],
        "scope": reports[0]["scope"],
    }


def warning_scan(runs):
    patterns = {
        "cuda_ipc_types": re.compile(r"CudaIPCTypes", re.I),
        "producer_terminated": re.compile(
            r"producer process has been terminated before all shared CUDA", re.I
        ),
        "counter_underflow": re.compile(
            r"(?:refcount|reference count).*underflow", re.I
        ),
        "leaked_cuda": re.compile(r"leak(?:ed|ing).*CUDA|CUDA.*leak(?:ed|ing)", re.I),
    }
    matches, files, archives = [], [], []

    def scan_text(label, content):
        files.append(label)
        for number, line in enumerate(content.decode(errors="replace").splitlines(), 1):
            for pattern_label, pattern in patterns.items():
                if pattern.search(line):
                    matches.append(
                        {"file": label, "line": number, "pattern": pattern_label}
                    )

    for run in runs:
        for path in run.rglob("*"):
            if path.is_file() and path.suffix in {".log", ".out", ".err"}:
                scan_text(str(path), path.read_bytes())
            elif path.is_file() and path.name.endswith(".tar.gz"):
                stem = path.name.removesuffix(".tar.gz")
                manifest_path = path.with_name(stem + ".members.json")
                if not manifest_path.exists():
                    manifest_path = path.with_name(stem + ".json")
                manifest = read(manifest_path)
                assert Path(manifest["archive_path"]).resolve() == path.resolve()
                assert path.stat().st_size == manifest["archive_bytes"]
                assert (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    == manifest["archive_sha256"]
                ), f"Raw log archive hash mismatch: {path}"
                records = [
                    row
                    for row in manifest["members"]
                    if row.get("kind", "file") == "file"
                ]
                expected = {row["path"]: row for row in records}
                assert len(expected) == len(records), "Duplicate manifest member"
                seen, log_members = set(), 0
                with tarfile.open(path, "r:gz") as archive:
                    for member in archive:
                        if not member.isfile():
                            continue
                        assert member.name not in seen, "Duplicate archive member"
                        seen.add(member.name)
                        record = expected[member.name]
                        stream = archive.extractfile(member)
                        assert stream is not None
                        content = stream.read()
                        assert len(content) == record["bytes"]
                        assert hashlib.sha256(content).hexdigest() == record["sha256"]
                        if Path(member.name).suffix in {".log", ".out", ".err"}:
                            scan_text(f"{path}::{member.name}", content)
                            log_members += 1
                assert seen == set(expected), "Incomplete archive member coverage"
                archives.append(
                    {
                        "path": str(path),
                        "manifest": str(manifest_path),
                        "members_verified": len(seen),
                        "log_members_scanned": log_members,
                    }
                )
    return {
        "files_scanned": len(files),
        "matches": matches,
        "verified_archives": archives,
        "scope": "Only retained raw logs were scanned. Absence of warnings does not prove clean IPC counters or complete worker teardown.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--controller-run", type=Path)
    parser.add_argument("--legacy-lifetime-pending", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    controller = args.controller_run.resolve() if args.controller_run else run
    runtime = read(run / "runtime.json")
    controller_runtime = read(controller / "runtime.json")
    sources = {
        str(path): verify_sources(path) for path in dict.fromkeys([run, controller])
    }
    result = {
        "status": "PASS_NUMERICAL_PENDING_LEGACY_LIFETIME_AUDIT"
        if args.legacy_lifetime_pending
        else "PASS",
        **accounting(run, args.job_id),
        "controller_run": str(controller),
        "tp": runtime["tp"],
        "cp": runtime["cp"],
        "dp": runtime["dp"],
        **controller_evidence(controller, controller_runtime),
        "optimizer_replay": replay_evidence(run, runtime),
        "independent_source_verification": sources,
        "cuda_ipc_warning_scan": warning_scan(list(dict.fromkeys([run, controller]))),
        "larger_memory_tests": "EXCLUDED",
    }
    if result["legacy_cross_tokenizer_consumer_reports"]:
        result["legacy_ipc_lifetime"] = legacy_lifetime_evidence(controller)
        if result["legacy_ipc_lifetime"]["status"] != "PASS":
            result["status"] = result["legacy_ipc_lifetime"]["status"]
    if result["cuda_ipc_warning_scan"]["matches"] and result["status"] == "PASS":
        result["status"] = "PASS_NUMERICAL_REQUIRES_IPC_WARNING_REVIEW"
    write(run / "results.json", result)
    update_runtime_status(run, runtime, result["status"])
    if controller != run:
        previous = read(controller / "results.json")
        previous.update(
            status=result["status"],
            optimizer_replay=result["optimizer_replay"],
            independent_source_verification=sources,
            cuda_ipc_warning_scan=result["cuda_ipc_warning_scan"],
        )
        if "legacy_ipc_lifetime" in result:
            previous["legacy_ipc_lifetime"] = result["legacy_ipc_lifetime"]
        previous["scope"] = (
            "Actual controller three updates plus evaluation and companion first-step preclip optimizer replay; separate lifetime status determines full acceptance."
            if result["status"] != "PASS"
            else "Actual controller three updates plus evaluation and companion first-step preclip optimizer replay."
        )
        write(controller / "results.json", previous)
        update_runtime_status(controller, controller_runtime, result["status"])
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "job_id",
                    "numerical_microbatch_reports",
                    "optimizer_replay",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
