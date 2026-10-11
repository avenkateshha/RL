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

"""Write a compact summary only after every agreed case has full acceptance."""

import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def normalize_early_r1(run: Path, result: dict, runtime: dict) -> dict:
    """Read the first run's older schema without rewriting its evidence."""
    assert run.name == "R1-tp1cp1"
    assert (runtime["tp"], runtime["cp"], runtime["dp"]) == (1, 1, 1)
    assert result["status"] == "PASS"
    assert result["source_and_owned_hashes_before_after"] == "PASS"
    source_path = run / "manual-final-source-verification.json"
    source = read(source_path)
    assert source["status"] == "PASS" and not source["mismatches"]
    assert source["source_files"] > 0 and source["owned_files"] > 0

    reports = [read(path) for path in (run / "controller-reference").glob("*.json")]
    assert len(reports) == result["numerical_microbatch_reports"] == 14
    assert all(report["status"] == "PASS" for report in reports)
    required_layout_keys = {
        "cp_load_balanced_to_contiguous",
        "rebuild_teacher_sparse_logits_from_ipc",
        "rebuild_teacher_full_logits_from_ipc",
        "allgather_cp_contiguous_tensor",
    }
    for report in reports:
        assert report["native_sparse_teacher_indices"] == [0]
        assert not report["legacy_sparse_teacher_indices"]
        assert set(report["production_layout_calls"]) == required_layout_keys
        assert not any(report["production_layout_calls"].values())
    training = [report for report in reports if report["dlogits"] is not None]
    assert len(training) == result["training_dlogits_reports"] == 6
    assert (
        max(report["dlogits"]["relative_l2"] for report in training)
        == result["max_dlogits_relative_l2"]
    )

    replay_run = Path(result["optimizer_preclip_replay"])
    replay_result = read(replay_run / "results.json")
    assert replay_result["status"] == "PASS_ACTUAL_OPTIMIZER_PRECLIP"
    assert replay_result["slurm_state"] == "COMPLETED"
    assert replay_result["exit_code"] == "0:0"
    assert replay_result["source_and_owned_hashes_before_after"] == "PASS"
    manifest_path = replay_run / "source-manifest.json"
    manifest = read(manifest_path)
    assert manifest["file_sha256"] and manifest["owned_file_sha256"]
    # Historical production source has since changed. Its frozen launcher
    # checked this manifest before and after the run; do not compare it to HEAD.
    log_path = replay_run / f"slurm-{replay_result['job_id']}.out"
    log = log_path.read_text()
    assert log.count("Focused source hashes verified.") == 2
    assert log.count("Owned reference launcher/config/fixture hashes verified.") == 2
    report_path = replay_run / "replay-rank0.json"
    replay = read(report_path)
    assert replay["status"] == "PASS"
    assert (replay["rank"], replay["tp"], replay["cp"], replay["dp"]) == (0, 1, 1, 1)
    assert replay["expected_microbatches"] == len(replay["microbatches"]) == 2
    assert replay["range_coverage"] == replay_result["range_coverage"]
    assert replay["range_coverage"].startswith("PASS:")
    index_path = Path(replay["capture_index"])
    assert digest(index_path) == replay["capture_index_sha256"]
    index = read(index_path)
    assert len(index["microbatches"]) == 2
    for actual, captured in zip(replay["microbatches"], index["microbatches"]):
        assert actual["ordinal"] == captured["ordinal"]
        assert actual["item_ids"] == captured["item_ids"]
        assert actual["pinned_initial_student_forward"]["relative_l2"] == 0
    assert len(index["optimizers"]) == 1
    metadata_path = Path(index["optimizers"][0]["metadata_path"])
    metadata = read(metadata_path)
    inventory = {item["name"]: item for item in metadata["inventory"]}
    assert metadata["all_finite"] and not metadata["found_inf"]
    assert metadata["dp_cp_group_ranks"] == metadata["optimizer_group_ranks"] == [0]
    assert len(inventory) == len(metadata["shards"]) == 98
    assert {item["name"] for item in metadata["shards"]} == set(inventory)
    for shard in metadata["shards"]:
        assert shard["start"] == 0
        assert shard["end"] == inventory[shard["name"]]["full_local_numel"]
    assert (
        sum(item["full_local_numel"] for item in inventory.values())
        == replay["covered_parameter_elements_per_tp_rank"]
        == replay_result["parameter_elements"]
    )
    assert sorted(
        (item["name"], item["start"], item["end"]) for item in replay["owned_shards"]
    ) == sorted(
        (item["name"], item["start"], item["end"]) for item in metadata["shards"]
    )
    gradients = replay["actual_optimizer_preclip_gradients"]
    assert gradients == result["actual_optimizer_preclip_gradients"]
    assert gradients == replay_result["actual_optimizer_preclip_gradients"]
    replay_source = {
        "status": "PASS",
        "scope": "Frozen manifest verified by the historical launcher before and after replay",
        "source_files": len(manifest["file_sha256"]),
        "owned_files": len(manifest["owned_file_sha256"]),
        "manifest_sha256": digest(manifest_path),
        "verification_log_sha256": digest(log_path),
    }
    return {
        **result,
        "optimizer_replay": {
            "status": "PASS",
            "run": str(replay_run),
            "ranks": 1,
            "microbatches_per_rank": 2,
            "range_coverage": replay["range_coverage"],
            "max_relative_l2": gradients["relative_l2"],
            "max_relative_norm_error": gradients["relative_norm"],
        },
        "evaluation_loss_metric_reports": len(reports) - len(training),
        "native_only_no_student_relayout_or_full_reconstruction_reports": len(reports),
        "independent_source_verification": {
            str(run): source,
            str(replay_run): replay_source,
        },
        "schema_normalization": {
            "scope": "Derived from original controller reports, companion replay, owned-range metadata, and historical source verification; original artifacts unchanged",
            "evidence_sha256": {
                str(path): digest(path)
                for path in (
                    source_path,
                    manifest_path,
                    log_path,
                    report_path,
                    index_path,
                    metadata_path,
                )
            },
        },
    }


validation = Path(__file__).resolve().parent
index_path = validation / "runs/recipe-index.json"
index = json.loads(index_path.read_text())
assert len(index) == len({entry["case"] for entry in index}) == 16
cases = []
for entry in index:
    assert entry["status"] == "PASS", entry["case"]
    run = validation / "runs" / entry["accepted_run"]
    result = read(run / "results.json")
    runtime = read(run / "runtime.json")
    if entry["case"] == "R1-tp1cp1":
        result = normalize_early_r1(run, result, runtime)
    replay = result["optimizer_replay"]
    assert result["status"] == replay["status"] == "PASS"
    assert result["optimizer_steps"] == [1, 2, 3]
    assert result["evaluation_steps"] == [3]
    assert replay["microbatches_per_rank"] == 2
    assert replay["ranks"] == runtime["tp"] * runtime["cp"] * runtime["dp"]
    assert replay["range_coverage"].startswith("PASS:")
    for field in ("max_relative_l2", "max_relative_norm_error"):
        assert math.isfinite(replay[field]) and replay[field] <= 0.02
    assert result["independent_source_verification"]
    assert all(
        value["status"] == "PASS"
        for value in result["independent_source_verification"].values()
    )
    legacy_reports = result.get("legacy_cross_tokenizer_consumer_reports", 0)
    if legacy_reports:
        assert result["legacy_ipc_lifetime"]["status"] == "PASS"
    else:
        assert (
            result["native_only_no_student_relayout_or_full_reconstruction_reports"]
            == result["numerical_microbatch_reports"]
        )
    replay_run = Path(replay["run"])
    replay_result = read(replay_run / "results.json")
    cases.append(
        {
            "case": entry["case"],
            "status": "PASS",
            "accepted_run": entry["accepted_run"],
            "controller_job_id": result["job_id"],
            "optimizer_replay_job_id": replay_result["job_id"],
            "nodes": runtime["nodes"],
            "tp": runtime["tp"],
            "cp": runtime["cp"],
            "dp": runtime["dp"],
            "global_batch_size": runtime["global_batch_size"],
            "student_microbatches_per_dp_step": 2,
            "max_sequence_length": runtime["max_sequence_length"],
            "teacher_topk_ipc_k": runtime["sparse_k"],
            "dynamic_loss_scaling": runtime["dynamic_loss_scaling"],
            "numerical_reports": result["numerical_microbatch_reports"],
            "training_reports": result["training_dlogits_reports"],
            "evaluation_reports": result["evaluation_loss_metric_reports"],
            "max_dlogits_relative_l2": result["max_dlogits_relative_l2"],
            "optimizer_relative_l2": replay["max_relative_l2"],
            "optimizer_relative_norm_error": replay["max_relative_norm_error"],
            "optimizer_replay_ranks": replay["ranks"],
            "legacy_consumer_reports": legacy_reports,
            "config_sha256": digest(run / "resolved_config.yaml"),
            "results": str((run / "results.json").relative_to(validation)),
            "results_sha256": digest(run / "results.json"),
            **(
                {"schema_normalization": result["schema_normalization"]}
                if "schema_normalization" in result
                else {}
            ),
        }
    )
summary = {
    "status": "PASS",
    "accepted_variants": len(cases),
    "numerical_reports": sum(case["numerical_reports"] for case in cases),
    "training_reports": sum(case["training_reports"] for case in cases),
    "evaluation_reports": sum(case["evaluation_reports"] for case in cases),
    "initial_replay_forwards": 2
    * sum(case["optimizer_replay_ranks"] for case in cases),
    "max_optimizer_relative_l2": max(case["optimizer_relative_l2"] for case in cases),
    "max_optimizer_relative_norm_error": max(
        case["optimizer_relative_norm_error"] for case in cases
    ),
    "optimizer_error_limit": 0.02,
    "source_index": "runs/recipe-index.json",
    "source_index_sha256": digest(index_path),
    "runtime": {
        "python": "3.13.13",
        "torch": "2.11.0+cu130",
        "transformer_engine": "2.15",
        "megatron_bridge_commit_prefix": "1f8873bb",
        "megatron_core_commit_prefix": "6a366090",
        "repository_requested_python": "3.13.14",
        "repository_requested_torch": "2.13",
        "provenance": "pinned-runtime-source-provenance.json",
    },
    "scope": "Each variant completed three optimizer updates and final evaluation, plus independent first-step accumulated pre-clip gradient replay with complete parameter ownership coverage. Native-only layout guards and required legacy lifetime proofs passed. Failed attempts remain recorded in the source index.",
    "larger_memory_tests": "EXCLUDED",
    "production_size_memory_fit": "NOT_TESTED",
    "project_type_check": "Scoped changed allowlisted module passes; 11 existing dependency import errors remain in the repository-wide check. See runs/checkpoint5-types-20261010/results.json.",
    "cases": cases,
}
(validation / "acceptance-summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(
    json.dumps(
        {key: value for key, value in summary.items() if key != "cases"}, indent=2
    )
)
