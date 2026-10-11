"""Compare accepted small R2 runs after reversing teacher order, without tensors."""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import yaml

validation = Path(__file__).resolve().parent
runs = [validation / "runs" / name for name in ("R2-tp2cp2", "R2-tp2cp2-reversed")]
configs = [yaml.safe_load((run / "resolved_config.yaml").read_text()) for run in runs]
results = [json.loads((run / "results.json").read_text()) for run in runs]
runtimes = [json.loads((run / "runtime.json").read_text()) for run in runs]
assert all(result["status"] == "PASS" for result in results)
models = [[teacher["model_name"] for teacher in cfg["teachers"]] for cfg in configs]
assert len(models[0]) == 2 and models[0] == models[1][::-1]
canonical = deepcopy(configs)
for cfg in canonical:
    cfg["teachers"].sort(key=lambda teacher: teacher["model_name"])
    cfg["logger"].pop("log_dir")
assert canonical[0] == canonical[1], "Unexpected configuration difference"

student_fields = (
    "input_ids",
    "input_lengths",
    "sample_mask",
    "token_mask",
    "kd_token_mask",
)
batches = []
for run in runs:
    logs = list((run / "driver-logs").rglob("ray-driver.log"))
    assert len(logs) == 1
    records = []
    for line in logs[0].read_text().splitlines():
        marker = "XTOKEN_LOGICAL_BATCH_DIGEST "
        if not line.startswith(marker):
            continue
        record = json.loads(line[len(marker) :])
        records.append(
            {
                "batch_uid": record["batch_uid"],
                "sample_ids": record["sample_ids"],
                "batch_item_ids": record["batch_item_ids"],
                "fields": {key: record["fields"][key] for key in student_fields},
            }
        )
    assert records
    batches.append(records)
assert batches[0] == batches[1], "Student batches differ"

report_maps = []
for run, teacher_models in zip(runs, models, strict=True):
    reports = {}
    for path in (run / "controller-reference").glob("*.json"):
        report = json.loads(path.read_text())
        assert report["status"] == "PASS"
        reports[path.name] = {
            key: report[key]
            for key in (
                "rank",
                "dp_rank",
                "tp_rank",
                "cp_rank",
                "item_ids",
                "token_denominator",
                "kd_denominator",
            )
        }
        assert set(report["chunk_denominators"]) == {"0", "1"}
        reports[path.name]["chunks_by_model"] = {
            model: report["chunk_denominators"][str(index)]
            for index, model in enumerate(teacher_models)
        }
    assert len(reports) == 80
    report_maps.append(reports)
assert report_maps[0] == report_maps[1], "Grouping or normalizers differ"

tolerances = runtimes[0]["tolerances"]["gpu_bfloat16"]
assert tolerances == runtimes[1]["tolerances"]["gpu_bfloat16"]
comparisons = []
for step, loss_a, loss_b, norm_a, norm_b in zip(
    results[0]["optimizer_steps"],
    results[0]["actual_step_loss"],
    results[1]["actual_step_loss"],
    results[0]["actual_preclip_grad_norms"],
    results[1]["actual_preclip_grad_norms"],
    strict=True,
):
    assert math.isclose(
        loss_a,
        loss_b,
        rel_tol=tolerances["loss_rtol"],
        abs_tol=tolerances["loss_atol"],
    )
    norm_error = abs(norm_a - norm_b) / max(abs(norm_a), 1e-30)
    assert norm_error <= tolerances["gradient_norm_relative_error_max"]
    comparisons.append(
        {
            "step": step,
            "combined_losses": [loss_a, loss_b],
            "preclip_norms": [norm_a, norm_b],
            "absolute_loss_difference": abs(loss_a - loss_b),
            "relative_norm_difference": norm_error,
        }
    )
assert len(comparisons) == 3
output = validation / "runs/R2-teacher-order-comparison-20261010"
output.mkdir(exist_ok=True)
result = {
    "status": "PASS",
    "scope": "Read-only teacher-order comparison of two accepted runs; no new GPU work or tensor loading.",
    "teacher_orders": models,
    "settings_by_model": {
        teacher["model_name"]: {
            key: teacher[key]
            for key in ("weight", "logprob_batch_size", "train_micro_batch_size")
        }
        for teacher in configs[0]["teachers"]
    },
    "configuration_equivalence": "Exact after sorting teachers by model and excluding output log directory",
    "identical_student_logical_batches": len(batches[0]),
    "identical_rank_item_and_model_normalizer_reports": len(report_maps[0]),
    "tolerances": tolerances,
    "step_comparisons": comparisons,
    "input_sha256": {
        str(path.relative_to(validation)): hashlib.sha256(path.read_bytes()).hexdigest()
        for run in runs
        for path in (run / "results.json", run / "resolved_config.yaml")
    },
}
(output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
