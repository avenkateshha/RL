"""Map each original small matrix case to immutable attempts and accepted evidence."""

import json
from pathlib import Path

validation = Path(__file__).resolve().parent
path = validation / "runs/recipe-index.json"
index = json.loads(path.read_text())
for item in index:
    attempts = []
    for directory in sorted((validation / "runs").iterdir()):
        runtime_path = directory / "runtime.json"
        if (
            not directory.is_dir()
            or not runtime_path.exists()
            or not (directory / "launcher/ray-runtime.sub").exists()
        ):
            continue
        runtime = json.loads(runtime_path.read_text())
        if (
            directory.name != item["case"]
            and runtime.get("original_case") != item["case"]
        ):
            continue
        result_path = directory / "results.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        submission_path = directory / "submission.json"
        submission = (
            json.loads(submission_path.read_text()) if submission_path.exists() else {}
        )
        status = result.get("status", "NOT_RUN")
        if status == "NOT_RUN" and submission:
            status = "SUBMITTED_RESULTS_PENDING"
        attempts.append(
            {
                "run": directory.name,
                "status": status,
                "job_id": result.get("job_id")
                or submission.get("stdout", "").strip().split(";", 1)[0]
                or None,
                "submitted_utc": submission.get("utc", ""),
                "result": str(result_path.relative_to(validation)),
                "config_sha256": runtime["config_sha256"],
            }
        )
    attempts.sort(key=lambda attempt: attempt["submitted_utc"])
    accepted = [attempt for attempt in attempts if attempt["status"] == "PASS"]
    item["status"] = "PASS" if accepted else attempts[-1]["status"]
    item["accepted_run"] = accepted[-1]["run"] if accepted else None
    item["attempts"] = attempts
path.write_text(json.dumps(index, indent=2) + "\n")
print(
    json.dumps(
        {
            "cases": len(index),
            "accepted": sum(item["status"] == "PASS" for item in index),
            "pending": [item["case"] for item in index if item["status"] != "PASS"],
        },
        indent=2,
    )
)
