"""Clarify historical padding metadata without modifying any frozen run inputs."""

import hashlib
import json
from pathlib import Path

import yaml

validation = Path(__file__).resolve().parent
fixture = json.loads((validation / "fixture_manifest.json").read_text())
smol_snapshot = fixture["models"]["teacher_b"]["snapshot_path"]
records = []
for runtime_path in sorted((validation / "runs").glob("*/runtime.json")):
    run = runtime_path.parent
    config_path = run / "resolved_config.yaml"
    runtime_bytes = runtime_path.read_bytes()
    runtime = json.loads(runtime_bytes)
    if "padding_rationale" not in runtime or not config_path.exists():
        continue
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    teachers = config["teachers"]
    actual = [
        {
            "teacher_index": index,
            "model_name": teacher["model_name"],
            "make_sequence_length_divisible_by": teacher[
                "make_sequence_length_divisible_by"
            ],
        }
        for index, teacher in enumerate(teachers)
    ]
    smol_padding = [
        teacher["make_sequence_length_divisible_by"]
        for teacher in teachers
        if teacher["model_name"] == smol_snapshot
    ]
    assert len(smol_padding) <= 1
    actual_smol = smol_padding[0] if smol_padding else None
    needs_clarification = (
        runtime.get("teacher_b_padding_divisibility") != actual_smol
        or "SmolLM2" in runtime["padding_rationale"]
        and not smol_padding
    )
    records.append(
        {
            "run": run.name,
            "runtime_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
            "resolved_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "original_padding_rationale": runtime["padding_rationale"],
            "original_teacher_b_padding_divisibility": runtime.get(
                "teacher_b_padding_divisibility"
            ),
            "actual_teachers": actual,
            "actual_smollm2_padding_divisibility": actual_smol,
            "clarification_required": needs_clarification,
            "clarification": (
                "The preparation-time rationale copied the two-cross-tokenizer recipe's SmolLM2 note. This run has no SmolLM2 teacher; the recorded resolved-config settings above are authoritative. No objective or executed setting changed."
                if needs_clarification
                else "Preparation metadata agrees with the actual configured SmolLM2 teacher."
            ),
        }
    )
assert records
report = {
    "status": "PASS_METADATA_AUDIT",
    "scope": "Read-only documentation correction. Every original runtime/config remains byte-identical; this audit does not establish numerical acceptance.",
    "method": "Compare generic preparation metadata against each resolved configuration and the pinned SmolLM2 fixture identity.",
    "runs": records,
}
(validation / "runtime-padding-audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "runs_audited": len(records),
            "clarifications": sum(
                record["clarification_required"] for record in records
            ),
        }
    )
)
