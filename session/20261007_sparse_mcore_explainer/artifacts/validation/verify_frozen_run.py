"""Independently verify a completed run's frozen source/config/launcher hashes."""

import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
run = args.run.resolve()
repo = Path(__file__).resolve().parents[4]
runtime = json.loads((run / "runtime.json").read_text())
if (run / "source-manifest.json").is_file():
    # Standalone replays inherit the controller's runtime description, but
    # freeze their own launch files in this separate authoritative manifest.
    manifest = json.loads((run / "source-manifest.json").read_text())
    sources, owned = manifest["file_sha256"], manifest["owned_file_sha256"]
else:
    sources = runtime["source_sha256_at_submission"]
    owned = dict(runtime["launcher_sha256_at_submission"])
    owned["resolved_config.yaml"] = runtime["resolved_config_sha256_at_submission"]
mismatches = []
for base, records in ((repo, sources), (run, owned)):
    for name, expected in records.items():
        path = base / name
        if (
            not path.exists()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            mismatches.append(str(path))
result = {
    "status": "FAIL" if mismatches else "PASS",
    "scope": "Independent post-run source/config/launcher verification against frozen pre-submission hashes",
    "source_files": len(sources),
    "owned_files": len(owned),
    "mismatches": mismatches,
}
(run / "manual-final-source-verification.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result))
assert not mismatches, "Frozen run input changed; do not mark run PASS"
