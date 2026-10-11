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

"""Build a reviewable small-artifact allowlist; never stages or publishes files."""

import hashlib
import json
import re
from pathlib import Path

from package_frozen_python import load_package

validation = Path(__file__).resolve().parent
repo = validation.parents[3]
package_paths = set()
if (validation / "frozen-python/manifest.json").exists():
    package = load_package(validation / "frozen-python")
    package_paths = {"frozen-python/manifest.json"} | {
        "frozen-python/" + item.blob for item in package.files
    }
# Raw logs stay local. Their locations, sizes and hashes establish provenance
# without committing third-party package warnings or bulky captured tensors.
safe_names = {
    "results.json",
    "accounting.txt",
    "command.txt",
    "command.json",
    "manifest.json",
    "runtime.json",
    "source-manifest.json",
    "source-provenance.json",
    "submission.json",
    "fixture-manifest.json",
    "recipe-index.json",
    "recipe-schema-validation.json",
    "collated_corpus.json",
    "resolved_config.yaml",
    "runtime_resolved_config.yaml",
    "controller-completion.json",
    "controller-process-exit.json",
    "ray-log-snapshot.json",
    "legacy-ipc-observations.json",
    "legacy-ipc-observations-summary.json",
    "evidence-manifest.json",
    "source-verification.json",
    "source.patch",
    "preflight-results.json",
    "preflight-junit.xml",
    "package-final-results.json",
    "source-before.json",
    "source-after.json",
    "scope.json",
    "junit.xml",
    "controller-metrics.jsonl",
    "test-plan.json",
    "environment.json",
    "provenance.json",
    "scheduler-test.json",
    "manual-final-source-verification.json",
}
secret_patterns = [
    re.compile(rb"github_pat_[A-Za-z0-9_]+"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(rb"https?://[^\s/]+:[^\s/@]+@"),
]
selected, frozen_python, logs, rejected = [], [], [], []
for path in sorted(validation.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(validation)
    is_packaged = relative.as_posix() in package_paths
    if any(
        part
        in (
            "runtime",
            "__pycache__",
            "teacher-observations",
        )
        for part in relative.parts
    ):
        continue
    if path.name in (
        "publication-allowlist.json",
        "raw-log-inventory.json",
        "native_mcore_docs_draft.md",
    ):
        continue
    is_log = path.suffix in (".log", ".out", ".err")
    archive_tree = any(
        part in ("archived-raw-logs", "ray-final-logs") for part in relative.parts
    )
    is_archive = archive_tree and path.name.endswith(".tar.gz")
    is_archive_manifest = archive_tree and path.suffix == ".json"
    if not (is_log or is_archive or is_archive_manifest) and any(
        part in ("training-logs", "driver-logs", "ray-final-logs")
        for part in relative.parts
    ):
        continue
    is_source = path.suffix in (".py", ".sh", ".sub") and (
        len(relative.parts) == 1
        or "launcher" in relative.parts
        or "fixtures" in relative.parts
    )
    is_result = (
        is_packaged
        or path.name in safe_names
        or is_archive_manifest
        or path.name == "examples.jsonl"
        and "fixtures" in relative.parts
        or path.suffix == ".jsonl"
        and "legacy-ipc-counters" in relative.parts
        or path.name.startswith(("reference-rank", "replay-rank"))
        and path.suffix == ".json"
        or path.suffix == ".json"
        and any(
            part in ("controller-reference", "optimizer-first-step")
            for part in relative.parts
        )
    )
    is_top_level = len(relative.parts) == 1 and path.suffix in (".md", ".json")
    if not (is_log or is_archive or is_source or is_result or is_top_level):
        continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    record = {
        "path": str(path.relative_to(repo)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
    if is_log or is_archive:
        record["local_path"] = str(path)
        if is_archive:
            candidates = [
                path.with_name(path.name[:-7] + suffix)
                for suffix in (".members.json", ".json")
            ]
            manifest_path = next(
                candidate for candidate in candidates if candidate.exists()
            )
            manifest = json.loads(manifest_path.read_text())
            assert manifest["archive_sha256"] == record["sha256"], path
            record.update(
                kind="verified_raw_log_archive",
                member_manifest=str(manifest_path.relative_to(repo)),
                members=len(manifest["members"]),
            )
        logs.append(record)
        continue
    if record["bytes"] > 1_000_000:
        rejected.append(
            {
                **record,
                "reason": "Over1MB: retain locally and select a concise excerpt if required",
            }
        )
        continue
    content = path.read_bytes()
    if any(pattern.search(content) for pattern in secret_patterns):
        rejected.append(
            {
                **record,
                "reason": "Potential credential pattern: manual review required; values never emitted",
            }
        )
        continue
    if relative.parts[0] == "runs" and path.suffix == ".py":
        frozen_python.append(record)
    else:
        selected.append(record)
report = {
    "status": "REVIEW_REQUIRED",
    "scope": "Candidate allowlist only; no files staged or published. Final results may supersede current hashes.",
    "exclusions": [
        "runtime dependency/source snapshots",
        "caches and pyc",
        "tensor/model/table binaries",
        "raw logs (hashed separately)",
        "historical original launchers",
    ],
    "candidate_files": selected,
    "frozen_python_sources": frozen_python,
    "rejected": rejected,
}
(validation / "publication-allowlist.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
(validation / "raw-log-inventory.json").write_text(
    json.dumps({"local_only_logs": logs}, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "candidates": len(selected),
            "candidate_bytes": sum(x["bytes"] for x in selected),
            "frozen_python_sources": len(frozen_python),
            "raw_logs_local_only": len(logs),
            "rejected_files": len(rejected),
        }
    )
)
