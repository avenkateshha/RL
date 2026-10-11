"""Build offline pinned-tokenizer tables and exhaustively verify historical Qwen rows."""

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoTokenizer

validation = Path(__file__).resolve().parents[1]
repo = validation.parents[3]
manifest_path = validation / "fixture_manifest.json"
manifest = json.loads(manifest_path.read_text())
builder = repo / "examples/xtoken_mcore_tpcp/build_subtoks_table.py"
spec = importlib.util.spec_from_file_location("table_builder", builder)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
report = {
    "status": "RUNNING",
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.executable,
    "torch": torch.__version__,
    "builder_sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
    "builder_options": {
        "N_max": 8,
        "use_exact_match": True,
        "use_special_role_map": True,
        "canonicalize": True,
    },
    "tables": [],
}
output = validation / "runs/fixture-preparation-20261010"
output.mkdir(parents=True, exist_ok=True)
report_path = output / "results.json"
report_path.write_text(json.dumps(report, indent=2) + "\n")
tokenizers = {}
for role in ("student", "teacher_a", "teacher_b", "teacher_c_same_tokenizer"):
    model = manifest["models"][role]
    snapshot = Path(model["snapshot_path"])
    for filename, expected in model["fingerprints"].items():
        assert (
            hashlib.sha256((snapshot / filename).read_bytes()).hexdigest() == expected
        ), (role, filename)
    tokenizers[role] = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True
    )
    assert len(tokenizers[role]) == model["real_tokenizer_vocab_size"]
report["tokenizer_fingerprints"] = "PASS"
for teacher in ("teacher_b", "teacher_a"):
    for index, direction in enumerate(("fwd", "rev")):
        src, dst = ("student", teacher) if direction == "fwd" else (teacher, "student")
        print(f"Building {teacher} {direction}", flush=True)
        subtoks, lengths = module.build_subtoks_table(
            tokenizers[src],
            tokenizers[dst],
            N_max=8,
            use_exact_match=True,
            use_special_role_map=True,
            canonicalize=True,
            verbose=True,
        )
        original = manifest["tables"][teacher][index]
        record = {
            "teacher": teacher,
            "direction": direction,
            "source": src,
            "destination": dst,
            "source_revision": manifest["models"][src]["revision"],
            "destination_revision": manifest["models"][dst]["revision"],
        }
        if teacher == "teacher_a":
            old = torch.load(original["path"], map_location="cpu", weights_only=False)
            assert (
                hashlib.sha256(Path(original["path"]).read_bytes()).hexdigest()
                == original["sha256"]
            )
            record["historical_exact_subtoks_match"] = torch.equal(
                old["subtoks"], subtoks
            )
            record["historical_exact_lengths_match"] = torch.equal(
                old["lengths"], lengths
            )
            record["historical_rows_different"] = int(
                (
                    (old["subtoks"] != subtoks).any(dim=1) | (old["lengths"] != lengths)
                ).sum()
            )
            if (
                record["historical_exact_subtoks_match"]
                and record["historical_exact_lengths_match"]
            ):
                table_path = Path(original["path"])
            else:
                table_path = (
                    validation
                    / "fixtures/tables"
                    / f"subtoks_llama3p2_1b_qwen3_4b_{direction}.pt"
                )
        else:
            table_path = Path(original["path"])
        if (
            teacher == "teacher_b"
            or not record.get("historical_exact_subtoks_match")
            or not record.get("historical_exact_lengths_match")
        ):
            table_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "subtoks": subtoks,
                    "lengths": lengths,
                    "src": manifest["models"][src]["snapshot_path"],
                    "dst": manifest["models"][dst]["snapshot_path"],
                    "N_max": 8,
                    "V_src": len(tokenizers[src]),
                    "V_dst": len(tokenizers[dst]),
                },
                table_path,
            )
        active = torch.arange(8)[None, :] < lengths[:, None]
        assert bool((subtoks[active] >= 0).all()) and bool(
            (subtoks[active] < len(tokenizers[dst])).all()
        )
        record.update(
            path=str(table_path),
            sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
            rows=len(lengths),
            mapped_rows=int((lengths > 0).sum()),
            status="PASS",
        )
        report["tables"].append(record)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
report["status"] = "PASS"
report["completed_utc"] = datetime.now(timezone.utc).isoformat()
report_path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
