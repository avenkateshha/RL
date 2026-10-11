"""Exercise frozen-source packaging and safe restoration in temporary trees."""

import hashlib
import json
import stat
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from package_frozen_python import (
    FORMAT,
    SOURCE_PREFIX,
    SnapshotRecord,
    build_package,
    load_package,
    package_digest,
    restore_package,
)


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    source = root / "original"
    source.mkdir()
    records = []
    for name, content, mode in [
        ("a.py", b"# Frozen bytes\nvalue = 1\n", 0o640),
        ("b.py", b"# Frozen bytes\nvalue = 1\n", 0o644),
        (
            "c.py",
            b"#!/usr/bin/env python\nraise RuntimeError('never execute')\n",
            0o755,
        ),
    ]:
        relative = SOURCE_PREFIX + "fixture/launcher/" + name
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(mode)
        records.append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    allowlist = root / "allowlist.json"
    allowlist.write_text(
        json.dumps({"candidate_files": [], "frozen_python_sources": records})
    )
    package = root / "package"
    build_package(allowlist, package_dir=package, repo_root=source)
    return source, package, allowlist


def _write_manifest(package: Path, records: tuple[SnapshotRecord, ...]) -> None:
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "format": FORMAT,
                "files": [asdict(record) for record in records],
                "source_package_sha256": package_digest(records),
            }
        )
    )


def check_case(case: str) -> None:
    """Verify one independent adversarial or roundtrip case without execution."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, package, allowlist = _fixture(root)
        manifest = load_package(package)
        assert len(manifest.files) == 3
        assert len({record.blob for record in manifest.files}) == 2
        destination = root / "restored"
        destination.mkdir()
        first = manifest.files[0]
        last = manifest.files[-1]
        expected_error = (ValueError, FileExistsError)
        if case == "roundtrip":
            assert restore_package(package, repo_root=destination) == (3, 0)
            assert restore_package(package, repo_root=destination) == (0, 3)
            for record in manifest.files:
                actual = destination / record.original_path
                original = source / record.original_path
                assert actual.read_bytes() == original.read_bytes()
                assert stat.S_IMODE(actual.stat().st_mode) == record.mode
            return
        if case == "corrupt_blob":
            (package / last.blob).write_bytes(b"corrupt")
        elif case == "traversal":
            changed = replace(last, original_path="../outside.py")
            _write_manifest(
                package,
                tuple(
                    sorted(
                        (*manifest.files[:2], changed), key=lambda r: r.original_path
                    )
                ),
            )
        elif case == "outside_source_scope":
            changed = replace(last, original_path="nemo_rl/unrelated.py")
            _write_manifest(
                package,
                tuple(
                    sorted(
                        (*manifest.files[:2], changed), key=lambda r: r.original_path
                    )
                ),
            )
        elif case == "duplicate_path":
            _write_manifest(package, (*manifest.files, last))
        elif case == "manifest_hash":
            data = json.loads((package / "manifest.json").read_text())
            data["source_package_sha256"] = "0" * 64
            (package / "manifest.json").write_text(json.dumps(data))
        elif case == "destination_symlink":
            outside = root / "outside"
            outside.mkdir()
            (destination / "session").symlink_to(outside, target_is_directory=True)
        elif case == "blob_symlink":
            blob = package / last.blob
            outside = root / "outside.txt"
            outside.write_bytes(blob.read_bytes())
            blob.unlink()
            blob.symlink_to(outside)
        elif case == "executable_blob":
            (package / last.blob).chmod(0o755)
        elif case in {"existing_bytes", "existing_mode"}:
            path = destination / last.original_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                b"different"
                if case == "existing_bytes"
                else (package / last.blob).read_bytes()
            )
            path.chmod(0o644)
        elif case == "stale_source_inventory":
            (source / first.original_path).write_bytes(b"changed after selection")
            try:
                build_package(allowlist, package_dir=package, repo_root=source)
            except ValueError:
                return
            raise AssertionError("Stale source inventory was accepted")
        else:
            raise AssertionError(case)
        try:
            restore_package(package, repo_root=destination)
        except expected_error:
            assert not (destination / first.original_path).exists(), (
                "Validation must reject all collisions before writing earlier files"
            )
        else:
            raise AssertionError(f"Unsafe package restored: {case}")


def main() -> None:
    """Run exact-byte/mode roundtrip and eleven fail-closed restoration checks."""
    cases = [
        "roundtrip",
        "corrupt_blob",
        "traversal",
        "outside_source_scope",
        "duplicate_path",
        "manifest_hash",
        "destination_symlink",
        "blob_symlink",
        "executable_blob",
        "existing_bytes",
        "existing_mode",
        "stale_source_inventory",
    ]
    for case in cases:
        check_case(case)
    print(json.dumps({"status": "PASS", "cases": cases, "executed_snapshots": 0}))


if __name__ == "__main__":
    main()
