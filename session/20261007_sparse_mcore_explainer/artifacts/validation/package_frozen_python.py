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

"""Publish exact historical Python bytes as data and explicitly restore them."""

import argparse
import hashlib
import json
import stat
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

FORMAT = "xtoken_frozen_python_v1"
SOURCE_PREFIX = "session/20261007_sparse_mcore_explainer/artifacts/validation/runs/"


@dataclass(frozen=True)
class SnapshotRecord:
    """An original run path and its exact immutable source bytes and mode."""

    original_path: str
    blob: str
    sha256: str
    bytes: int
    mode: int


@dataclass(frozen=True)
class FrozenPythonPackage:
    """A validated ordered inventory with a digest covering every record."""

    files: tuple[SnapshotRecord, ...]
    source_package_sha256: str


def _relative_path(value: str) -> PurePosixPath:
    """Require one canonical relative path without traversal components."""
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("Snapshot paths must be POSIX relative paths")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise ValueError(f"Unsafe snapshot path: {value!r}")
    return path


def _contained_path(root: Path, relative: str) -> Path:
    """Reject symlinks and non-directory ancestors before touching a path."""
    parts = _relative_path(relative).parts
    current = root
    for index, part in enumerate(parts):
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink in snapshot path: {relative}")
        if index < len(parts) - 1 and current.exists() and not current.is_dir():
            raise ValueError(f"Non-directory snapshot ancestor: {relative}")
    if not current.resolve().is_relative_to(root):
        raise ValueError(f"Snapshot escapes its root: {relative}")
    return current


def _source_path(root: Path, value: str) -> Path:
    _relative_path(value)
    if not value.startswith(SOURCE_PREFIX) or not value.endswith(".py"):
        raise ValueError("Only this validation session's run-owned Python is eligible")
    return _contained_path(root, value)


def package_digest(records: tuple[SnapshotRecord, ...]) -> str:
    """Hash the canonical file mapping, byte hashes, sizes, and original modes."""
    payload = {"format": FORMAT, "files": [asdict(record) for record in records]}
    content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def _validate_record(record: SnapshotRecord) -> None:
    _relative_path(record.original_path)
    if (
        len(record.sha256) != 64
        or any(character not in "0123456789abcdef" for character in record.sha256)
        or record.blob != f"blobs/{record.sha256}.py.txt"
        or type(record.bytes) is not int
        or record.bytes < 0
        or type(record.mode) is not int
        or not 0 <= record.mode <= 0o777
    ):
        raise ValueError("Invalid frozen Python record")


def load_package(package_dir: Path) -> FrozenPythonPackage:
    """Validate a package manifest without restoring or executing its contents."""
    if package_dir.is_symlink():
        raise ValueError("Package root must not be a symlink")
    root = package_dir.resolve(strict=True)
    manifest = json.loads(_contained_path(root, "manifest.json").read_text())
    if manifest["format"] != FORMAT:
        raise ValueError("Unsupported frozen Python package format")
    records = tuple(SnapshotRecord(**record) for record in manifest["files"])
    for record in records:
        _validate_record(record)
    paths = [record.original_path for record in records]
    if len(set(paths)) != len(paths) or paths != sorted(paths):
        raise ValueError("Frozen Python paths must be unique and sorted")
    digest = package_digest(records)
    if digest != manifest["source_package_sha256"]:
        raise ValueError("Frozen Python package digest mismatch")
    return FrozenPythonPackage(records, digest)


def build_package(
    allowlist_path: Path, *, package_dir: Path, repo_root: Path
) -> FrozenPythonPackage:
    """Copy selected historical bytes into deduplicated, non-executable blobs."""
    root = repo_root.resolve(strict=True)
    allowlist = json.loads(allowlist_path.read_text())
    selected = {}
    for item in allowlist["candidate_files"] + allowlist.get(
        "frozen_python_sources", []
    ):
        name = item["path"]
        if name.startswith(SOURCE_PREFIX) and name.endswith(".py"):
            if name in selected and selected[name] != item:
                raise ValueError(f"Conflicting source inventory: {name}")
            selected[name] = item
    if not selected:
        raise ValueError("Allowlist contains no frozen run-owned Python sources")
    records = []
    contents = {}
    for name, item in sorted(selected.items()):
        source = _source_path(root, name)
        if not source.is_file():
            raise ValueError(f"Missing regular snapshot source: {name}")
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != item["sha256"] or len(content) != item["bytes"]:
            raise ValueError(f"Source differs from its selected allowlist: {name}")
        record = SnapshotRecord(
            original_path=name,
            blob=f"blobs/{digest}.py.txt",
            sha256=digest,
            bytes=len(content),
            mode=stat.S_IMODE(source.stat().st_mode),
        )
        _validate_record(record)
        records.append(record)
        contents[record.blob] = content
    if package_dir.is_symlink():
        raise ValueError("Package root must not be a symlink")
    package_dir.mkdir(parents=True, exist_ok=True)
    package_root = package_dir.resolve(strict=True)
    for name, content in contents.items():
        target = _contained_path(package_root, name)
        if target.exists():
            if not target.is_file() or target.read_bytes() != content:
                raise ValueError(f"Existing snapshot blob is corrupt: {name}")
            if target.stat().st_mode & 0o111:
                raise ValueError(f"Snapshot blob must not be executable: {name}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
            target.chmod(0o644)
    files = tuple(records)
    package = FrozenPythonPackage(files, package_digest(files))
    manifest_path = _contained_path(package_root, "manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "format": FORMAT,
                "source_package_sha256": package.source_package_sha256,
                "files": [asdict(record) for record in package.files],
            },
            indent=2,
        )
        + "\n"
    )
    return package


def restore_package(package_dir: Path, *, repo_root: Path) -> tuple[int, int]:
    """Restore absent originals; existing files must have identical bytes/modes.

    Validate every source and destination before writing any file. This function
    never imports or executes restored source and never overwrites a collision.
    """
    package = load_package(package_dir)
    source_root = package_dir.resolve(strict=True)
    if repo_root.is_symlink():
        raise ValueError("Restore root must not be a symlink")
    root = repo_root.resolve(strict=True)
    pending = []
    unchanged = 0
    for record in package.files:
        blob = _contained_path(source_root, record.blob)
        if not blob.is_file() or blob.stat().st_mode & 0o111:
            raise ValueError(f"Snapshot blob is not non-executable data: {record.blob}")
        content = blob.read_bytes()
        if (
            len(content) != record.bytes
            or hashlib.sha256(content).hexdigest() != record.sha256
        ):
            raise ValueError(f"Snapshot blob digest mismatch: {record.blob}")
        target = _source_path(root, record.original_path)
        if target.exists():
            if (
                not target.is_file()
                or target.read_bytes() != content
                or stat.S_IMODE(target.stat().st_mode) != record.mode
            ):
                raise FileExistsError(f"Refusing to overwrite snapshot: {target}")
            unchanged += 1
        else:
            pending.append((target, record, content))
    for target, record, content in pending:
        _contained_path(root, record.original_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
        target.chmod(record.mode)
    return len(pending), unchanged


def main() -> None:
    """Build publication data or restore it only after an explicit subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--allowlist", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--repo-root", type=Path, required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--package", type=Path, required=True)
    restore.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        package = build_package(
            args.allowlist, package_dir=args.output, repo_root=args.repo_root
        )
        print(
            json.dumps(
                {
                    "files": len(package.files),
                    "unique_blobs": len({record.blob for record in package.files}),
                    "source_package_sha256": package.source_package_sha256,
                }
            )
        )
    else:
        restored, unchanged = restore_package(args.package, repo_root=args.repo_root)
        print(json.dumps({"restored": restored, "already_identical": unchanged}))


if __name__ == "__main__":
    main()
