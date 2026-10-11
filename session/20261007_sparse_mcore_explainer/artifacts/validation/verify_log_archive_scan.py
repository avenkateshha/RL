"""Check warning detection and integrity gates on synthetic archived raw logs."""

import hashlib
import io
import json
import tarfile
import tempfile
from pathlib import Path

from summarize_matrix_run import warning_scan


def main():
    with tempfile.TemporaryDirectory(prefix="xtoken-log-scan-") as directory:
        run = Path(directory)
        (run / "driver.log").write_text("ordinary driver output\n")
        archive_path = run / "session.tar.gz"
        payload = b"ordinary line\nCudaIPCTypes warning\nreference count underflow\n"
        with tarfile.open(archive_path, "w:gz") as archive:
            member = tarfile.TarInfo("session/logs/worker-1.err")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        metadata = {
            "archive_path": str(archive_path),
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "archive_bytes": archive_path.stat().st_size,
            "members": [
                {
                    "path": member.name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ],
        }
        for suffix in (".members.json", ".json"):
            manifest = run / ("session" + suffix)
            manifest.write_text(json.dumps(metadata))
            result = warning_scan([run])
            assert result["files_scanned"] == 2
            assert [r["line"] for r in result["matches"]] == [2, 3]
            assert result["verified_archives"][0]["members_verified"] == 1
            assert result["verified_archives"][0]["log_members_scanned"] == 1
            manifest.unlink()
        manifest.write_text(json.dumps({**metadata, "archive_sha256": "incorrect"}))
        try:
            warning_scan([run])
        except AssertionError:
            pass
        else:
            raise AssertionError("Archive digest corruption was accepted")
        altered = {
            **metadata,
            "members": [{**metadata["members"][0], "sha256": "incorrect"}],
        }
        manifest.write_text(json.dumps(altered))
        try:
            warning_scan([run])
        except AssertionError:
            pass
        else:
            raise AssertionError("Member digest corruption was accepted")
    print(
        "PASS two archive manifest formats, warning locations, archive/member corruption rejection"
    )


if __name__ == "__main__":
    main()
