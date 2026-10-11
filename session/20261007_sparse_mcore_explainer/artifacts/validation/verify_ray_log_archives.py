"""Verify raw log tar snapshots and fingerprints on a tiny local fixture."""

import hashlib
import json
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from controller_teardown import ControllerTeardown

with TemporaryDirectory(prefix="xtoken-log-archive-") as temporary:
    root = Path(temporary)
    logs = root / "ray/session_2026_01/logs"
    logs.mkdir(parents=True)
    (logs / "worker-123.err").write_text("actual actor stderr fixture\n")
    (logs / "driver.log").write_text("driver fixture\n")
    run = root / "run"
    run.mkdir()
    (run / "controller-process-exit.json").write_text('{"status":"PASS"}')
    helper = ControllerTeardown(run)
    helper.copy_ray_logs(ray_log_root=root / "ray")
    result = json.loads((run / "ray-log-snapshot.json").read_text())
    assert result["exits_confirmed"]
    node = result["nodes"][0]
    assert len(node["archives"]) == len(node["actor_stderr_members"]) == 1
    record = node["archives"][0]
    archive_path = run / record["path"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == record["sha256"]
    manifest = json.loads((run / record["member_manifest"]).read_text())
    assert len(manifest["members"]) == record["files"] == 2
    with tarfile.open(archive_path) as archive:
        for member in manifest["members"]:
            content = archive.extractfile(member["path"]).read()
            assert len(content) == member["bytes"]
            assert hashlib.sha256(content).hexdigest() == member["sha256"]
        stderr = node["actor_stderr_members"][0]
        assert stderr["archive"] == record["path"]
        assert (
            archive.extractfile(stderr["member"]).read()
            == (logs / "worker-123.err").read_bytes()
        )
    assert not list((run / "ray-final-logs").rglob("*.err"))
print(
    "PASS raw log tar snapshot, per-member fingerprints and stderr lookup; local fixture only"
)
