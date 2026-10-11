"""Create one immutable small numerical-reference launcher; never submits."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("case")
parser.add_argument("run_id")
parser.add_argument("--steps", type=int, default=1)
parser.add_argument("--capture-run", type=Path)
parser.add_argument(
    "--preflight-pytest", help="Exact pytest node ID run on node zero before reference"
)
args = parser.parse_args()
validation = Path(__file__).resolve().parent
repo = validation.parents[3]
case = validation / "runs" / args.case
run = validation / "runs" / args.run_id
assert not run.exists(), "Use a fresh run ID, preserving every previous launcher"
(run / "launcher").mkdir(parents=True)
runtime = json.loads((case / "runtime.json").read_text())
for name in ("resolved_config.yaml", "runtime.json"):
    shutil.copy2(case / name, run / name)
shutil.copy2(validation / "fixture_manifest.json", run / "fixture-manifest.json")
shutil.copy2(
    validation / "runs/fixture-preparation-20261010/collated_corpus.json",
    run / "collated_corpus.json",
)
for name in (
    "real_model_reference.py",
    "model_oracle.py",
    "replay_controller_gradients.py",
):
    shutil.copy2(validation / name, run / "launcher" / name)
template = validation / "runs/checkpoint2-primitives-20261010/launcher"
for name in ("runtime-env.sh", "check_sources.py"):
    shutil.copy2(template / name, run / "launcher" / name)
checker = run / "launcher/check_sources.py"
checker.write_text(
    checker.read_text()
    + "\nfor name, expected in manifest['owned_file_sha256'].items():\n"
    + "    actual = hashlib.sha256((run / name).read_bytes()).hexdigest()\n"
    + "    assert actual == expected, (name, expected, actual)\n"
    + "print('Owned reference launcher/config/fixture hashes verified.', flush=True)\n"
)
environment = json.loads((template / "environment.json").read_text())
environment.update(
    PREFLIGHT_RUN=str(run),
    REFERENCE_NODES=str(runtime["nodes"]),
    REFERENCE_LOCAL_PROCESSES=str(runtime["gpus_per_node_used"]),
    REFERENCE_STEPS=str(args.steps),
)
(run / "launcher/environment.json").write_text(json.dumps(environment, indent=2) + "\n")
(run / "launcher/driver.sh").write_text("""#!/bin/bash
set -euo pipefail
source "$PREFLIGHT_RUN/launcher/runtime-env.sh"
cd "$REPO_ROOT"
unset XTOKEN_NUMERICAL_CAPTURE_RUN RAY_ADDRESS RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK MASTER_ADDR MASTER_PORT
ACTOR_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
trap 'xtoken_reference_status=$?; trap - EXIT; "$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py" || exit 1; exit "$xtoken_reference_status"' EXIT
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
uv run --no-project --python "$ACTOR_PY" python -m torch.distributed.run \
  --nnodes="$REFERENCE_NODES" --nproc_per_node="$REFERENCE_LOCAL_PROCESSES" \
  --node_rank="$SLURM_PROCID" --master_addr="$REFERENCE_MASTER_ADDR" --master_port="$REFERENCE_MASTER_PORT" \
  "$PREFLIGHT_RUN/launcher/real_model_reference.py" "$PREFLIGHT_RUN" \
  --steps="$REFERENCE_STEPS" --fixture-manifest="$PREFLIGHT_RUN/fixture-manifest.json" --corpus="$PREFLIGHT_RUN/collated_corpus.json"
""")
if args.preflight_pytest:
    import shlex

    driver = run / "launcher/driver.sh"
    text = driver.read_text()
    text = text.replace(
        'uv run --no-project --python "$ACTOR_PY" python -m torch.distributed.run',
        'if [[ "$SLURM_PROCID" == 0 ]]; then\n'
        '  uv run --no-project --python "$ACTOR_PY" python -m pytest '
        "--confcutdir=tests/unit/algorithms/x_token -q "
        + shlex.quote(args.preflight_pytest)
        + ' > "$PREFLIGHT_RUN/preflight-pytest.log" 2>&1\nfi\n'
        'uv run --no-project --python "$ACTOR_PY" python -m torch.distributed.run',
    )
    driver.write_text(text)
if args.capture_run is not None:
    import shlex

    driver = run / "launcher/driver.sh"
    text = driver.read_text()
    text = text.replace(
        '"$PREFLIGHT_RUN/launcher/real_model_reference.py"',
        '"$PREFLIGHT_RUN/launcher/replay_controller_gradients.py"',
    )
    text = text.replace(
        '--steps="$REFERENCE_STEPS" --fixture-manifest="$PREFLIGHT_RUN/fixture-manifest.json" --corpus="$PREFLIGHT_RUN/collated_corpus.json"',
        "--capture-run=" + shlex.quote(str(args.capture_run.resolve())),
    )
    driver.write_text(text)
(run / "launcher/reference.sub").write_text("""#!/bin/bash
set -euo pipefail
REFERENCE_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export REFERENCE_MASTER_ADDR
export REFERENCE_MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
srun --no-container-mount-home --mpi=pmix --ntasks="$REFERENCE_NODES" --ntasks-per-node=1 --nodes="$REFERENCE_NODES" --gres=gpu:8 \
  --container-image="$CONTAINER" --container-mounts="$MOUNTS" --container-workdir="$REPO_ROOT" \
  bash "$PREFLIGHT_RUN/launcher/driver.sh"
""")
submit = (
    (template / "submit.py")
    .read_text()
    .replace('"--nodes=1"', f'"--nodes={runtime["nodes"]}"')
    .replace('"--time=00:15:00"', '"--time=00:30:00"')
    .replace("xtoken-native-sparse-primitives", "xtoken-reference-" + args.case)
    .replace("launcher/preflight.sub", "launcher/reference.sub")
)
(run / "launcher/submit.py").write_text(submit)
sources = list((repo / "nemo_rl").rglob("*.py")) + [
    repo / "tests/unit/algorithms/x_token/native_sparse_fixtures.py"
]
if args.preflight_pytest:
    sources.append(repo / args.preflight_pytest.split("::", 1)[0])
(run / "source-manifest.json").write_text(
    json.dumps(
        {
            "root_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "file_sha256": {
                str(path.relative_to(repo)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sources
            },
            "owned_file_sha256": {
                str(path.relative_to(run)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in run.rglob("*")
                if path.is_file()
            },
            "reference_case": args.case,
            "scope": "Actual pinned-model numerical reference with preclip dlogits and every parameter gradient; production controller steps/evaluation remain companion case.",
            "tolerances": runtime["tolerances"],
        },
        indent=2,
    )
    + "\n"
)
(run / "results.json").write_text(
    json.dumps(
        {
            "status": "NOT_RUN",
            "reason": "Prepared only; await completed native consumers and reusable IPC validation.",
        },
        indent=2,
    )
    + "\n"
)
print(run)
