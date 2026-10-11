"""Create new run-owned launchers from successful xToken job19467093; never submit."""

import argparse
import hashlib
import json
import shlex
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("case_ids", nargs="+")
parser.add_argument(
    "--controller-reference",
    action="store_true",
    help="Capture actual mixed-backend observations and independent loss/dlogits evidence",
)
parser.add_argument(
    "--optimizer-replay",
    action="store_true",
    help="After the one-node controller, replay first-step optimizer shards in the same allocation",
)
parser.add_argument(
    "--preflight-pytest",
    help="One focused pytest file/node in an isolated subprocess before the controller",
)
parser.add_argument(
    "--preflight-backend", choices=("mcore", "dtensor"), default="mcore"
)
parser.add_argument("--preflight-gpu", action="store_true")
parser.add_argument("--preflight-expected-tests", type=int)
parser.add_argument(
    "--legacy-ipc-audit",
    action="store_true",
    help="Require actual legacy producer counter coverage before optimizer replay",
)
args = parser.parse_args()
validation = Path(__file__).resolve().parent
repo = validation.parents[3]
historical = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/experiments/xtoken-smoke/xtoken-distillation/xtoken-offset-rebased-2n100-20260928-092950"
)
pinned = validation / "runs/runtime-preflight-pinned-20261010"
test_deps = validation / "runs/runtime-preflight-current-20261010/runtime/python"
mcore = "/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
dtensor = "/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
old_environment = json.loads((historical / "launcher/environment.json").read_text())
for case in args.case_ids:
    run = validation / "runs" / case
    assert (run / "resolved_config.yaml").is_file(), case
    assert not (run / "submission.json").exists(), (
        "Already submitted; use a new case id for a retry"
    )
    runtime = json.loads((run / "runtime.json").read_text())
    if args.optimizer_replay:
        assert runtime["nodes"] == 1, "Two-node replay uses its own torchrun launcher"
    runtime["optimizer_replay_in_allocation"] = args.optimizer_replay
    runtime["preflight_pytest"] = args.preflight_pytest
    runtime["preflight_backend"] = args.preflight_backend
    runtime["preflight_gpu"] = args.preflight_gpu
    runtime["preflight_expected_tests"] = args.preflight_expected_tests
    assert not args.preflight_gpu or args.preflight_pytest
    runtime["legacy_ipc_audit"] = args.legacy_ipc_audit
    if args.legacy_ipc_audit:
        assert args.optimizer_replay, "Counter audit runs between controller and replay"
    (run / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    launcher = run / "launcher"
    launcher.mkdir(exist_ok=True)
    source = (historical / "launcher/ray-runtime.sub").read_text()
    source = source.replace(dtensor, mcore)
    (launcher / "ray-runtime.sub").write_text(source)
    env = {
        k: v
        for k, v in old_environment.items()
        if k
        not in (
            "PYTHONPATH",
            "COMMAND",
            "SETUP_COMMAND",
            "BASE_LOG_DIR",
            "UV_PROJECT_ENVIRONMENT",
        )
    }
    env.update(
        PYTHONPATH=":".join(
            [
                str(repo),
                str(pinned / "runtime/bridge/src"),
                str(pinned / "runtime/mcore"),
                str(test_deps),
                str(historical / "runtime/automodel"),
                str(historical / "runtime/python"),
            ]
        ),
        UV_PROJECT_ENVIRONMENT=mcore,
        NEMO_RL_PY_EXECUTABLES_SYSTEM="1",
        NRL_IGNORE_VERSION_MISMATCH="1",
        NRL_MEGATRON_CHECKPOINT_DIR=str(validation / "runtime/model-conversion-cache"),
        BASE_LOG_DIR=str(run / "driver-logs"),
        COMMAND="bash " + str(launcher / "driver.sh"),
        SETUP_COMMAND="",
        RUN_DIR=str(run),
        REPO_ROOT=str(repo),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        RAY_LOG_SYNC_FREQUENCY="",
    )
    env["MOUNTS"] += (
        ",/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/guihongl:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/guihongl"
    )
    if args.controller_reference or args.optimizer_replay:
        capture = launcher / "reference_capture"
        capture.mkdir(exist_ok=True)
        for name in (
            "controller_reference.py",
            "model_oracle.py",
            "optimizer_gradient_capture.py",
            "legacy_ipc_counter_observer.py",
        ):
            shutil.copy2(validation / name, capture / name)
        if args.optimizer_replay:
            for name in ("real_model_reference.py", "replay_controller_gradients.py"):
                shutil.copy2(validation / name, capture / name)
        if args.legacy_ipc_audit:
            shutil.copy2(
                validation / "verify_legacy_ipc_observations.py",
                capture / "verify_legacy_ipc_observations.py",
            )
        shutil.copy2(
            validation / "reference_sitecustomize.py", capture / "sitecustomize.py"
        )
        env["PYTHONPATH"] = str(capture) + ":" + env["PYTHONPATH"]
        env["XTOKEN_NUMERICAL_CAPTURE_RUN"] = str(run)
    shutil.copy2(
        validation / "controller_teardown.py", launcher / "controller_teardown.py"
    )
    (launcher / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    env_shell = (
        "\n".join(
            "export " + k + "=" + shlex.quote(v)
            for k, v in env.items()
            if k not in ("COMMAND", "SETUP_COMMAND")
        )
        + "\n"
    )
    (launcher / "runtime-env.sh").write_text(env_shell)
    (launcher / "driver.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\nsource "
        + shlex.quote(str(launcher / "runtime-env.sh"))
        + '\ncd "$REPO_ROOT"\nexec '
        + mcore
        + "/bin/python "
        + shlex.quote(str(launcher / "run_driver.py"))
        + "\n"
    )
    if args.preflight_pytest:
        driver = launcher / "driver.sh"
        script = driver.read_text()
        preflight_python = (
            mcore if args.preflight_backend == "mcore" else dtensor
        ) + "/bin/python"
        preflight = (
            shlex.quote(mcore + "/bin/python")
            + " "
            + shlex.quote(str(launcher / "run_driver.py"))
            + " --verify-only\n"
            + "env -u RAY_ADDRESS -u XTOKEN_NUMERICAL_CAPTURE_RUN "
            + ("" if args.preflight_gpu else "CUDA_VISIBLE_DEVICES='' ")
            + shlex.quote(preflight_python)
            + " -m pytest --confcutdir="
            + shlex.quote(str(Path(args.preflight_pytest.split("::", 1)[0]).parent))
            + " -q "
            + shlex.quote(args.preflight_pytest)
            + " --junitxml="
            + shlex.quote(str(run / "preflight-junit.xml"))
            + " > "
            + shlex.quote(str(run / "preflight-pytest.log"))
            + " 2>&1\n"
            + shlex.quote(mcore + "/bin/python")
            + " "
            + shlex.quote(str(launcher / "check_preflight.py"))
            + "\n"
        )
        (
            launcher / "check_preflight.py"
        ).write_text('''"""Require complete focused runtime coverage before actual model execution."""
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

run = Path(os.environ["RUN_DIR"])
runtime = json.loads((run / "runtime.json").read_text())
cases = list(ET.parse(run / "preflight-junit.xml").getroot().iter("testcase"))
counts = {kind: sum(case.find(kind) is not None for case in cases) for kind in ("skipped", "failure", "error")}
expected = runtime["preflight_expected_tests"]
passed = bool(cases) and not any(counts.values()) and (expected is None or len(cases) == expected)
result = {"status": "PASS" if passed else "FAIL", "tests": len(cases), "expected_tests": expected, **counts, "backend": runtime["preflight_backend"], "cuda_visible": runtime["preflight_gpu"], "test_file": runtime["preflight_pytest"], "scope": "Exact cached actor interpreter, isolated from numerical capture and Ray driver address; every focused test must execute without skips", "log": "preflight-pytest.log", "junit": "preflight-junit.xml"}
(run / "preflight-results.json").write_text(json.dumps(result, indent=2) + "\\n")
print(json.dumps(result), flush=True)
assert passed, result
''')
        driver.write_text(script.replace("\nexec ", "\n" + preflight + "exec "))
    if args.optimizer_replay:
        driver = launcher / "driver.sh"
        script = driver.read_text().replace("\nexec ", "\n")
        verify = (
            shlex.quote(mcore + "/bin/python")
            + " "
            + shlex.quote(str(launcher / "run_driver.py"))
            + " --verify-only"
        )
        script = script.replace(
            '\ncd "$REPO_ROOT"\n',
            '\ncd "$REPO_ROOT"\ntrap '
            + shlex.quote(
                "xtoken_driver_status=$?; trap - EXIT; "
                + verify
                + ' || exit 1; exit "$xtoken_driver_status"'
            )
            + " EXIT\n",
        )
        if args.legacy_ipc_audit:
            script += (
                shlex.quote(mcore + "/bin/python")
                + " "
                + shlex.quote(
                    str(
                        launcher / "reference_capture/verify_legacy_ipc_observations.py"
                    )
                )
                + " "
                + shlex.quote(str(run))
                + " > "
                + shlex.quote(str(run / "legacy-ipc-verification.log"))
                + " 2>&1\n"
            )
        script += (
            "env -u XTOKEN_NUMERICAL_CAPTURE_RUN -u RANK -u LOCAL_RANK "
            "-u WORLD_SIZE -u LOCAL_WORLD_SIZE -u GROUP_RANK -u ROLE_RANK "
            "-u MASTER_ADDR -u MASTER_PORT uv run --no-project --python "
            + shlex.quote(mcore + "/bin/python")
            + " python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="
            + str(runtime["gpus_per_node_used"])
            + " "
            + shlex.quote(
                str(launcher / "reference_capture/replay_controller_gradients.py")
            )
            + " "
            + shlex.quote(str(run))
            + " --capture-run="
            + shlex.quote(str(run))
            + " > "
            + shlex.quote(str(run / "optimizer-replay.log"))
            + " 2>&1\n"
        )
        driver.write_text(script)
    (launcher / "run_driver.py").write_text(
        '''"""Use cached backend-specific interpreters through the public actor registry."""
import os
from pathlib import Path
import runpy
import sys
import hashlib
import json
import math
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
from nemo_rl.utils.logger import Logger
from controller_teardown import ControllerTeardown

run=Path(os.environ['RUN_DIR'])
runtime=json.loads((run/'runtime.json').read_text())
def verify_sources():
    repo=Path(os.environ['REPO_ROOT'])
    assert hashlib.sha256((run/'resolved_config.yaml').read_bytes()).hexdigest()==runtime['resolved_config_sha256_at_submission'], 'resolved_config.yaml'
    for name,expected in runtime['source_sha256_at_submission'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==expected, name
    for name,expected in runtime['launcher_sha256_at_submission'].items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest()==expected, name
    print('Controller source and owned launcher hashes verified.', flush=True)
verify_sources()
if '--verify-only' in sys.argv:
    raise SystemExit(0)
teardown=ControllerTeardown(run)
teardown.install()
original_log_metrics=Logger.log_metrics
metric_records=[]
def json_metric(value):
    if hasattr(value,'detach'):
        value=value.detach().cpu()
    if hasattr(value,'tolist'):
        return value.tolist()
    raise TypeError('Unsupported metric type: '+type(value).__name__)
def recorded_log_metrics(self,metrics,step,prefix='',step_metric=None,step_finished=False):
    record={'step':step,'prefix':prefix,'metrics':metrics}
    encoded=json.dumps(record,default=json_metric)
    metric_records.append(json.loads(encoded))
    with (run/'controller-metrics.jsonl').open('a') as stream:
        stream.write(encoded+'\\n')
    return original_log_metrics(self,metrics,step,prefix,step_metric,step_finished)
Logger.log_metrics=recorded_log_metrics
ACTOR_ENVIRONMENT_REGISTRY['nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker'] = '''
        + repr(mcore + "/bin/python")
        + """
ACTOR_ENVIRONMENT_REGISTRY['nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2'] = """
        + repr(dtensor + "/bin/python")
        + """
sys.argv=['examples/run_xtoken_off_policy_distillation.py','--config',str(run/'resolved_config.yaml'),'--resolved-config-output',str(run/'runtime_resolved_config.yaml')]
try:
    runpy.run_path(str(Path(os.environ['REPO_ROOT'])/'examples/run_xtoken_off_policy_distillation.py'),run_name='__main__')
    teardown.verify_and_shutdown()
    training=[record for record in metric_records if record['prefix']=='train']
    validation=[record for record in metric_records if record['prefix']=='validation']
    assert [record['step'] for record in training]==list(range(1,runtime['optimizer_steps']+1)), training
    assert any(record['step']==runtime['optimizer_steps'] for record in validation), validation
    for record in training:
        for name in ('loss','ce_loss','kl_loss','grad_norm'):
            assert math.isfinite(float(record['metrics'][name])), (record['step'],name)
    (run/'controller-completion.json').write_text(json.dumps({'status':'PASS','optimizer_steps_recorded':[r['step'] for r in training],'evaluation_steps_recorded':[r['step'] for r in validation],'metrics_file':'controller-metrics.jsonl','scope':'Actual controller updates/evaluation; independent gradient parity is recorded by the companion reference/capture harness.'},indent=2)+'\\n')
finally:
    controller_error=sys.exc_info()[1]
    final_errors=[]
    for action in (teardown.restore,teardown.copy_ray_logs,verify_sources):
        try:
            action()
        except Exception as final_error:
            if controller_error is not None:
                controller_error.add_note(f'Final artifact verification failed: {type(final_error).__name__}: {final_error}')
            else:
                final_errors.append(final_error)
    if final_errors:
        raise ExceptionGroup('Final artifact verification failed',final_errors)
"""
    )
    (
        launcher / "submit.py"
    ).write_text('''"""Adapted run-owned job19467093 submission workflow; duplicate submission forbidden."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib

run=Path(__file__).resolve().parents[1]
env=dict(os.environ)
for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','WANDB_API_KEY','GH_TOKEN','GITHUB_TOKEN'):
    env.pop(key,None)
env.update(json.loads((run/'launcher/environment.json').read_text()))
runtime=json.loads((run/'runtime.json').read_text())
if '--test-only' not in sys.argv:
    repo=Path(env['REPO_ROOT'])
    runtime['source_head_at_submission']=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    tracked=subprocess.check_output(['git','-C',str(repo),'ls-files','nemo_rl','pyproject.toml','uv.lock'],text=True).splitlines()
    tracked.append('tests/unit/algorithms/x_token/native_sparse_fixtures.py')
    tracked.append('examples/run_xtoken_off_policy_distillation.py')
    if runtime.get('preflight_pytest'):
        tracked.append(runtime['preflight_pytest'].split('::',1)[0])
    runtime['source_sha256_at_submission']={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in tracked if name.endswith('.py') or name in ('pyproject.toml','uv.lock')}
    runtime['resolved_config_sha256_at_submission']=hashlib.sha256((run/'resolved_config.yaml').read_bytes()).hexdigest()
    runtime['launcher_sha256_at_submission']={str(path.relative_to(run)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (run/'launcher').rglob('*') if path.is_file() and '__pycache__' not in path.parts}
    (run/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\\n')
    (run/'fixture-manifest.json').write_bytes((run.parents[1]/'fixture_manifest.json').read_bytes())
command=['sbatch','--nodes='+str(runtime['nodes']),'--gres=gpu:8','--exclusive','--account=coreai_dlalgo_genai','--partition=interactive','--time=00:45:00','--job-name=xtoken-small-'+run.name,'--output='+str(run/'slurm-%j.out'),'--chdir='+env['REPO_ROOT']]
if '--test-only' in sys.argv:
    command.append('--test-only')
    record_path=run/'scheduler-test.json'
else:
    assert not (run/'submission.json').exists(),'Already submitted; create a new run for retry'
    command.append('--parsable')
    record_path=run/'submission.json'
command.append(str(run/'launcher/ray-runtime.sub'))
result=subprocess.run(command,env=env,text=True,capture_output=True)
record={'command':command,'utc':datetime.now(timezone.utc).isoformat(),'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
record_path.write_text(json.dumps(record,indent=2)+'\\n')
print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
sys.exit(result.returncode)
''')
    (launcher / "provenance.json").write_text(
        json.dumps(
            {
                "historical_job": "19467093",
                "historical_launcher": str(historical / "launcher/ray-runtime.sub"),
                "historical_launcher_sha256": hashlib.sha256(
                    (historical / "launcher/ray-runtime.sub").read_bytes()
                ).hexdigest(),
                "adaptations": [
                    "8GPU node Slurm and Ray topology retained",
                    "MCore driver/head interpreter selected",
                    "Backend-specific actor registry uses cached MCore/DTensor interpreters",
                    "Exact committed Bridge/MCore source snapshots and pinned Lens/Automodel/Transformers overlay",
                    "45min small-test resource limit",
                    "Pinned fixture config and run-owned outputs",
                ],
                "dependency_source_manifest": str(
                    pinned / "runtime/source-provenance.json"
                ),
                "runtime_status": "Require PASS MCore preflight and model conversion before acceptance; generated launcher does not establish runtime compatibility.",
            },
            indent=2,
        )
        + "\n"
    )
    print("Prepared launcher:", case)
