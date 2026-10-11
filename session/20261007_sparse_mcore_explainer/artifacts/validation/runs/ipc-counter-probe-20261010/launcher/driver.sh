#!/bin/bash
set -euo pipefail
source "$PREFLIGHT_RUN/launcher/runtime-env.sh"
cd "$REPO_ROOT"
ACTOR_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
set +e
uv run --no-project --python "$ACTOR_PY" python "$PREFLIGHT_RUN/launcher/ipc_refcount_probe.py" > "$PREFLIGHT_RUN/probe.log" 2>&1
probe_rc=$?
printf '%s\n' "$probe_rc" > "$PREFLIGHT_RUN/probe-exit-code.txt"
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/algorithms/x_token tests/unit/algorithms/x_token/test_megatron_cp_normalization.py -k 'wrapper_keeps_schedule_compensation or loss_input_enables_per_term' -q --junitxml="$PREFLIGHT_RUN/wrapper-junit.xml" > "$PREFLIGHT_RUN/wrapper-pytest.log" 2>&1
wrapper_rc=$?
printf '%s\n' "$wrapper_rc" > "$PREFLIGHT_RUN/wrapper-exit-code.txt"
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
if [[ $probe_rc -ne 0 ]]; then exit "$probe_rc"; fi
exit "$wrapper_rc"
