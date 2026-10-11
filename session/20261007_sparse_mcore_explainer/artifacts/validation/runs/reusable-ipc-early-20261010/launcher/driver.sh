#!/bin/bash
set -euo pipefail
source "$PREFLIGHT_RUN/launcher/runtime-env.sh"
cd "$REPO_ROOT"
ACTOR_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
set +e
total_rc=0
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/distributed tests/unit/distributed/test_reusable_cuda_ipc.py -q --junitxml="$PREFLIGHT_RUN/raw-junit.xml" > "$PREFLIGHT_RUN/raw.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/raw-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
uv run --no-project --python "$ACTOR_PY" python "$PREFLIGHT_RUN/launcher/ipc_refcount_probe.py" > "$PREFLIGHT_RUN/legacy_probe.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/legacy_probe-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/algorithms/x_token tests/unit/algorithms/x_token/test_xtoken_off_policy_distillation.py -q --junitxml="$PREFLIGHT_RUN/orchestration-junit.xml" > "$PREFLIGHT_RUN/orchestration.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/orchestration-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/models/policy tests/unit/models/policy/test_sparse_ipc_dispatch.py tests/unit/models/policy/test_utils.py::TestAggregatePerSampleHandles -q --junitxml="$PREFLIGHT_RUN/policy-junit.xml" > "$PREFLIGHT_RUN/policy.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/policy-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/algorithms/x_token tests/unit/algorithms/x_token/test_megatron_cp_normalization.py -k 'wrapper_keeps_schedule_compensation or loss_input_enables_per_term' -q --junitxml="$PREFLIGHT_RUN/wrapper-junit.xml" > "$PREFLIGHT_RUN/wrapper.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/wrapper-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
source_rc=$?
if [[ $source_rc -ne 0 ]]; then exit "$source_rc"; fi
exit "$total_rc"
