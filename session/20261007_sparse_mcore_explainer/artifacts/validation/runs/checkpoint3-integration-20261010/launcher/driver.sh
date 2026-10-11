#!/bin/bash
set -euo pipefail
source "$PREFLIGHT_RUN/launcher/runtime-env.sh"
cd "$REPO_ROOT"
ACTOR_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
set +e
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/models/megatron tests/unit/models/megatron/test_xtoken_padding.py tests/unit/models/megatron/test_sparse_teacher_producer.py -q --junitxml="$PREFLIGHT_RUN/megatron-junit.xml" > "$PREFLIGHT_RUN/megatron-pytest.log" 2>&1
megatron_rc=$?
printf '%s\n' "$megatron_rc" > "$PREFLIGHT_RUN/megatron-exit-code.txt"
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/algorithms/x_token tests/unit/algorithms/x_token/test_utils.py tests/unit/algorithms/x_token/test_native_dense_reader.py tests/unit/algorithms/x_token/test_native_loss_input.py -q --junitxml="$PREFLIGHT_RUN/utils-junit.xml" > "$PREFLIGHT_RUN/utils-pytest.log" 2>&1
utils_rc=$?
printf '%s\n' "$utils_rc" > "$PREFLIGHT_RUN/utils-exit-code.txt"
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/models/policy tests/unit/models/policy/test_sparse_ipc_dispatch.py -q --junitxml="$PREFLIGHT_RUN/policy-junit.xml" > "$PREFLIGHT_RUN/policy-pytest.log" 2>&1
policy_rc=$?
printf '%s\n' "$policy_rc" > "$PREFLIGHT_RUN/policy-exit-code.txt"
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
source_rc=$?
if [[ $source_rc -ne 0 ]]; then exit "$source_rc"; fi
if [[ $megatron_rc -ne 0 ]]; then exit "$megatron_rc"; fi
if [[ $utils_rc -ne 0 ]]; then exit "$utils_rc"; fi
exit "$policy_rc"
