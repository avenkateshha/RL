#!/bin/bash
set -euo pipefail
source "$PREFLIGHT_RUN/launcher/runtime-env.sh"
cd "$REPO_ROOT"
ACTOR_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
set +e
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/distributed tests/unit/distributed/test_native_sparse_primitives.py tests/unit/distributed/test_native_sparse_reader.py -q --junitxml="$PREFLIGHT_RUN/junit.xml" > "$PREFLIGHT_RUN/pytest.log" 2>&1
pytest_rc=$?
printf '%s\n' "$pytest_rc" > "$PREFLIGHT_RUN/pytest-exit-code.txt"
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
source_rc=$?
if [[ $source_rc -ne 0 ]]; then exit "$source_rc"; fi
exit "$pytest_rc"
