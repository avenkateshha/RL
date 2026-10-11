#!/bin/bash
set -euo pipefail
source "$PREFLIGHT_RUN/launcher/runtime-env.sh"
cd "$REPO_ROOT"
ACTOR_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python
export PYTHONPATH="$PREFLIGHT_RUN/launcher:$PYTHONPATH"
export XTOKEN_NUMERICAL_CAPTURE_RUN="$PREFLIGHT_RUN"
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
set +e
uv run --no-project --python "$ACTOR_PY" python "$PREFLIGHT_RUN/launcher/dtensor_preflight.py" "$PREFLIGHT_RUN" > "$PREFLIGHT_RUN/import.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/import-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then exit "$case_rc"; fi
total_rc=0
for role in teacher_b teacher_a; do
 uv run --no-project --python "$ACTOR_PY" python -m torch.distributed.run --standalone --nproc_per_node=4 "$PREFLIGHT_RUN/launcher/dtensor_preflight.py" "$PREFLIGHT_RUN" --role="$role" > "$PREFLIGHT_RUN/$role.log" 2>&1
 case_rc=$?
 printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/$role-exit-code.txt"
 if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
done
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
source_rc=$?
if [[ $source_rc -ne 0 ]]; then exit "$source_rc"; fi
exit "$total_rc"
