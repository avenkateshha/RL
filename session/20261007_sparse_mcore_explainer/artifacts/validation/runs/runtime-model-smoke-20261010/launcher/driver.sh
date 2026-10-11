#!/bin/bash
set -euo pipefail
source "$REPO_ROOT/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/R2-tp2cp2/launcher/runtime-env.sh"
cd "$REPO_ROOT"
python_mcore=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
set +e
uv run --no-sync --python "$python_mcore" -m pytest tests/unit/models/policy/test_megatron_split_state.py --mcore-only -k xtoken > "$PREFLIGHT_RUN/split-guards.log" 2>&1
printf '%s\n' "$?" > "$PREFLIGHT_RUN/split-guards-exit-code.txt"
for role in student teacher_b teacher_a teacher_c_same_tokenizer; do
  timeout 240s "$python_mcore" -m torch.distributed.run --standalone --nproc-per-node=4 "$PREFLIGHT_RUN/launcher/model_forward_preflight.py" "$role" > "$PREFLIGHT_RUN/$role-model-forward.log" 2>&1
  model_rc=$?
  printf '%s %s\n' "$role" "$model_rc" >> "$PREFLIGHT_RUN/model-forward-exit-codes.txt"
  if [[ $model_rc -ne 0 ]]; then exit "$model_rc"; fi
done
