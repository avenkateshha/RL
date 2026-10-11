#!/bin/bash
set -euo pipefail
historical_runtime=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/experiments/xtoken-smoke/xtoken-distillation/xtoken-offset-rebased-2n100-20260928-092950/runtime
export PYTHONPATH="$REPO_ROOT:$PREFLIGHT_RUN/runtime/bridge/src:$PREFLIGHT_RUN/runtime/mcore:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/runtime-preflight-current-20261010/runtime/python:$historical_runtime/python"
export PREFLIGHT_OVERLAY=committed_sources_and_pinned_lens
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_MODE=disabled
python_mcore=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
export UV_PROJECT_ENVIRONMENT=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
"$python_mcore" "$PREFLIGHT_RUN/launcher/runtime_preflight.py"
set +e
CUDA_VISIBLE_DEVICES= uv run --no-sync --python "$python_mcore" -m tests.unit.algorithms.x_token.test_megatron_cp_normalization > "$PREFLIGHT_RUN/normalization.log" 2>&1
normalization_rc=$?
uv run --no-sync --python "$python_mcore" -m pytest tests/unit/models/policy/test_megatron_split_state.py tests/unit/algorithms/x_token/test_megatron_cp_normalization.py -k 'xtoken or wrapper' > "$PREFLIGHT_RUN/focused-pytest.log" 2>&1
pytest_rc=$?
printf '{"normalization_returncode":%s,"focused_pytest_returncode":%s}\n' "$normalization_rc" "$pytest_rc" > "$PREFLIGHT_RUN/checkpoint1-tests.json"
uv run --no-sync --python "$python_mcore" "$PREFLIGHT_RUN/launcher/model_provider_preflight.py" > "$PREFLIGHT_RUN/model-provider.log" 2>&1
uv run --no-sync --python "$python_mcore" -m pytest tests/unit/models/policy/test_megatron_split_state.py --mcore-only -k xtoken > "$PREFLIGHT_RUN/split-guards.log" 2>&1
printf '%s\n' "$?" > "$PREFLIGHT_RUN/split-guards-exit-code.txt"
# Actual pinned-weight compatibility smoke; one model at a time, four GPUs,
# one sequence of 32 tokens. This is independent of pending loss integration.
for role in student teacher_b teacher_a teacher_c_same_tokenizer; do
  timeout 360s "$python_mcore" -m torch.distributed.run --standalone --nproc-per-node=4 "$PREFLIGHT_RUN/launcher/model_forward_preflight.py" "$role" > "$PREFLIGHT_RUN/$role-model-forward.log" 2>&1
  model_rc=$?
  printf '%s %s\n' "$role" "$model_rc" >> "$PREFLIGHT_RUN/model-forward-exit-codes.txt"
  if [[ $model_rc -ne 0 ]]; then break; fi
done
