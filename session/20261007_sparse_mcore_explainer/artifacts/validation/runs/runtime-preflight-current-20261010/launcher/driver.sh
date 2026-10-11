#!/bin/bash
set -euo pipefail
historical_runtime=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/experiments/xtoken-smoke/xtoken-distillation/xtoken-offset-rebased-2n100-20260928-092950/runtime
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:$REPO_ROOT/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:$PREFLIGHT_RUN/runtime/python:$historical_runtime/python"
export PREFLIGHT_OVERLAY=current_sources_and_pinned_lens
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_MODE=disabled
python_mcore=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
export UV_PROJECT_ENVIRONMENT=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker
"$python_mcore" "$PREFLIGHT_RUN/launcher/runtime_preflight.py"
set +e
uv run --no-sync --python "$python_mcore" -m tests.unit.algorithms.x_token.test_megatron_cp_normalization > "$PREFLIGHT_RUN/normalization.log" 2>&1
normalization_rc=$?
uv run --no-sync --python "$python_mcore" -m pytest tests/unit/models/policy/test_megatron_split_state.py tests/unit/algorithms/x_token/test_megatron_cp_normalization.py -k 'xtoken or wrapper' > "$PREFLIGHT_RUN/focused-pytest.log" 2>&1
pytest_rc=$?
printf '{"normalization_returncode":%s,"focused_pytest_returncode":%s}\n' "$normalization_rc" "$pytest_rc" > "$PREFLIGHT_RUN/checkpoint1-tests.json"
uv run --no-sync --python "$python_mcore" "$PREFLIGHT_RUN/launcher/model_provider_preflight.py" > "$PREFLIGHT_RUN/model-provider.log" 2>&1
