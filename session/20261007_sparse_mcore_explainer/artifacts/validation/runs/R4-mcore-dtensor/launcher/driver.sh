#!/bin/bash
set -euo pipefail
source /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/R4-mcore-dtensor/launcher/runtime-env.sh
cd "$REPO_ROOT"
exec /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/R4-mcore-dtensor/launcher/run_driver.py
