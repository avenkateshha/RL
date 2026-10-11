#!/bin/bash
set -euo pipefail
source "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/R3-fixed-cuda-support-20261010/launcher/runtime-env.sh"
unset XTOKEN_NUMERICAL_CAPTURE_RUN
exec /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/dtensor-driver-import-probe-20261010/launcher/probe.py" "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/dtensor-driver-import-probe-20261010/results.json"
