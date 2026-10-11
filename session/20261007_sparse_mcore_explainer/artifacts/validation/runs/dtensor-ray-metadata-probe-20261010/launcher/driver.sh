#!/bin/bash
set -euo pipefail
source "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/R4-dtensor-mcore/launcher/runtime-env.sh"
DTENSOR_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python
"$DTENSOR_PY" "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/dtensor-ray-metadata-probe-20261010/launcher/probe_dtensor_ray_metadata.py" --output "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/dtensor-ray-metadata-probe-20261010/observer-enabled.json"
unset XTOKEN_NUMERICAL_CAPTURE_RUN
"$DTENSOR_PY" "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/dtensor-ray-metadata-probe-20261010/launcher/probe_dtensor_ray_metadata.py" --output "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/dtensor-ray-metadata-probe-20261010/observer-disabled.json"
