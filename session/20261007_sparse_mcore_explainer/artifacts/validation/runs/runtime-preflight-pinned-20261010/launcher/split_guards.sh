#!/bin/bash
set -euo pipefail
source "$REPO_ROOT/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/R2-tp2cp2/launcher/runtime-env.sh"
cd "$REPO_ROOT"
exec /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python -m pytest tests/unit/models/policy/test_megatron_split_state.py --mcore-only -k xtoken
