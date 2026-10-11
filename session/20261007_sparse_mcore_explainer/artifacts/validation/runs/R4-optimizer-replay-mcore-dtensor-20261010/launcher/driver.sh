#!/bin/bash
set -euo pipefail
source "$PREFLIGHT_RUN/launcher/runtime-env.sh"
cd "$REPO_ROOT"
ACTOR_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
trap 'xtoken_reference_status=$?; trap - EXIT; "$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py" || exit 1; exit "$xtoken_reference_status"' EXIT
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
uv run --no-project --python "$ACTOR_PY" python -m torch.distributed.run   --nnodes="$REFERENCE_NODES" --nproc_per_node="$REFERENCE_LOCAL_PROCESSES"   --node_rank="$SLURM_PROCID" --master_addr="$REFERENCE_MASTER_ADDR" --master_port="$REFERENCE_MASTER_PORT"   "$PREFLIGHT_RUN/launcher/replay_controller_gradients.py" "$PREFLIGHT_RUN"   --capture-run=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/RL/session/20261007_sparse_mcore_explainer/artifacts/validation/runs/R4-mcore-dtensor
