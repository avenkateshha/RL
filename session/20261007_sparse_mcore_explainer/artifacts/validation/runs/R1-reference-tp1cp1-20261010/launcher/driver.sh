#!/bin/bash
set -euo pipefail
source "$PREFLIGHT_RUN/launcher/runtime-env.sh"
cd "$REPO_ROOT"
ACTOR_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
trap '"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"' EXIT
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
uv run --no-project --python "$ACTOR_PY" python -m torch.distributed.run   --nnodes="$REFERENCE_NODES" --nproc_per_node="$REFERENCE_LOCAL_PROCESSES"   --node_rank="$SLURM_PROCID" --master_addr="$REFERENCE_MASTER_ADDR" --master_port="$REFERENCE_MASTER_PORT"   "$PREFLIGHT_RUN/launcher/real_model_reference.py" "$PREFLIGHT_RUN"   --steps="$REFERENCE_STEPS" --fixture-manifest="$PREFLIGHT_RUN/fixture-manifest.json" --corpus="$PREFLIGHT_RUN/collated_corpus.json"
