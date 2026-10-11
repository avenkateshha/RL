#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

# Existing tokenizer-pair artifacts are required; no untracked table builder or
# machine-specific model/data path is assumed. A small corpus needs >=8 rows.
: "${XTOKEN_TEXT_DATA:?Set XTOKEN_TEXT_DATA to the raw-text fixture path}"
: "${XTOKEN_QWEN_FORWARD_TABLE:?Set the Llama-to-Qwen subtoken table path}"
: "${XTOKEN_QWEN_REVERSE_TABLE:?Set the Qwen-to-Llama subtoken table path}"
: "${XTOKEN_SMOLLM_FORWARD_TABLE:?Set the Llama-to-SmolLM2 subtoken table path}"
: "${XTOKEN_SMOLLM_REVERSE_TABLE:?Set the SmolLM2-to-Llama subtoken table path}"
for artifact in "$XTOKEN_TEXT_DATA" "$XTOKEN_QWEN_FORWARD_TABLE" \
    "$XTOKEN_QWEN_REVERSE_TABLE" "$XTOKEN_SMOLLM_FORWARD_TABLE" \
    "$XTOKEN_SMOLLM_REVERSE_TABLE"; do
    if [[ ! -f "$artifact" ]]; then
        echo "[ERROR] Required xToken fixture does not exist: $artifact" >&2
        exit 1
    fi
done

NUM_NODES=1
NUM_MINUTES=30
MAX_STEPS=${MAX_STEPS:-3}
exit_if_max_steps_reached
cd "$PROJECT_ROOT"

uv run examples/run_xtoken_off_policy_distillation.py \
    --config "$CONFIG_PATH" \
    "distillation.max_num_steps=$MAX_STEPS" \
    "logger.log_dir=$LOG_DIR" \
    logger.wandb_enabled=False \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

# This is a correctness/integration smoke test, with no convergence or peak-VRAM
# assertions. Missing steps, missing teacher terms, or missing validation fail.
uv run tests/check_metrics.py "$JSON_METRICS" \
    "max({step: float(step) for step in data['train/loss']}) >= $MAX_STEPS" \
    'all_finite(data["train/loss"])' \
    'min(data["train/loss"]) > 0' \
    'all_finite(data["train/kl_loss_t0"])' \
    'min(data["train/kl_loss_t0"]) > 0' \
    'all_finite(data["train/kl_loss_t1"])' \
    'min(data["train/kl_loss_t1"]) > 0' \
    'all_finite(data["train/grad_norm"])' \
    'min(data["train/grad_norm"]) > 0' \
    'all_finite(data["validation/loss"])' \
    'min(data["validation/loss"]) > 0' \
    'all_finite(data["validation/kl_loss"])' \
    'min(data["validation/kl_loss"]) > 0' \
    "max({step: float(step) for step in data['validation/loss']}) >= $MAX_STEPS"
