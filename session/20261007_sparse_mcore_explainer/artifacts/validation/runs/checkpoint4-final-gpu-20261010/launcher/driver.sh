#!/bin/bash
set -euo pipefail
source "$PREFLIGHT_RUN/launcher/runtime-env.sh"
cd "$REPO_ROOT"
ACTOR_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
set +e
total_rc=0
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/algorithms/x_token tests/unit/algorithms/x_token/test_native_sparse_loss.py tests/unit/algorithms/x_token/test_native_student.py tests/unit/algorithms/x_token/test_native_same_tokenizer.py -q --junitxml="$PREFLIGHT_RUN/native_math-junit.xml" > "$PREFLIGHT_RUN/native_math.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/native_math-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
CUDA_VISIBLE_DEVICES="" uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/algorithms/x_token tests/unit/algorithms/x_token/test_native_loss_contract.py tests/unit/algorithms/x_token/test_native_dense_contract.py tests/unit/algorithms/x_token/test_native_sparse_dispatch.py tests/unit/algorithms/x_token/test_native_sparse_legacy_mix.py tests/unit/algorithms/x_token/test_native_same_mixed.py tests/unit/algorithms/x_token/test_native_loss_input.py tests/unit/algorithms/x_token/test_native_dense_reader.py -q --junitxml="$PREFLIGHT_RUN/dispatch-junit.xml" > "$PREFLIGHT_RUN/dispatch.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/dispatch-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/algorithms/x_token tests/unit/algorithms/x_token/test_megatron_cp_normalization.py -k 'wrapper_keeps_schedule_compensation or loss_input_enables_per_term' -q --junitxml="$PREFLIGHT_RUN/wrapper-junit.xml" > "$PREFLIGHT_RUN/wrapper.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/wrapper-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
CUDA_VISIBLE_DEVICES="" uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/algorithms/x_token tests/unit/algorithms/x_token/test_xtoken_off_policy_distillation.py -q --junitxml="$PREFLIGHT_RUN/controller-junit.xml" > "$PREFLIGHT_RUN/controller.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/controller-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/algorithms/x_token tests/unit/algorithms/x_token/test_native_dense_reader_cuda.py -q --junitxml="$PREFLIGHT_RUN/dense_cuda-junit.xml" > "$PREFLIGHT_RUN/dense_cuda.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/dense_cuda-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
source_rc=$?
if [[ $source_rc -ne 0 ]]; then exit "$source_rc"; fi
exit "$total_rc"
