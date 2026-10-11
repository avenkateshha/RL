#!/bin/bash
set -euo pipefail
source "$PREFLIGHT_RUN/launcher/runtime-env.sh"
cd "$REPO_ROOT"
ACTOR_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
set +e
total_rc=0
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/distributed tests/unit/distributed/test_reusable_cuda_ipc.py -q --junitxml="$PREFLIGHT_RUN/raw-junit.xml" > "$PREFLIGHT_RUN/raw.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/raw-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/distributed tests/unit/distributed/test_native_sparse_primitives.py tests/unit/distributed/test_native_sparse_reader.py -q --junitxml="$PREFLIGHT_RUN/primitives-junit.xml" > "$PREFLIGHT_RUN/primitives.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/primitives-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/models/megatron tests/unit/models/megatron/test_sparse_teacher_producer.py tests/unit/models/megatron/test_xtoken_padding.py -q --junitxml="$PREFLIGHT_RUN/producer-junit.xml" > "$PREFLIGHT_RUN/producer.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/producer-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/algorithms/x_token tests/unit/algorithms/x_token/test_native_dense_reader_cuda.py -q --junitxml="$PREFLIGHT_RUN/dense_cuda-junit.xml" > "$PREFLIGHT_RUN/dense_cuda.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/dense_cuda-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
CUDA_VISIBLE_DEVICES="" uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/algorithms/x_token tests/unit/algorithms/x_token/test_native_dense_reader.py tests/unit/algorithms/x_token/test_native_loss_input.py -q --junitxml="$PREFLIGHT_RUN/dense_adapter-junit.xml" > "$PREFLIGHT_RUN/dense_adapter.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/dense_adapter-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/distributed tests/unit/distributed/test_process_lifetime.py tests/unit/distributed/test_recreate_worker.py -q --junitxml="$PREFLIGHT_RUN/lifetime-junit.xml" > "$PREFLIGHT_RUN/lifetime.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/lifetime-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/algorithms/x_token tests/unit/algorithms/x_token/test_xtoken_off_policy_distillation.py -q --junitxml="$PREFLIGHT_RUN/orchestration-junit.xml" > "$PREFLIGHT_RUN/orchestration.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/orchestration-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/models/policy tests/unit/models/policy/test_sparse_ipc_dispatch.py tests/unit/models/policy/test_utils.py::TestAggregatePerSampleHandles -q --junitxml="$PREFLIGHT_RUN/policy-junit.xml" > "$PREFLIGHT_RUN/policy.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/policy-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
uv run --no-project --python "$ACTOR_PY" python -m pytest --confcutdir=tests/unit/algorithms/x_token tests/unit/algorithms/x_token/test_megatron_cp_normalization.py -k 'wrapper_keeps_schedule_compensation or loss_input_enables_per_term' -q --junitxml="$PREFLIGHT_RUN/wrapper-junit.xml" > "$PREFLIGHT_RUN/wrapper.log" 2>&1
case_rc=$?
printf '%s\n' "$case_rc" > "$PREFLIGHT_RUN/wrapper-exit-code.txt"
if [[ $case_rc -ne 0 ]]; then total_rc=$case_rc; fi
"$ACTOR_PY" "$PREFLIGHT_RUN/launcher/check_sources.py"
source_rc=$?
if [[ $source_rc -ne 0 ]]; then exit "$source_rc"; fi
exit "$total_rc"
