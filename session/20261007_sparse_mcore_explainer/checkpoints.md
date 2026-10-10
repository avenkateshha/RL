# Native CP xToken implementation checkpoints

Starting branch: `avenkateshha/xtoken-v6-loss`, verified local and remote HEAD
`da97802f5288e2c372055f01b7c1a5c0ee1647fd` on 2026-10-10.
Publication destination: `myfork`, `github.com/avenkateshha/RL`, same branch.
Pinned port reference: `5acd78e948d2bc9061255e4d0474ed3aa1d9586b`, available locally.
Unrelated dirty submodules and untracked files are preserved.

Order: **1 → 2 → 3 → 4a → 4b → 5**. Each checkpoint is reviewed, checked,
committed with sign-off, and pushed before subsequent implementation starts.
Runtime and fixture preparation may proceed independently. Large memory tests
and a K/sequence-size ladder are excluded.

## Checkpoint 1 — per-term Megatron CP normalization

Implemented. Existing same-tokenizer KL owns disjoint predictor
windows; its gradient must not receive the old blanket objective `/CP`.
Legacy replicated-gradient CE and cross-tokenizer v6 retain their own `/CP`
before aggregation. The CP SUM-then-select relayout backward and MCore
`num_microbatches / CP` schedule compensation remain intact.

Fixed-weight same-tokenizer KD gradients at CP2 recover the missing factor of
two. This is not a claim that dynamic-scaled gradients or optimizer updates
universally double. Dynamic CE/KD scaling uses one detached CP-complete ratio
per student microbatch and DP replica. Reported loss terms are CP-complete.
The unsupported split training API rejects xToken before modifying step state.

Validation: CPU-gloo FP32 passed TP/CP `(1,1)`, `(2,1)`, `(1,2)`, `(2,2)`:
32 parameter combinations × two microbatches × four grids = 256 loss/gradient
comparisons. Covers weighted sum and true averaged logits, common K5/full K12,
both KL directions, fixed/dynamic scaling, empty masks and zero teacher weights.
Additional checks prove the fixed-weight factor-of-CP correction and unchanged
legacy CE plus common/mismatch v6 gradients. Tolerances are explicit in
`test_megatron_cp_normalization.py`: same-tokenizer/CE `rtol=1e-4, atol=1e-5`,
legacy v6 `rtol=atol=1e-4`, factor-of-CP `rtol=1e-5, atol=1e-6`.

Command: `PYTHONPATH=<repo>:<historical-run>/runtime/python UV_CACHE_DIR=/tmp/xtoken-port-uv-cache uv run --no-sync --offline python -m tests.unit.algorithms.x_token.test_megatron_cp_normalization`.
The historical run is `xtoken-offset-rebased-2n100-20260928-092950`, listed in
the implementation plan. This overlay supplies the pinned Lens implementation;
shared dependencies were not modified. The CPU fixture redirects the TP gather's
CUDA-device allocation to CPU while retaining real collectives and autograd.
The direct adapter flag test passed for both explicit-TP settings. Changed-file
Ruff checks/format and `git diff --check` passed; independent diff review found
no actionable issues.

Runtime limitations: the host wrapper import lacks `megatron.bridge`; the two
worker split-API guard tests are present but the host module skips without that
dependency. Container wrapper/worker execution and real-model pre-clip gradient
checks remain pending the runtime repair and final GPU matrix. No CUDA IPC,
model-conversion or production memory acceptance is claimed by this checkpoint.

Publication: pending signed-off commit/push; the resulting SHA will be recorded
in the next checkpoint update (avoiding a self-referential commit hash).

## Runtime and fixture preparation

- Historical submission reference: job `19467093`, with new run-owned Slurm
  artifacts under `artifacts/validation/runs/`.
- Runtime preflight `19989358`: CUDA and eight H100 GPUs available. The base
  `/opt/nemo_rl_venv` lacks Lens and MCore; actor interpreter has MCore/Bridge,
  but its older source fails the current worker's `FullyShardedDataParallelV1`
  import. Historical pinned Lens overlay resolves `SpanRegistry`.
- Current-source overlay preflight `19989442`: modified dependency checkouts
  have the same API mismatch. Provider configuration conversion passed for all
  four pinned models; this does not load model weights or execute a forward.
- Branch-pinned source preflight `19989499`: worker, Bridge, MCore, Lens and
  CUDA imports **PASS**, using isolated cached git snapshots of committed
  Bridge `1f8873bb` and MCore `6a366090`. Shared installs, submodules and the
  historical run are preserved. Wrapper and worker tests are running.
- Pinned Qwen forward/reverse table validation and offline SmolLM2 table
  generation completed; exact evidence resides in fixture-preparation results.
- Real-model R1–R6 acceptance remains **NOT_RUN** until implementation and
  compatible runtime are ready. CPU tensor checks do not certify CUDA IPC or
  MCore scheduling/model conversion.

## Source contract checked

Local MCore commit `002255075c3728fded9a2e435677840b08560d55`:
`pipeline_parallel/schedules.py` multiplies a two-result loss by CP and divides
by the microbatch count. `distributed/distributed_data_parallel.py` uses unit
gradient scaling when `calculate_per_token_loss=True`; NeMo-RL setup sets this
and disables `average_in_collective`. Actual container imports are recorded
separately and must match the source contract before GPU acceptance. The same
schedule and unit-gradient-scaling contracts were verified in committed MCore
`6a366090` (`schedules.py:346-347`, `distributed_data_parallel.py:250-255`).
