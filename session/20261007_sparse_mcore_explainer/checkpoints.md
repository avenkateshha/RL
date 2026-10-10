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

Publication: signed-off commit `09de9ec5558f6768aab8bf105e933528432f1d37`
pushed to `myfork/avenkateshha/xtoken-v6-loss`; remote tip verified. The final
numerical harness also passed all four grids inside pinned container job
`19989499`; its schedule-compensation wrapper test subsequently passed. The
first pytest command filtered MCore-marked split guard tests, so those were
not counted as passing. The frozen follow-up job `19989617` ran
`--mcore-only -k xtoken`: **2 passed, 90 deselected**, exit 0.

## Checkpoint 2 — transport and selected-probability math

Implemented after checkpoint 1's verified push. Adds typed native sparse-v2
records and invocation-scoped requested-row readers, exact TP selected-log-prob
autograd, deterministic score/ID top-K with exact temperature-scaled logZ,
independent two-label forced sidecars and teacher-native CP segment mapping.
Padded vocabulary columns have zero probability/gradient. Selected requests
accumulate repeated row/token gradients and rematerialize bounded row tiles.
Cross-CP scalar exchange reuses the existing forward/backward SUM primitive.

Readers validate K, temperature, vocabulary, membership support, sample count,
backing offsets and complete nonoverlapping row coverage. They replace the
natural cutoff independently for each required-label request, preserving the
original noise membership. The forced-label builder accepts HEAD's derived
absent spans, retains two labels at origin collisions and validates teacher IDs.

Validation: `uv run --no-sync --offline pytest
--confcutdir=tests/unit/distributed
tests/unit/distributed/test_native_sparse_primitives.py
tests/unit/distributed/test_native_sparse_reader.py -q`: **35 passed, 4 CUDA
skipped**, 18.20 s. Includes CPU-gloo TP/CP `(1,1)`, `(2,1)`, `(1,2)`, `(2,2)`,
noncontiguous logits, padded columns, tied BF16 scores, repeated/empty requests,
owner-only cross-CP backward, two teachers with distinct lengths/microbatch
slots, independent generations and malformed metadata. Primitive FP32 values
use `rtol=atol=1e-6`; selected gradients use `rtol=atol=2e-6`.

Independent pure-Torch two-teacher sparse-objective fixture/oracle self-checks
pass: `uv run --no-sync python -m
tests.unit.algorithms.x_token.native_sparse_fixtures`, loss `1.5517744005`.
It covers selected support plus REST, prefix/mismatch cases, distinct teacher
normalizers, teacher permutation, fixed-ratio partitioning, empty contributions
and finite-difference gradients. Production-loss parity is reserved for 4a.
Changed-file Ruff/format, targeted Pyrefly (**0 errors**) and diff checks pass.
CUDA primitive execution, persistent producer streaming, CUDA IPC and complete
loss/model integration are not claimed by this checkpoint.

Publication: signed-off commit `7e6492b1b5c192bc9949d87405a469ab018c9770`
pushed to `myfork/avenkateshha/xtoken-v6-loss`; remote tip verified.

## Checkpoint 3 — teacher producer and reader integration

Implemented after checkpoint 2's verified push. Teacher-native streaming
producer, public backend dispatch, route-scoped validation, typed adapter and
dense requested-row reader, padding/locality guards and enclosing IPC lifetime.

Host checks: 140 existing orchestration tests plus 31 new setup/lifetime tests
passed; six public sparse dispatch tests passed; 38 adapter/reader tests and
eight legacy caller regressions passed. Frozen container job `19989936`
completed with **76 passed, 0 skipped**, including 14 actual CUDA producer and
padding tests, 56 utility/adapter tests and six policy dispatch tests. Source
hashes matched before and after execution. A CUDA IPC shutdown warning exposed
unsafe reuse of PyTorch's one-use exported reference counters; publication is
held until reusable producer-owned descriptors and teardown probes resolve it.
Native losses remain disabled until checkpoints 4a/4b.

Follow-up `19990303`: **217 passed, 0 skipped** (two reusable raw CUDA IPC
tests, 181 orchestration tests, 27 public policy/aggregation tests, seven
wrapper/eligibility tests). Before/after source hashes matched. The corrected
legacy diagnostic measured exported counters **1 → 0 → -1** over two fresh
consumers, confirming the underflow; the raw CUDA replacement passed repeated
fanout, nondefault-stream copies, view lifetime, storage growth and final free
without the shutdown warning. Unpacked sample occurrence IDs now survive
teacher export and aggregation. Final producer/dense-reader integration and
explicit OS process-exit checks were completed in the final frozen run below.

Final frozen job `19990413`: **COMPLETED 0:0**, 5m35s, **340 passed, zero
failures/errors/skips**. The nine suites cover reusable raw CUDA IPC (2),
CPU/Gloo plus CUDA/NCCL primitives/readers (40), actual producer and padding
(14), dense CUDA reader (1), dense reader/adapter (42), process lifetime and
worker recreation including real Ray actors (23), orchestration (184), policy
and aggregation (27), and wrapper/eligibility (7). All nine commands exited 0;
394 source/test/launcher hashes matched before and after. No IPC producer-exit,
counter-underflow or leaked-handle warning occurred in the final run logs.

The actual Ray check confirms OS process exit before producer release; when
exit cannot be confirmed the controller preserves the original error and
retains producer-owned storage. New reusable dense export remains internal and
opt-in; native same-tokenizer activation is reserved for checkpoint 4b. Changed
source/test Ruff and format checks and `git diff --check` pass. Targeted
Pyrefly reports zero errors for all four new modules (existing unrelated
baseline typing issues are not claimed fixed). Independent reviews found no
remaining actionable issues. Native loss numerical and real-model R1–R6
acceptance remains reserved for checkpoints 4a/4b/5.

Publication: signed-off commit `310d4dcfdf0b0433d1d1c15beea9dffd6202aed7`
pushed to `myfork/avenkateshha/xtoken-v6-loss`; remote tip verified.

## Checkpoint 4a — native sparse KD, CE, accuracy and scaling

Implemented after checkpoint 3's verified push. The MCore static/unpacked PP1
wrapper enables the native sparse consumer. Native same-tokenizer activation
remains reserved for checkpoint 4b.

The loss keeps native student rows and computes CE/accuracy with global next-token
labels, real-vocabulary masking and no rank-3 student CP relayout or full-sequence
CE-target gathering. Each native teacher has one invocation-local validated
reader. Common chunks belong to their predictor owner; mismatch chunks belong
to their final predictor owner, with one unconditional differentiable CP prefix
SUM in forward and backward, including empty and zero terms. Requested row tiles
are bounded at 64; common support construction remains on device. The historical
selected-support plus REST objective, forced-label/noise distinctions, M-to-N
handling, teacher-specific full-step denominators and existing loss-mode fallback
semantics are preserved.

Native CE and KD have no extra /CP. The retained legacy v6 consumer keeps /CP;
MCore schedule compensation is unchanged. Detached CE/KD ratios and reported
terms are CP-complete per microbatch and DP replica. Static native weights stay
FP32 even with BF16 forward logits. Actual legacy sparse consumers retain one
shared compatibility view and their fully reconstructed sparse sequence.

Final frozen-source CPU validation passed all four TP/CP grids with distributed
collective diagnostics enabled: native dispatcher 4 passed (143.33 s), mixed
native/legacy dispatcher 4 passed (141.00 s); both processes exited 0. All 11
source/config/test hashes matched before and after. Durable commands/logs and
hashes are recorded in `artifacts/validation/runs/checkpoint4a-cpu-20261010/`. The native suite performs 144 combined loss/gradient
comparisons (12 cases × full B2 plus two B1 microbatches × four grids). It covers
unequal weights and normalizers, fixed/dynamic scaling, vocabulary scaling,
reversed teacher order, zero/empty terms, filtering, consecutive generations,
BF16 input and averaged-logits fallback, and forbids legacy relayout/CE helpers.
The mixed suite performs 96 comparisons across sum/fallback modes, both teacher
orders, fixed/dynamic scaling and identical microbatch groupings, using the exact
CP1 retained legacy consumer plus an independent native teacher oracle. Its
initial test fixture incorrectly CP-sliced the legacy full sparse payload; that
fixture was corrected without a production collective-ordering change.

Additional focused evidence: final vectorized native sparse objective/gradient
tests 20 passed (135.44 s, four CUDA cases deselected),
native CE/accuracy 7 CPU passed (four CUDA cases await the final GPU run), reader
contract/BF16-weight tests 18 passed, and orchestration/full-step-normalizer
regressions 204 passed. FP32 comparisons use rtol 1e-4/atol 1e-5; BF16 combined
gradients require relative L2 and norm error <= 0.02. Both new production modules
pass targeted Pyrefly; the loss-functions file retains exactly its 63 pre-existing
type errors. Changed-file Ruff/format checks pass. Real-model loss, optimizer,
pre-clip parameter-gradient and native CUDA acceptance remain for checkpoint 5.
No larger memory test or production-capacity claim is included.

Publication: signed-off commit `6bf7cb1c616ba026db808d6808158bedd55ed80f`
pushed to `myfork/avenkateshha/xtoken-v6-loss`; remote tip verified.

## Checkpoint 4b — native same-tokenizer KL

Implemented after checkpoint 4a's verified push. Same-tokenizer KL uses the
student's native CP owners and local TP vocabulary. Common-K support preserves
the valid-predictor microbatch/CP maximum; true averaged logits combine frozen
teacher rows before full-real-vocabulary KL. Bounded FP32 forward/backward tiles
avoid a full student vocabulary gather, including when a TP shard is entirely
padding. Both KL directions, temperature squared, separate CE/KD full-step
normalizers and corrected no-extra-CP normalization are retained.

Dense readers and copied rows are local to one loss invocation and reused for
frozen teacher scoring and KD. CE/selection use global next labels and shifted
KD masks; entropy/max-prob preserve unshifted masks, including a valid final
predictor. Training and validation opt eligible same-tokenizer teachers into
reusable dense IPC independently of sparse K. Mixed native/legacy same-tokenizer
true averaging stays entirely on the retained dense route. Native sparse plus
native same, and native same plus legacy dense cross-tokenizer, share one native
CE/accuracy calculation; only the legacy consumer requests a compatibility view.

CPU validation: same-only suite **16 passed**, including **156** actual
adapter/dispatcher comparisons against an independent dense oracle and corrected
contiguous baseline across TP/CP `(1,1)`, `(2,1)`, `(1,2)`, `(2,2)` (100.04 s).
The four CUDA grids are reserved for checkpoint 5. Mixed native same/sparse and
native same/legacy-dense suites **8 passed**, **432** combined loss/gradient
comparisons (183.83 s), with both teacher orders, fixed/dynamic scaling,
common-K/full-vocabulary KL, reverse KL, vocabulary scaling, fallback semantics,
zero weights and empty masks. Each microbatch is compared with its own support
and dynamic ratio; no unsupported common-K partition invariance is claimed.

Controller regressions **200 passed**; twelve selector checks were rerun after
correcting a test mode label. Dense contract/scoring/cache tests **9 passed**,
including CP2 global-score/weight/selected-KL gradient checks. Existing
adapter/contract regressions **45 passed**. These overlapping suites are not a
unique-test total. FP32 loss/gradient checks use rtol 1e-4/atol 1e-5 or tighter;
BF16 same-only gradients require relative L2 and norm error <= 0.02.

Final same helper/test hashes are frozen, and eight source/test hashes were
rechecked unchanged after the mixed run. Changed-file Ruff/format and diff
checks pass; the new helper has zero Pyrefly errors, while loss_functions keeps
its 63 pre-existing errors. Independent code reviews found no actionable issue.
CPU harness fixes only adapt the retained legacy CUDA allocation/gather to Gloo;
production legacy code remains unchanged. Actual MCore optimizer/pre-clip model
gradient acceptance and real CUDA native losses remain for checkpoint 5.

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
  historical run are preserved. Wrapper test passed. Standalone model creation
  hit the provider's APEX fusion default; a follow-up uses the existing exemplar
  setting `gradient_accumulation_fusion=false`. A late edit of the active driver
  also produced a shell EOF; the follow-up launcher is frozen before submission.
- Follow-up `19989617`: **COMPLETED 0:0**, split guards **2 passed**. All four
  pinned models (Llama-3.2-1B, SmolLM2-1.7B, Qwen3-4B, Llama-3.2-3B) loaded
  actual weights and completed finite native TP2/CP2 forwards at B1/T32.
  These checks do not establish KD, backward, optimizer or IPC acceptance.
- Frozen checkpoint-2 CPU/CUDA primitive job `19989749`: **COMPLETED 0:0**,
  **39 passed, 0 skipped**, including all four CUDA/NCCL TP/CP grids; source
  hashes matched before/after. No production loss acceptance is claimed.
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
