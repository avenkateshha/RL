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

Publication: signed-off commit `fcc943c5eab4690a4faa1892d80b9d56cb8d1564`
pushed to `myfork/avenkateshha/xtoken-v6-loss`; remote tip verified.

## Checkpoint 5 — small CUDA and real-model acceptance

Complete after checkpoint 4b's verified push. All 16 agreed R1–R6 small model
variants pass, including three updates, final evaluation and independent
accumulated pre-clip parameter-gradient replay. The
[acceptance summary](artifacts/validation/acceptance-summary.json) records
1,134 numerical reports and 226 exact initial replay forwards.
Maximum gradient relative L2 is 1.1994% and norm error is
0.1290%, below the unchanged 2% limits. Larger memory tests and a
K/sequence-size ladder remain excluded.

Frozen final focused job `19991092` completed **0:0**, **348 passed, 0 skipped**:
55 native math (including all 12 CUDA TP/CP grids), 85 dispatch, 7 wrapper,
200 controller and 1 actual dense CUDA IPC test. All 415 source and run-owned
file checks passed before/after; no CUDA IPC shutdown warning was found.

Real-model reference pilot `19991170` completed **0:0**, comparing both
microbatches at TP1/CP1/DP1 with the independent sparse objective. Maximum
dlogit relative L2 error was **3.62e-5**; maximum model-parameter VJP relative
L2 error was **0.008299** and norm error **0.001517**, below the predeclared
0.02 limits. Each microbatch covered 98 parameter tensors / 1,235,814,400
elements. This pilot checks bare-model VJPs, not the controller's accumulated
pre-clip optimizer gradients; it does not replace the matrix's optimizer proof.
Two preserved failed pilots (`19991091`, `19991125`) exposed artifact-only
device typing and Transformer Engine graph-reuse errors. The passing pilot
uses a fresh reference forward with restored RNG and matching initial logits.

Final review found an empty-common-K edge: summing finite FP16 logits before
multiplying by zero can overflow. The zero-loss branch now reduces in FP32;
its targeted CPU regression passed and its CUDA regression passed in job
`19991376`. This concerns same-tokenizer `vocab_topk=0`, a separate setting
from dense export's `teacher_topk_ipc_k=0`.

R1 TP1/CP1/DP1 controller job `19991383` completed **0:0**: three actual
optimizer updates, final evaluation, 14 microbatch numerical checks (six
training and eight evaluation), and no CUDA IPC shutdown warning. Its actual
first-step optimizer capture records all 98 parameter tensors / 1,235,814,400
elements after `prepare_grads`, before clipping or updating. Independent replay
`19991459` completed **0:0**: both initial forwards match exactly, every owned
parameter range is covered, gradient relative L2 error is **0.00965641** and
norm error **0.000669906**. The captured norm **39.30782427** matches the
controller's reported pre-clip norm. Both microbatches retain the same full-step
CE/KD token denominator 64 and teacher chunk denominator 78. Source and launcher
hashes passed before/after. Distributed CP/DP optimizer proof is recorded in the
subsequent matrix cases.

Standalone R1 TP2/CP2/DP1 reference `19991376` completed **0:0**: four rank
reports, two microbatches each, maximum dlogit relative L2 **0.0001241**, model
parameter relative L2 **0.009124**, and norm error **0.000993**. A read-only
`nvidia-smi` diagnostic ran in a no-GPU cgroup and returned 6; the corrected
diagnostic, actual tests, and overall batch exited successfully. This scope
remains bare-model VJPs; controller optimizer evidence is recorded separately.

Bundled R1 TP1/CP2/DP1 controller/replay `19991553` completed **0:0**.
Both replay ranks cover the full owned parameter ranges; accumulated optimizer
gradient relative L2 error is **0.00991114**, norm error **0.000244704**. Root
independently recomputed all **399** source/launcher hashes and the configuration
hash after completion. This pilot's shell EXIT trap did not itself propagate a
verification failure; all actual hashes matched, and future launchers explicitly
preserve workload failures and propagate verification failures.

R4 MCore-Qwen/DTensor-SmolLM2 controller `19991476` completed **0:0** at
TP2/CP2/DP2: three updates, final evaluation, 80 microbatch numerical checks,
and maximum training dlogit relative L2 **0.00303126**. Companion optimizer
replay `19991626` completed **0:0** on all eight ranks: two microbatches each,
exact owned-range coverage, accumulated gradient relative L2 **0.0104928**
and norm error **0.0000162928**. All frozen source/config/launcher hashes pass.
These runs did not retain raw Ray worker teardown stderr; the numerical result
does not close the legacy Torch IPC lifetime audit. The reversed-backend case
records actual export counters and worker logs to investigate that separately.

R3 first attempt `19991885` retained a failing dlogit comparison (**0.0758815**
relative L2). A bounded probe in allocation `19992100` reproduced the exact
cause: on the second DP replica, four BF16 teacher vocabulary scores tie at
the common-K cutoff, and CPU/CUDA `topk` select different tied IDs. The reference
now computes importance independently on CPU, uses the controller device only
for discrete selection, validates the selected support against the cutoff, and
keeps independent CPU loss/gradient math. Production code and the 2% tolerance
are unchanged. The first corrected attempt `19992163` passed the
formerly failing microbatch on every rank (rank6 relative L2 **0.0000251599**),
then stopped on **EDQUOT** while writing a validation report. The shared
project's inode quota was exhausted; user and byte quotas had room. Its final
source hashes passed. The failed run remains incomplete. Completed task logs
were archived and verified before their expanded duplicates were removed.

The R6 reference extension passed **24** CPU dense/mixed dispatcher scalar,
gradient and metric checks, **48** same-tokenizer oracle checks, dense consumer
row/identity checks, and capture plumbing including retained failure operands.
All reference-source hashes matched before/after. Dense legacy v6 is the
retained CP1 implementation in both paths; these checks establish aggregation
and reference plumbing, not independent dense-v6 mathematics or GPU acceptance.

Actual mixed-backend teardown exposed a Ray callback-binding issue: the MCore
driver cannot import the DTensor actor's optional `torchao` dependency, so Ray
uses placeholder method metadata accepting keyword arguments only. The worker
itself has a healthy interpreter. `record_worker_processes` now passes its
callback as `fn=...`; the exact Ray metadata reproduction and **17** complete
process-lifetime tests pass, including real Ray actors and OS-exit verification.
The intended-container repetition passes all **17 tests, 0 skipped** in job
`19992724`; mixed-backend teardown remains required.

R1's final TP2/CP2/DP1 case `19992724` completed **0:0** with all 56 numerical
reports and four-rank accumulated optimizer replay passing: relative L2
**0.0106910**, norm error **0.000864822**. This completes R1's four TP/CP grids.
Corrected R3 fixed-scaling retry `19992723` completed **0:0** with all 80 reports
and eight-rank optimizer replay passing: relative L2 **0.0110410**, norm error
**0.000196053**. Both use two microbatches per rank, cover every optimizer-owned
parameter range, match pinned initial forward logits exactly, and avoid all
prohibited native layout/reconstruction calls. Frozen hashes and actual worker
OS exits pass. Every archived log member was read back and verified, with no
IPC warning matches. The earlier failed R3 attempts remain recorded separately.

The remaining named legacy regressions pass **31 tests**, with two explicitly
optional external-upstream checks skipped. They cover chat/EOT, global chunk
normalization, matrix-free and retained dense/sparse v6 paths, including bounded
TP2/CP2 CE/v6 gradient checks. All 388 recorded source hashes are unchanged.
Public recipe syntax and dry-run checks pass; its earlier schema validation is
preserved with explicit provenance. This does not claim that the unpinned public
recipe was executed verbatim in the pinned acceptance runtime.

R2 two distinct MCore teachers at TP2/CP2/DP2 (`19992929`) completed **0:0**:
80 numerical reports and eight-rank accumulated optimizer replay pass, with
relative L2 **0.0104982** and norm error **0.000454316**. An auxiliary no-GPU
container-import probe was cancelled during startup as the main job ended;
it executed no Python and establishes no additional runtime result.

The corrected observer in R4 inverse retry `19992921` established a real legacy
DTensor sparse lifetime failure despite three updates, evaluation and optimizer
replay passing (relative L2 **0.0107180**, norm error **0.000595532**). Every
selected TP0 Torch counter starts at +1 and reaches **-3** after four independent
student TP/CP consumers; unused TP1 exports remain **+1**. All eight worker
observations are present, with no read errors. Actual cleanup/OS exits and source
hashes pass, but this run is explicitly `PASS_NUMERICAL_LEGACY_IPC_LIFETIME_FAILURE`.
After both jobs completed and source checks passed, the freeze was released for
a narrow reusable-buffer compatibility correction; affected acceptance was held
until the fix and reruns recorded below.

The targeted correction makes DTensor sparse TP0 ranks export whole reusable
producer-owned slabs with explicit sample/slice metadata. Other TP ranks finish
the same forward collectives and return without concatenating, allocating or
exporting IPC storage. The legacy reader opens each slab once per invocation,
reconstructs the same full sparse sequence and still accepts old Torch tuples.
Both independent code reviews found no actionable issue. Host checks pass
11 focused cases and 13 retained tuple/zero-copy regressions. The actual cached
DTensor interpreter then passes **all 12 focused tests, 0 skipped**, including
two separate CUDA consumers, repeated reuse, growth and final release, in the
prelude to `19993411`. That corrected inverse R4 run completed **0:0** with
80 numerical reports, eight-rank accumulated optimizer relative L2
**0.0107179700** and norm error **0.0005955324**, full owned-range coverage,
and all 387 source plus 18 run-owned hashes unchanged. All eight workers and
five generations satisfy reusable/nonpublisher coverage without counter
anomalies; backend cleanup and actual OS process exits pass.

The first R6 dense run, `19993412`, failed before producing numerical reports:
the reference incorrectly applied the native position-zero restriction to the
retained dense objective. Its complete eight-worker observation also exposed
the dense Torch handle fanout counter defect (**+1 to -3**). This run remains
failed, with source hashes verified; neither mathematical nor optimizer
acceptance is inferred. The reference guard now applies only to native sparse
terms. All 48 dense reference cases pass, with nonzero KD and gradient changes
in every one of the 24 position-zero on/off pairs; native rejection is retained.
The scoped dense transport correction keeps legacy row geometry and loss math,
opens reusable raw handles in the existing reader, and enables compatible static
MCore teacher/student pairs through the existing internal opt-in. Its 33 host
checks and 205 controller regressions pass. R6 retry `19993795` preserves the
objective, models, data and grouping while changing only the log destination.
It completed **0:0**: all 13 focused tests ran without skips, including
both CUDA ownership cases; all 80 numerical reports and eight-rank optimizer
replay pass (relative L2 **0.0103154555**, norm error **0.000263456**). All eight
dense producers cover five generations and final release using reusable storage,
with no counter anomaly. Cleanup, OS exits, all 387 source plus 18 run-owned
hashes, and archived-log verification pass.

The corrected forward R4 run, `19993796`, also completed **0:0**, with all 80
numerical reports and eight-rank accumulated optimizer replay passing (relative
L2 **0.0104927611**, norm error **0.0000162928**). All eight DTensor workers,
five generations and final release have complete raw-storage/nonpublisher
coverage without Torch counters. Cleanup, OS exits and frozen hashes pass.
Both supported mixed-backend directions now have complete acceptance evidence.
After releasing that freeze, final import sorting in the two worker modules
passed Ruff, import sorting, formatting and diff checks across all 12 changed
production/test files. Repository-wide Pyrefly reports 11 missing/incompatible
dependency imports in four files unchanged since the initial commit; the changed
allowlisted native same-tokenizer module passes its scoped type check.

The required two-node R2 controller `19993994` completed **0:0** in 8m08s at
TP2/CP2/DP4, global batch 16, K64 and two student microbatches per DP replica.
All 128 reports pass (96 training, 32 evaluation), with no native student
relayout or full sparse reconstruction. Independent archive inspection proves
matching student/two-teacher rank placement across both nodes and all 48 worker
OS exits. The Ray worker service step was cancelled after successful head exit;
the driver and overall job exited zero. All 1,084 archive members and controller
source/launcher/config hashes verify.

Separate 16-rank optimizer replay `19994298` completed **0:0** in 3m18s. All 32
initial forwards match exactly. Each TP shard covers all 98 parameter tensors /
617,940,992 elements exactly once across eight CP+DP owners. The accumulated
pre-clip gradient relative L2 is **0.00980961782015** and relative norm error is
**0.00057112858812**, below the unchanged 2% thresholds. Its own 383 source and
13 run-owned hashes verify independently. This closes the small two-node
correctness/topology requirement; it makes no production-size memory claim.

True same-tokenizer averaged-logits R5 (`19994428`) completed **0:0** in 7m05s:
all 80 reports use both native dense teachers and no forbidden student layout
operations. Actual pinned Llama 1B and 3B observations are distinct. The first
microbatch's independent full-vocabulary KL of their 0.3/0.7 averaged logits is
**0.0423298851**, distinct from weighted individual KL (**0.0627468310**) and
common-64 KL (**0.0226116050**). All eight optimizer replay ranks pass, with
relative L2 **0.01118162572** and norm error **0.0007001578373**. Capture, source,
archive and all 24 worker placement/exit checks pass.

Dynamic mixed sparse/same-tokenizer R3 (`19994429`) also passes all 80 numerical
reports and accumulated optimizer replay, with relative L2 **0.0118200284** and
norm error **0.000572755** under the unchanged 2% thresholds. The larger actual
pre-clip norm (274.29) is included in that comparison. Frozen hashes verify.
Its independent capture audit confirms first-step ratios **14.40337/12.24592**
for DP0 and **20.11620/9.74387** for DP1, shared across each CP group. They differ
from the counterfactual whole-step ratio **13.89826**. Reproducible commands and
capture/source hashes for this audit and the true-average contrast are retained
under the corresponding independent-audit run directories.

Fixed same-tokenizer R5 (`19994577`) completed **0:0** in 6m31s. All 80
numerical reports and eight-rank optimizer replay pass, with relative L2
**0.0111510960** and norm error **0.0011863382**. Native layout guards,
complete parameter ownership, frozen hashes and actual worker exits pass.

Preserved DTensor sparse R6 (`19994584`) completed **0:0** in 7m21s. All 80
numerical reports and eight-rank optimizer replay pass, with relative L2
**0.00999389515** and norm error **0.0007103212**. All eight producer workers
have complete reusable-storage/nonpublisher lifetime coverage; frozen source,
archive hashes and actual OS exits verify.

Dynamic same-tokenizer R5 (`19994716`) completed **0:0**. All 80
numerical reports and eight-rank optimizer replay pass, with relative L2
**0.01199397652** and norm error **0.00128972732**.
Frozen source/config/launcher checks, native layout guards, archive verification
and actual worker OS exits pass.

Reversed-teacher R2 (`19994688`) completed **0:0**. All 80
numerical reports and eight-rank optimizer replay pass, with relative L2
**0.01049818003** and norm error **0.0004543163047**.
Frozen source/config/launcher checks, native layout guards, archive verification
and actual worker OS exits pass.

The final teacher-order audit verifies equal configurations after sorting
teachers and excluding the log directory, all three identical student logical
batches, and all 80 rank/item/model-normalizer reports. Across three updates,
maximum combined-loss difference is **0** and relative pre-clip norm
difference is **0**, within the original tolerances. No additional GPU
work was required. This closes all 16 variants in the agreed matrix.

Final Ruff, import ordering and formatting checks pass on all **51** changed
Python files, validation helpers and fixture scripts, with byte hashes unchanged
across the checks. Final publication selects small reproducibility artifacts;
raw logs, tensors, model/table binaries and dependency snapshots remain local.
Historical Python launchers retain exact bytes in a verified restore package.

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
- All 16 agreed real-model R1–R6 variants pass. The recipe index identifies
  accepted runs and preserves incomplete/failed attempts; each result states
  its own proof scope. CPU-only checks remain distinct from CUDA/model proof.

## Source contract checked

Local MCore commit `002255075c3728fded9a2e435677840b08560d55`:
`pipeline_parallel/schedules.py` multiplies a two-result loss by CP and divides
by the microbatch count. `distributed/distributed_data_parallel.py` uses unit
gradient scaling when `calculate_per_token_loss=True`; NeMo-RL setup sets this
and disables `average_in_collective`. Actual container imports are recorded
separately and must match the source contract before GPU acceptance. The same
schedule and unit-gradient-scaling contracts were verified in committed MCore
`6a366090` (`schedules.py:346-347`, `distributed_data_parallel.py:250-255`).
