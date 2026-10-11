# Small correctness-test artifacts

This directory records the small correctness and integration validation described in the [implementation checkpoint record](../../checkpoints.md). Larger memory tests, a staged K ladder, and a production-size VRAM-fit gate are excluded.

## Completed acceptance

All **16 variants pass**: **1,134 numerical reports**
(678 training and 456 evaluation),
three optimizer updates and final evaluation per variant, and
226 exact initial replay forwards. Every accumulated
pre-clip gradient comparison covers all optimizer-owned parameter ranges and
uses two microbatches per DP replica. Maximum gradient relative L2 is
**1.1994%** and relative norm error is
**0.1290%**, below the unchanged 2% limits.
Native layout guards and required legacy IPC lifetime checks pass.

The machine-readable [acceptance summary](acceptance-summary.json) links each
accepted result and records its configuration/result hashes.
`summarize_acceptance.py` reproduces it from the retained evidence, including a
read-only adapter for the first R1 run's historical result schema. The original
R1 source-verification log is retained locally and hashed in the summary and raw
log inventory. The [recipe index](runs/recipe-index.json) preserves failed and
superseded attempts alongside the accepted runs.

| Variant | Controller job | TP/CP/DP | Gradient relative L2 | Norm error |
| --- | --- | --- | --- | --- |
| R1-tp1cp1 | 19991383 | 1/1/1 | 0.9656% | 0.0670% |
| R1-tp2cp1 | 19992100 | 2/1/1 | 1.0260% | 0.0140% |
| R1-tp1cp2 | 19991553 | 1/2/1 | 0.9911% | 0.0245% |
| R1-tp2cp2 | 19992724 | 2/2/1 | 1.0691% | 0.0865% |
| R2-tp2cp2 | 19992929 | 2/2/2 | 1.0498% | 0.0454% |
| R2-tp2cp2-reversed | 19994688 | 2/2/2 | 1.0498% | 0.0454% |
| R2-2n-tp2cp2 | 19993994 | 2/2/4 | 0.9810% | 0.0571% |
| R3-fixed | 19992723 | 2/2/2 | 1.1041% | 0.0196% |
| R3-dynamic | 19994429 | 2/2/2 | 1.1820% | 0.0573% |
| R4-mcore-dtensor | 19993796 | 2/2/2 | 1.0493% | 0.0016% |
| R4-dtensor-mcore | 19993411 | 2/2/2 | 1.0718% | 0.0596% |
| R5-fixed | 19994577 | 2/2/2 | 1.1151% | 0.1186% |
| R5-dynamic | 19994716 | 2/2/2 | 1.1994% | 0.1290% |
| R5-averaged | 19994428 | 2/2/2 | 1.1182% | 0.0700% |
| R6-dense-cross | 19993795 | 2/2/2 | 1.0315% | 0.0263% |
| R6-dtensor-sparse | 19994584 | 2/2/2 | 0.9994% | 0.0710% |

The reversed-order comparison preserves all three student batches and all 80
rank/item/per-model normalization reports. Maximum loss difference is
0; maximum relative pre-clip norm difference is 0.
See [order-comparison evidence](runs/R2-teacher-order-comparison-20261010/results.json).
The two-node case uses 16 GPUs at TP2/CP2/DP4, global batch 16 and K64; its
separate 16-rank optimizer replay is job19994298. Larger memory tests remain
excluded, and no production-size memory fit is claimed.

Ruff, import ordering and formatting pass for all 51 changed Python files,
validation helpers and fixture scripts. The configured repository-wide Pyrefly
check retains 11 dependency-import errors in four unchanged files; the changed
allowlisted native same-tokenizer module passes its scoped check. See
`runs/checkpoint5-publication-quality-20261010/results.json` and
`runs/checkpoint5-types-20261010/results.json`.

## Fixtures and recipes

- `fixture_manifest.json` pins the four model snapshots, tokenizer files, corpus, and table hashes. The offline table preparation passed: both Llama↔SmolLM2 directions were built, and fresh Llama↔Qwen tables matched every historical table row exactly, including special-role mapping.
- `runs/fixture-preparation-20261010/` records the builder options, hashes, interpreter, table comparisons, and production text collator output for all 16 fixed samples. `collated_corpus.json` contains IDs, masks, and independent Qwen/SmolLM2 alignments; same-tokenizer rows reuse the student IDs.
- `prepare_recipes.py` materializes small R1–R6 variants. All 16 prepared configurations passed `MasterConfig` validation, recorded in `runs/recipe-schema-validation.json`. Each uses at most 256 tokens, two student microbatches per DP replica, three updates and final evaluation. Seeded epoch shuffling changes microbatch contents while retaining reproducibility. R2 includes the two-node/16-GPU TP2/CP2/DP4 case at global batch16 and K64. `runs/recipe-index.json` identifies each accepted run and preserves every earlier attempt's outcome.
- `prepare_launchers.py` creates new run-owned `launcher/` files by adapting successful job19467093's submission/environment/Ray/container workflow. The historical run is preserved. Backend-specific public actor-registry overrides select the cached MCore or DTensor interpreter, including mixed-backend cases. Scripts and imports must be exercised; generation alone does not certify runtime compatibility.

## Runtime evidence

- Job19989358, `runs/runtime-preflight-20261010/`: cached base `/opt/nemo_rl_venv` has Torch2.11/CUDA13 and eight H100s, but lacks MCore and Lens. The cached MCore actor venv plus pinned Lens imports MCore, but its older source cannot satisfy the current worker's `FullyShardedDataParallelV1/V2` imports.
- Job19989442, `runs/runtime-preflight-current-20261010/`: using the user's current dependency checkouts still fails that worker import. The four pinned configs convert to finalized TP2/CP2/PP1 model providers. This is configuration conversion, without weights or model forwards.
- Job19989499, `runs/runtime-preflight-pinned-20261010/`: isolated source snapshots of the root HEAD's committed Bridge1f8873bb and nested MCore6a366090 resolve worker imports; Lens `SpanRegistry` and all required imports pass. Archive and file fingerprints are published in `pinned-runtime-source-provenance.json`, an exact-byte copy of the run’s `runtime/source-provenance.json`; user submodule checkouts are untouched. All four TP/CP CPU-gloo normalization cases and the schedule-compensation wrapper pass. The initial standalone model construction used the provider's fusion default and stopped because APEX fusion is unavailable; the production exemplar already disables that option. A late edit to the running shell also caused an EOF failure, so the batch exit is not an overall pass. Successful subtest logs retain their individual evidence.
- Job19989617, `runs/runtime-model-smoke-20261010/`: frozen follow-up uses the exemplar's `gradient_accumulation_fusion=false`, checks the split guards with the repository-required `--mcore-only` option, and attempts actual pinned-weight native TP2/CP2 forwards one model at a time, batch1/length32. Completed0:0 in6m24s: the2 split guards and all4 actual pinned-weight native forwards pass. This resolves the model/runtime prerequisite; this job does not establish KD/backward integration.

The run-owned pytest target and dependency-source snapshots are ignored build artifacts. Their source revisions, archive hashes, and dependency versions are recorded. The cached environment uses Python3.13.13/Torch2.11/TE2.15, while the repository currently requests Python3.13.14/Torch2.13; this deviation remains explicit even when a focused check passes. Installed MCore/Bridge package metadata describes the cached distributions; imported source paths and pinned git objects identify the actual Python implementation.

`runtime-padding-audit.json` reads the preserved runtime metadata and resolved
configs, records their hashes, and clarifies historical generic SmolLM2 padding
notes against each run's actual teachers. Executed configs and runtime files are
unchanged. Future recipe generation derives those notes from the configured
teachers.

## Acceptance and reporting

Each case stores its resolved config, command, runtime/fixture manifest, predetermined tolerances, and `PASS`, `FAIL`, or `NOT_RUN` result. CPU FP32 comparisons use `rtol=atol=1e-4`; planned BF16 GPU comparisons use loss `rtol=0.02, atol=0.005`, gradient relative L2 ≤0.02 and relative norm error ≤0.02. Compare pre-clip gradients using the same support, full-step denominators and microbatch/DP grouping. A successful optimizer update alone does not establish gradient parity. Passing imports or model forward does not establish sparse IPC, backward, no-relayout, multi-teacher, or final R1–R6 acceptance.

Submissions are authorized under the user's prompt. No launch-jobs skill is used. Submit only the agreed small cases using their new launcher; do not rerun historical submit scripts or restore excluded memory work.

The immutable checkpoint-2 primitive run `19989749` completed with exit `0:0`: **39 passed, 0 skipped** in 60.19 seconds, including CUDA/NCCL TP/CP `(1,1)`, `(2,1)`, `(1,2)`, `(2,2)` value and pre-clip gradient checks. Its source manifest passed before and after pytest. Evidence: `runs/checkpoint2-primitives-20261010/`. These are primitive checks; CUDA IPC producer/consumer and complete model integration remain later checks.

Stage3 job `19989936` passed all **76 tests, 0 skipped** (14 producer/padding, 56 guards/adapter/reader, 6 public dispatch), with source hashes verified before and after. A CUDA IPC producer-lifetime warning appeared at pytest shutdown and was investigated by the later reusable raw IPC correction; see `runs/checkpoint3-integration-20261010/results.json` and the follow-up evidence below.

The corrected legacy IPC diagnostic in `reusable-ipc-early-20261010` confirms the one-use Torch export counter changes `1 → 0 → -1` under two fresh consumers and emits the lifetime warning. The replacement raw reusable CUDA transport passes its 2 GPU/metadata tests without that warning; all 217 focused tests in job19990303 pass. The final producer/dense/Ray teardown batch is recorded below. The existing R6 packed adapter/lockstep CPU fixtures pass14 tests.

Real-model numerical harness sources are `model_oracle.py` and `real_model_reference.py`; `prepare_reference_launcher.py` freezes new run-owned copies. The generalized oracle passes four fixture loss/gradient self-checks at `rtol=1e-10, atol=1e-12`. Executed cases record their evidence in each run’s `results.json`; a prepared launcher alone establishes no acceptance. R1 holds DP1 while TP/CP varies; native sparse recipes use the supported position-zero-off, mismatch alpha0/beta1 objective, while the dense R6 recipe retains its legacy objective.


Stage3 final job **19990413 completed0:0 in5m35s** with **340 passed,0 skipped**: raw transport2, distributed primitive/reader40, actual producer/padding14, dense CUDA1, dense reader/adapter42, process lifetime/recreate23, orchestration184, public policy/aggregation27 and wrapper7. Source fingerprints passed before and after all commands. None of the final logs contains a CUDA IPC producer-terminated, refcount-underflow or leak warning. The real Ray check captures two live actor process identities, kills the actors and verifies OS process disappearance before producer release. Evidence: `runs/checkpoint3-final-20261010/`. This closes focused Stage3 lifetime integration; it does not mark any R1–R6 actual-model case complete.

Mixed-backend numerical instrumentation is artifact-only: `controller_reference.py` captures full teacher observations inside the actual MCore/DTensor postprocessors and checks the actual controller's loss/dlogits against `model_oracle.py`; `reference_sitecustomize.py` installs it only when a run explicitly enables the capture environment. `optimizer_gradient_capture.py` observes the actual distributed optimizer immediately after `prepare_grads()` and before clipping/update. `replay_controller_gradients.py` replays every microbatch in that first optimizer step through the pinned initial student, accumulates independent VJPs in the actual per-parameter buffer dtype, sums CP+DP contributions using the already-global denominators, verifies complete owned-range coverage, and compares every owned prepared gradient. Each replay forward must match the saved actual logits. Both actual and reference gradients use the exact captured teacher observations; no teacher model is substituted. Standalone `real_model_reference.py` results are separately labeled per-microbatch bare-MCore VJP evidence and do not establish the optimizer accumulation contract. Actual run results identify which proof completed.

For mixed R4 references, the native teacher uses the independent oracle; the preserved DTensor sparse term uses the retained CP1 implementation on the exact captured full-sequence sparse tuple. This specifically checks TP/CP and transport compatibility without substituting the native forced-sidecar objective for legacy semantics. Legacy sparse rows remain full-sequence on every CP rank; only the production alignment IDs use local contiguous windows.

R6 dense-cross references likewise use the retained CP1 dense-v6 objective with independently captured full teacher observations, independent CE, and independent term aggregation. Before reference evaluation, the harness compares the actual reconstructed consumer rows exactly with the corresponding contiguous CP window and real vocabulary of those observations. CPU artifact selfchecks cover 48 dense/mixed scalar, gradient, and metric cases (including 24 nonzero position-zero comparisons), native-cross rejection guards, 48 native same-tokenizer cases, row/identity validation, and first-step capture boundaries. These checks validate the reference harness; they do not establish GPU or optimizer acceptance.

Common-K selection has discrete cutoff ties in the actual BF16 teacher outputs. In R3's first attempt, independently computed CPU importance selected token602 while CUDA selected token26 from four equally important columns at the final slot. The bounded GPU probe reproduces that difference. The reference now computes full-microbatch importance independently, performs only the discrete `topk` on the controller's device, validates cardinality and cutoff membership, and computes loss/gradients independently on CPU. Production support selection and numerical tolerances are unchanged. Failed gradient comparisons retain exact operands and a `FAIL` report before raising.

`summarize_matrix_run.py` verifies scheduler completion, every rank's training/evaluation report, complete optimizer-range coverage, the predetermined error limits, and frozen inputs. Legacy cross-tokenizer acceptance additionally requires complete counter observations and verified worker identities, backend cleanup and OS exits. Negative or outstanding counters retain a numerical pass with an explicit lifetime failure; missing evidence retains pending status. It scans retained raw logs but explicitly does not infer IPC counter correctness from absent warnings. Standalone replay `runtime.json` files are immutable inputs and retain preparation-time status; their `results.json` files carry completed outcomes. The helper prefers each standalone replay's own source manifest over inherited controller provenance.


DTensor runtime follow-up **19990683 completed0:0**: the actual cached DTensor-V2 interpreter imports current worker/Automodel APIs and `SpanRegistry`; both pinned Qwen3-4B and SmolLM2-1.7B workers load real weights and execute TP2/CP2/DP1 sparse exports. Four separate GPU consumers per model read and close the actual descriptors, then producers release after join; no CUDA IPC lifetime warning appears. `runs/runtime-dtensor-consumers-20261010/` records all8 producer and8 consumer results, source/API paths, observation hashes and both source checks. The preceding job19990609 failed only when the test fixture attempted a same-process IPC open after both real forwards/exports succeeded; that failed run is preserved. This runtime proof uses one fresh consumer per descriptor and does not certify the legacy repeated/fanout contract or R4 combined loss.


`controller_teardown.py` retains successfully initialized policies until the ordinary controller has returned, then records live process identities, invokes original backend cleanup, and verifies OS process exit for the student before teachers. A node-pinned task archives raw actor logs from each live Ray node after confirmed exits. Archive and member hashes are verified without extracting the files; a shared project inode-quota failure motivated this storage change. Earlier completed raw-log trees were archived and verified before their expanded copies were removed. Publication retains member manifests and the local archive inventory.

`legacy_ipc_counter_observer.py` independently reads existing Torch export counters before producer reuse/release, without opening CUDA handles or adding consumers. The hook patches Ray's copied actor methods and method metadata; a plain implementation-class patch proved insufficient and that failed observation is retained. Coverage verification requires every expected worker and every export/boundary pair, including final release. Actor termination and warning-free stderr do not establish counter correctness; the counter observation summary reports that evidence separately. The larger memory tests remain excluded.

## Restoring frozen Python launchers

Run-owned Python files are historical execution evidence, including the exact
versions used in failed attempts. Their bytes are published as non-executable
`frozen-python/blobs/<sha256>.py.txt` data. The package manifest records every
original repository-relative path, SHA-256, byte count, and permission mode;
its package digest covers that entire mapping. Identical snapshots share one
blob. Original local run files remain unchanged, and their existing source
manifests and commands continue to refer to the original paths. Current helper
masters remain ordinary, linted Python; their present contents do not replace
the frozen execution inputs.

After checkout, restore the original Python paths before using a recorded
launcher or source verifier:

```bash
uv run --no-sync --offline python \
  session/20261007_sparse_mcore_explainer/artifacts/validation/package_frozen_python.py \
  restore \
  --package session/20261007_sparse_mcore_explainer/artifacts/validation/frozen-python \
  --repo-root "$PWD"
```

Restoration validates the whole package before writing. Existing destinations
must already match both bytes and mode; differing files, traversal paths, and
symlink escapes are rejected. Restoration never executes or submits a launcher.
The recorded runtime, model, table, and filesystem prerequisites still apply.

Publication refresh uses `prepare_publication_allowlist.py`, then the package
helper's `build` subcommand with `--allowlist publication-allowlist.json`,
`--output frozen-python`, and `--repo-root` pointing at the repository, followed
by another allowlist refresh. These paths are relative to this directory when
running the refresh there. `frozen_python_sources` retains the original selected
source inventory; only the package manifest and its referenced blobs are
published in place of those run-owned `.py` files. No repository-wide lint
exemption is required. `verify_frozen_python_package.py` exercises byte/mode
roundtrip, idempotence, corruption and unsafe-path rejection in temporary trees.
