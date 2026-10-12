# Realized-only mismatch validation

Date: 2026-10-12 (America/Los_Angeles). Base HEAD: `f8d1a073f`, branch `avenkateshha/xtoken-v6-loss`.

Run directory: `/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/experiments/realized-only-mismatch/xtoken-distillation/focused-20261012`.

## Results

- Pinned Mingyu manifest: **PASS**, 21 files / 1,324,108 bytes, every SHA-256 matched commit `5acd78e948d2bc9061255e4d0474ed3aa1d9586b`.
- Local single-process native loss suite: **PASS**, 50 passed / 8 deselected (distributed cases excluded for this preliminary check).
- CUDA/NCCL native loss and student gradient grids: **PASS**, 4 passed / 54 deselected, 72.09 seconds. TP/CP `(1,1)`, `(2,1)`, `(1,2)`, `(2,2)`; each checks false and true, masks, empty teachers, zero weights, filtering, BF16, prefix-only ownership and zero denominator.
- Container CPU/Gloo native grids, public dispatcher, setup/controller and mixed-route regressions: **PASS**, 332 passed / 4 CUDA cases deselected, 269.20 seconds. Public dispatcher checks 51 combined loss/gradient comparisons per TP/CP layout. Both CPU and GPU invocation exit codes are 0.
- Repaired artifact-omission integration cases rerun locally: **PASS**, 2 passed / 239 deselected, 125.03 seconds.
- Ruff lint and format: **PASS**, all seven modified Python files.
- Pyrefly for allowlisted `native_sparse_loss.py`: **PASS**, exit 0; zero reported errors, one ignored by repository configuration.
- Both exemplar YAML defaults parse as boolean `false`: **PASS**.
- Independent reference/patch review: **LGTM**, no unresolved findings. Two test-fixture issues (message regex and omitted setup default) fixed before final container CPU collection.
- Final source checksum verification and `git diff --check`: **PASS**. Pre-existing Qwen parity diff is byte-for-byte unchanged; dirty submodules were not modified.

Feature commit: `fc7b3d387` — `feat(xtoken): add realized-only mismatch objective`.
Feature diff: **10 tracked files, 678 insertions, 36 deletions**, excluding session notes and pre-existing changes.

## Runtime and evidence

Slurm job `20050101`, interactive partition, node `pool0-01812`, four visible GPUs: **COMPLETED**, exit `0:0`, elapsed `00:07:39`; allocation released. Both JUnit reports contain zero errors, failures or skips. No model download or production-model training. Launchers, runtime log, JUnit XML, stdout logs, accounting, tested diff and source checksums are in the run directory.

Container: existing `nemo_rl_main_20260803_amd64.sqsh`; actor interpreter `/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python`.

Actual runtime: Python **3.13.13**, PyTorch **2.11.0+cu130**, Transformer Engine **2.15.0+42b84005**. Existing source overlays: Bridge `1f8873bb00a8ddf3af811649f0a7efdb2363570c`, MCore `6a3660905a2736b5670baed1ca5954372937918b`, Lens `b0f977d414b2f89938604a0b7eaa78ee08bc8700`, Transformers 5.12.1. These reuse historical runtime assets; no dependencies or dirty submodules were modified.

**NOT_RUN:** repository-requested Python 3.13.14 / PyTorch 2.13 runtime, which is not the established available environment. Production-model/16-variant historical replay is outside this feature's requested qualification scope and was not run.

Initial local `.venv` attempts were blocked before feature assertions by missing `nemo.lens.groups.SpanRegistry`; using the existing Lens overlay resolved this. The local setup attempt then exposed a test fixture that replaced rather than merged required setup defaults (91 passed before failure); it was corrected. These preliminary failures are not claimed as passing validation.

## Exact commands

Reference verification:

```bash
uv run --no-project --python /usr/bin/python3 session/20261007_sparse_mcore_explainer/fetch_reference.py --verify
```

Preliminary local loss check (existing overlay):

```bash
PYTHONPATH="$PWD:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/experiments/xtoken-smoke/xtoken-distillation/xtoken-offset-rebased-2n100-20260928-092950/runtime/python" \
uv run --no-sync --offline python -m pytest --confcutdir=tests/unit/algorithms/x_token \
  tests/unit/algorithms/x_token/test_native_sparse_loss.py -k 'not cuda and not tp_cp' -q
```

Job launch (uses the run-owned scripts and historical container/source environment):

```bash
sbatch --parsable --job-name=xtoken-realized-mismatch \
  --output=/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/experiments/realized-only-mismatch/xtoken-distillation/focused-20261012/slurm-%j.out \
  /lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/experiments/realized-only-mismatch/xtoken-distillation/focused-20261012/launcher/validation.sub
```

The driver invokes, using `uv run --no-project --python "$ACTOR_PY" python`:

```bash
-m pytest --confcutdir=tests/unit/algorithms/x_token \
  tests/unit/algorithms/x_token/test_native_sparse_loss.py -k cuda -q \
  --junitxml="$REALIZED_VALIDATION_RUN/gpu-junit.xml"

# CUDA_VISIBLE_DEVICES="" for CPU/Gloo coverage:
-m pytest --confcutdir=tests/unit/algorithms/x_token \
  tests/unit/algorithms/x_token/test_native_sparse_loss.py \
  tests/unit/algorithms/x_token/test_native_sparse_dispatch.py \
  tests/unit/algorithms/x_token/test_xtoken_off_policy_distillation.py \
  tests/unit/algorithms/x_token/test_native_sparse_legacy_mix.py \
  tests/unit/algorithms/x_token/test_native_same_mixed.py \
  tests/unit/algorithms/x_token/test_native_loss_contract.py \
  -k 'not cuda' -q --junitxml="$REALIZED_VALIDATION_RUN/cpu-junit.xml"
```

Quality checks: `uv run --no-sync --offline ruff check` and `ruff format --check` against the seven Python files below; `uv run --no-sync --offline pyrefly check nemo_rl/algorithms/x_token/native_sparse_loss.py`; `git diff --check`.

- `nemo_rl/algorithms/loss/loss_functions.py`
- `nemo_rl/algorithms/x_token/native_sparse_loss.py`
- `nemo_rl/algorithms/xtoken_off_policy_distillation.py`
- `tests/unit/algorithms/x_token/native_sparse_fixtures.py`
- `tests/unit/algorithms/x_token/test_native_sparse_loss.py`
- `tests/unit/algorithms/x_token/test_native_sparse_dispatch.py`
- `tests/unit/algorithms/x_token/test_xtoken_off_policy_distillation.py`

All seven file hashes matched the run's `source-sha256.txt` again before committing.
