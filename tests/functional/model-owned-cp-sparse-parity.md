# Model-owned CP sparse export parity

`model_owned_cp_sparse_parity.py` compares the actual Qwen3.5/3.6 Bridge wrapper
at teacher CP1 and CP2, holding TP, checkpoint, padded inputs, BF16 precision,
temperature, and K fixed. It requires two GPUs at TP1, a complete local Hugging
Face snapshot with MTP weights, and a compatible Megatron worker environment.
A Qwen3.5-4B checkpoint fits this small test: two sequences of 32 tokens.

Full IDs enter the wrapper; hooks check exact zigzag IDs, absolute MRoPE
positions, and CP-local MTP masks at the inner language model. A separate hook
must observe an actual MTP forward. `eval()` does not bypass MTP here.

The existing `SparseTeacherLogitsPostProcessor`, `SparseTeacherStorage`, and
`SparseTeacherRowReader` produce and consume the payload. Dense logits are saved
solely for independent diagnostic comparison, with no production CP relayout.
Wire fields are copied to CPU files; this harness does not test cross-process
CUDA IPC handle opening or lifetime.

## Prepare pinned source archives

Use Bridge `1f8873bb00a8ddf3af811649f0a7efdb2363570c` and MCore
`6a3660905a2736b5670baed1ca5954372937918b`. Preserve existing submodules.
From the repository root, archive into a new run directory:

```bash
export PARITY_RUN=/absolute/path/to/new-parity-run
export BRIDGE_REPO="$PWD/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
export MCORE_REPO="$BRIDGE_REPO/3rdparty/Megatron-LM"
mkdir -p "$PARITY_RUN/runtime/bridge" "$PARITY_RUN/runtime/mcore"
git -C "$BRIDGE_REPO" archive 1f8873bb00a8ddf3af811649f0a7efdb2363570c src pyproject.toml |
  tar -x -C "$PARITY_RUN/runtime/bridge"
git -C "$MCORE_REPO" archive 6a3660905a2736b5670baed1ca5954372937918b megatron pyproject.toml |
  tar -x -C "$PARITY_RUN/runtime/mcore"
uv run --no-sync python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["PARITY_RUN"]) / "runtime"
revisions = {
    "bridge": "1f8873bb00a8ddf3af811649f0a7efdb2363570c",
    "mcore": "6a3660905a2736b5670baed1ca5954372937918b",
}
manifest = {
    name: {
        "source_git_object": revision,
        "file_hashes": {
            str(path.relative_to(root / name)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / name).rglob("*")) if path.is_file()
        },
    }
    for name, revision in revisions.items()
}
(root / "source-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
```

If the nested repository lacks the object, fetch it into an isolated clone and
point `MCORE_REPO` there. Keep the compatible binary runtime (Transformer Engine,
CUDA, GDN kernels, etc.); archives supply source, not compiled dependencies.
The harness verifies all manifest hashes and actual imported package origins.
Wheel version strings are recorded separately because source overlays can differ.

## Run

Use a compatible worker environment; `uv run --no-sync` leaves it unchanged.
`CHECKPOINT` must be an immutable `snapshots/<40-character-revision>` directory.

```bash
export CHECKPOINT=/absolute/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
export PYTHONPATH="$PWD:$PARITY_RUN/runtime/bridge/src:$PARITY_RUN/runtime/mcore:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 CUDA_DEVICE_MAX_CONNECTIONS=1
export UV_PROJECT_ENVIRONMENT=/absolute/path/to/compatible/megatron-worker-venv
HARNESS=tests/functional/model_owned_cp_sparse_parity.py
uv run --no-sync "$HARNESS" prepare --checkpoint "$CHECKPOINT" --output "$PARITY_RUN/fixture.pt"
for CP in 1 2; do
  uv run --no-sync -m torch.distributed.run --standalone --nproc-per-node="$CP" \
    "$HARNESS" capture --checkpoint "$CHECKPOINT" \
    --source-manifest "$PARITY_RUN/runtime/source-provenance.json" \
    --fixture "$PARITY_RUN/fixture.pt" --tp 1 --cp "$CP" --k 32 --temperature 1 \
    --output "$PARITY_RUN/cp$CP" > "$PARITY_RUN/cp$CP.log" 2>&1
done
uv run --no-sync "$HARNESS" compare --cp1 "$PARITY_RUN/cp1" \
  --cp2 "$PARITY_RUN/cp2" --output "$PARITY_RUN/comparison" \
  > "$PARITY_RUN/comparison.log" 2>&1
```

The report records revisions, import paths, checkpoint/weight identities, input
byte digest, exact ownership counts, numeric errors, top-K changes, and student
loss/gradient differences. Numerical failures are accumulated so all comparisons
and backward still run; the report says `FAIL` and the command exits nonzero.
Ownership/source failures stop immediately. Use new output directories for
independent runs.

## Numerical checks and downstream scope

Global row coverage, segments, inner token IDs, forced-label ownership, and
input tensors are checked exactly. Each exported natural top-K support is checked
against its forward's dense logits. CP1/CP2 support changes are counted and only
allowed near both Kth boundaries, within twice the raw-score error budget.

Both full dense logits and the union of exported top-K token scores are checked
separately. Default raw-score tolerance is `atol=0.25, rtol=0.01`: BF16 spacing is 0.125 around
magnitude 16, and CP changes attention/GDN reduction order. LogZ is checked with
`atol=0.08, rtol=0`. These explicit smoke-test budgets are not universal error
bounds; inspect actual errors and tie counts rather than increasing thresholds
to pass. Within each capture, sparse scores and LogZ must match dense FP32
calculations at `atol=1e-5, rtol=1e-6`.

The downstream check runs production `compute_native_sparse_teacher_loss` and
backward through a fixed-seed FP32 `Linear(8, vocabulary)` student at TP1/CP1,
with an identity subtoken table. Features, weights, masks, and alignment are
identical for both teachers. Loss tolerance is `atol=0.002, rtol=0.03`; gradients
must satisfy `L2(delta) <= 1e-5 + 0.15 * L2(reference)`. The norm check allows
small gradient mass to move between near-tied token IDs. This checks native loss
integration with a tiny student, not pretrained-student convergence or mismatch
chains across different tokenizers.

Packing, dense IPC enablement, arbitrary wrapper output layouts, and production
memory testing remain outside scope.

## Observed result (2026-10-11)

Qwen3.5-4B snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, using the pinned
sources above, completed both BF16 captures at TP1, CP1/CP2, K32, temperature1.
All 64 global rows and 128 forced labels aligned exactly, MTP executed on every
rank, and each sparse payload matched its own dense forward at the tighter
within-capture bounds.

The overall numerical result was **FAIL** with the unchanged default bounds:
maximum raw/exported-support score difference was **0.4375**, and maximum LogZ
difference was **0.132677**. One of 4,096 compared exported support entries and
two of 64 LogZ rows exceeded their bounds. Maximum forced-score difference was
0.1875 and passed. Top-K support changed in 33 of 64 rows; all exchanges met the
tie check, with maximum distance from the cutoff of 0.1875. These observations
do not establish general BF16 numerical parity.

The fixed tiny student's native loss/backward checks passed: losses were
0.558725715 and 0.558327734. Gradient difference L2 was `1.65631e-5`, versus
reference L2 `9.46708e-5`, satisfying the documented combined absolute/relative
bound (`1e-5 + 0.15 * reference`). This narrower downstream result does not
override the teacher score and LogZ failures.
