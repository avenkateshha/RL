"""Small actual Qwen3.5/3.6 wrapper sparse-export parity harness.

Run ``prepare``, then separate torchrun ``capture --cp 1`` / ``--cp 2`` jobs,
then ``compare``. See model-owned-cp-sparse-parity.md for pinned-runtime setup.
Only this diagnostic gathers dense logits; production export remains native CP.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from nemo_rl.algorithms.x_token.sparse_teacher import (
    SparseTeacherIPC,
    SparseTeacherRowReader,
)
from nemo_rl.distributed.selected_logprobs import cp_native_global_positions

BRIDGE_REVISION = "1f8873bb00a8ddf3af811649f0a7efdb2363570c"
MCORE_REVISION = "6a3660905a2736b5670baed1ca5954372937918b"


def tensor_digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()


def checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    """Require an immutable HF snapshot, recording resolved weight blob identities."""
    checkpoint = checkpoint.resolve()
    assert len(checkpoint.name) == 40 and checkpoint.parent.name == "snapshots", (
        "Use a local HF snapshots/<40-character-revision> directory."
    )
    shards = sorted(checkpoint.glob("*.safetensors"))
    assert shards, "Checkpoint has no cached safetensors weights"
    config = json.loads((checkpoint / "config.json").read_text())
    assert config["model_type"] in {"qwen3_5", "qwen3_5_moe"}
    return {
        "path": str(checkpoint),
        "revision": checkpoint.name,
        "config_sha256": hashlib.sha256(
            (checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        "weights": [(p.name, str(p.resolve()), p.stat().st_size) for p in shards],
    }


def prepare(args: argparse.Namespace) -> None:
    identity = checkpoint_identity(args.checkpoint)
    config = json.loads((args.checkpoint / "config.json").read_text())
    text_config = config["text_config"]
    assert args.sequence >= 16 and args.sequence % 4 == 0
    ids = (
        torch.arange(2 * args.sequence).reshape(2, args.sequence) * 37 + 11
    ) % 997 + 100
    lengths = torch.tensor([args.sequence - 3, args.sequence - 9])
    token_mask = torch.arange(args.sequence)[None, :] < lengths[:, None]
    ids.masked_fill_(~token_mask, int(text_config["eos_token_id"]))
    token_mask[:, :3] = False
    next_ids = ids.roll(-1, 1)
    forced = torch.stack(
        (next_ids, (next_ids + 1) % int(text_config["vocab_size"])), dim=-1
    )
    fixture = {
        "checkpoint": identity,
        "input_ids": ids,
        "input_lengths": lengths,
        "mtp_loss_mask": token_mask.float(),
        "force_include_token_ids": forced,
        "real_vocab_size": int(text_config["vocab_size"]),
        "input_sha256": tensor_digest(ids),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixture, args.output)
    print(
        json.dumps(
            {"fixture": str(args.output), "input_sha256": fixture["input_sha256"]}
        )
    )


def verify_runtime(source_manifest: Path) -> dict[str, Any]:
    """Check every archived Python source plus actual imported package origins."""
    # Optional Megatron imports are confined to the GPU capture path.
    import megatron.bridge
    import megatron.core

    provenance = json.loads(source_manifest.read_text())
    origins = {
        "bridge": Path(megatron.bridge.__file__),
        "mcore": Path(megatron.core.__file__),
    }
    roots = {
        "bridge": source_manifest.parent / "bridge",
        "mcore": source_manifest.parent / "mcore",
    }
    for name, revision in (("bridge", BRIDGE_REVISION), ("mcore", MCORE_REVISION)):
        assert provenance[name]["source_git_object"] == revision
        assert origins[name].resolve().is_relative_to(roots[name].resolve()), origins[
            name
        ]
        for relative, expected in provenance[name]["file_hashes"].items():
            path = roots[name] / relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    return {
        "bridge_revision": BRIDGE_REVISION,
        "mcore_revision": MCORE_REVISION,
        "imported_paths": {name: str(path.resolve()) for name, path in origins.items()},
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "transformer-engine",
                "megatron-core",
                "megatron-bridge",
            )
        },
    }


def capture(args: argparse.Namespace) -> None:
    # These optional dependencies require the MCore GPU worker runtime.
    from megatron.bridge import AutoBridge
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    from megatron.core import parallel_state
    from megatron.core.utils import unwrap_model

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron.data import process_microbatch
    from nemo_rl.models.megatron.sparse_teacher import SparseTeacherStorage
    from nemo_rl.models.megatron.train import SparseTeacherLogitsPostProcessor
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        _model_self_packs_for_cp,
    )

    assert torch.cuda.is_available(), "GPU capture requires CUDA"
    runtime = verify_runtime(args.source_manifest)
    fixture = torch.load(args.fixture, weights_only=False)
    assert fixture["checkpoint"] == checkpoint_identity(args.checkpoint)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    assert dist.get_world_size() == args.tp * args.cp
    provider = AutoBridge.from_hf_pretrained(
        str(args.checkpoint), torch_dtype=torch.bfloat16, local_files_only=True
    ).to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = args.tp
    provider.context_parallel_size = args.cp
    provider.pipeline_model_parallel_size = 1
    provider.params_dtype = provider.pipeline_dtype = torch.bfloat16
    provider.bf16, provider.fp16 = True, False
    provider.sequence_parallel = False
    provider.apply_rope_fusion = False
    # Teacher forward needs no fused weight-gradient extension (Apex is optional).
    provider.gradient_accumulation_fusion = False
    provider.seq_length = fixture["input_ids"].shape[1]
    assert provider.mtp_num_layers and provider.mtp_num_layers > 0, (
        "MTP must be enabled"
    )
    provider.finalize()
    provider.initialize_model_parallel(seed=1234)
    models = provider.provide_distributed_model(wrap_with_ddp=False)
    assert len(models) == 1 and _model_self_packs_for_cp(models)
    model = models[0]
    wrapper = unwrap_model(model)
    assert isinstance(wrapper, Qwen3VLModel), type(wrapper)
    assert wrapper.language_model.mtp_process
    model.eval()
    cp_rank = parallel_state.get_context_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    ids = fixture["input_ids"].cuda()
    sequence = ids.shape[1]
    positions = cp_native_global_positions(
        sequence, cp_rank, args.cp, device=ids.device
    )
    batch = BatchedDataDict(
        {
            key: fixture[key].cuda()
            for key in (
                "input_ids",
                "input_lengths",
                "mtp_loss_mask",
                "force_include_token_ids",
            )
        }
    )
    processed = process_microbatch(
        batch,
        seq_length_key="input_lengths",
        pack_sequences=False,
        pad_individual_seqs_to_multiple_of=1,
        pad_packed_seq_to_multiple_of=1,
        pad_full_seq_to=None,
        straggler_timer=None,
        delegate_pack_to_model=True,
        mtp_enabled=True,
    )
    assert torch.equal(processed.input_ids_cp_sharded, ids)
    if args.cp > 1:
        assert processed.position_ids is None and processed.attention_mask is None
    expected_mask = fixture["mtp_loss_mask"].cuda().index_select(1, positions)
    assert torch.equal(processed.mtp_loss_mask, expected_mask)
    observations = {"language_model_calls": 0, "mtp_calls": 0}

    def inspect_language_inputs(
        module: torch.nn.Module, positional: tuple, kwargs: dict
    ) -> None:
        observations["language_model_calls"] += 1
        assert torch.equal(kwargs["input_ids"], ids.index_select(1, positions))
        assert torch.equal(kwargs["loss_mask"], expected_mask)
        expected_positions = positions[None, None, :].expand(3, ids.shape[0], -1)
        assert torch.equal(kwargs["position_ids"], expected_positions)

    def inspect_mtp_inputs(
        module: torch.nn.Module, positional: tuple, kwargs: dict
    ) -> None:
        observations["mtp_calls"] += 1
        assert torch.equal(kwargs["input_ids"], ids.index_select(1, positions))

    hooks = [
        wrapper.language_model.register_forward_pre_hook(
            inspect_language_inputs, with_kwargs=True
        ),
        wrapper.language_model.mtp.register_forward_pre_hook(
            inspect_mtp_inputs, with_kwargs=True
        ),
    ]
    with torch.no_grad():
        logits = model(
            input_ids=processed.input_ids_cp_sharded,
            position_ids=processed.position_ids,
            attention_mask=processed.attention_mask,
            loss_mask=processed.mtp_loss_mask,
        )
    for hook in hooks:
        hook.remove()
    assert observations == {"language_model_calls": 1, "mtp_calls": 1}
    assert logits.shape[:2] == (ids.shape[0], sequence // args.cp)
    assert torch.isfinite(logits).all()
    storage = (
        SparseTeacherStorage.allocate(
            batch_size=ids.shape[0],
            local_sequence_length=sequence // args.cp,
            k=args.k,
            device=ids.device,
        )
        if tp_rank == 0
        else None
    )
    processor = SparseTeacherLogitsPostProcessor(
        k=args.k,
        temperature=args.temperature,
        real_vocab_size=fixture["real_vocab_size"],
        noise_filter_k=0,
        output_storage=storage,
    )
    processor(batch, None)(logits)
    records = []
    if storage is not None:
        # Serialize snapshots of the native wire fields for an offline consumer.
        storage.export_handles(lambda tensor: tensor.cpu().clone())
        records = [
            storage.sample_record(
                sample_index=index,
                full_sequence_length=sequence,
                cp_rank=cp_rank,
                cp_size=args.cp,
                real_vocab_size=fixture["real_vocab_size"],
                temperature=args.temperature,
                membership_k=args.k,
            )
            for index in range(ids.shape[0])
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "runtime": runtime,
            "fixture": fixture,
            "tp": args.tp,
            "cp": args.cp,
            "tp_rank": tp_rank,
            "cp_rank": cp_rank,
            "k": args.k,
            "temperature": args.temperature,
            "precision": "bfloat16",
            "provider_settings": {
                "sequence_parallel": False,
                "apply_rope_fusion": False,
                "gradient_accumulation_fusion": False,
                "mtp_num_layers": provider.mtp_num_layers,
            },
            "wrapper": type(wrapper).__qualname__,
            "observations": observations,
            "positions": positions.cpu(),
            "logits": logits.float().cpu(),
            "records": records,
        },
        args.output / f"rank-{dist.get_rank()}.pt",
    )
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@dataclass
class Capture:
    metadata: dict[str, Any]
    logits: torch.Tensor
    payload: SparseTeacherIPC
    natural_ids: torch.Tensor


def read_capture(directory: Path) -> Capture:
    ranks = [
        torch.load(path, weights_only=False)
        for path in sorted(directory.glob("rank-*.pt"))
    ]
    assert ranks, f"No capture records in {directory}"
    first = ranks[0]
    fixture = first["fixture"]
    batch_size, sequence = fixture["input_ids"].shape
    assert len(ranks) == first["tp"] * first["cp"]
    shard_vocab = first["logits"].shape[-1]
    dense = torch.empty(batch_size, sequence, shard_vocab * first["tp"])
    coverage = torch.zeros(sequence, first["tp"], dtype=torch.long)
    natural_ids = torch.empty(batch_size, sequence, first["k"], dtype=torch.long)
    samples = [{"teacher_shards": []} for _ in range(batch_size)]
    for rank in ranks:
        for key in (
            "tp",
            "cp",
            "k",
            "temperature",
            "precision",
            "wrapper",
            "runtime",
            "provider_settings",
        ):
            assert rank.get(key) == first.get(key), key
        for key in (
            "input_ids",
            "input_lengths",
            "mtp_loss_mask",
            "force_include_token_ids",
        ):
            assert torch.equal(rank["fixture"][key], fixture[key]), key
        positions = rank["positions"]
        cp_rank, tp_rank = rank["cp_rank"], rank["tp_rank"]
        assert rank["observations"] == {"language_model_calls": 1, "mtp_calls": 1}
        # Independently derive expected zigzag row ownership.
        chunks = torch.arange(sequence).chunk(2 * first["cp"])
        expected = (
            torch.arange(sequence)
            if first["cp"] == 1
            else torch.cat((chunks[cp_rank], chunks[-cp_rank - 1]))
        )
        assert torch.equal(positions, expected)
        coverage[positions, tp_rank] += 1
        dense[:, positions, tp_rank * shard_vocab : (tp_rank + 1) * shard_vocab] = rank[
            "logits"
        ]
        if tp_rank == 0:
            for index, record in enumerate(rank["records"]):
                declared = torch.full_like(positions, -1)
                for local_start, global_start, length in record["sequence_segments"]:
                    declared[local_start : local_start + length] = torch.arange(
                        global_start, global_start + length
                    )
                assert torch.equal(declared, positions)
                forced = record["forced_indices_ipc"][0, index].long()
                assert torch.equal(
                    forced, fixture["force_include_token_ids"][index, positions]
                )
                samples[index]["teacher_shards"].append(record)
                natural_ids[index, positions] = record["topk_indices_ipc"][
                    0, index
                ].long()
    assert torch.equal(coverage, torch.ones_like(coverage)), (
        "Duplicated or missing global rows"
    )
    dense = dense[:, :, : fixture["real_vocab_size"]]
    threshold = dense.topk(first["k"], dim=-1).values[:, :, -1:]
    selected = dense.gather(-1, natural_ids)
    assert bool((selected >= threshold - 1e-5).all()), "Exported support is not top-K"
    assert bool((natural_ids.sort(-1).values.diff(dim=-1) > 0).all())
    return Capture(first, dense, SparseTeacherIPC(samples), natural_ids)


def downstream(
    capture_result: Capture, table_path: Path
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed tiny FP32 student, TP1/CP1, through the production native sparse loss."""
    # Loss dependencies are unnecessary for prepare/capture commands.
    from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn
    from nemo_rl.algorithms.x_token.loss_utils import (
        LocalizedAlignment,
        NativeStudentContext,
    )
    from nemo_rl.algorithms.x_token.native_sparse_loss import (
        compute_native_sparse_teacher_loss,
    )

    metadata = capture_result.metadata
    fixture = metadata["fixture"]
    ids = fixture["input_ids"]
    batch_size, sequence = ids.shape
    vocab = fixture["real_vocab_size"]
    torch.manual_seed(9876)
    student = torch.nn.Linear(8, vocab, bias=False)
    features = torch.randn(batch_size, sequence, 8)
    logits = student(features)
    loss_fn = CrossTokenizerDistillationLossFn(
        {
            "temperature": metadata["temperature"],
            "vocab_topk": metadata["k"],
            "reverse_kl": True,
            "kl_loss_weight": 1.0,
            "ce_loss_scale": 0.0,
            "dynamic_loss_scaling": False,
            "student_vocab_size": vocab,
            "kd_loss_mode": "sum",
            "normalize_teacher_by_vocab": False,
            "alpha": 1.0,
            "projection_matrix_paths": [None],
            "teacher_is_cross_tokenizer": [True],
            "teacher_vocab_sizes": [vocab],
            "teacher_weights": [1.0],
            "pseudo_target_paths": [str(table_path)],
            "reverse_pseudo_target_paths": [str(table_path)],
            "common_indices_from_subtoks": True,
            "teacher_topk_ipc_k": metadata["k"],
            "teacher_topk_ipc_keep_realized": True,
            "kl_chunk_shift": True,
        }
    )
    mask = fixture["mtp_loss_mask"]
    spans = torch.stack(
        (torch.arange(1, sequence), torch.arange(2, sequence + 1)), dim=-1
    )[None].expand(batch_size, -1, -1)
    valid = mask[:, 1:].bool()
    alignment = LocalizedAlignment(
        sample_mask=torch.ones(batch_size),
        pair_valid=valid,
        student_input_ids=ids,
        teacher_input_ids=ids,
        student_token_mask=mask,
        student_kd_token_mask=mask,
        student_spans=spans,
        teacher_spans=spans,
    )
    context = NativeStudentContext(
        logits=logits,
        global_positions=torch.arange(sequence),
        input_ids=ids,
        next_token_ids=ids.roll(-1, 1),
        next_token_mask=mask.roll(-1, 1),
        next_kd_token_mask=mask.roll(-1, 1),
        sample_mask=torch.ones(batch_size),
        full_seq_len=sequence,
        real_vocab_size=vocab,
    )
    loss, _ = compute_native_sparse_teacher_loss(
        loss_fn,
        0,
        context,
        alignment,
        SparseTeacherRowReader(capture_result.payload, device=torch.device("cpu")),
        global_valid_chunks=valid.sum(),
        tp_group=None,
        cp_group=None,
    )
    loss.backward()
    assert student.weight.grad is not None and torch.isfinite(student.weight.grad).all()
    return loss.detach(), student.weight.grad.detach().clone()


def compare(args: argparse.Namespace) -> None:
    one, two = read_capture(args.cp1), read_capture(args.cp2)
    assert one.metadata["cp"] == 1 and two.metadata["cp"] == 2
    for key in (
        "tp",
        "k",
        "temperature",
        "precision",
        "wrapper",
        "runtime",
        "provider_settings",
    ):
        assert one.metadata.get(key) == two.metadata.get(key), key
    for key in ("checkpoint", "input_sha256", "real_vocab_size"):
        assert one.metadata["fixture"][key] == two.metadata["fixture"][key], key
    for key in (
        "input_ids",
        "input_lengths",
        "mtp_loss_mask",
        "force_include_token_ids",
    ):
        assert torch.equal(
            one.metadata["fixture"][key], two.metadata["fixture"][key]
        ), key
    failures: dict[str, str] = {}
    checks: dict[str, bool] = {}

    def check_close(
        name: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        atol: float,
        rtol: float,
    ) -> None:
        try:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            checks[name] = True
        except AssertionError as error:
            checks[name] = False
            failures[name] = str(error)

    check_close(
        "dense_scores",
        two.logits,
        one.logits,
        atol=args.score_atol,
        rtol=args.score_rtol,
    )
    selected_ids = torch.cat((one.natural_ids, two.natural_ids), dim=-1)
    selected_scores = [value.logits.gather(-1, selected_ids) for value in (one, two)]
    check_close(
        "exported_support_scores",
        selected_scores[1],
        selected_scores[0],
        atol=args.score_atol,
        rtol=args.score_rtol,
    )
    fixture, k = one.metadata["fixture"], one.metadata["k"]
    batch_size, sequence = fixture["input_ids"].shape
    b = torch.arange(batch_size).repeat_interleave(sequence)
    p = torch.arange(sequence).repeat(batch_size)
    readers = [
        SparseTeacherRowReader(value.payload, device=torch.device("cpu"))
        for value in (one, two)
    ]
    max_logz_error = max_forced_error = 0.0
    for slot in range(2):
        labels = fixture["force_include_token_ids"][:, :, slot].reshape(-1)
        results = [reader.gather_rows(b, p, labels) for reader in readers]
        check_close(
            f"log_z_slot_{slot}",
            results[1][2],
            results[0][2],
            atol=args.logz_atol,
            rtol=0,
        )
        max_logz_error = max(
            max_logz_error, float((results[1][2] - results[0][2]).abs().max())
        )
        forced_scores = []
        for cp_index, (captured, (values, indices, log_z, _)) in enumerate(
            zip((one, two), results, strict=True)
        ):
            assert torch.equal(
                (indices == labels[:, None]).sum(-1), torch.ones_like(labels)
            )
            check_close(
                f"cp{cp_index + 1}_sparse_dense_values_slot_{slot}",
                values,
                captured.logits[b, p].gather(-1, indices.long()),
                atol=1e-5,
                rtol=1e-6,
            )
            check_close(
                f"cp{cp_index + 1}_sparse_dense_log_z_slot_{slot}",
                log_z,
                torch.logsumexp(
                    captured.logits[b, p] / one.metadata["temperature"], -1
                ),
                atol=1e-5,
                rtol=1e-6,
            )
            forced_scores.append(
                values.gather(
                    -1, (indices == labels[:, None]).long().argmax(-1)[:, None]
                )
            )
        check_close(
            f"forced_scores_slot_{slot}",
            forced_scores[1],
            forced_scores[0],
            atol=args.score_atol,
            rtol=args.score_rtol,
        )
        max_forced_error = max(
            max_forced_error, float((forced_scores[1] - forced_scores[0]).abs().max())
        )
    # A changed membership is acceptable only within the numerical error budget
    # of the Kth boundary in BOTH directions; count every affected row.
    boundaries = [value.logits.topk(k, dim=-1).values[:, :, -1] for value in (one, two)]
    changed_rows = 0
    max_exchanged_boundary_gap = 0.0
    for row in range(batch_size * sequence):
        supports = [
            set(value.natural_ids.reshape(-1, k)[row].tolist()) for value in (one, two)
        ]
        if supports[0] != supports[1]:
            changed_rows += 1
            for index, value in enumerate((one, two)):
                boundary = boundaries[index].reshape(-1)[row]
                tolerance = 2 * (
                    args.score_atol + args.score_rtol * abs(float(boundary))
                )
                exchanged = torch.tensor(sorted(supports[0] ^ supports[1]))
                scores = value.logits.reshape(-1, value.logits.shape[-1])[
                    row, exchanged
                ]
                max_exchanged_boundary_gap = max(
                    max_exchanged_boundary_gap, float((scores - boundary).abs().max())
                )
                if not bool(((scores - boundary).abs() <= tolerance).all()):
                    failures[f"topk_boundary_row_{row}_cp{index + 1}"] = (
                        "Top-K changed away from a numerical tie"
                    )
    checks["topk_boundary_changes"] = not any(
        key.startswith("topk_boundary") for key in failures
    )
    args.output.mkdir(parents=True, exist_ok=True)
    table = args.output / "identity-subtokens.pt"
    torch.save(
        {
            "subtoks": torch.arange(fixture["real_vocab_size"])[:, None],
            "lengths": torch.ones(fixture["real_vocab_size"], dtype=torch.long),
        },
        table,
    )
    losses_and_gradients = [downstream(value, table) for value in (one, two)]
    check_close(
        "student_loss",
        losses_and_gradients[1][0],
        losses_and_gradients[0][0],
        atol=args.loss_atol,
        rtol=args.loss_rtol,
    )
    # Top-K boundary exchanges can move a small amount of gradient mass to a
    # different token; assess the complete parameter-gradient norm accordingly.
    gradient_delta = float(
        (losses_and_gradients[1][1] - losses_and_gradients[0][1]).norm()
    )
    gradient_norm = float(losses_and_gradients[0][1].norm())
    checks["student_gradient"] = (
        gradient_delta <= args.gradient_atol + args.gradient_rtol * gradient_norm
    )
    if not checks["student_gradient"]:
        failures["student_gradient"] = (
            f"L2(delta)={gradient_delta}, L2(reference)={gradient_norm}"
        )
    report = {
        "status": "FAIL" if failures else "PASS",
        "checks": checks,
        "failures": failures,
        "checkpoint": fixture["checkpoint"],
        "runtime": one.metadata["runtime"],
        "teacher_tp": one.metadata["tp"],
        "teacher_cp": [1, 2],
        "student_tp": 1,
        "student_cp": 1,
        "precision": one.metadata["precision"],
        "provider_settings": one.metadata.get("provider_settings"),
        "k": k,
        "temperature": one.metadata["temperature"],
        "input_sha256": fixture["input_sha256"],
        "exact_global_rows": batch_size * sequence,
        "exact_forced_labels": 2 * batch_size * sequence,
        "mtp_calls_per_rank": 1,
        "max_score_error": float((two.logits - one.logits).abs().max()),
        "max_exported_support_score_error": float(
            (selected_scores[1] - selected_scores[0]).abs().max()
        ),
        "max_logz_error": max_logz_error,
        "max_forced_score_error": max_forced_error,
        "topk_changed_rows_near_boundary": changed_rows,
        "max_exchanged_topk_boundary_gap": max_exchanged_boundary_gap,
        "student_losses": [float(value[0]) for value in losses_and_gradients],
        "student_gradient_delta_l2": gradient_delta,
        "student_gradient_reference_l2": gradient_norm,
        "tolerances": {
            key: value
            for key, value in vars(args).items()
            if key.endswith(("atol", "rtol"))
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise AssertionError(
            f"Parity checks failed: {', '.join(failures)}; see report.json"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--sequence", type=int, default=32)
    prepare_parser.add_argument("--output", type=Path, required=True)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--checkpoint", type=Path, required=True)
    capture_parser.add_argument("--source-manifest", type=Path, required=True)
    capture_parser.add_argument("--fixture", type=Path, required=True)
    capture_parser.add_argument("--cp", type=int, choices=(1, 2), required=True)
    capture_parser.add_argument("--tp", type=int, default=1)
    capture_parser.add_argument("--k", type=int, default=32)
    capture_parser.add_argument("--temperature", type=float, default=1.0)
    capture_parser.add_argument("--output", type=Path, required=True)
    compare_parser = sub.add_parser("compare")
    for option in ("cp1", "cp2", "output"):
        compare_parser.add_argument(f"--{option}", type=Path, required=True)
    for option, default in (
        ("score-atol", 0.25),
        ("score-rtol", 0.01),
        ("logz-atol", 0.08),
        ("loss-atol", 0.002),
        ("loss-rtol", 0.03),
        ("gradient-atol", 1e-5),
        ("gradient-rtol", 0.15),
    ):
        compare_parser.add_argument(f"--{option}", type=float, default=default)
    args = parser.parse_args()
    {"prepare": prepare, "capture": capture, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
