"""Materialize the agreed small integration matrix from the repository exemplar."""

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

validation = Path(__file__).resolve().parent
repo = validation.parents[3]
manifest = json.loads((validation / "fixture_manifest.json").read_text())
preparation = json.loads(
    (validation / "runs/fixture-preparation-20261010/results.json").read_text()
)
assert preparation["status"] == "PASS"
register_omegaconf_resolvers()
base = OmegaConf.to_container(
    load_config(repo / "examples/configs/xtoken_off_policy_distillation.yaml"),
    resolve=True,
)
tables = {
    (item["teacher"], item["direction"]): item["path"] for item in preparation["tables"]
}
# name, teacher roles/backend, nodes, used GPUs/node, TP, CP, K, dynamic, mode.
cases = [
    ("R1-tp1cp1", [("teacher_a", "mcore")], 1, 1, 1, 1, 64, False, "sum"),
    ("R1-tp2cp1", [("teacher_a", "mcore")], 1, 2, 2, 1, 64, False, "sum"),
    ("R1-tp1cp2", [("teacher_a", "mcore")], 1, 2, 1, 2, 64, False, "sum"),
    ("R1-tp2cp2", [("teacher_a", "mcore")], 1, 4, 2, 2, 64, False, "sum"),
    (
        "R2-tp2cp2",
        [("teacher_a", "mcore"), ("teacher_b", "mcore")],
        1,
        8,
        2,
        2,
        64,
        False,
        "sum",
    ),
    (
        "R2-tp2cp2-reversed",
        [("teacher_b", "mcore"), ("teacher_a", "mcore")],
        1,
        8,
        2,
        2,
        64,
        False,
        "sum",
    ),
    (
        "R2-2n-tp2cp2",
        [("teacher_a", "mcore"), ("teacher_b", "mcore")],
        2,
        8,
        2,
        2,
        64,
        False,
        "sum",
    ),
    (
        "R3-fixed",
        [("teacher_a", "mcore"), ("teacher_c_same_tokenizer", "mcore")],
        1,
        8,
        2,
        2,
        64,
        False,
        "sum",
    ),
    (
        "R3-dynamic",
        [("teacher_a", "mcore"), ("teacher_c_same_tokenizer", "mcore")],
        1,
        8,
        2,
        2,
        64,
        True,
        "sum",
    ),
    (
        "R4-mcore-dtensor",
        [("teacher_a", "mcore"), ("teacher_b", "dtensor")],
        1,
        8,
        2,
        2,
        64,
        False,
        "sum",
    ),
    (
        "R4-dtensor-mcore",
        [("teacher_a", "dtensor"), ("teacher_b", "mcore")],
        1,
        8,
        2,
        2,
        64,
        False,
        "sum",
    ),
    ("R5-fixed", [("teacher_c_same_tokenizer", "mcore")], 1, 8, 2, 2, 0, False, "sum"),
    ("R5-dynamic", [("teacher_c_same_tokenizer", "mcore")], 1, 8, 2, 2, 0, True, "sum"),
    (
        "R5-averaged",
        [("student", "mcore"), ("teacher_c_same_tokenizer", "mcore")],
        1,
        8,
        2,
        2,
        0,
        False,
        "averaged_logits",
    ),
    ("R6-dense-cross", [("teacher_a", "mcore")], 1, 8, 2, 2, 0, False, "sum"),
    ("R6-dtensor-sparse", [("teacher_a", "dtensor")], 1, 8, 2, 2, 64, False, "sum"),
]
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
index = []
for name, roles, nodes, gpus, tp, cp, k, dynamic, mode in cases:
    run = validation / "runs" / name
    assert not (run / "submission.json").exists(), (
        "Submitted recipes are immutable; use a new run ID"
    )
    run.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(base)
    dp = nodes * gpus // (tp * cp)
    gbs = 4 * dp
    cfg["distillation"].update(
        num_prompts_per_step=gbs,
        max_num_steps=3,
        max_num_epochs=6,
        seed=1234,
        val_period=0,
        val_at_start=False,
        val_at_end=True,
        offload_student_after_step=True,
    )
    cfg["loss_fn"].update(
        teacher_topk_ipc_k=k,
        vocab_topk=64,
        dynamic_loss_scaling=dynamic,
        kl_loss_weight=1.0,
        ce_loss_scale=0.1,
        common_indices_from_subtoks=True,
        kl_chunk_shift=True,
        prefix_bidir_v3_position_0_kl=name == "R6-dense-cross",
        kd_loss_mode=mode,
    )
    if k > 0:
        cfg["loss_fn"].update(
            prefix_bidir_v3_mismatch_pos0_alpha=0.0,
            prefix_bidir_v3_mismatch_loss_beta=1.0,
        )
        cfg["loss_fn"].pop("prefix_bidir_v3_mismatch_pos0_weight", None)
        cfg["loss_fn"].pop("prefix_bidir_v3_mismatch_loss_scale", None)
    cfg["cluster"].update(num_nodes=nodes, gpus_per_node=gpus)
    cfg["checkpointing"]["enabled"] = False
    cfg["logger"].update(
        log_dir=str(run / "training-logs"),
        wandb_enabled=False,
        tensorboard_enabled=False,
        monitor_gpus=False,
    )
    cfg["data"].update(max_input_seq_length=256, shuffle=True, num_workers=0)
    cfg["data"]["train"].update(
        data_files=manifest["plaintext_corpus"]["path"],
        characters_per_sample=None,
        seed=1234,
        split_validation_size=0.0,
    )
    cfg["data"]["validation"] = deepcopy(cfg["data"]["train"])

    def policy_settings(pol, role, backend, microbatch):
        """Apply pinned model and small-test parallelism to one policy."""
        snapshot = manifest["models"][role]["snapshot_path"]
        pol["model_name"] = snapshot
        pol["tokenizer"]["name"] = snapshot
        pol.update(
            train_global_batch_size=gbs,
            train_micro_batch_size=microbatch,
            logprob_batch_size=microbatch,
            max_total_sequence_length=256,
            make_sequence_length_divisible_by=tp * cp * 2,
        )
        pol["sequence_packing"]["enabled"] = False
        pol["dynamic_batching"]["enabled"] = False
        pol["dtensor_cfg"].update(
            enabled=backend == "dtensor",
            tensor_parallel_size=tp,
            context_parallel_size=cp,
            load_precision="bfloat16",
        )
        pol["megatron_cfg"].update(
            enabled=backend == "mcore",
            tensor_model_parallel_size=tp,
            context_parallel_size=cp,
            pipeline_model_parallel_size=1,
            sequence_parallel=False,
            defer_fp32_logits=False,
        )
        pol["megatron_cfg"]["optimizer"].update(lr=5e-6, min_lr=5e-6)
        pol["megatron_cfg"]["scheduler"].update(lr_warmup_iters=0, lr_decay_iters=3)

    policy_settings(cfg["policy"], "student", "mcore", 2)
    cfg["teachers"] = []
    for i, (role, backend) in enumerate(roles):
        teacher = deepcopy(base["teachers"][0])
        policy_settings(teacher, role, backend, 1 if role == "teacher_a" else 2)
        if role == "teacher_b":
            teacher["make_sequence_length_divisible_by"] = tp * cp * 4
        cross = role in ("teacher_a", "teacher_b")
        teacher["is_cross_tokenizer"] = cross
        teacher["weight"] = (
            1.0
            if len(roles) == 1
            else (0.3 if role in ("teacher_a", "student") else 0.7)
        )
        teacher["aligner"].update(
            projection_matrix_path=None,
            pseudo_target_path=tables[(role, "fwd")] if cross else None,
            reverse_pseudo_target_path=tables[(role, "rev")] if cross else None,
        )
        cfg["teachers"].append(teacher)
    OmegaConf.save(OmegaConf.create(cfg), run / "resolved_config.yaml")
    config_sha = hashlib.sha256((run / "resolved_config.yaml").read_bytes()).hexdigest()
    (run / "command.txt").write_text(
        "uv run --no-sync examples/run_xtoken_off_policy_distillation.py --config "
        + str(run / "resolved_config.yaml")
        + " --resolved-config-output "
        + str(run / "runtime_resolved_config.yaml")
        + "\n"
    )
    runtime = {
        "status": "NOT_RUN",
        "reason": "Implementation checkpoints and MCore runtime validation pending.",
        "source_head_at_preparation": head,
        "config_sha256": config_sha,
        "fixture_manifest": str(validation / "fixture_manifest.json"),
        "fixture_preparation": str(
            validation / "runs/fixture-preparation-20261010/results.json"
        ),
        "tp": tp,
        "cp": cp,
        "dp": dp,
        "pp": 1,
        "nodes": nodes,
        "gpus_per_node_used": gpus,
        "gpus_per_node_allocated": 8,
        "global_batch_size": gbs,
        "student_micro_batch_size": 2,
        "student_microbatches_per_dp_step": 2,
        "max_sequence_length": 256,
        "teacher_padding_divisibility": [
            teacher["make_sequence_length_divisible_by"] for teacher in cfg["teachers"]
        ],
        "teacher_b_padding_divisibility": tp * cp * 4
        if any(role == "teacher_b" for role, _ in roles)
        else None,
        "padding_rationale": (
            "Distinct SmolLM2 padded sequence length exercises separate teacher CP geometry using existing padding setting."
            if any(role == "teacher_b" for role, _ in roles)
            else "Each configured teacher uses the existing padding setting; no SmolLM2 teacher is present."
        ),
        "sparse_k": k,
        "optimizer_steps": 3,
        "evaluate": True,
        "dynamic_loss_scaling": dynamic,
        "full_step_normalizers": "Compute independently per teacher with production collator; retain in run log.",
        "tolerances": {
            "cpu_gloo_fp32": {"rtol": 1e-4, "atol": 1e-4},
            "gpu_bfloat16": {
                "loss_rtol": 0.02,
                "loss_atol": 0.005,
                "gradient_relative_l2_max": 0.02,
                "gradient_norm_relative_error_max": 0.02,
            },
            "note": "Predetermined before submission. Compare pre-clip gradients; matching clipped updates do not establish parity.",
        },
        "required_evidence": [
            "3 optimizer steps and final evaluation",
            "per-teacher weighted KD and CE once",
            "pre-clip gradient parity against same-grouping reference",
            "student native row/no-relayout checks",
            "consecutive IPC generations without stale or cross-teacher values",
        ],
        "larger_memory_tests": "EXCLUDED",
    }
    (run / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    (run / "results.json").write_text(
        json.dumps({"status": "NOT_RUN", "reason": runtime["reason"]}, indent=2) + "\n"
    )
    index.append({"case": name, "config_sha256": config_sha, "status": "NOT_RUN"})
(validation / "runs/recipe-index.json").write_text(json.dumps(index, indent=2) + "\n")
print(f"Prepared {len(index)} small recipes; no jobs submitted.")
