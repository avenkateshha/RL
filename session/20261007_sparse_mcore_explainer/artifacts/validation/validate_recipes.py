"""Validate prepared small recipes with the production config schema."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from nemo_rl.algorithms.xtoken_off_policy_distillation import (
    MasterConfig,
    _supports_native_xtoken_rows,
    _use_reusable_dense_teacher_ipc,
    validate_xtoken_sparse_setup,
)
from nemo_rl.utils.config import register_omegaconf_resolvers

register_omegaconf_resolvers()
validation = Path(__file__).resolve().parent
results = []
manifest = json.loads((validation / "fixture_manifest.json").read_text())


def policy_view(cfg):
    return SimpleNamespace(
        cfg=cfg,
        use_sequence_packing=cfg["sequence_packing"]["enabled"],
        use_dynamic_batches=cfg["dynamic_batching"]["enabled"],
    )


for path in sorted((validation / "runs").glob("R*/resolved_config.yaml")):
    config = MasterConfig(**OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    vocabularies = [
        next(
            info["real_tokenizer_vocab_size"]
            for info in manifest["models"].values()
            if info["snapshot_path"] == teacher.model_name
        )
        for teacher in config.teachers
    ]
    validate_xtoken_sparse_setup(config, vocabularies)
    student = policy_view(config.policy)
    teachers = [policy_view(teacher.policy_config()) for teacher in config.teachers]
    loss = SimpleNamespace(
        kd_loss_mode=config.loss_fn["kd_loss_mode"],
        teacher_is_cross_tokenizer=[
            teacher.uses_cross_tokenizer for teacher in config.teachers
        ],
    )
    reusable = _use_reusable_dense_teacher_ipc(student, teachers, loss)
    routes = []
    for teacher, pol, cross in zip(
        config.teachers, teachers, loss.teacher_is_cross_tokenizer, strict=True
    ):
        if cross:
            route = (
                (
                    "native_sparse"
                    if pol.cfg["megatron_cfg"]["enabled"]
                    else "legacy_dtensor_sparse"
                )
                if config.loss_fn["teacher_topk_ipc_k"] > 0
                else "legacy_dense_cross"
            )
        else:
            assert reusable and _supports_native_xtoken_rows(pol), path.parent.name
            route = "native_same_reusable_dense"
        routes.append(route)
    assert _supports_native_xtoken_rows(student)
    results.append(
        {
            "case": path.parent.name,
            "status": "PASS",
            "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "teacher_routes": routes,
            "native_student_eligible": True,
            "scope": "Schema and actual controller eligibility helpers on resolved policy settings, without actor/model execution",
        }
    )
(validation / "runs/recipe-schema-validation.json").write_text(
    json.dumps(results, indent=2) + "\n"
)
print(f"PASS: {len(results)} prepared configurations.")
