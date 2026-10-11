"""CPU plumbing audit of the actual extracted controller capture functions.

Use tiny saved teacher observations, single-rank collective stubs, and the
oracle itself as the input loss. This checks same-tokenizer keys, metric/index
serialization and first-step capture boundaries; it is not independent loss,
MCore, CUDA, optimizer, or distributed numerical evidence.
"""

import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import model_oracle
import torch
import yaml


def main():
    torch.set_num_threads(2)
    root = Path(__file__).parent
    source = root / "controller_reference.py"
    names = {
        "_run",
        "_sha256",
        "_save_atomic",
        "_positions",
        "_check_loss",
        "_check_legacy_dense_rows",
        "_optimizer_first_step_complete",
    }
    nodes = [
        node
        for node in ast.parse(source.read_text(), filename=str(source)).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(nodes) == len(names)
    parallel = SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 0,
        get_context_parallel_rank=lambda: 0,
        get_context_parallel_world_size=lambda: 1,
        get_data_parallel_rank=lambda: 0,
        get_context_parallel_group=lambda: None,
    )
    megatron, core = ModuleType("megatron"), ModuleType("megatron.core")
    core.parallel_state = parallel
    megatron.core = core
    identities = [
        {"teacher_index": i, "model_name": f"cpu_fixture_{i}", "real_vocab_size": 4}
        for i in range(2)
    ]
    scope = {
        "torch": torch,
        "json": json,
        "hashlib": hashlib,
        "os": os,
        "Path": Path,
        "model_oracle": model_oracle,
        "dist": SimpleNamespace(get_rank=lambda: 0, all_reduce=lambda *a, **k: None),
        "_first_step_complete": False,
        "_first_step_microbatches": [],
        "_first_step_optimizers": [],
        "_teacher_inventory": lambda: identities,
        "_mcore_full": lambda value: value.detach().clone(),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    config = yaml.safe_load(
        (root / "runs/R5-averaged/resolved_config.yaml").read_text()
    )
    cfg = {
        **config["loss_fn"],
        "student_vocab_size": 4,
        "teacher_vocab_sizes": [4, 4],
        "teacher_is_cross_tokenizer": [False, False],
        "teacher_weights": [0.3, 0.7],
        "pseudo_target_paths": [None, None],
        "reverse_pseudo_target_paths": [None, None],
        "vocab_topk": 0,  # True average still uses full-vocabulary KL.
    }
    generator = torch.Generator().manual_seed(5802)
    with (
        TemporaryDirectory(prefix="xtoken-capture-") as directory,
        patch.dict(sys.modules, {"megatron": megatron, "megatron.core": core}),
        patch.dict(os.environ, {"XTOKEN_NUMERICAL_CAPTURE_RUN": directory}),
    ):
        run = Path(directory)

        def microbatch(item_id, *, bad_gradient=False):
            data = {
                "input_ids": torch.tensor([[0, 2, 1, 3]]),
                "token_mask": torch.tensor([[0, 1, 1, 1]]),
                "kd_token_mask": torch.tensor([[0, 1, 0, 1]]),
                "sample_mask": torch.ones(1),
                "batch_item_id": torch.tensor([item_id]),
            }
            assert not any(name.startswith("teacher_") for name in data)
            student = torch.randn(1, 4, 6, generator=generator).requires_grad_()
            teachers = [torch.randn(1, 4, 6, generator=generator) for _ in range(2)]
            for i, teacher in enumerate(teachers):
                scope["_save_atomic"](
                    {
                        **identities[i],
                        "logits": teacher[0],
                        "input_ids": data["input_ids"][0],
                        "batch_item_id": item_id,
                        "backend": "cpu_plumbing_fixture_not_a_model",
                    },
                    run / "teacher-observations" / f"teacher{i}" / f"{item_id}.pt",
                )
            fixture = model_oracle.sparse_fixture(
                data, student, teachers, cfg, 6.0, {}, {}
            )
            value, metrics = model_oracle.full_objective(
                fixture, teachers, data, student, cfg, 4.0
            )
            if bad_gradient:
                value = value + 0.1 * (student.sum() - student.detach().sum())
            scope["_check_loss"](
                {
                    "next_token_logits": student,
                    "data": data,
                    "loss_fn": SimpleNamespace(cfg=cfg),
                    "global_valid_toks": torch.tensor(6.0),
                    "global_valid_kd_toks": torch.tensor(4.0),
                    "global_valid_chunks_by_idx": None,
                },
                value,
                {key: float(val.detach()) for key, val in metrics.items()},
                {},
                [],
                [0, 1],
                {},
            )
            return run / "controller-reference" / f"items{item_id}-{item_id}-rank0"

        for occurrence in (100, 101):
            stem = microbatch(occurrence)
            report = json.loads(stem.with_suffix(".json").read_text())
            assert report["chunk_denominators"] == {}
            assert report["native_dense_teacher_indices"] == [0, 1]
            assert report["dlogits"]["relative_l2"] == 0
            assert report["capture_path"] == str(stem.with_suffix(".pt"))
            assert scope["_sha256"](stem.with_suffix(".pt")) == report["capture_sha256"]
            assert "teacher_0/weighted_kl" in report["reference_metrics"]
        optimizer = {"found_inf": False, "optimizer_index": 0, "scope": "mock callback"}
        (run / "optimizer-first-step").mkdir()
        scope["_optimizer_first_step_complete"](optimizer)
        index_path = run / "optimizer-first-step" / "rank0-index.json"
        index = json.loads(index_path.read_text())
        assert [entry["ordinal"] for entry in index["microbatches"]] == [0, 1]
        assert [entry["item_ids"] for entry in index["microbatches"]] == [[100], [101]]
        assert index["optimizers"] == [optimizer]
        stem = microbatch(102)
        report = json.loads(stem.with_suffix(".json").read_text())
        assert report["capture_path"] is None and not stem.with_suffix(".pt").exists()
        assert len(scope["_first_step_microbatches"]) == 2
        scope["_optimizer_first_step_complete"]({**optimizer, "optimizer_index": 1})
        index = json.loads(index_path.read_text())
        assert len(index["microbatches"]) == len(index["optimizers"]) == 2
        try:
            microbatch(103, bad_gradient=True)
        except AssertionError:
            pass
        else:
            raise AssertionError("A mismatched gradient was accepted")
        failed = run / "controller-reference" / "items103-103-rank0"
        report = json.loads(failed.with_suffix(".json").read_text())
        assert report["status"] == "FAIL" and report["dlogits"]["relative_l2"] > 0.02
        assert failed.with_suffix(".pt").exists()
        assert len(scope["_first_step_microbatches"]) == 2
    print(
        "PASS R5 capture plumbing: same IDs, None chunks, dense metrics/dlogits, two MBs, chained callback, first-step boundary, failed-gradient operands retained; CPU stubs only"
    )


if __name__ == "__main__":
    main()
