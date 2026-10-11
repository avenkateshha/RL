"""CPU audit of actual observation identity/storage helpers; no model claims."""

import ast
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import torch
import yaml


def main():
    root = Path(__file__).parent
    source = root / "controller_reference.py"
    names = {
        "_run",
        "_teacher_inventory",
        "_teacher_identity",
        "_save_atomic",
        "_save_teacher",
        "_check_legacy_dense_rows",
    }
    tree = ast.parse(source.read_text(), filename=str(source))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(nodes) == len(names)
    scope = {
        "lru_cache": lru_cache,
        "json": json,
        "os": os,
        "Path": Path,
        "torch": torch,
        "yaml": yaml,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    dense_rows = torch.arange(2 * 8 * 6).reshape(2, 8, 6).float()
    dense_cfg = {"teacher_is_cross_tokenizer": [True], "teacher_vocab_sizes": [5]}
    check_dense = scope["_check_legacy_dense_rows"]
    for cp_size in (1, 2):
        width = 8 // cp_size
        for cp_rank in range(cp_size):
            actual = dense_rows[:, cp_rank * width : (cp_rank + 1) * width].clone()
            actual[..., 5] = -99  # Padding vocabulary is outside the objective.
            check_dense([dense_rows], {0: actual}, dense_cfg, cp_rank, cp_size)
            for bad in (actual.flip(1), actual[:1], actual[:, :-1], actual[..., :4]):
                try:
                    check_dense([dense_rows], {0: bad}, dense_cfg, cp_rank, cp_size)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("Dense observation mismatch was accepted")
    previous = os.environ.get("XTOKEN_NUMERICAL_CAPTURE_RUN")
    try:
        for case in (
            "R1-tp1cp1",
            "R2-tp2cp2",
            "R3-fixed",
            "R4-dtensor-mcore",
            "R5-averaged",
        ):
            os.environ["XTOKEN_NUMERICAL_CAPTURE_RUN"] = str(root / "runs" / case)
            scope["_teacher_inventory"].cache_clear()
            for identity in scope["_teacher_inventory"]():
                resolved = scope["_teacher_identity"](
                    identity["real_vocab_size"], identity["model_name"]
                )
                assert resolved == identity
                if identity["is_cross_tokenizer"]:
                    assert (
                        scope["_teacher_identity"](identity["real_vocab_size"])
                        == identity
                    )
        with tempfile.TemporaryDirectory(prefix="xtoken-observation-") as directory:
            run = Path(directory)
            fixture = root / "runs" / "R5-averaged"
            for name in ("resolved_config.yaml", "runtime.json"):
                (run / name).write_text((fixture / name).read_text())
            os.environ["XTOKEN_NUMERICAL_CAPTURE_RUN"] = str(run)
            scope["_teacher_inventory"].cache_clear()
            identities = scope["_teacher_inventory"]()
            assert identities[0]["real_vocab_size"] == identities[1]["real_vocab_size"]
            assert identities[0]["model_name"] != identities[1]["model_name"]
            data = {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "batch_item_id": torch.tensor([91, 92]),
            }
            for identity in identities:
                index = identity["teacher_index"]
                scope["_save_teacher"](
                    torch.full((2, 2, 3), float(index + 1)),
                    data,
                    identity["real_vocab_size"],
                    "cpu_storage_selfcheck",
                    True,
                    model_name=identity["model_name"],
                )
                for item_id in (91, 92):
                    record = torch.load(
                        run
                        / "teacher-observations"
                        / f"teacher{index}"
                        / f"{item_id}.pt",
                        weights_only=True,
                    )
                    assert record["model_name"] == identity["model_name"]
                    assert record["teacher_index"] == index
                    assert record["batch_item_id"] == item_id
                    assert bool((record["logits"] == index + 1).all())
            assert len(list((run / "teacher-observations").glob("*/*.pt"))) == 4
            try:
                scope["_teacher_identity"](identities[0]["real_vocab_size"])
            except AssertionError:
                pass
            else:
                raise AssertionError(
                    "Shared-vocab observations require a model identity"
                )
    finally:
        scope["_teacher_inventory"].cache_clear()
        if previous is None:
            os.environ.pop("XTOKEN_NUMERICAL_CAPTURE_RUN", None)
        else:
            os.environ["XTOKEN_NUMERICAL_CAPTURE_RUN"] = previous
    print(
        "PASS: five matrix teacher inventories, distinct equal-vocab storage, ambiguous identity rejection; CP1/CP2 dense consumer row checks and 12 mismatch rejections"
    )


if __name__ == "__main__":
    main()
