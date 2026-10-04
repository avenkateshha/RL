"""Cascade preparation preserves source identity through caching and mixtures."""

import errno
import json
import os
import shutil

import pytest
from datasets import Dataset

from nemo_rl.data.datasets.response_datasets import (
    load_response_dataset,
    nemotron_cascade2_sft as cascade,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import chat_kd_processor


def _row(index=0):
    return {
        "messages": [
            {"role": "user", "content": f"question {index}"},
            {
                "role": "assistant",
                "content": f"<think>reason {index}</think>answer {index}",
            },
        ]
    }


def _tool_row(name="lookup", *, explicit_tools=False):
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
    row = {
        "messages": [
            {"role": "system", "content": f"<tools>{json.dumps(tool)}</tools>"},
            {"role": "user", "content": "Check."},
            {
                "role": "assistant",
                "content": "<tool_call>"
                + json.dumps({"name": name, "arguments": {"query": "status"}})
                + "</tool_call>",
            },
            {"role": "tool", "content": "<tool_response>okay</tool_response>"},
            {"role": "assistant", "content": "<think>checked</think>Done."},
        ],
        "message_loss_mask": [0, 0, 1, 0, 1],
    }
    if explicit_tools:
        row["messages"][0]["content"] = "Be helpful."
        row["tools"] = [tool]
    return row


def _write(root, subset, rows):
    directory = root / subset
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "data.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def _load(root, **options):
    return cascade.NemotronCascade2SFTDataset(
        split_validation_size=0, cascade={"dataset_path": str(root), **options}
    )


def test_legacy_math_defaults_and_stable_ids(monkeypatch):
    raw = Dataset.from_list([_row(index) for index in range(40)])
    calls = []

    def load(path, subset, **kwargs):
        calls.append((path, subset, kwargs))
        return raw

    monkeypatch.setattr(cascade, "load_dataset", load)
    prepared = cascade.NemotronCascade2SFTMathDataset()
    assert calls == [("nvidia/Nemotron-Cascade-2-SFT-Data", "math", {"split": "train"})]
    assert prepared.task_name == "Nemotron-Cascade-2-SFT-Math"
    assert prepared.get_task_names() == (prepared.task_name,)
    assert len(prepared.dataset) == 38 and len(prepared.val_dataset) == 2
    assert "<think>" in prepared.dataset[0]["messages"][-1]["content"]
    assert set(prepared.dataset["sample_id"]).isdisjoint(
        prepared.val_dataset["sample_id"]
    )
    limited = cascade.NemotronCascade2SFTMathDataset(
        max_samples=10, split_validation_size=0.2
    )
    assert len(limited.dataset) == 8 and len(limited.val_dataset) == 2
    assert set(limited.dataset["sample_id"]) <= set(
        prepared.dataset["sample_id"]
    ) | set(prepared.val_dataset["sample_id"])


def test_stage2_alias_and_revision(monkeypatch):
    calls = []

    def load(path, subset, **kwargs):
        calls.append((path, subset, kwargs))
        return Dataset.from_list([_row()])

    monkeypatch.setattr(cascade, "load_dataset", load)
    prepared = load_response_dataset(
        {
            "dataset_name": "Nemotron-Cascade-SFT-Stage-2",
            "split_validation_size": 0,
            "processor": "chat_kd_processor",
            "cascade": {"subset": "code", "revision": "pinned"},
        }
    )
    assert calls == [
        (
            "nvidia/Nemotron-Cascade-SFT-Stage-2",
            "code",
            {"split": "train", "revision": "pinned"},
        )
    ]
    assert prepared.task_name == "Nemotron-Cascade-SFT-Stage-2-code"
    assert prepared.processor is chat_kd_processor


@pytest.mark.parametrize("explicit_tools", [False, True])
def test_tools_ids_masks_and_thinking(tmp_path, explicit_tools):
    row = _tool_row(explicit_tools=explicit_tools)
    row["id"] = "durable-source"
    _write(tmp_path, "tools", [row])
    prepared = _load(
        tmp_path, subset="tools", normalize_tool_calls=True, strip_thinking=True
    )
    data = prepared.dataset[0]
    assert data["sample_id"] == "durable-source"
    assert data["source_subset"] == "tools"
    assert data["message_loss_mask"] == (
        [0, 0, 1, 0, 1] if explicit_tools else [0, 1, 0, 1]
    )
    result = chat_kd_processor(data, TaskDataSpec(), None, None, 0)
    assert result["message_log"][-1]["content"] == "Done."
    assert result["message_log"][-2]["content"] == "okay"
    call = result["message_log"][-3]["tool_calls"][0]
    assert call["function"]["arguments"] == {"query": "status"}
    assert result["tools"][0]["function"]["name"] == "lookup"
    result["tools"][0]["function"]["name"] = "mutated"
    assert json.loads(data["tools_json"])[0]["function"]["name"] == "lookup"


def test_drop_counts_and_cache_round_trip(tmp_path, caplog, monkeypatch):
    source, cache = tmp_path / "source", tmp_path / "cache"
    _write(source, "tools", [_tool_row("undefined"), _tool_row()])
    kwargs = {
        "subset": "tools",
        "normalize_tool_calls": True,
        "cached_path": str(cache),
    }
    prepared = _load(source, **kwargs)
    assert len(prepared.dataset) == 1
    assert prepared.dataset[0]["sample_id"].endswith(":1")
    assert "undefined_tool_call" in caplog.text and "dropping 1/2" in caplog.text
    manifest_path = next(cache.rglob(cascade._CACHE_MANIFEST))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["rejection_reasons"] == {"undefined_tool_call": 1}
    assert manifest["source_rows"] == 2 and manifest["accepted_rows"] == 1
    assert manifest["local_content_sha256"] and manifest["source_fingerprint"]

    def fail(*args, **kwargs):
        raise AssertionError("Cache hit must not repeat preparation")

    monkeypatch.setattr(cascade, "_format_cascade_row", fail)
    reloaded = _load(source, **kwargs)
    assert reloaded.dataset.to_list() == prepared.dataset.to_list()
    assert (
        os.stat(manifest_path.parent).st_mode & 0o777
        == os.stat(manifest_path.parent.parent).st_mode & 0o777
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"strip_thinking": True},
        {"normalize_tool_calls": True},
        {"task_name_prefix": "different"},
        {"revision": "different"},
        {"tool_call_invalid_policy": "error"},
    ],
)
def test_cache_rejects_transform_or_revision_mismatch(tmp_path, changed):
    source, cache = tmp_path / "source", tmp_path / "cache"
    _write(source, "math", [_row()])
    _load(source, cached_path=str(cache))
    with pytest.raises(RuntimeError, match="cache identity mismatch"):
        _load(source, cached_path=str(cache), **changed)


def test_cache_rejects_changed_source_and_missing_manifest(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    _write(source, "math", [_row()])
    _load(source, cached_path=str(cache))
    _write(source, "math", [_row(9)])
    with pytest.raises(RuntimeError, match="cache identity mismatch"):
        _load(source, cached_path=str(cache))
    next(cache.rglob(cascade._CACHE_MANIFEST)).unlink()
    with pytest.raises(RuntimeError, match="missing cascade_preparation_manifest"):
        _load(source, cached_path=str(cache))


def test_remote_fingerprint_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cascade, "load_dataset", lambda *args, **kwargs: Dataset.from_list([_row()])
    )
    cascade.NemotronCascade2SFTDataset(
        split_validation_size=0, cascade={"cached_path": str(tmp_path)}
    )
    monkeypatch.setattr(
        cascade, "load_dataset", lambda *args, **kwargs: Dataset.from_list([_row(1)])
    )
    with pytest.raises(RuntimeError, match="source_fingerprint"):
        cascade.NemotronCascade2SFTDataset(
            split_validation_size=0, cascade={"cached_path": str(tmp_path)}
        )


def test_concurrent_atomic_publication_and_cleanup(tmp_path, monkeypatch):
    source, cache = tmp_path / "source", tmp_path / "cache"
    _write(source, "math", [_row()])
    rename = os.rename

    def publish_winner(first, second):
        if ".tmp-" in str(first):
            shutil.copytree(first, second)
            raise OSError(errno.ENOTEMPTY, "concurrent winner")
        rename(first, second)

    monkeypatch.setattr(cascade.os, "rename", publish_winner)
    assert len(_load(source, cached_path=str(cache)).dataset) == 1
    assert not list(cache.rglob("*.tmp-*"))


def test_error_policy_all_invalid_and_invalid_masks(tmp_path):
    _write(tmp_path, "tools", [_tool_row("undefined")])
    with pytest.raises(ValueError, match="undefined_tool_call"):
        _load(
            tmp_path,
            subset="tools",
            normalize_tool_calls=True,
            tool_call_invalid_policy="error",
        )
    with pytest.raises(ValueError, match="no valid rows.*undefined_tool_call"):
        _load(tmp_path, subset="tools", normalize_tool_calls=True)
    row = _tool_row()
    row["message_loss_mask"] = [0, 1]
    _write(tmp_path, "tools", [row])
    with pytest.raises(ValueError, match="one entry per message"):
        _load(tmp_path, subset="tools", normalize_tool_calls=True)


def test_mixture_determinism_caps_provenance_and_disjoint_ids(tmp_path):
    for subset in ("code", "math"):
        _write(tmp_path, subset, [_row(index) for index in range(20)])
    config = {
        "dataset_path": str(tmp_path),
        "max_samples_per_subset": 8,
        "subsets": [
            {"name": "code", "weight": 0.8, "max_samples": 12},
            {"name": "math", "weight": 0.2},
            {"name": "absent", "weight": 0},
        ],
    }
    first = cascade.NemotronCascade2SFTDataset(
        cascade=config, split_validation_size=0.25
    )
    second = cascade.NemotronCascade2SFTDataset(
        cascade=config, split_validation_size=0.25
    )
    assert first.dataset.to_list() == second.dataset.to_list()
    assert len(first.val_dataset) == 5
    assert len(set(first.dataset["sample_id"])) == 15
    assert set(first.dataset["sample_id"]).isdisjoint(first.val_dataset["sample_id"])
    assert set(first.dataset["source_subset"]) == {"code", "math"}
    assert set(first.dataset["task_name"]) == set(first.get_task_names())
    assert len(first.dataset) > 15  # Repeats remain confined to training.


def test_duplicate_source_id_leakage_is_rejected(tmp_path):
    _write(
        tmp_path, "math", [{**_row(index), "id": "duplicate"} for index in range(10)]
    )
    with pytest.raises(ValueError, match="sample_id overlap"):
        cascade.NemotronCascade2SFTDataset(
            cascade={"dataset_path": str(tmp_path)}, split_validation_size=0.2
        )


def test_parallel_preparation_preserves_ordinal_ids(tmp_path):
    _write(tmp_path, "math", [_row(index) for index in range(8)])
    serial = _load(tmp_path)
    parallel = _load(tmp_path, map_num_proc=2)
    assert parallel.dataset.to_list() == serial.dataset.to_list()


def test_native_schemas_survive_local_arrow_transport(tmp_path):
    first, second = _tool_row(explicit_tools=True), _tool_row(explicit_tools=True)
    second["tools"][0]["function"]["parameters"]["properties"] = {
        "other": {"type": "integer"}
    }
    second["tools"][0]["function"]["parameters"]["required"] = ["other"]
    for row, arguments in [(first, {"query": "status"}), (second, {"other": 2})]:
        row["messages"][2] = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "arguments": arguments},
                }
            ],
        }
    _write(tmp_path, "tools", [first, second])
    prepared = _load(tmp_path, subset="tools")
    results = [
        chat_kd_processor(row, TaskDataSpec(), None, None, index)
        for index, row in enumerate(prepared.dataset)
    ]
    for result, original in zip(results, [first, second]):
        assert result["tools"] == original["tools"]
        assert (
            result["message_log"][2]["tool_calls"][0]["function"]["arguments"]
            == original["messages"][2]["tool_calls"][0]["function"]["arguments"]
        )


def test_math_and_tools_mix_aligns_message_schemas(tmp_path):
    _write(tmp_path, "math", [_row(index) for index in range(4)])
    _write(tmp_path, "tools", [_tool_row() for _ in range(4)])
    prepared = cascade.NemotronCascade2SFTDataset(
        cascade={
            "dataset_path": str(tmp_path),
            "subsets": [
                {"name": "math", "weight": 0.5},
                {"name": "tools", "weight": 0.5, "normalize_tool_calls": True},
            ],
        },
        split_validation_size=0.25,
    )
    assert len(prepared.val_dataset) == 2
    assert set(prepared.dataset["task_name"]) == set(prepared.get_task_names())
    for index, row in enumerate(prepared.dataset):
        processed = chat_kd_processor(row, TaskDataSpec(), None, None, index)
        if row["source_subset"] == "tools":
            assert processed["tools"][0]["function"]["name"] == "lookup"
            assert processed["message_log"][1]["tool_calls"][0]["function"][
                "arguments"
            ] == {"query": "status"}


def test_malformed_native_tool_row_reaches_invalid_policy(tmp_path, caplog):
    invalid = _tool_row(explicit_tools=True)
    invalid["messages"][2] = {"role": "assistant", "content": "", "tool_calls": [None]}
    _write(tmp_path, "tools", [invalid, _tool_row()])
    prepared = _load(tmp_path, subset="tools", normalize_tool_calls=True)
    assert len(prepared.dataset) == 1
    assert "invalid_tool_call" in caplog.text
    with pytest.raises(ValueError, match="invalid_tool_call"):
        _load(
            tmp_path,
            subset="tools",
            normalize_tool_calls=True,
            tool_call_invalid_policy="error",
        )
