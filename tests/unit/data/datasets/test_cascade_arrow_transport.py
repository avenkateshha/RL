"""Arrow transport must not infer conversation structure from early rows."""

import json
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from torch.utils.data import ConcatDataset

from nemo_rl.data.datasets import merge_datasets
from nemo_rl.data.datasets.response_datasets.nemotron_cascade2_sft import (
    NemotronCascade2SFTDataset,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import chat_kd_processor


def _tool_row(index: int, *, valid: bool) -> dict[str, Any]:
    return {
        "sample_id": f"tool-row-{index}",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Check"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {}}
                        if valid
                        else "invalid-function",
                    }
                ],
            },
        ],
        "message_loss_mask": [0, 1],
    }


def _write(root: Path, rows: list[dict[str, Any]]) -> None:
    subset = root / "tools"
    subset.mkdir(parents=True)
    (subset / "data.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("workers", [1, 2])
def test_initial_invalid_batches_are_dropped_before_schema_inference(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, workers: int
) -> None:
    _write(
        tmp_path,
        [
            *[_tool_row(index, valid=False) for index in range(1000)],
            _tool_row(1000, valid=True),
        ],
    )
    prepared = NemotronCascade2SFTDataset(
        split_validation_size=0,
        cascade={
            "dataset_path": str(tmp_path),
            "subset": "tools",
            "normalize_tool_calls": True,
            "map_num_proc": workers,
        },
    )
    assert len(prepared.dataset) == 1
    assert prepared.dataset[0]["sample_id"] == "tool-row-1000"
    assert "dropping 1000/1001" in caplog.text
    result = chat_kd_processor(prepared.dataset[0], TaskDataSpec(), None, None, 0)
    assert result["message_log"][1]["tool_calls"][0]["function"]["arguments"] == {}
    assert result["message_loss_mask"] == [0, 1]


def test_late_optional_fields_and_argument_shapes_survive_cache_and_outer_merge(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    rows = [
        {
            "sample_id": f"plain-{index}",
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ],
        }
        for index in range(1000)
    ]
    last = _tool_row(1000, valid=True)
    last["messages"][1]["reasoning_content"] = "Use lookup"
    last["messages"][1]["tool_calls"][0]["function"]["arguments"] = {
        "query": {"nested": [1, None, True]},
        "metadata": {"unseen_key": "literal <think>text</think>"},
    }
    rows.append(last)
    _write(source, rows)
    config = {
        "dataset_path": str(source),
        "subset": "tools",
        "cached_path": str(tmp_path / "cache"),
    }
    prepared = NemotronCascade2SFTDataset(split_validation_size=0, cascade=config)
    reloaded = NemotronCascade2SFTDataset(split_validation_size=0, cascade=config)
    first = reloaded.dataset[0]
    assert first["messages"] == rows[0]["messages"]
    assert "tool_calls" not in first["messages"][1]
    assert first["message_loss_mask"] is None
    assert reloaded.dataset[-1] == prepared.dataset[-1]
    assert reloaded.dataset[-1]["messages"] == last["messages"]

    ordinary = Dataset.from_list([{"sample_id": "ordinary", "messages": []}])
    merged = merge_datasets([ordinary, reloaded.dataset])
    assert isinstance(merged, ConcatDataset)
    assert len(merged) == 1002
    assert merged[-1]["messages"] == last["messages"]
    result = chat_kd_processor(merged[-1], TaskDataSpec(), None, None, len(merged) - 1)
    assert result["message_log"] == last["messages"]
    assert result["tools"] == last["tools"]
    result["message_log"][1]["tool_calls"][0]["function"]["arguments"]["query"][
        "nested"
    ][0] = "changed"
    assert merged[-1]["messages"] == last["messages"]
