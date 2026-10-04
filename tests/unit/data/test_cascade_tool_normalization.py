# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from multiprocess import get_context
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from nemo_rl.data.cascade_tool_normalization import (
    CascadeToolNormalizationError,
    normalize_cascade_tool_messages,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.native_chat import _render_and_tokenize_chat
from nemo_rl.data.processors import chat_kd_processor


def test_normalization_error_survives_pickle_round_trip() -> None:
    error = CascadeToolNormalizationError("invalid_tool_call", "missing name")
    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, CascadeToolNormalizationError)
    assert restored.code == "invalid_tool_call"
    assert str(restored) == "invalid_tool_call: missing name"


def test_normalization_error_propagates_from_preparation_worker() -> None:
    with get_context("spawn").Pool(processes=1) as pool:
        pending = pool.apply_async(
            normalize_cascade_tool_messages,
            ([{"role": "assistant", "content": "no definitions"}],),
        )
        with pytest.raises(
            CascadeToolNormalizationError, match="tool_block_count"
        ) as error:
            pending.get(timeout=30)
    assert error.value.code == "tool_block_count"


def _definition(
    name: str,
    *,
    required: list[str] | None = None,
    additional_properties: bool | None = None,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": required or [],
    }
    if additional_properties is not None:
        parameters["additionalProperties"] = additional_properties
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Call {name}",
            "parameters": parameters,
        },
    }


def _system(definitions: list[dict[str, Any]]) -> str:
    serialized = "\n".join(json.dumps(definition) for definition in definitions)
    return (
        "You are a helpful and harmless assistant.\n\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{serialized}\n</tools>\n\n"
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def _messages(
    definitions: list[dict[str, Any]],
    *,
    call_name: str = "lookup",
    arguments: Any = '{"query": "status"}',
    tool_response: str = '<tool_response>\n{"ok": true}\n</tool_response>',
) -> list[dict[str, Any]]:
    call = json.dumps({"name": call_name, "arguments": arguments})
    return [
        {"role": "system", "content": _system(definitions)},
        {"role": "user", "content": "Check the status."},
        {
            "role": "assistant",
            "content": f"<think>\nUse the tool.\n</think>\n\n<tool_call>\n{call}\n</tool_call>",
        },
        {"role": "tool", "content": tool_response},
        {
            "role": "assistant",
            "content": "<think>\nDone.\n</think>\n\nThe status is good.",
        },
    ]


def test_normalizes_real_cascade_protocol_to_native_logical_row() -> None:
    result = normalize_cascade_tool_messages(_messages([_definition("lookup")]))

    assert result.definition_count == 1
    assert result.call_count == 1
    assert result.response_count == 1
    assert result.duplicate_definition_count == 0
    assert result.tools == [_definition("lookup")]
    assert result.messages[0] == {
        "role": "system",
        "content": "You are a helpful and harmless assistant.",
    }
    assert result.messages[2]["content"] == "<think>\nUse the tool.\n</think>\n\n"
    assert result.messages[2]["tool_calls"] == [
        {
            "type": "function",
            "function": {"name": "lookup", "arguments": {"query": "status"}},
        }
    ]
    assert result.messages[3]["content"] == '{"ok": true}'
    assert result.retained_message_indices == [0, 1, 2, 3, 4]


def test_object_arguments_and_multiple_calls_preserve_order() -> None:
    definitions = [_definition("first"), _definition("second")]
    calls = "\n".join(
        (
            "<tool_call>",
            json.dumps(
                {
                    "name": "first",
                    "arguments": {"query": "one", "limit": 2},
                }
            ),
            "</tool_call>",
            "<tool_call>",
            json.dumps({"name": "second", "arguments": {"query": "two"}}),
            "</tool_call>",
        )
    )
    messages = [
        {"role": "system", "content": _system(definitions)},
        {"role": "user", "content": "Run both."},
        {"role": "assistant", "content": calls},
    ]

    result = normalize_cascade_tool_messages(messages)

    assert result.call_count == 2
    calls = result.messages[-1]["tool_calls"]
    assert [call["function"]["name"] for call in calls] == ["first", "second"]
    assert calls[0]["function"]["arguments"] == {"query": "one", "limit": 2}


def test_duplicate_definitions_are_preserved_and_any_compatible_schema_accepts() -> (
    None
):
    strict = _definition(
        "lookup",
        required=["query", "limit"],
        additional_properties=False,
    )
    query_only = _definition(
        "lookup",
        required=["query"],
        additional_properties=False,
    )

    result = normalize_cascade_tool_messages(
        _messages([strict, query_only], arguments={"query": "status"})
    )

    assert result.tools == [strict, query_only]
    assert result.duplicate_definition_count == 1


@pytest.mark.parametrize(
    "definitions,call_name,arguments,error_code",
    [
        (
            [_definition("lookup")],
            "missing",
            {"query": "status"},
            "undefined_tool_call",
        ),
        (
            [_definition("lookup", required=["query"])],
            "lookup",
            {},
            "missing_required_arguments",
        ),
        (
            [
                _definition(
                    "lookup",
                    required=["query"],
                    additional_properties=False,
                )
            ],
            "lookup",
            {"query": "status", "unknown": 1},
            "additional_properties_forbidden",
        ),
    ],
)
def test_invalid_calls_have_stable_rejection_codes(
    definitions: list[dict[str, Any]],
    call_name: str,
    arguments: dict[str, Any],
    error_code: str,
) -> None:
    with pytest.raises(CascadeToolNormalizationError) as error:
        normalize_cascade_tool_messages(
            _messages(
                definitions,
                call_name=call_name,
                arguments=arguments,
            )
        )
    assert error.value.code == error_code


def test_additional_properties_default_to_allowed() -> None:
    result = normalize_cascade_tool_messages(
        _messages(
            [_definition("lookup", required=["query"])],
            arguments={"query": "status", "unknown": 1},
        )
    )
    assert result.messages[2]["tool_calls"][0]["function"]["arguments"]["unknown"] == 1


@pytest.mark.parametrize(
    "tool_response,error_code",
    [
        (
            "<tool_response><tool_response>x</tool_response></tool_response>",
            "nested_tool_response_wrapper",
        ),
        (
            "prefix <tool_response>x</tool_response>",
            "malformed_tool_response_wrapper",
        ),
        (
            "<tool_response>x",
            "malformed_tool_response_wrapper",
        ),
    ],
)
def test_malformed_tool_response_wrappers_are_rejected(
    tool_response: str,
    error_code: str,
) -> None:
    with pytest.raises(CascadeToolNormalizationError) as error:
        normalize_cascade_tool_messages(
            _messages(
                [_definition("lookup")],
                tool_response=tool_response,
            )
        )
    assert error.value.code == error_code


def test_unwrapped_tool_response_is_preserved() -> None:
    result = normalize_cascade_tool_messages(
        _messages(
            [_definition("lookup")],
            tool_response='{"already": "raw"}',
        )
    )
    assert result.messages[3]["content"] == '{"already": "raw"}'


def test_removed_system_turn_exposes_mask_mapping_and_copies_metadata() -> None:
    messages = _messages([_definition("lookup")])
    messages[0]["content"] = "<tools>" + json.dumps(_definition("lookup")) + "</tools>"
    messages[2]["metadata"] = {"source": ["cascade"]}
    original = deepcopy(messages)

    result = normalize_cascade_tool_messages(messages)

    assert result.retained_message_indices == [1, 2, 3, 4]
    assert [([0, 0, 1, 0, 1])[i] for i in result.retained_message_indices] == [
        0,
        1,
        0,
        1,
    ]
    result.messages[1]["metadata"]["source"].append("changed")
    result.tools[0]["function"]["parameters"]["properties"].clear()
    assert messages == original


def test_merges_existing_and_embedded_tools_without_dropping_duplicate_schemas() -> (
    None
):
    existing = [_definition("lookup", required=["limit"]), _definition("other")]
    original = deepcopy(existing)
    result = normalize_cascade_tool_messages(
        _messages([_definition("lookup")]), tools=existing
    )
    assert result.tools == [*existing, _definition("lookup")]
    assert result.duplicate_definition_count == 1
    assert existing == original


def test_native_calls_and_their_metadata_are_preserved_without_embedded_schema() -> (
    None
):
    messages = [
        {"role": "user", "content": "Check"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Use lookup",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query": "status"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
    ]
    original = deepcopy(messages)
    result = normalize_cascade_tool_messages(messages, tools=[_definition("lookup")])
    expected = deepcopy(original)
    expected[1]["tool_calls"][0]["function"]["arguments"] = {"query": "status"}
    assert result.messages == expected
    assert result.call_count == 1
    assert result.response_count == 1
    result.messages[1]["tool_calls"][0]["id"] = "changed"
    assert messages == original


@pytest.mark.parametrize(
    "arguments",
    [
        json.dumps(json.dumps({"query": "status"})),
        {"query": {"nested": [1, None, True]}},
    ],
)
def test_nested_and_double_encoded_arguments_are_preserved(arguments: Any) -> None:
    result = normalize_cascade_tool_messages(
        _messages([_definition("lookup")], arguments=arguments)
    )
    expected = (
        json.loads(json.loads(arguments)) if isinstance(arguments, str) else arguments
    )
    assert result.messages[2]["tool_calls"][0]["function"]["arguments"] == expected


@pytest.mark.parametrize(
    "content,error_code",
    [
        (
            '<tool_call>{"name":"lookup","arguments":{}}</tool_call>after',
            "interleaved_tool_call_content",
        ),
        (
            '<tool_call>{"name":"lookup","arguments":[]}</tool_call>',
            "invalid_tool_arguments",
        ),
        ("<tool_call>not json</tool_call>", "invalid_tool_call_json"),
        ('<tool_call>{"name":"lookup"}', "malformed_tool_call_wrapper"),
    ],
)
def test_invalid_embedded_calls_are_rejected(content: str, error_code: str) -> None:
    messages = _messages([_definition("lookup")])
    messages[2]["content"] = content
    with pytest.raises(CascadeToolNormalizationError) as error:
        normalize_cascade_tool_messages(messages)
    assert error.value.code == error_code


def test_processor_to_native_renderer_uses_logical_tools_and_selected_turns() -> None:
    result = normalize_cascade_tool_messages(_messages([_definition("lookup")]))
    transported = deepcopy(result.messages)
    transported[2]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"query": "status"}
    )
    datum = chat_kd_processor(
        {
            "sample_id": "cascade-fixture#0",
            "messages": transported,
            "tools_json": json.dumps(result.tools),
            "message_loss_mask": [0, 0, 1, 0, 0],
        },
        TaskDataSpec(),
        None,
        None,
        0,
    )
    tokenizer_backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    tokenizer.chat_template = (
        Path(__file__).parent / "fixtures/xtoken/qwen_thinking_history_reference.jinja"
    ).read_text()
    document = _render_and_tokenize_chat(
        tokenizer,
        datum["message_log"],
        4096,
        tools=datum["tools"],
        message_loss_mask=datum["message_loss_mask"],
        include_thinking_in_loss=True,
        native_thinking_alignment=True,
        skip_overlength=True,
    )
    assert document.source_turn_indices == [2]
    assert (
        '<tool_call>\n{"name": "lookup", "arguments": {"query": "status"}}\n</tool_call>'
        in document.rendered_text
    )
    assert document.rendered_text.count('"parameters":') == 1
    assert document.rendered_text.count("<tool_response>") == 1
    supervised = "".join(
        document.rendered_text[start:end]
        for (start, end), include in zip(document.offsets, document.assistant_mask)
        if include
    )
    assert "status" in supervised
    assert "Use" in supervised
    assert "harmless" not in supervised
    assert '"ok"' not in supervised
    assert "good" not in supervised
    assert datum["sample_id"] == "cascade-fixture#0"
