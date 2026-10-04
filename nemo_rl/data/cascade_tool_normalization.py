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

"""Normalize Cascade's embedded tool protocol into native chat-template inputs."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, NoReturn, Sequence, cast

CASCADE_TOOL_NORMALIZATION_VERSION = "cascade-tool-native-v2"
CASCADE_TOOLS_JSON_KEY = "tools_json"

_TOOLS_BLOCK_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_RESPONSE_FULL_RE = re.compile(
    r"\A\s*<tool_response>\s*(.*?)\s*</tool_response>\s*\Z",
    re.DOTALL | re.IGNORECASE,
)
_CALL_FORMAT_INSTRUCTION_RE = re.compile(
    r"""
    (?:^|\n)[ \t]*
    For[ \t]+each[ \t]+function[ \t]+call,[ \t]*
    .*?
    <tool_call>\s*
    \{\s*"name"\s*:\s*<function-name>\s*,\s*
    "arguments"\s*:\s*<args-json-object>\s*\}\s*
    </tool_call>[ \t]*(?=\n|$)
    """,
    re.DOTALL | re.IGNORECASE | re.VERBOSE,
)
_TOOLS_HEADING_RE = re.compile(r"(?im)^[ \t]*#[ \t]*Tools[ \t]*(?:\n|$)")
_TOOLS_PREAMBLE_RE = re.compile(
    r"""(?im)^[ \t]*
    (?:
       You[ \t]+have[ \t]+access[ \t]+to[ \t]+the[ \t]+following
       [ \t]+(?:functions|tools)
     | You[ \t]+may[ \t]+call[ \t]+the[ \t]+following[ \t]+(?:functions|tools)
     | You[ \t]+may[ \t]+call[ \t]+one[ \t]+or[ \t]+more[ \t]+functions
       [ \t]+to[ \t]+assist[ \t]+with[ \t]+the[ \t]+user[ \t]+query\.
     | You[ \t]+are[ \t]+provided[ \t]+with[ \t]+function[ \t]+signatures
       [ \t]+within[ \t]+<tools></tools>[ \t]+XML[ \t]+tags
    )
    :?[ \t]*(?:\n|$)
    """,
    re.VERBOSE,
)


class CascadeToolNormalizationError(ValueError):
    """A row cannot be normalized without changing or dropping semantics."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        # Exception pickling reconstructs the type from args. Keep both
        # constructor arguments so worker failures reach the parent process.
        super().__init__(code, message)

    def __str__(self) -> str:
        return f"{self.code}: {self.args[1]}"


@dataclass(frozen=True)
class CascadeToolNormalizationResult:
    """Normalized logical conversation plus audit counters."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    retained_message_indices: list[int]
    definition_count: int
    call_count: int
    response_count: int
    duplicate_definition_count: int


def _fail(code: str, message: str) -> NoReturn:
    raise CascadeToolNormalizationError(code, message)


def _parse_json_objects(text: str) -> list[dict[str, Any]]:
    """Parse adjacent JSON objects (or one JSON array) without line assumptions."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 3:
            _fail("invalid_tool_definitions", "empty fenced <tools> body")
        stripped = "\n".join(lines[1:-1]).strip()
    if not stripped:
        _fail("missing_tool_definitions", "<tools> body is empty")

    decoder = json.JSONDecoder()
    values: list[Any] = []
    position = 0
    while position < len(stripped):
        while position < len(stripped) and (
            stripped[position].isspace() or stripped[position] == ","
        ):
            position += 1
        if position == len(stripped):
            break
        try:
            value, position = decoder.raw_decode(stripped, position)
        except json.JSONDecodeError as error:
            _fail(
                "invalid_tool_definitions",
                f"cannot parse <tools> JSON at character {error.pos}: {error.msg}",
            )
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)

    definitions: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            _fail(
                "invalid_tool_definition",
                f"definition {index} is {type(value).__name__}, expected object",
            )
        definitions.append(value)
    if not definitions:
        _fail("missing_tool_definitions", "<tools> contains no definitions")
    return definitions


def _normalize_tool_definition(
    definition: dict[str, Any],
    *,
    index: int,
) -> dict[str, Any]:
    if "function" in definition:
        if definition.get("type", "function") != "function":
            _fail(
                "unsupported_tool_type",
                f"definition {index} has type={definition.get('type')!r}",
            )
        function = definition["function"]
    else:
        function = definition
    if not isinstance(function, dict):
        _fail(
            "invalid_tool_definition",
            f"definition {index} function is not an object",
        )

    normalized_function = copy.deepcopy(function)
    name = normalized_function.get("name")
    if not isinstance(name, str) or not name:
        _fail(
            "invalid_tool_definition",
            f"definition {index} is missing a non-empty function name",
        )

    parameters = normalized_function.get("parameters")
    if parameters is None:
        parameters = {"type": "object", "properties": {}}
        normalized_function["parameters"] = parameters
    if not isinstance(parameters, dict):
        _fail(
            "invalid_tool_definition",
            f"definition {index} parameters must be an object",
        )
    properties = parameters.get("properties", {})
    if properties is not None and not isinstance(properties, dict):
        _fail(
            "invalid_tool_definition",
            f"definition {index} parameters.properties must be an object",
        )
    required = parameters.get("required", [])
    if not isinstance(required, list) or any(
        not isinstance(field, str) for field in required
    ):
        _fail(
            "invalid_tool_definition",
            f"definition {index} parameters.required must be a string list",
        )
    additional = parameters.get("additionalProperties", True)
    if not isinstance(additional, (bool, dict)):
        _fail(
            "invalid_tool_definition",
            f"definition {index} additionalProperties must be boolean or object",
        )
    return {"type": "function", "function": normalized_function}


def _clean_system_content(content: str, tools_match: re.Match[str]) -> str:
    cleaned = content[: tools_match.start()] + content[tools_match.end() :]
    cleaned, instruction_count = _CALL_FORMAT_INSTRUCTION_RE.subn("\n", cleaned)
    if "For each function call" in cleaned:
        _fail(
            "unsupported_call_instruction",
            "found an unrecognized Cascade function-call instruction",
        )
    if instruction_count > 1:
        _fail(
            "multiple_call_instructions",
            f"found {instruction_count} Cascade function-call instructions",
        )
    cleaned = _TOOLS_HEADING_RE.sub("", cleaned)
    cleaned = _TOOLS_PREAMBLE_RE.sub("", cleaned)
    cleaned = re.sub(r"\n[ \t]+\n", "\n\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _parse_call_payload(
    payload: str, *, message_index: int
) -> tuple[str, dict[str, Any]]:
    try:
        event = json.loads(payload)
    except json.JSONDecodeError as error:
        _fail(
            "invalid_tool_call_json",
            f"message {message_index} call JSON failed at character {error.pos}: {error.msg}",
        )
    return _parse_call_event(event, message_index=message_index)


def _parse_call_event(event: Any, *, message_index: int) -> tuple[str, dict[str, Any]]:
    if not isinstance(event, dict):
        _fail(
            "invalid_tool_call",
            f"message {message_index} call payload must be an object",
        )
    name = event.get("name")
    if not isinstance(name, str) or not name:
        _fail(
            "invalid_tool_call",
            f"message {message_index} call is missing a non-empty name",
        )

    arguments = event.get("arguments", {})
    decode_depth = 0
    while isinstance(arguments, str) and decode_depth < 2:
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            _fail(
                "invalid_tool_arguments_json",
                f"message {message_index} call {name!r} arguments failed JSON decoding: {error.msg}",
            )
        decode_depth += 1
    if not isinstance(arguments, dict):
        _fail(
            "invalid_tool_arguments",
            f"message {message_index} call {name!r} arguments must decode to an object",
        )
    return cast(str, name), cast(dict[str, Any], arguments)


def _definition_accepts_arguments(
    *,
    definition: dict[str, Any],
    arguments: dict[str, Any],
) -> tuple[bool, set[str], set[str]]:
    function = definition["function"]
    parameters = function.get("parameters") or {}
    required = set(parameters.get("required") or [])
    missing = required.difference(arguments)
    properties = parameters.get("properties") or {}
    extra = set(arguments).difference(properties)
    reject_extra = parameters.get("additionalProperties", True) is False
    return (
        not missing and not (reject_extra and extra),
        missing,
        extra if reject_extra else set(),
    )


def _validate_call(
    name: str,
    arguments: dict[str, Any],
    definitions_by_name: dict[str, list[dict[str, Any]]],
    *,
    message_index: int,
) -> None:
    candidates = definitions_by_name.get(name, [])
    if not candidates:
        _fail(
            "undefined_tool_call",
            f"message {message_index} calls undefined function {name!r}",
        )

    failures: list[tuple[set[str], set[str]]] = []
    for definition in candidates:
        accepted, missing, extra = _definition_accepts_arguments(
            definition=definition,
            arguments=arguments,
        )
        if accepted:
            return
        failures.append((missing, extra))

    all_missing = sorted(set.intersection(*(missing for missing, _ in failures)))
    if all_missing:
        _fail(
            "missing_required_arguments",
            f"message {message_index} call {name!r} is missing required fields {all_missing}",
        )
    rejected_extra = sorted(set.intersection(*(extra for _, extra in failures)))
    if rejected_extra:
        _fail(
            "additional_properties_forbidden",
            f"message {message_index} call {name!r} has forbidden fields {rejected_extra}",
        )
    _fail(
        "tool_call_schema_mismatch",
        f"message {message_index} call {name!r} matches no duplicate definition schema",
    )


def _normalize_assistant_calls(
    content: str,
    definitions_by_name: dict[str, list[dict[str, Any]]],
    *,
    message_index: int,
) -> tuple[str, list[dict[str, Any]]]:
    matches = list(_TOOL_CALL_BLOCK_RE.finditer(content))
    if not matches:
        if "<tool_call" in content.lower() or "</tool_call>" in content.lower():
            _fail(
                "malformed_tool_call_wrapper",
                f"message {message_index} contains an unmatched tool-call marker",
            )
        return content, []

    # Native templates render prose before the call list. Refuse content that
    # cannot be expressed in that order instead of silently moving its text.
    prefix = content[: matches[0].start()]
    if "<tool_call" in prefix.lower() or "</tool_call>" in prefix.lower():
        _fail(
            "malformed_tool_call_wrapper",
            f"message {message_index} contains an unmatched tool-call marker",
        )
    calls: list[dict[str, Any]] = []
    position = matches[0].start()
    for match in matches:
        if content[position : match.start()].strip():
            _fail(
                "interleaved_tool_call_content",
                f"message {message_index} has prose between tool calls",
            )
        name, arguments = _parse_call_payload(
            match.group(1),
            message_index=message_index,
        )
        _validate_call(
            name,
            arguments,
            definitions_by_name,
            message_index=message_index,
        )
        calls.append(
            {"type": "function", "function": {"name": name, "arguments": arguments}}
        )
        position = match.end()
    if content[position:].strip():
        _fail(
            "interleaved_tool_call_content",
            f"message {message_index} has prose after tool calls",
        )
    return prefix, calls


def _unwrap_tool_response(content: str, *, message_index: int) -> tuple[str, bool]:
    lowered = content.lower()
    has_open = "<tool_response>" in lowered
    has_close = "</tool_response>" in lowered
    if not has_open and not has_close:
        return content, False
    match = _TOOL_RESPONSE_FULL_RE.fullmatch(content)
    if match is None:
        _fail(
            "malformed_tool_response_wrapper",
            f"message {message_index} tool response is not enclosed exactly once",
        )
    payload = match.group(1)
    payload_lower = payload.lower()
    if "<tool_response>" in payload_lower or "</tool_response>" in payload_lower:
        _fail(
            "nested_tool_response_wrapper",
            f"message {message_index} contains nested tool-response wrappers",
        )
    return payload.strip(), True


def normalize_cascade_tool_messages(
    messages: Sequence[dict[str, Any]],
    *,
    tools: Sequence[dict[str, Any]] | None = None,
) -> CascadeToolNormalizationResult:
    """Convert Cascade wrappers to native tool schemas and logical calls.

    Existing schemas precede extracted schemas, including duplicate names.
    Existing logical calls retain their fields. All returned payloads are deep
    copies; ``retained_message_indices`` maps each output turn to its input so
    callers can realign masks when an emptied system message is removed.
    """
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        _fail("invalid_messages", "messages must be a sequence")

    copied_messages: list[dict[str, Any]] = []
    tool_block_locations: list[tuple[int, re.Match[str]]] = []
    typed_messages = cast(Sequence[dict[str, Any]], messages)
    for message_index, source_message in enumerate(typed_messages):
        if not isinstance(source_message, dict):
            _fail(
                "invalid_message",
                f"message {message_index} is not an object",
            )
        message = copy.deepcopy(source_message)
        content = message.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            _fail(
                "invalid_message_content",
                f"message {message_index} content is not a string",
            )
        copied_messages.append(message)
        if message.get("role") == "system":
            outside_blocks = _TOOLS_BLOCK_RE.sub("", content).lower()
            if "<tools>" in outside_blocks or "</tools>" in outside_blocks:
                _fail(
                    "malformed_tools_wrapper",
                    f"message {message_index} contains an unmatched tools marker",
                )
            tool_block_locations.extend(
                (message_index, match)
                for match in _TOOLS_BLOCK_RE.finditer(content)
                if match.group(1).strip()
            )

    if len(tool_block_locations) > 1 or (not tool_block_locations and tools is None):
        _fail(
            "tool_block_count",
            f"expected exactly one system <tools> block, found {len(tool_block_locations)}",
        )
    if tools is not None and (
        not isinstance(tools, Sequence)
        or isinstance(tools, (str, bytes))
        or any(not isinstance(tool, dict) for tool in tools)
    ):
        _fail("invalid_tool_definitions", "tools must be a sequence of objects")
    source_definitions = (
        list(cast(Sequence[dict[str, Any]], tools)) if tools is not None else []
    )
    if tool_block_locations:
        system_index, tools_match = tool_block_locations[0]
        source_definitions.extend(_parse_json_objects(tools_match.group(1)))
        copied_messages[system_index]["content"] = _clean_system_content(
            copied_messages[system_index]["content"], tools_match
        )
    normalized_tools = [
        _normalize_tool_definition(definition, index=index)
        for index, definition in enumerate(source_definitions)
    ]
    names = [tool["function"]["name"] for tool in normalized_tools]
    name_counts = Counter(names)
    duplicate_definition_count = sum(count - 1 for count in name_counts.values())

    definitions_by_name: dict[str, list[dict[str, Any]]] = {}
    for tool in normalized_tools:
        definitions_by_name.setdefault(tool["function"]["name"], []).append(tool)

    call_count = 0
    response_count = 0
    normalized_messages: list[dict[str, Any]] = []
    retained_message_indices: list[int] = []
    for message_index, message in enumerate(copied_messages):
        role = message.get("role")
        if role == "assistant":
            content, calls = _normalize_assistant_calls(
                message.get("content") or "",
                definitions_by_name,
                message_index=message_index,
            )
            native_calls = message.get("tool_calls")
            if native_calls is not None:
                if not isinstance(native_calls, list) or any(
                    not isinstance(call, dict) for call in native_calls
                ):
                    _fail("invalid_tool_call", "tool_calls must be a list of objects")
                for call in native_calls:
                    if call.get("type", "function") != "function":
                        _fail(
                            "unsupported_tool_type", "only function calls are supported"
                        )
                    function = call.get("function", call)
                    name, arguments = _parse_call_event(
                        function, message_index=message_index
                    )
                    _validate_call(
                        name,
                        arguments,
                        definitions_by_name,
                        message_index=message_index,
                    )
                    function["arguments"] = arguments
                call_count += len(native_calls)
            if calls:
                if native_calls:
                    _fail(
                        "mixed_tool_call_formats",
                        f"message {message_index} combines embedded and logical tool calls",
                    )
                message["content"] = content
                message["tool_calls"] = calls
                call_count += len(calls)
        elif role == "tool":
            message["content"], _ = _unwrap_tool_response(
                message.get("content") or "",
                message_index=message_index,
            )
            response_count += 1
        if role == "system" and not message.get("content"):
            continue
        normalized_messages.append(message)
        retained_message_indices.append(message_index)

    return CascadeToolNormalizationResult(
        messages=normalized_messages,
        tools=normalized_tools,
        retained_message_indices=retained_message_indices,
        definition_count=len(normalized_tools),
        call_count=call_count,
        response_count=response_count,
        duplicate_definition_count=duplicate_definition_count,
    )
