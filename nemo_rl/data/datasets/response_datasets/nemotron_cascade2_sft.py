# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Prepare Cascade chat sources for the existing response-data pipeline."""

from __future__ import annotations

import errno
import gzip
import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset, load_from_disk

from nemo_rl.data import CascadeDatasetConfig, CascadeSubsetConfig
from nemo_rl.data.cascade_tool_normalization import (
    CASCADE_TOOL_NORMALIZATION_VERSION,
    CASCADE_TOOLS_JSON_KEY,
    CascadeToolNormalizationError,
    normalize_cascade_tool_messages,
)
from nemo_rl.data.chat_utils import normalize_message_loss_mask
from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.response_datasets.response_dataset import (
    attach_source_sample_id,
)
from nemo_rl.data.datasets.utils import merge_datasets, weighted_merge_datasets

_LOG = logging.getLogger(__name__)
_DEFAULT_DATASET_PATH = "nvidia/Nemotron-Cascade-2-SFT-Data"
_STAGE2_DATASET_PATH = "nvidia/Nemotron-Cascade-SFT-Stage-2"
_CACHE_VERSION = "cascade-prepared-v1"
_CACHE_MANIFEST = "cascade_preparation_manifest.json"
_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_DIAGNOSTIC_COLUMNS = (
    "_cascade_error",
    "_cascade_definitions",
    "_cascade_calls",
    "_cascade_responses",
    "_cascade_duplicates",
)


class CascadePreparedDataset:
    """Expose lossless chat rows from an Arrow-backed, JSON-encoded dataset.

    Like ``PreservingDataset``, this implements the map-style dataset interface
    accepted by ``merge_datasets`` and ``AllTaskProcessedDataset``. Rows remain
    disk-backed instead of materializing the entire corpus as Python objects.
    JSON storage avoids Arrow changing arbitrary message/tool argument schemas
    or inferring a null schema from an initial batch of dropped rows.
    """

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int | str) -> Any:
        if isinstance(index, str):
            if index in {"messages", "message_loss_mask"}:
                return [json.loads(value) for value in self._dataset[f"{index}_json"]]
            return self._dataset[index]
        row = dict(self._dataset[index])
        row["messages"] = json.loads(row.pop("messages_json"))
        row["message_loss_mask"] = json.loads(row.pop("message_loss_mask_json"))
        return row

    def to_list(self) -> list[dict[str, Any]]:
        """Materialize logical rows for inspection of small prepared datasets."""
        return [self[index] for index in range(len(self))]


def _read_local_rows(
    files: list[str], *, data_path: str, subset: str
) -> Iterator[dict[str, Any]]:
    """Attach identity and protect schemas before local JSON reaches Arrow."""
    ordinal = 0
    for filename in files:
        opener = gzip.open if filename.endswith(".gz") else open
        with opener(filename, "rt", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                row.update(
                    attach_source_sample_id(
                        row, ordinal, data_path=data_path, subset=subset, split="train"
                    )
                )
                ordinal += 1
                # Keep only the adapter's logical input columns; unrelated source
                # metadata may also have incompatible nested Arrow schemas.
                yield {
                    "sample_id": row["sample_id"],
                    # Delay schema inference until invalid tool rows have been
                    # handled; malformed native calls must reach the policy.
                    "_cascade_source_messages_json": json.dumps(
                        row.get("messages"), ensure_ascii=False
                    ),
                    CASCADE_TOOLS_JSON_KEY: json.dumps(
                        row.get("tools"), ensure_ascii=False
                    ),
                    "message_loss_mask_json": json.dumps(row.get("message_loss_mask")),
                }


def _task_name(dataset_path: str, subset: str, prefix: str | None) -> str:
    if prefix is None and dataset_path == _DEFAULT_DATASET_PATH and subset == "math":
        return "Nemotron-Cascade-2-SFT-Math"
    if prefix is None:
        prefix = (
            "Nemotron-Cascade-2-SFT"
            if dataset_path == _DEFAULT_DATASET_PATH
            else Path(dataset_path).name
        )
    return f"{prefix}-{subset.replace('_', '-')}"


def _format_cascade_row(
    data: dict[str, Any],
    *,
    task_name: str,
    subset: str,
    strip_thinking: bool,
    normalize_tool_calls: bool,
    tool_call_invalid_policy: str,
) -> dict[str, Any]:
    messages = (
        json.loads(data["_cascade_source_messages_json"])
        if "_cascade_source_messages_json" in data
        else data["messages"]
    )
    mask = (
        json.loads(data["message_loss_mask_json"])
        if "message_loss_mask_json" in data
        else data.get("message_loss_mask")
    )
    if mask is not None:
        mask = normalize_message_loss_mask(messages, mask)
    tools = data.get("tools")
    if CASCADE_TOOLS_JSON_KEY in data:
        tools = json.loads(data[CASCADE_TOOLS_JSON_KEY])
    output = {
        "sample_id": data["sample_id"],
        "task_name": task_name,
        "source_subset": subset,
        "messages_json": "[]",
        "message_loss_mask_json": "null",
        CASCADE_TOOLS_JSON_KEY: "",
        "_cascade_error": "",
        "_cascade_definitions": 0,
        "_cascade_calls": 0,
        "_cascade_responses": 0,
        "_cascade_duplicates": 0,
    }
    try:
        if not isinstance(messages, list) or any(
            not isinstance(message, dict) for message in messages
        ):
            raise CascadeToolNormalizationError(
                "invalid_messages", "messages must be a list of objects"
            )
        if not messages or messages[-1].get("role") != "assistant":
            raise CascadeToolNormalizationError(
                "row_not_assistant_terminated",
                "Cascade rows must end with an assistant message",
            )
        if normalize_tool_calls:
            normalized = normalize_cascade_tool_messages(messages, tools=tools)
            messages, tools = normalized.messages, normalized.tools
            if mask is not None:
                mask = [mask[index] for index in normalized.retained_message_indices]
            output.update(
                {
                    "_cascade_definitions": normalized.definition_count,
                    "_cascade_calls": normalized.call_count,
                    "_cascade_responses": normalized.response_count,
                    "_cascade_duplicates": normalized.duplicate_definition_count,
                }
            )
    except CascadeToolNormalizationError as error:
        if not normalize_tool_calls or tool_call_invalid_policy == "error":
            raise
        output["_cascade_error"] = error.code
        return output
    if tools is not None and (
        not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools)
    ):
        raise TypeError("Cascade tools must be a list of objects or None")
    if strip_thinking:
        messages = deepcopy(messages)
        for message in messages:
            if message.get("role") == "assistant" and isinstance(
                message.get("content"), str
            ):
                message["content"] = _THINK_PATTERN.sub("", message["content"]).lstrip()
    output.update(
        {
            "messages_json": json.dumps(
                messages, ensure_ascii=False, separators=(",", ":")
            ),
            "message_loss_mask_json": json.dumps(mask),
            CASCADE_TOOLS_JSON_KEY: json.dumps(
                tools, ensure_ascii=False, separators=(",", ":")
            )
            if tools is not None
            else "",
        }
    )
    return output


def _cache_path(root: str, *, dataset_path: str, subset: str, split: str) -> Path:
    namespace = json.dumps([dataset_path, subset, split], ensure_ascii=True)
    digest = hashlib.sha256(namespace.encode()).hexdigest()
    return Path(root) / _CACHE_VERSION / digest


def _load_cached_subset(path: Path, expected: dict[str, Any]) -> Dataset:
    manifest_path = path / _CACHE_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(f"Cascade cache is missing {_CACHE_MANIFEST}: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise RuntimeError(
            f"Cascade cache identity mismatch at {path}: {', '.join(mismatches)}"
        )
    dataset = load_from_disk(str(path))
    required = {
        "sample_id",
        "messages_json",
        "task_name",
        "source_subset",
        "message_loss_mask_json",
        CASCADE_TOOLS_JSON_KEY,
    }
    if not required.issubset(dataset.column_names):
        raise RuntimeError(
            f"Cascade cache missing required columns at {path}: {sorted(required - set(dataset.column_names))}"
        )
    if len(dataset) != manifest["accepted_rows"]:
        raise RuntimeError(f"Cascade cache row count mismatch at {path}")
    if any(not value or not value.strip() for value in dataset["sample_id"]):
        raise RuntimeError(f"Cascade cache contains invalid sample_id at {path}")
    return dataset


def _publish_cache(dataset: Dataset, path: Path, manifest: dict[str, Any]) -> Dataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{path.name}.tmp-", dir=path.parent))
    try:
        dataset.save_to_disk(str(temporary))
        (temporary / _CACHE_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.chmod(temporary, path.parent.stat().st_mode & 0o777)
        try:
            os.rename(temporary, path)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not path.is_dir():
                raise
            return _load_cached_subset(path, manifest)
    finally:
        if temporary.is_dir():
            shutil.rmtree(temporary)
    return dataset


def _load_subset(
    *,
    dataset_path: str,
    revision: str | None,
    subset: str,
    split: str,
    task_name: str,
    config: CascadeDatasetConfig,
    normalize_tool_calls: bool,
    tool_call_invalid_policy: str,
) -> Dataset:
    local_digest = None
    if os.path.isdir(dataset_path):
        if split != "train":
            raise ValueError(
                "Local subset-organized Cascade JSONL supports split='train' only"
            )
        root = Path(dataset_path) / subset
        files = (
            sorted(
                str(path)
                for path in root.iterdir()
                if path.name.endswith((".jsonl", ".jsonl.gz"))
            )
            if root.is_dir()
            else []
        )
        if not files:
            raise FileNotFoundError(
                f"No JSONL files for subset={subset!r} under {dataset_path}"
            )
        digest = hashlib.sha256()
        for filename in files:
            digest.update(filename.encode())
            with open(filename, "rb") as source:
                digest.update(hashlib.file_digest(source, "sha256").digest())
        local_digest = digest.hexdigest()
        # Include the ingress schema version as well as content in HF's cache key.
        source_fingerprint = hashlib.sha256(
            f"{_CACHE_VERSION}:{local_digest}".encode()
        ).hexdigest()
        dataset = Dataset.from_generator(
            _read_local_rows,
            gen_kwargs={"files": files, "data_path": dataset_path, "subset": subset},
            fingerprint=source_fingerprint,
        )
    else:
        load_kwargs = {"split": split}
        if revision is not None:
            load_kwargs["revision"] = revision
        dataset = load_dataset(dataset_path, subset, **load_kwargs)
    expected = {
        "cache_version": _CACHE_VERSION,
        "normalization_version": CASCADE_TOOL_NORMALIZATION_VERSION,
        "dataset_path": dataset_path,
        "revision": revision,
        "source_fingerprint": dataset._fingerprint,
        "local_content_sha256": local_digest,
        "subset": subset,
        "split": split,
        "task_name": task_name,
        "strip_thinking": config.strip_thinking,
        "normalize_tool_calls": normalize_tool_calls,
        "tool_call_invalid_policy": tool_call_invalid_policy,
    }
    cache_path = (
        _cache_path(
            config.cached_path, dataset_path=dataset_path, subset=subset, split=split
        )
        if config.cached_path
        else None
    )
    if cache_path is not None and cache_path.exists():
        return _load_cached_subset(cache_path, expected)
    if not len(dataset):
        raise ValueError(f"Cascade subset {subset!r} is empty")
    source_rows = len(dataset)
    dataset = dataset.map(
        attach_source_sample_id,
        with_indices=True,
        remove_columns=["sample_id"] if "sample_id" in dataset.column_names else None,
        fn_kwargs={"data_path": dataset_path, "subset": subset, "split": split},
        num_proc=config.map_num_proc,
    )
    dataset = dataset.map(
        _format_cascade_row,
        fn_kwargs={
            "task_name": task_name,
            "subset": subset,
            "strip_thinking": config.strip_thinking,
            "normalize_tool_calls": normalize_tool_calls,
            "tool_call_invalid_policy": tool_call_invalid_policy,
        },
        remove_columns=dataset.column_names,
        num_proc=config.map_num_proc,
    )
    reasons = Counter(reason for reason in dataset["_cascade_error"] if reason)
    totals = {
        column.removeprefix("_cascade_"): sum(dataset[column])
        for column in _DIAGNOSTIC_COLUMNS[1:]
    }
    if reasons:
        _LOG.warning(
            "Cascade subset %s: dropping %d/%d invalid rows; reasons=%s",
            subset,
            sum(reasons.values()),
            source_rows,
            dict(sorted(reasons.items())),
        )
        dataset = dataset.filter(
            lambda reason: not reason,
            input_columns=["_cascade_error"],
            num_proc=config.map_num_proc,
        )
    dataset = dataset.remove_columns(list(_DIAGNOSTIC_COLUMNS))
    if not len(dataset):
        raise ValueError(
            f"Cascade subset {subset!r} has no valid rows; rejection reasons={dict(reasons)}"
        )
    manifest = {
        **expected,
        "source_rows": source_rows,
        "accepted_rows": len(dataset),
        "rejected_rows": sum(reasons.values()),
        "rejection_reasons": dict(reasons),
        "accepted_totals": totals,
    }
    _LOG.info("Cascade subset %s preparation: %s", subset, manifest)
    return (
        _publish_cache(dataset, cache_path, manifest)
        if cache_path is not None
        else dataset
    )


class NemotronCascade2SFTDataset(RawDataset):
    """Load Cascade subsets; hold out sources before any weighted repetition.

    The SFT name identifies the source corpus, not the training objective.
    Use ``chat_kd_processor`` for off-policy distillation. Structured preparation
    options live in :class:`CascadeDatasetConfig` under the ``cascade`` key.
    """

    default_dataset_path = _DEFAULT_DATASET_PATH

    def __init__(
        self,
        split: str = "train",
        split_validation_size: float = 0.05,
        seed: int = 42,
        max_samples: int | None = None,
        cascade: CascadeDatasetConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        config = (
            CascadeDatasetConfig.model_validate(cascade)
            if cascade is not None
            else CascadeDatasetConfig()
        )
        if not math.isfinite(split_validation_size) or split_validation_size < 0:
            raise ValueError("split_validation_size must be finite and nonnegative")
        if split_validation_size > 1 and not float(split_validation_size).is_integer():
            raise ValueError("An absolute split_validation_size must be an integer")
        dataset_path = config.dataset_path or self.default_dataset_path
        specs = config.subsets or [CascadeSubsetConfig(name=config.subset)]
        training: list[Dataset] = []
        validation: list[Dataset] = []
        weights: list[float] = []
        names: list[str] = []
        for index, spec in enumerate(specs):
            if spec.weight == 0:
                continue
            path = spec.dataset_path or dataset_path
            if os.path.isdir(path):
                path = str(Path(path).resolve())
            task_name = _task_name(path, spec.name, config.task_name_prefix)
            prepared = _load_subset(
                dataset_path=path,
                revision=spec.revision
                if spec.revision is not None
                else config.revision,
                subset=spec.name,
                split=split,
                task_name=task_name,
                config=config,
                normalize_tool_calls=spec.normalize_tool_calls
                if spec.normalize_tool_calls is not None
                else config.normalize_tool_calls,
                tool_call_invalid_policy=spec.tool_call_invalid_policy
                if spec.tool_call_invalid_policy is not None
                else config.tool_call_invalid_policy,
            )
            cap = (
                spec.max_samples
                if spec.max_samples is not None
                else config.max_samples_per_subset
            )
            if cap is not None:
                prepared = prepared.shuffle(seed=seed + index).select(
                    range(min(cap, len(prepared)))
                )
            # Preserve the historical math adapter's pre-split sample cap.
            if config.subsets is None and max_samples is not None and max_samples > 0:
                prepared = prepared.shuffle(seed=seed).select(
                    range(min(max_samples, len(prepared)))
                )
            if split_validation_size > 0:
                test_size = (
                    int(split_validation_size)
                    if split_validation_size >= 1
                    else split_validation_size
                )
                parts = prepared.train_test_split(test_size=test_size, seed=seed)
                training.append(parts["train"])
                validation.append(parts["test"])
            else:
                training.append(prepared)
            weights.append(spec.weight)
            names.append(task_name)
        if len(names) != len(set(names)):
            raise ValueError(
                "Cascade subsets must produce distinct task names; change subset names or task_name_prefix"
            )
        if validation:
            train_ids = {
                sample_id for data in training for sample_id in data["sample_id"]
            }
            val_ids = {
                sample_id for data in validation for sample_id in data["sample_id"]
            }
            if train_ids.intersection(val_ids):
                raise ValueError(
                    "Cascade source sample_id overlap between training and validation; source IDs must identify distinct examples"
                )
        self._subset_task_names = tuple(names)
        self.task_name = (
            names[0] if config.subsets is None else f"Cascade-Mix({','.join(names)})"
        )
        self.dataset = (
            weighted_merge_datasets(
                training, weights, seed=seed, stopping_strategy=config.stopping_strategy
            )
            if config.subsets is not None
            else training[0]
        )
        if config.subsets is not None and max_samples is not None and max_samples > 0:
            self.dataset = self.dataset.shuffle(seed=seed).select(
                range(min(max_samples, len(self.dataset)))
            )
        self.val_dataset = merge_datasets(validation) if validation else None
        self.dataset = CascadePreparedDataset(self.dataset)
        if self.val_dataset is not None:
            self.val_dataset = CascadePreparedDataset(self.val_dataset)

    def get_task_names(self) -> tuple[str, ...]:
        """Return the declared task names carried by prepared subset rows."""
        return self._subset_task_names


class NemotronCascadeSFTStage2Dataset(NemotronCascade2SFTDataset):
    """Load the earlier Cascade Stage-2 release with the same preparation API."""

    default_dataset_path = _STAGE2_DATASET_PATH


# Preserve the existing registry name, constructor defaults, and math task alias.
NemotronCascade2SFTMathDataset = NemotronCascade2SFTDataset
