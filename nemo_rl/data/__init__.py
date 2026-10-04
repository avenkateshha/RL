# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math
from collections.abc import Mapping
from typing import Annotated, Literal, NotRequired, Self, TypedDict, Union

from pydantic import BaseModel, BeforeValidator, Field, model_validator

from nemo_rl.data.energon.config import EnergonLoaderConfig, EnergonSourceConfig

_CascadeString = Annotated[str, Field(min_length=1, pattern=r"\S")]


class CascadeSubsetConfig(BaseModel, extra="forbid"):
    """One Cascade subset and its relative sampling weight.

    Optional source and preparation settings override the parent dataset's
    settings. A zero weight excludes the subset without loading it.
    """

    name: _CascadeString
    weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1.0
    dataset_path: _CascadeString | None = None
    revision: _CascadeString | None = None
    max_samples: Annotated[int, Field(ge=1)] | None = None
    normalize_tool_calls: bool | None = None
    tool_call_invalid_policy: Literal["drop", "error"] | None = None


class CascadeDatasetConfig(BaseModel, extra="forbid"):
    """Cascade source and CPU preparation settings shared across subsets.

    ``dataset_path=None`` selects the release associated with the registered
    dataset name. ``subsets=None`` loads ``subset`` without weighted repetition.
    Prepared caches are optional and validated against the current source and
    transformation settings before reuse.
    """

    dataset_path: _CascadeString | None = None
    revision: _CascadeString | None = None
    subset: _CascadeString = "math"
    subsets: Annotated[list[CascadeSubsetConfig], Field(min_length=1)] | None = None
    cached_path: _CascadeString | None = None
    map_num_proc: Annotated[int, Field(ge=1)] = 1
    strip_thinking: bool = False
    task_name_prefix: _CascadeString | None = None
    normalize_tool_calls: bool = False
    tool_call_invalid_policy: Literal["drop", "error"] = "drop"
    max_samples_per_subset: Annotated[int, Field(ge=1)] | None = None
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "all_exhausted"

    @model_validator(mode="after")
    def _validate_subsets(self) -> Self:
        if self.subsets is not None:
            names = [subset.name for subset in self.subsets]
            if len(names) != len(set(names)):
                raise ValueError("Cascade subset names must be unique")
            total = sum(subset.weight for subset in self.subsets)
            if not math.isfinite(total) or total <= 0:
                raise ValueError(
                    "Cascade subset weights must have a finite positive total"
                )
        return self


class ResponseDatasetConfig(TypedDict):
    dataset_name: NotRequired[str]
    data_path: NotRequired[str]
    input_key: NotRequired[str]
    output_key: NotRequired[str]
    subset: NotRequired[str | None]
    split: NotRequired[str]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]
    env_name: NotRequired[str]
    processor: NotRequired[str]  # remove once processor is refactored
    download_dir: NotRequired[str]
    # Size of the validation data
    split_validation_size: NotRequired[float]
    # Seed for train/validation split when split_validation_size > 0
    seed: NotRequired[int]
    # TODO(rohitrango): Move model-specific media controls to ProcessorInterface.
    num_frames: NotRequired[int]
    video_sampling_style: NotRequired[Literal["nemotron_vl"]]
    video_target_num_patches: NotRequired[int | None]
    video_temporal_patch_size: NotRequired[int]
    video_maintain_aspect_ratio: NotRequired[bool]
    min_generation_tokens: NotRequired[int]
    max_samples: NotRequired[int | None]
    # Arrow/raw-text dataset fields used by xToken distillation.
    data_files: NotRequired[str | list[str] | None]
    text_key: NotRequired[str | None]
    characters_per_sample: NotRequired[int | None]
    # Native OpenAI-format chat adapter fields.
    chat_key: NotRequired[str]
    use_preserving_dataset: NotRequired[bool]
    system_key: NotRequired[str | None]
    system_prompt: NotRequired[str | None]
    tool_key: NotRequired[str | None]
    # Cascade releases, subset mixtures, and CPU preparation controls.
    cascade: NotRequired[CascadeDatasetConfig]


class PreferenceDatasetConfig(TypedDict):
    dataset_name: NotRequired[str]
    data_path: NotRequired[str]
    prompt_key: NotRequired[str]
    chosen_key: NotRequired[str]
    rejected_key: NotRequired[str]
    subset: NotRequired[str | None]
    split: NotRequired[str]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]
    # TODO(rohitrango): Move model-specific media controls to ProcessorInterface.
    num_frames: NotRequired[int]
    video_sampling_style: NotRequired[Literal["nemotron_vl"]]
    video_target_num_patches: NotRequired[int | None]
    video_temporal_patch_size: NotRequired[int]
    video_maintain_aspect_ratio: NotRequired[bool]
    min_generation_tokens: NotRequired[int]
    split_validation_size: NotRequired[float | int]
    legacy_validation_split: NotRequired[bool]
    seed: NotRequired[int]
    max_samples: NotRequired[int | None]
    cache_dir: NotRequired[str | None]


def _validate_cascade_before_dataset_union(value: object) -> object:
    """Reject invalid Cascade settings before a permissive TypedDict fallback.

    A response config with invalid Cascade settings would otherwise match the
    preference-config branch, silently discarding the entire Cascade block.
    """
    if isinstance(value, list):
        for item in value:
            _validate_cascade_before_dataset_union(item)
    elif isinstance(value, Mapping) and "cascade" in value:
        CascadeDatasetConfig.model_validate(value["cascade"])
    return value


class DataConfig(TypedDict):
    backend: NotRequired[Literal["hf", "energon"]]
    energon: NotRequired[EnergonLoaderConfig]
    max_input_seq_length: int | None
    add_bos: NotRequired[bool]
    add_eos: NotRequired[bool]
    add_generation_prompt: NotRequired[bool]
    add_system_prompt: NotRequired[bool]
    shuffle: bool
    # Number of data loader workers.
    # Set to 8 or 10 for large batches to improve loading speed.
    # This saturates CPU threads without consuming too much memory
    # However, setting it too high might cause memory issues for long seqlens.
    num_workers: NotRequired[int]
    # multiple dataloader configs
    # currently only supported for GRPO
    use_multiple_dataloader: NotRequired[bool]
    num_prompts_per_dataloader: NotRequired[int]
    custom_dataloader: NotRequired[str]
    # Cross-tokenizer collator controls.
    collator_mode: NotRequired[Literal["text", "chat"]]
    drop_first_assistant_chunk_kl: NotRequired[bool]
    include_thinking_in_loss: NotRequired[bool]
    native_thinking_alignment: NotRequired[bool]
    kd_alignment_regions: NotRequired[list[str] | None]
    num_packed_rows: NotRequired[int]
    # dataset configs
    train: Annotated[
        ResponseDatasetConfig
        | PreferenceDatasetConfig
        | EnergonSourceConfig
        | list[ResponseDatasetConfig],
        BeforeValidator(_validate_cascade_before_dataset_union),
    ]
    validation: NotRequired[
        Annotated[
            ResponseDatasetConfig
            | PreferenceDatasetConfig
            | EnergonSourceConfig
            | list[ResponseDatasetConfig]
            | None,
            BeforeValidator(_validate_cascade_before_dataset_union),
        ]
    ]
    # default settings for all datasets, will be overridden by dataset-specific settings
    default: NotRequired[
        Annotated[
            ResponseDatasetConfig | PreferenceDatasetConfig | None,
            BeforeValidator(_validate_cascade_before_dataset_union),
        ]
    ]


# ===============================================================================
# Eval Dataset Configs
# ===============================================================================
# These configs are used by the evaluation entrypoint. Migrated datasets such
# as AIME use the response registry; the rest still use eval_datasets/.
# Note: TypedDict doesn't allow narrowing types in child classes, so each config
# is defined independently with common fields repeated.


class MMLUEvalDataConfig(TypedDict):
    """Config for MMLU and multilingual MMLU datasets.

    Supports dataset_name: "mmlu" or "mmlu_{language}" where language is one of:
    AR-XY, BN-BD, DE-DE, EN-US, ES-LA, FR-FR, HI-IN, ID-ID, IT-IT, JA-JP,
    KO-KR, PT-BR, ZH-CN, SW-KE, YO-NG
    """

    max_input_seq_length: int
    dataset_name: Literal[
        "mmlu",
        "mmlu_AR-XY",
        "mmlu_BN-BD",
        "mmlu_DE-DE",
        "mmlu_EN-US",
        "mmlu_ES-LA",
        "mmlu_FR-FR",
        "mmlu_HI-IN",
        "mmlu_ID-ID",
        "mmlu_IT-IT",
        "mmlu_JA-JP",
        "mmlu_KO-KR",
        "mmlu_PT-BR",
        "mmlu_ZH-CN",
        "mmlu_SW-KE",
        "mmlu_YO-NG",
    ]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]


class MMLUProEvalDataConfig(TypedDict):
    """Config for MMLU Pro dataset."""

    max_input_seq_length: int
    dataset_name: Literal["mmlu_pro"]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]


class AIMEEvalDataConfig(TypedDict):
    """Config for AIME datasets loaded from the response registry."""

    max_input_seq_length: int
    dataset_name: Literal["AIME2024", "AIME2025", "AIME2026"]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]
    processor: NotRequired[str]
    repeat: NotRequired[int]


class GPQAEvalDataConfig(TypedDict):
    """Config for GPQA datasets."""

    max_input_seq_length: int
    dataset_name: Literal["gpqa", "gpqa_diamond"]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]


class MathEvalDataConfig(TypedDict):
    """Config for Math datasets."""

    max_input_seq_length: int
    dataset_name: Literal["math", "math500"]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]


class LocalMathEvalDataConfig(TypedDict):
    """Config for local math datasets loaded from files.

    dataset_name can be a URL or local file path.
    Requires additional fields: problem_key, solution_key, file_format, split.
    """

    max_input_seq_length: int
    dataset_name: str  # URL or file path
    problem_key: str
    solution_key: str
    file_format: Literal["csv", "json"]
    split: NotRequired[str | None]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]


class MMAUEvalDataConfig(TypedDict):
    """Config for MMAU (Massive Multitask Audio Understanding) datasets."""

    max_input_seq_length: int
    dataset_name: Literal["mmau", "TwinkStart/MMAU"]
    split: NotRequired[str | None]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]
    env_name: NotRequired[str]
    max_samples: NotRequired[int | None]


class DailyOmniEvalDataConfig(TypedDict):
    """Config for the Daily-Omni audio-visual eval dataset.

    Mirrors the MMAU multimodal schema but with its own ``dataset_name`` literal
    so the eval-config union resolves daily-omni unambiguously. Kept as a
    ``TypedDict`` for consistency with the other (still v1) eval-data configs in
    this union, whose consumers access the resolved config by key
    (``config.data["dataset_name"]``).

    Fields:
        max_input_seq_length: Max prompt length passed to the generation backend.
        dataset_name: Must be ``"daily-omni"``.
        split: HuggingFace split to load.
        prompt_file: Optional prompt template path.
        system_prompt_file: Optional system prompt path.
        env_name: Reward/eval environment name (e.g. ``"vlm"``).
        max_samples: Cap on the number of rows evaluated. None evaluates the
            whole split.
    """

    max_input_seq_length: int
    dataset_name: Literal["daily-omni"]
    split: NotRequired[str | None]
    prompt_file: NotRequired[str | None]
    system_prompt_file: NotRequired[str | None]
    env_name: NotRequired[str]
    max_samples: NotRequired[int | None]


# Union type for all eval dataset configs
EvalDataConfigType = Union[
    MMLUEvalDataConfig,
    MMLUProEvalDataConfig,
    AIMEEvalDataConfig,
    GPQAEvalDataConfig,
    MathEvalDataConfig,
    MMAUEvalDataConfig,
    DailyOmniEvalDataConfig,
    LocalMathEvalDataConfig,
]
