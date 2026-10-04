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

import pytest
from pydantic import TypeAdapter, ValidationError

from nemo_rl.data import CascadeDatasetConfig, DataConfig


def test_data_config_preserves_native_openai_chat_adapter_fields() -> None:
    train = {
        "dataset_name": "openai_format",
        "data_path": "<ASSET_DIR>/openmath-chat.jsonl",
        "data_files": None,
        "text_key": None,
        "characters_per_sample": None,
        "processor": "kd_data_processor",
        "split": "train",
        "chat_key": "messages",
        "use_preserving_dataset": False,
        "system_key": None,
        "system_prompt": None,
        "tool_key": None,
        "prompt_file": None,
    }

    validated = TypeAdapter(DataConfig).validate_python(
        {
            "max_input_seq_length": 4096,
            "shuffle": False,
            "collator_mode": "chat",
            "train": train,
        }
    )

    assert validated["train"] == train


def test_data_config_preserves_typed_cascade_settings() -> None:
    adapter = TypeAdapter(DataConfig)
    validated = adapter.validate_python(
        {
            "max_input_seq_length": 4096,
            "shuffle": False,
            "train": {
                "dataset_name": "Nemotron-Cascade-SFT-Stage-2",
                "processor": "chat_kd_processor",
                "cascade": {
                    "map_num_proc": 2,
                    "subsets": [
                        {"name": "math", "weight": 3.0},
                        {
                            "name": "tool_calling",
                            "weight": 1.0,
                            "normalize_tool_calls": True,
                        },
                    ],
                },
            },
        }
    )

    cascade = validated["train"]["cascade"]
    assert isinstance(cascade, CascadeDatasetConfig)
    assert cascade.dataset_path is None  # The registered adapter selects its release.
    assert cascade.map_num_proc == 2
    assert [subset.weight for subset in cascade.subsets] == [3.0, 1.0]
    assert cascade.subsets[1].normalize_tool_calls is True
    assert adapter.validate_python(adapter.dump_python(validated)) == validated


@pytest.mark.parametrize("field", ["train", "validation", "default"])
@pytest.mark.parametrize(
    "cascade",
    [
        {"map_num_proc": 0},
        {"max_samples_per_subset": 0},
        {"tool_call_invalid_policy": "ignore"},
        {"stopping_strategy": "forever"},
        {"dataset_path": " "},
        {"subsets": []},
        {"subsets": [{"name": "math", "weight": -1.0}]},
        {"subsets": [{"name": "math", "weight": float("nan")}]},
        {"subsets": [{"name": "math", "weight": float("inf")}]},
        {"subsets": [{"name": "math", "weight": 0.0}]},
        {"subsets": [{"name": "math"}, {"name": "math"}]},
        {"subsets": [{"name": "math", "max_samples": 0}]},
        {
            "subsets": [
                {"name": "math", "weight": 1e308},
                {"name": "code", "weight": 1e308},
            ]
        },
        {"strip_think": True},
    ],
)
def test_invalid_cascade_config_cannot_fall_back_to_preference_config(
    field: str, cascade: dict
) -> None:
    config = {
        "max_input_seq_length": 4096,
        "shuffle": False,
        "train": {"dataset_name": "Nemotron-Cascade-2-SFT-Math"},
    }
    config[field] = {"dataset_name": "Nemotron-Cascade-2-SFT", "cascade": cascade}
    with pytest.raises(ValidationError):
        TypeAdapter(DataConfig).validate_python(config)


def test_invalid_cascade_in_dataset_list_is_rejected() -> None:
    with pytest.raises(ValidationError, match="finite positive total"):
        TypeAdapter(DataConfig).validate_python(
            {
                "max_input_seq_length": 4096,
                "shuffle": False,
                "train": [
                    {
                        "dataset_name": "Nemotron-Cascade-2-SFT",
                        "cascade": {"subsets": [{"name": "math", "weight": 0.0}]},
                    }
                ],
            }
        )
