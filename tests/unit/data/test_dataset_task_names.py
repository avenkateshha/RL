"""Raw datasets explicitly declare the row task names routed by data setup."""

from typing import Any

import pytest
from datasets import Dataset

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.utils import setup_response_data


def _process(
    row: dict[str, Any],
    task_spec: TaskDataSpec,
    tokenizer: Any,
    max_seq_length: int | None,
    idx: int,
) -> dict[str, Any]:
    assert row["preprocessed"]
    assert row["task_name"] == task_spec.task_name
    return {"task_name": task_spec.task_name, "sample_id": row["sample_id"]}


def _preprocess(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "preprocessed": True}


class _MixtureDataset(RawDataset):
    def __init__(self, name: str) -> None:
        self.task_name = name
        self.dataset = Dataset.from_list(
            [
                {"task_name": task_name, "sample_id": f"{task_name}:train"}
                for task_name in self.get_task_names()
            ]
        )
        self.val_dataset = Dataset.from_list(
            [
                {"task_name": task_name, "sample_id": f"{task_name}:val"}
                for task_name in self.get_task_names()
            ]
        )
        self.task_spec = TaskDataSpec(task_name=self.task_name)
        self.processor = _process
        self.preprocessor = _preprocess

    def get_task_names(self) -> tuple[str, ...]:
        return (f"{self.task_name}/math", f"{self.task_name}/code")


class _LegacyExternalDataset:
    """External adapters did not have to inherit RawDataset or add a preprocessor."""

    def __init__(self, name: str) -> None:
        self.task_name = name
        self.dataset = Dataset.from_list(
            [{"task_name": name, "sample_id": f"{name}:train", "preprocessed": True}]
        )
        self.val_dataset = Dataset.from_list(
            [{"task_name": name, "sample_id": f"{name}:val", "preprocessed": True}]
        )
        self.task_spec = TaskDataSpec(task_name=name)
        self.processor = _process


def test_single_task_dataset_interface_preserves_existing_name() -> None:
    dataset = RawDataset()
    dataset.task_name = "existing-math-name"
    assert dataset.get_task_names() == ("existing-math-name",)


@pytest.mark.parametrize("multiple_dataloaders", [False, True])
@pytest.mark.parametrize("has_envs", [False, True])
def test_setup_registers_subset_processors_and_environments(
    monkeypatch: pytest.MonkeyPatch, multiple_dataloaders: bool, has_envs: bool
) -> None:
    train_source = _MixtureDataset("train-mixture")
    validation_source = _MixtureDataset("eval-mixture")
    sources = {"train": train_source, "validation": validation_source}
    monkeypatch.setattr(
        "nemo_rl.data.utils.load_response_dataset",
        lambda cfg: sources[cfg["dataset_name"]],
    )
    environment = object()
    monkeypatch.setattr("nemo_rl.data.utils.create_env", lambda **kwargs: environment)
    config = {
        "train": {"dataset_name": "train", "env_name": "test"},
        "validation": {"dataset_name": "validation", "env_name": "test"},
        "max_input_seq_length": 128,
        "use_multiple_dataloader": multiple_dataloaders,
    }
    result = setup_response_data(
        None, config, env_configs={"test": {}} if has_envs else None
    )
    train, validation = result[:2]
    if multiple_dataloaders:
        assert set(train) == {"train-mixture"}
        train = train["train-mixture"]

    train_tasks = train_source.get_task_names()
    validation_tasks = (*train_tasks, *validation_source.get_task_names())
    assert [train[index]["task_name"] for index in range(len(train))] == list(
        train_tasks
    )
    assert [validation[index]["task_name"] for index in range(len(validation))] == list(
        validation_tasks
    )
    assert set(train.task_data_processors) == set(train_tasks)
    assert set(validation.task_data_processors) == set(validation_tasks)
    assert train_source.task_spec.task_name == "train-mixture"
    assert validation_source.task_spec.task_name == "eval-mixture"
    if has_envs:
        assert result[2] == {task_name: environment for task_name in train_tasks}
        assert result[3] == {task_name: environment for task_name in validation_tasks}


def test_setup_retains_external_single_task_adapters_without_raw_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {
        "external-train": _LegacyExternalDataset("external-train"),
        "external-eval": _LegacyExternalDataset("external-eval"),
    }
    monkeypatch.setattr(
        "nemo_rl.data.utils.load_response_dataset",
        lambda cfg: sources[cfg["dataset_name"]],
    )
    train, validation = setup_response_data(
        None,
        {
            "train": {"dataset_name": "external-train"},
            "validation": {"dataset_name": "external-eval"},
            "max_input_seq_length": 128,
        },
    )
    assert train[0]["sample_id"] == "external-train:train"
    assert [validation[index]["sample_id"] for index in range(len(validation))] == [
        "external-train:val",
        "external-eval:train",
    ]
