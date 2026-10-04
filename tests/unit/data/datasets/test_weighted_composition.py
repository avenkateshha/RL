"""Probability-based composition must retain source rows and their identities."""

import pytest
from datasets import Dataset

from nemo_rl.data.datasets import merge_datasets, weighted_merge_datasets


def _source(name: str, size: int) -> Dataset:
    return Dataset.from_dict(
        {
            "sample_id": [f"{name}:{index}" for index in range(size)],
            "task_name": [name] * size,
            "messages": [
                [{"role": "assistant", "content": str(index)}] for index in range(size)
            ],
            "message_loss_mask": [[1] for _ in range(size)],
        }
    )


@pytest.mark.parametrize("strategy", ["first_exhausted", "all_exhausted"])
def test_weighted_composition_reproducible_and_preserves_rows(strategy: str) -> None:
    sources = [_source("math", 10), _source("code", 20)]
    mixed = weighted_merge_datasets(
        sources, [1, 2], seed=42, stopping_strategy=strategy
    )
    repeated = weighted_merge_datasets(
        sources, [10, 20], seed=42, stopping_strategy=strategy
    )
    different_seed = weighted_merge_datasets(
        sources, [1, 2], seed=7, stopping_strategy=strategy
    )

    assert list(mixed) == list(repeated)
    assert list(mixed) != list(different_seed)
    original_rows = {row["sample_id"]: row for source in sources for row in source}
    assert all(row == original_rows[row["sample_id"]] for row in mixed)


def test_all_exhausted_repeats_in_source_order_until_all_rows_seen() -> None:
    sources = [_source("short", 1), _source("long", 30)]
    mixed = weighted_merge_datasets(
        sources, [1, 1], seed=42, stopping_strategy="all_exhausted"
    )
    assert list(mixed["sample_id"]).count("short:0") > 1
    assert [row["sample_id"] for row in mixed if row["task_name"] == "long"] == [
        f"long:{index}" for index in range(30)
    ]
    assert set(mixed["sample_id"]) == {
        row["sample_id"] for source in sources for row in source
    }
    assert mixed[-1]["sample_id"] == "long:29"


def test_first_exhausted_stops_without_repetition() -> None:
    mixed = weighted_merge_datasets(
        [_source("short", 1), _source("long", 30)],
        [1, 1],
        seed=42,
        stopping_strategy="first_exhausted",
    )
    assert mixed[-1]["sample_id"] == "short:0"
    assert len(mixed) < 31
    assert len(set(mixed["sample_id"])) == len(mixed)


def test_relative_weights_control_source_selection() -> None:
    mixed = weighted_merge_datasets(
        [_source("rare", 100), _source("frequent", 100)],
        [1, 9],
        seed=42,
        stopping_strategy="first_exhausted",
    )
    tasks = list(mixed["task_name"])
    assert tasks.count("frequent") == 100
    assert tasks.count("rare") < 30


@pytest.mark.parametrize("excluded_size", [0, 3])
def test_zero_weight_sources_are_excluded(excluded_size: int) -> None:
    included = _source("included", 3)
    mixed = weighted_merge_datasets(
        [_source("excluded", excluded_size), included],
        [0, 5],
        seed=42,
        stopping_strategy="all_exhausted",
    )
    assert mixed is included


@pytest.mark.parametrize(
    "weights",
    [[], [1], [0, 0], [-1, 2], [float("nan"), 1], [float("inf"), 1]],
)
def test_invalid_weights_fail(weights: list[float]) -> None:
    with pytest.raises(ValueError, match="weight"):
        weighted_merge_datasets(
            [_source("a", 2), _source("b", 2)],
            weights,
            seed=42,
            stopping_strategy="all_exhausted",
        )


def test_finite_large_weights_do_not_overflow() -> None:
    sources = [_source("a", 4), _source("b", 6)]
    mixed = weighted_merge_datasets(
        sources, [1e308, 1e308], seed=42, stopping_strategy="all_exhausted"
    )
    expected = weighted_merge_datasets(
        sources, [1, 1], seed=42, stopping_strategy="all_exhausted"
    )
    assert list(mixed) == list(expected)


def test_underflowing_positive_weight_fails_instead_of_looping_forever() -> None:
    with pytest.raises(ValueError, match="too small"):
        weighted_merge_datasets(
            [_source("a", 1), _source("b", 1)],
            [1e-300, 1e300],
            seed=42,
            stopping_strategy="all_exhausted",
        )


def test_positive_weight_rounded_out_of_sampling_distribution_fails() -> None:
    with pytest.raises(ValueError, match="too small"):
        weighted_merge_datasets(
            [_source("a", 1), _source("b", 1)],
            [1, 1e-20],
            seed=42,
            # Both strategies reject unreachable sources; first_exhausted also
            # keeps this regression test finite if the guard is removed.
            stopping_strategy="first_exhausted",
        )


def test_positive_weight_empty_source_fails() -> None:
    with pytest.raises(ValueError, match="Positive-weight dataset 0 is empty"):
        weighted_merge_datasets(
            [_source("empty", 0), _source("a", 2)],
            [1, 1],
            seed=42,
            stopping_strategy="all_exhausted",
        )


def test_empty_sources_and_unknown_stopping_strategy_fail() -> None:
    with pytest.raises(ValueError, match="at least one dataset"):
        weighted_merge_datasets([], [], seed=42, stopping_strategy="all_exhausted")
    with pytest.raises(ValueError, match="stopping_strategy"):
        weighted_merge_datasets(
            [_source("a", 1)], [1], seed=42, stopping_strategy="unknown"
        )


def test_ordinary_merge_keeps_concatenation_order_without_repetition() -> None:
    sources = [_source("a", 2), _source("b", 3)]
    assert list(merge_datasets(sources)) == [
        row for source in sources for row in source
    ]
