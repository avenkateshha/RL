"""xToken sequence padding preserves independent teacher and alignment axes."""

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict

pytestmark = pytest.mark.mcore


@pytest.fixture
def pad_sequences():
    from nemo_rl.models.megatron.data import _pad_sequence_aligned_tensors

    return _pad_sequence_aligned_tensors


def test_padding_five_to_eight_preserves_two_forced_id_slots(pad_sequences):
    ids = torch.arange(10).reshape(2, 5)
    forced = torch.stack((ids, ids + 20), dim=-1)
    batch = BatchedDataDict(
        input_ids=ids,
        token_mask=torch.ones((2, 5), dtype=torch.bool),
        force_include_token_ids=forced,
    )

    pad_sequences(batch, 4)

    torch.testing.assert_close(batch["input_ids"][:, :5], ids)
    assert batch["input_ids"].shape == (2, 8)
    assert batch["input_ids"][:, 5:].eq(0).all()
    assert batch["token_mask"].shape == (2, 8)
    assert not batch["token_mask"][:, 5:].any()
    assert batch["force_include_token_ids"].shape == (2, 8, 2)
    torch.testing.assert_close(batch["force_include_token_ids"][:, :5], forced)
    assert batch["force_include_token_ids"][:, 5:].eq(-1).all()
    assert batch["force_include_token_ids"].is_contiguous()


def test_student_chunk_padding_uses_invalid_sentinel(pad_sequences):
    chunk_ids = torch.tensor([[0, 0, 1, 2, -1], [0, 1, 1, -1, -1]])
    batch = BatchedDataDict(
        input_ids=torch.ones((2, 5), dtype=torch.long),
        student_chunk_id=chunk_ids.clone(),
        alignment_0_student_chunk_id=chunk_ids.clone(),
        alignment_2_student_chunk_id=chunk_ids.clone(),
        alignment_0_student_exact_partition_mask=chunk_ids.ge(0),
    )

    pad_sequences(batch, 4)

    for key in (
        "student_chunk_id",
        "alignment_0_student_chunk_id",
        "alignment_2_student_chunk_id",
    ):
        assert batch[key].shape == (2, 8)
        torch.testing.assert_close(batch[key][:, :5], chunk_ids)
        assert batch[key][:, 5:].eq(-1).all()
    exact = batch["alignment_0_student_exact_partition_mask"]
    assert exact.shape == (2, 8)
    torch.testing.assert_close(exact[:, :5], chunk_ids.ge(0))
    assert not exact[:, 5:].any()


def test_equal_width_teacher_and_pair_metadata_remain_untouched(pad_sequences):
    metadata = {
        "teacher_0_input_ids": torch.arange(10).reshape(2, 5),
        "teacher_1_token_mask": torch.ones((2, 5), dtype=torch.bool),
        "teacher_1_force_include_token_ids": torch.ones((2, 5, 2), dtype=torch.long),
        "alignment_0_teacher_chunk_id": torch.arange(10).reshape(2, 5),
        "alignment_0_pair_valid": torch.ones((2, 5), dtype=torch.bool),
        "alignment_1_pair_is_correct": torch.zeros((2, 5), dtype=torch.bool),
        "teacher_chunk_id": torch.arange(10).reshape(2, 5),
        "pair_valid": torch.ones((2, 5), dtype=torch.bool),
        "pair_is_correct": torch.zeros((2, 5), dtype=torch.bool),
        "global_valid_chunks": torch.ones((2, 5)),
    }
    batch = BatchedDataDict(input_ids=torch.ones((2, 5), dtype=torch.long), **metadata)

    pad_sequences(batch, 4)

    assert batch["input_ids"].shape == (2, 8)
    for key, value in metadata.items():
        assert batch[key] is value, key


def test_padding_retains_generic_model_fields_and_nonsequence_values(pad_sequences):
    batch = BatchedDataDict(
        input_ids=torch.ones((2, 5), dtype=torch.long),
        routed_experts=torch.ones((2, 5, 3), dtype=torch.int32),
        input_lengths=torch.tensor([5, 3]),
        sample_ids=["first", "second"],
    )
    lengths = batch["input_lengths"]
    sample_ids = batch["sample_ids"]

    pad_sequences(batch, 4)

    assert batch["routed_experts"].shape == (2, 8, 3)
    assert batch["routed_experts"].dtype == torch.int32
    assert batch["routed_experts"][:, :5].eq(1).all()
    assert batch["routed_experts"][:, 5:].eq(0).all()
    assert batch["input_lengths"] is lengths
    assert batch["sample_ids"] is sample_ids


@pytest.mark.parametrize("multiple", [1, 2, 4, 8])
def test_already_aligned_sequence_is_noop(pad_sequences, multiple):
    values = {
        "input_ids": torch.ones((2, 8), dtype=torch.long),
        "force_include_token_ids": torch.full((2, 8, 2), -1, dtype=torch.long),
        "alignment_0_student_chunk_id": torch.full((2, 8), -1, dtype=torch.long),
    }
    batch = BatchedDataDict(**values)

    pad_sequences(batch, multiple)

    for key, value in values.items():
        assert batch[key] is value, key
