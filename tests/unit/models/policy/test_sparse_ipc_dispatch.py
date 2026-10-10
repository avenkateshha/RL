# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Public sparse policy dispatch, without constructing model actors."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.lm_policy import Policy


def _policy(*, mcore):
    group = MagicMock()
    group.get_all_worker_results.return_value = [
        {"dp_rank": 0, "per_sample_handles": [{"row": "cp0"}]},
        {"dp_rank": 0, "per_sample_handles": [{"row": "cp1"}]},
    ]
    return SimpleNamespace(
        cfg={
            "dtensor_cfg": {"enabled": not mcore, "_v2": True},
            "megatron_cfg": {"enabled": mcore, "pipeline_model_parallel_size": 1},
        },
        use_dynamic_batches=False,
        use_sequence_packing=False,
        data_parallel_size=1,
        worker_group=group,
    )


@pytest.mark.parametrize("mcore", [False, True])
def test_sparse_public_api_reaches_worker_and_preserves_native_cp_records(mcore):
    policy = _policy(mcore=mcore)
    data = BatchedDataDict({"input_ids": torch.tensor([[1, 2, 3, 4]])})
    result = Policy.get_topk_logits_ipc(
        policy,
        data,
        k=2,
        temperature=1.5,
        vocab_size=8,
        micro_batch_size=1,
        support_mode="row_topk",
        gt_filter_topk=3,
    )
    assert result == [{"teacher_shards": [{"row": "cp0"}, {"row": "cp1"}]}]
    call = policy.worker_group.run_all_workers_sharded_data.call_args
    assert call.args == ("get_topk_logits_ipc",)
    assert call.kwargs["common_kwargs"] == {
        "k": 2,
        "temperature": 1.5,
        "vocab_size": 8,
        "micro_batch_size": 1,
        "support_mode": "row_topk",
        "gt_filter_topk": 3,
    }
    assert call.kwargs["output_is_replicated"] == [
        "tensor_parallel",
        "pipeline_parallel",
    ]
    assert "context_parallel" in call.kwargs["replicate_on_axes"]


@pytest.mark.parametrize("failure", ["pp", "packing", "dynamic", "backend"])
def test_sparse_public_api_rejects_unsupported_execution_before_dispatch(failure):
    policy = _policy(mcore=True)
    if failure == "pp":
        policy.cfg["megatron_cfg"]["pipeline_model_parallel_size"] = 2
    elif failure == "packing":
        policy.use_sequence_packing = True
    elif failure == "dynamic":
        policy.use_dynamic_batches = True
    else:
        policy.cfg["megatron_cfg"]["enabled"] = False
    with pytest.raises((ValueError, NotImplementedError)):
        Policy.get_topk_logits_ipc(
            policy,
            BatchedDataDict({"input_ids": torch.ones(1, 4)}),
            k=2,
            temperature=1.0,
            support_mode="row_topk",
        )
    policy.worker_group.run_all_workers_sharded_data.assert_not_called()


@pytest.mark.parametrize("reusable", [False, True])
def test_dense_public_api_threads_reusable_storage_opt_in(reusable):
    policy = _policy(mcore=True)
    result = Policy.get_full_logits_ipc(
        policy,
        BatchedDataDict({"input_ids": torch.ones(1, 4)}),
        micro_batch_size=1,
        reusable_ipc=reusable,
    )
    assert result == [{"teacher_shards": [{"row": "cp0"}, {"row": "cp1"}]}]
    call = policy.worker_group.run_all_workers_sharded_data.call_args
    assert call.args == ("get_full_logits_ipc",)
    assert call.kwargs["common_kwargs"] == {
        "micro_batch_size": 1,
        **({"reusable_ipc": True} if reusable else {}),
    }


@pytest.mark.parametrize("failure", ["pp", "packing", "backend"])
def test_reusable_dense_public_api_rejects_unsupported_producer(failure):
    policy = _policy(mcore=True)
    if failure == "pp":
        policy.cfg["megatron_cfg"]["pipeline_model_parallel_size"] = 2
    elif failure == "packing":
        policy.use_sequence_packing = True
    else:
        policy.cfg["megatron_cfg"]["enabled"] = False
    with pytest.raises(ValueError, match="Reusable dense IPC"):
        Policy.get_full_logits_ipc(
            policy,
            BatchedDataDict({"input_ids": torch.ones(1, 4)}),
            reusable_ipc=True,
        )
    policy.worker_group.run_all_workers_sharded_data.assert_not_called()
