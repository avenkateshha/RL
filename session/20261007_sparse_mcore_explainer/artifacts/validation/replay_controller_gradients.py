"""Compare actual first-step preclip optimizer shards with a full reference replay.

Every first-step microbatch retains real teacher observations and independent
reference dlogits. A fresh pinned student accumulates their reference VJPs in
the recorded gradient-buffer dtype, sums CP+DP, and compares every owned range.
"""

import argparse
import gc
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import model_oracle
import torch
import torch.distributed as dist
import yaml
from megatron.core import parallel_state as ps
from real_model_reference import forward, load_model, oracle_full_logits


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_name(name):
    while name.startswith("module."):
        name = name[len("module.") :]
    return name


def verify_range_coverage(reports, inventory):
    """Every TP-local parameter element has exactly one CP+DP optimizer owner."""
    ranges = {item["name"]: [] for item in inventory}
    shapes = {item["name"]: item["full_local_shape"] for item in inventory}
    for rank_reports in reports:
        for report in rank_reports:
            for shard in report["shards"]:
                name = shard["name"]
                assert shard["full_local_shape"] == shapes[name], shard
                ranges[name].append((shard["start"], shard["end"]))
    for item in inventory:
        name, cursor = item["name"], 0
        assert ranges[name], f"No optimizer owns {name}"
        for start, end in sorted(ranges[name]):
            assert start == cursor and end > start, (name, cursor, start, end)
            cursor = end
        assert cursor == item["full_local_numel"], (name, cursor, item)
    return sum(item["full_local_numel"] for item in inventory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("--capture-run", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load((args.case / "resolved_config.yaml").read_text())
    torch.set_num_threads(4)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    tp = config["policy"]["megatron_cfg"]["tensor_model_parallel_size"]
    cp = config["policy"]["megatron_cfg"]["context_parallel_size"]
    student = load_model(config["policy"]["model_name"], tp, cp).train()
    rank = dist.get_rank()
    index_path = args.capture_run / "optimizer-first-step" / f"rank{rank}-index.json"
    index = json.loads(index_path.read_text())
    assert index["rank"] == rank and index["optimizers"]
    expected_microbatches = config["policy"]["train_global_batch_size"] // (
        ps.get_data_parallel_world_size() * config["policy"]["train_micro_batch_size"]
    )
    assert len(index["microbatches"]) == expected_microbatches, index
    parameters = {
        canonical_name(name): parameter
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    optimizer_metadata, actual_gradients = [], {}
    for entry in index["optimizers"]:
        assert not entry["found_inf"], entry
        metadata = json.loads(Path(entry["metadata_path"]).read_text())
        assert metadata["all_finite"] and not metadata["found_inf"], metadata
        assert metadata["tp_rank"] == ps.get_tensor_model_parallel_rank()
        assert metadata["cp_rank"] == ps.get_context_parallel_rank()
        assert metadata["dp_rank"] == ps.get_data_parallel_rank()
        assert metadata["calculate_per_token_loss"]
        assert not metadata["average_in_collective"]
        assert not metadata["gradient_accumulation_fusion"]
        assert not metadata["sequence_parallel"] and not metadata["qk_layernorm"]
        assert sha256(entry["tensor_path"]) == entry["sha256"]
        tensors = torch.load(
            entry["tensor_path"], map_location="cpu", weights_only=True
        )
        assert set(tensors) == {shard["name"] for shard in metadata["shards"]}
        assert not set(tensors).intersection(actual_gradients)
        actual_gradients.update(tensors)
        optimizer_metadata.append(metadata)
    inventory = optimizer_metadata[0]["inventory"]
    assert set(parameters) == {item["name"] for item in inventory}
    for item in inventory:
        assert not item["average_gradients_across_tp_domain"], item
        assert not item["sequence_parallel"] and not item["qk_layernorm_name"], item
        assert list(parameters[item["name"]].shape) == item["full_local_shape"]
        assert str(parameters[item["name"]].dtype) == item["parameter_dtype"]
    assert len({entry["grad_reduce_in_fp32"] for entry in optimizer_metadata}) == 1
    accumulation_dtypes = {
        name: torch.float32
        if optimizer_metadata[0]["grad_reduce_in_fp32"]
        else parameter.dtype
        for name, parameter in parameters.items()
    }
    group = ps.get_data_parallel_group(with_context_parallel=True)
    assert all(
        entry["optimizer_group_ranks"] == dist.get_process_group_ranks(group)
        for entry in optimizer_metadata
    ), "Fixture requires one distributed optimizer instance per CP+DP group"
    group_reports = [None] * dist.get_world_size(group)
    dist.all_gather_object(group_reports, optimizer_metadata, group=group)
    covered_elements = verify_range_coverage(group_reports, inventory)
    accumulated = {
        name: torch.zeros_like(parameter, dtype=accumulation_dtypes[name])
        for name, parameter in parameters.items()
    }
    forward_reports, sample_ids = [], []
    for ordinal, entry in enumerate(index["microbatches"]):
        assert entry["ordinal"] == ordinal
        assert sha256(entry["capture_path"]) == entry["capture_sha256"]
        capture = torch.load(
            entry["capture_path"], map_location="cpu", weights_only=False
        )
        assert capture["status"] == "PASS"
        assert capture["tp_rank"] == ps.get_tensor_model_parallel_rank()
        assert capture["cp_rank"] == ps.get_context_parallel_rank()
        assert capture["dp_rank"] == ps.get_data_parallel_rank()
        assert capture["item_ids"] == entry["item_ids"]
        sample_ids.extend(capture["item_ids"])
        logits = forward(student, capture["data"]["input_ids"])
        full = oracle_full_logits(logits)
        value_error = model_oracle.gradient_errors(full, capture["student_full_logits"])
        assert value_error["relative_l2"] <= 0.002, value_error
        torch.testing.assert_close(
            full, capture["student_full_logits"], rtol=0.02, atol=0.05
        )
        reference = capture["reference_local_dlogits"].cuda()
        assert tuple(reference.shape) == tuple(logits.shape)
        logits.backward(reference.to(logits.dtype))
        for name, parameter in parameters.items():
            assert parameter.grad is not None, name
            accumulated[name].add_(parameter.grad)
        student.zero_grad(set_to_none=True)
        forward_reports.append(
            {
                "ordinal": ordinal,
                "item_ids": capture["item_ids"],
                "pinned_initial_student_forward": value_error,
                "dlogits": capture["dlogits"],
                "teacher_backends": capture["teacher_backends"],
            }
        )
        del capture, logits, reference, full
    assert len(sample_ids) == len(set(sample_ids)), "Repeated first-step occurrence IDs"
    # Actual DDP sums MBs in the recorded dtype, then reduce-scatters with SUM.
    # All-reduce supplies the same sum for slicing each owned optimizer range.
    # Schedule factors cancel: no extra MB, CP or DP divisor belongs here.
    for value in accumulated.values():
        dist.all_reduce(value, group=group)
    local_sums = torch.zeros(3, dtype=torch.float64, device="cuda")
    shard_reports = []
    for metadata in optimizer_metadata:
        for shard in metadata["shards"]:
            name = shard["name"]
            actual = actual_gradients[name].float()
            expected = (
                accumulated[name]
                .reshape(-1)[shard["start"] : shard["end"]]
                .float()
                .cpu()
            )
            assert str(actual_gradients[name].dtype) == shard["gradient_dtype"]
            assert actual.shape == expected.shape
            difference = actual.double() - expected.double()
            sums = torch.tensor(
                [
                    difference.square().sum(),
                    actual.double().square().sum(),
                    expected.double().square().sum(),
                ],
                dtype=torch.float64,
                device="cuda",
            )
            local_sums += sums
            shard_reports.append(
                {
                    "name": name,
                    "start": shard["start"],
                    "end": shard["end"],
                    "gradient_dtype": shard["gradient_dtype"],
                    "max_absolute": float(difference.abs().max()),
                }
            )
    # Count each CP+DP-owned element once; replicated TP parameters appear once
    # per TP rank consistently in both actual and reference global norms.
    dist.all_reduce(local_sums)
    difference_sq, actual_sq, reference_sq = local_sums.tolist()
    errors = {
        "relative_l2": (difference_sq / max(reference_sq, 1e-30)) ** 0.5,
        "relative_norm": abs(actual_sq**0.5 - reference_sq**0.5)
        / max(reference_sq**0.5, 1e-15),
        "actual_norm_including_tp_replicas": actual_sq**0.5,
        "reference_norm_including_tp_replicas": reference_sq**0.5,
    }
    assert errors["relative_l2"] <= 0.02, errors
    assert errors["relative_norm"] <= 0.02, errors
    report = {
        "status": "PASS",
        "rank": rank,
        "tp": tp,
        "cp": cp,
        "dp": ps.get_data_parallel_world_size(),
        "capture_index": str(index_path.resolve()),
        "capture_index_sha256": sha256(index_path),
        "microbatches": forward_reports,
        "expected_microbatches": expected_microbatches,
        "accumulation_dtypes": {
            name: str(dtype) for name, dtype in accumulation_dtypes.items()
        },
        "normalization": "Sum all MB VJPs, then CP+DP SUM; schedule factors cancel and token-finalization factor is one",
        "range_coverage": "PASS: every TP-local element owned exactly once across CP+DP",
        "covered_parameter_elements_per_tp_rank": covered_elements,
        "actual_optimizer_preclip_gradients": errors,
        "owned_shards": shard_reports,
        "scope": "Actual first optimizer step after prepare_grads before clipping/update; all MBs and owned parameter ranges compared to independent-reference VJP at pinned initial weights.",
    }
    (args.case / f"replay-rank{rank}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    del accumulated, actual_gradients, student
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier()
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
