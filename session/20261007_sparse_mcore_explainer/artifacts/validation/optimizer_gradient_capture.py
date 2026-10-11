"""Run-owned first-step gradient capture for pinned MCore 6a366090.

Install in the student Ray process before its first optimizer step. The wrapper
calls the actual DistributedOptimizer.prepare_grads and observes its result
before either MixedPrecisionOptimizer.step or ChainedOptimizer.step clips or
updates parameters. It does not substitute gradients or communication.

Every optimizer in a chain produces one independent report. Parameter gradients
are owned flattened ranges, not complete synchronized model.main_grad tensors.
The companion replay must verify coverage across the DP+CP optimizer group.
This instrumentation supports the ordinary PP1 dense BF16/FP32 DDP fixture;
Megatron-FSDP and multi-chunk models fail explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypedDict
from weakref import WeakKeyDictionary

import torch


class CaptureReport(TypedDict):
    """Paths and identity passed to the controller's completion callback."""

    metadata_path: str
    tensor_path: str
    sha256: str
    rank: int
    optimizer_index: int
    found_inf: bool
    num_parameters: int
    num_elements: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_name(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module.") :]
    return name


def _parameter_metadata(name: str, parameter: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "full_local_shape": list(parameter.shape),
        "full_local_numel": parameter.numel(),
        "parameter_dtype": str(parameter.dtype),
        "tensor_model_parallel": bool(
            getattr(parameter, "tensor_model_parallel", False)
        ),
        "partition_dim": int(getattr(parameter, "partition_dim", -1)),
        "partition_stride": int(getattr(parameter, "partition_stride", 1)),
        "sequence_parallel": bool(getattr(parameter, "sequence_parallel", False)),
        "average_gradients_across_tp_domain": bool(
            getattr(parameter, "average_gradients_across_tp_domain", False)
        ),
        "qk_layernorm_name": "q_layernorm" in name or "k_layernorm" in name,
    }


def _collect_owned_gradients(optimizer) -> tuple[dict, dict[str, torch.Tensor]]:
    """Copy actual prepared optimizer shards using its pinned range mapping."""
    assert not optimizer.ddp_config.use_megatron_fsdp, "Megatron-FSDP unsupported"
    assert len(optimizer.model_chunks) == 1, "Capture requires the PP1 single chunk"
    model = optimizer.model_chunks[0]
    names = {}
    inventory = []
    for raw_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        name = _canonical_name(raw_name)
        assert name not in {entry["name"] for entry in inventory}, name
        assert id(parameter) not in names, name
        names[id(parameter)] = name
        inventory.append(_parameter_metadata(name, parameter))

    precision_aware = optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    field = "decoupled_grad" if precision_aware else "grad"
    metadata = {
        "format": "mcore_preclip_owned_gradients_v1",
        "boundary": "DistributedOptimizer.prepare_grads returned; before clipping/update",
        "stub_optimizer": bool(optimizer.is_stub_optimizer),
        "gradient_field": field,
        "precision_aware": bool(precision_aware),
        "grad_reduce_in_fp32": bool(optimizer.ddp_config.grad_reduce_in_fp32),
        "average_in_collective": bool(optimizer.ddp_config.average_in_collective),
        "sequence_parallel": bool(model.config.sequence_parallel),
        "qk_layernorm": bool(model.config.qk_layernorm),
        "calculate_per_token_loss": bool(model.config.calculate_per_token_loss),
        "gradient_accumulation_fusion": bool(model.config.gradient_accumulation_fusion),
        "inventory": inventory,
        "shards": [],
    }
    gradients = {}
    if optimizer.is_stub_optimizer:
        return metadata, gradients

    # These are the exact pairings in pinned distrib_optimizer.py:2804-2811.
    half_shards = (
        optimizer.shard_float16_groups
        if precision_aware
        else optimizer.shard_fp32_from_float16_groups
    )
    pairings = (
        (optimizer.model_float16_groups, half_shards),
        (optimizer.model_fp32_groups, optimizer.shard_fp32_groups),
    )
    observed_parameters = set()
    for model_groups, shard_groups in pairings:
        for model_group, shard_group in zip(model_groups, shard_groups, strict=True):
            for model_parameter, main_shard in zip(
                model_group, shard_group, strict=True
            ):
                name = names[id(model_parameter)]
                assert name not in gradients, name
                observed_parameters.add(model_parameter)
                owned = optimizer._get_model_param_range_map(model_parameter)["param"]
                assert 0 <= owned.start < owned.end <= model_parameter.numel(), name
                assert owned.size == owned.end - owned.start == main_shard.numel()
                gradient = getattr(main_shard, field)
                assert gradient is not None, f"Missing prepared {field} for {name}"
                assert gradient.numel() == owned.size, name
                # Copy one shard at a time, preserving its actual accumulation dtype.
                # copy=True also prevents CPU fixture mutations altering the snapshot.
                gradients[name] = gradient.detach().reshape(-1).to("cpu", copy=True)
                item = _parameter_metadata(name, model_parameter)
                item.update(
                    start=owned.start,
                    end=owned.end,
                    gradient_dtype=str(gradient.dtype),
                    gradient_field=field,
                    all_finite=bool(torch.isfinite(gradients[name]).all()),
                )
                metadata["shards"].append(item)
    assert observed_parameters == set(optimizer.model_param_gbuf_map), (
        "Prepared optimizer group mapping omitted or duplicated owned parameters"
    )
    return metadata, gradients


def _parallel_metadata(optimizer) -> dict:
    import torch.distributed as dist
    from megatron.core import parallel_state as ps

    # Stub optimizers may not initialize their data_parallel_group member.
    optimizer_group = (
        ps.get_data_parallel_group(with_context_parallel=True)
        if optimizer.is_stub_optimizer
        else optimizer.data_parallel_group
    )
    return {
        "rank": dist.get_rank(),
        "world_size": dist.get_world_size(),
        "tp_rank": ps.get_tensor_model_parallel_rank(),
        "tp_size": ps.get_tensor_model_parallel_world_size(),
        "cp_rank": ps.get_context_parallel_rank(),
        "cp_size": ps.get_context_parallel_world_size(),
        "dp_rank": ps.get_data_parallel_rank(),
        "dp_size": ps.get_data_parallel_world_size(),
        "optimizer_group_ranks": dist.get_process_group_ranks(optimizer_group),
        "dp_cp_group_ranks": dist.get_process_group_ranks(
            ps.get_data_parallel_group(with_context_parallel=True)
        ),
        "distributed_optimizer_instance_id": optimizer.distributed_optimizer_instance_id,
    }


class CaptureInstallation:
    """Invocation-local instrumentation state, never attached to an optimizer."""

    def __init__(
        self,
        optimizer_class,
        output_dir: Path,
        callback: Callable[[CaptureReport], None] | None,
        parallel_metadata: Callable,
    ):
        self.optimizer_class = optimizer_class
        self.output_dir = output_dir.resolve()
        self.callback = callback
        self.parallel_metadata = parallel_metadata
        self.captures: list[CaptureReport] = []
        self.completed = WeakKeyDictionary()
        self.original = optimizer_class.prepare_grads
        self.original_own_method = optimizer_class.__dict__.get("prepare_grads")

        @wraps(self.original)
        def prepare(optimizer):
            found_inf = self.original(optimizer)
            if optimizer not in self.completed:
                self.capture(optimizer, bool(found_inf))
            return found_inf

        self.wrapper = prepare
        optimizer_class.prepare_grads = prepare

    def capture(self, optimizer, found_inf: bool) -> None:
        """Save the first prepared step, then notify the controller observer."""
        metadata, gradients = _collect_owned_gradients(optimizer)
        parallel = self.parallel_metadata(optimizer)
        index = len(self.captures)
        stem = f"rank{parallel['rank']}-optimizer{index}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tensor_path = self.output_dir / f"{stem}.pt"
        metadata_path = self.output_dir / f"{stem}.json"
        assert not tensor_path.exists() and not metadata_path.exists(), stem
        temporary = tensor_path.with_suffix(f".pt.{os.getpid()}.tmp")
        torch.save(gradients, temporary)
        temporary.replace(tensor_path)
        report: CaptureReport = {
            "metadata_path": str(metadata_path),
            "tensor_path": str(tensor_path),
            "sha256": _sha256(tensor_path),
            "rank": parallel["rank"],
            "optimizer_index": index,
            "found_inf": found_inf,
            "num_parameters": len(gradients),
            "num_elements": sum(value.numel() for value in gradients.values()),
        }
        metadata.update(parallel)
        metadata.update(report)
        # Empty ownership is explicit and legal; replay must prove global coverage.
        metadata["empty_owner"] = not gradients
        metadata["all_finite"] = all(x["all_finite"] for x in metadata["shards"])
        temporary = metadata_path.with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(metadata_path)
        self.completed[optimizer] = report
        self.captures.append(report)
        if self.callback is not None:
            self.callback(report)

    def uninstall(self) -> None:
        """Restore the class method; used only by the standalone CPU selfcheck."""
        assert self.optimizer_class.prepare_grads is self.wrapper
        if self.original_own_method is None:
            del self.optimizer_class.prepare_grads
        else:
            self.optimizer_class.prepare_grads = self.original_own_method


G_INSTALLATION: CaptureInstallation | None = None


def install_optimizer_gradient_capture(
    *,
    output_dir: Path,
    on_first_step_complete: Callable[[CaptureReport], None] | None = None,
) -> CaptureInstallation:
    """Install once before training; invoke callback once per chained optimizer.

    The callback receives metadata_path, tensor_path, their tensor sha256, rank,
    optimizer_index, found_inf, and shard counts. Every first-step microbatch has
    completed before the callback, including when the optimizer chain has more
    than one member. Gradients retain their true BF16 or FP32 dtype.
    """
    global G_INSTALLATION
    assert G_INSTALLATION is None, "Optimizer capture must only be installed once"
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    G_INSTALLATION = CaptureInstallation(
        DistributedOptimizer,
        Path(output_dir),
        on_first_step_complete,
        _parallel_metadata,
    )
    return G_INSTALLATION


def _selfcheck() -> None:
    """Exercise both gradient fields, exact owned ranges, ordering, and empty owners."""
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace

    events = []

    class FakeOptimizer:
        def __init__(self, precision_aware, empty=False):
            half = torch.nn.Parameter(torch.arange(6, dtype=torch.bfloat16))
            full = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
            module = torch.nn.Module()
            module.register_parameter("weight_half", half)
            module.register_parameter("weight_full", full)
            module.config = SimpleNamespace(
                sequence_parallel=False,
                qk_layernorm=False,
                calculate_per_token_loss=True,
                gradient_accumulation_fusion=False,
            )
            self.model_chunks = [module]
            self.config = SimpleNamespace(
                use_precision_aware_optimizer_no_fp8_or_ds_fp8=precision_aware
            )
            self.ddp_config = SimpleNamespace(
                use_megatron_fsdp=False,
                grad_reduce_in_fp32=False,
                average_in_collective=False,
            )
            self.is_stub_optimizer = False
            self.model_float16_groups = [[] if empty else [half]]
            self.model_fp32_groups = [[] if empty else [full]]
            half_shard = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
            master = torch.nn.Parameter(torch.zeros(3))
            full_shard = torch.nn.Parameter(torch.zeros(2))
            self.shard_float16_groups = [[] if empty else [half_shard]]
            self.shard_fp32_from_float16_groups = [[] if empty else [master]]
            self.shard_fp32_groups = [[] if empty else [full_shard]]
            self.model_param_gbuf_map = {} if empty else {half: None, full: None}
            self.ranges = {
                half: SimpleNamespace(start=2, end=5, size=3),
                full: SimpleNamespace(start=1, end=3, size=2),
            }

        def _get_model_param_range_map(self, parameter):
            return {"param": self.ranges[parameter]}

        def prepare_grads(self):
            events.append("prepare")
            aware = self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
            half_groups = (
                self.shard_float16_groups
                if aware
                else self.shard_fp32_from_float16_groups
            )
            for groups in (half_groups, self.shard_fp32_groups):
                for group in groups:
                    for shard in group:
                        value = torch.arange(1, shard.numel() + 1, dtype=shard.dtype)
                        if aware:
                            shard.decoupled_grad = value
                        else:
                            shard.grad = value
            return False

        def step(self):
            found_inf = self.prepare_grads()
            events.append("clip")
            return found_inf

    with TemporaryDirectory() as directory:
        installation = CaptureInstallation(
            FakeOptimizer,
            Path(directory),
            lambda report: events.append("capture"),
            lambda optimizer: {"rank": 0},
        )
        try:
            for aware in (False, True):
                optimizer = FakeOptimizer(aware)
                events.clear()
                assert optimizer.step() is False
                assert events == ["prepare", "capture", "clip"]
                report = installation.captures[-1]
                metadata = json.loads(Path(report["metadata_path"]).read_text())
                tensors = torch.load(report["tensor_path"], weights_only=True)
                assert report["sha256"] == _sha256(Path(report["tensor_path"]))
                assert report["num_elements"] == 5 and report["num_parameters"] == 2
                assert [(x["start"], x["end"]) for x in metadata["shards"]] == [
                    (2, 5),
                    (1, 3),
                ]
                torch.testing.assert_close(
                    tensors["weight_half"].float(), torch.arange(1, 4).float()
                )
                assert tensors["weight_half"].dtype == (
                    torch.bfloat16 if aware else torch.float32
                )
                events.clear()
                optimizer.step()
                assert events == ["prepare", "clip"]
            empty = FakeOptimizer(False, empty=True)
            empty.step()
            report = installation.captures[-1]
            metadata = json.loads(Path(report["metadata_path"]).read_text())
            assert metadata["empty_owner"] and not metadata["stub_optimizer"]
            assert len(metadata["inventory"]) == 2 and not metadata["shards"]
            assert report["num_elements"] == 0
            assert len(installation.captures) == 3
            broken = FakeOptimizer(False)
            broken.shard_fp32_groups[0].clear()
            try:
                _collect_owned_gradients(broken)
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError("Malformed owned-gradient mapping was accepted")
        finally:
            installation.uninstall()
    print(
        "PASS optimizer capture CPU selfcheck: precision-aware/FP32, ranges, first-step ordering, empty owner, malformed mapping"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true", required=True)
    parser.parse_args()
    _selfcheck()
