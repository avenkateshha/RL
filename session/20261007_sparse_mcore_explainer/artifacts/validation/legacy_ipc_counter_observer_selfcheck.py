"""CPU-only hook/metadata check; never exports or imports a CUDA allocation."""

import inspect
import json
import os
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import legacy_ipc_counter_observer as observer
import ray
import ray.cloudpickle as cloudpickle
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor


def main():
    signature = inspect.signature(rebuild_cuda_tensor)
    with (
        tempfile.NamedTemporaryFile(
            prefix="torch_counter_observer_", dir="/dev/shm"
        ) as shm,
        tempfile.TemporaryDirectory() as run,
        patch.dict(os.environ, {"XTOKEN_NUMERICAL_CAPTURE_RUN": run}),
    ):
        shm.truncate(64 + 64 * 8)
        name = ("/" + Path(shm.name).name).encode()

        shm_path = shm.name

        def set_value(slot, value):
            with open(shm_path, "r+b") as stream:
                os.pwrite(stream.fileno(), struct.pack("q", value), 64 + slot * 8)

        def handle(slot):
            values = {
                "tensor_cls": torch.Tensor,
                "tensor_size": (3, 2),
                "tensor_stride": (2, 1),
                "tensor_offset": 0,
                "storage_cls": torch.UntypedStorage,
                "dtype": torch.float32,
                "storage_device": 0,
                "storage_handle": bytes([slot + 1]) * 64,
                "storage_size_bytes": 24,
                "storage_offset_bytes": 0,
                "requires_grad": False,
                "ref_counter_handle": name,
                "ref_counter_offset": slot,
                "event_handle": bytes(64),
                "event_sync_required": False,
            }
            return (tuple(values[key] for key in signature.parameters),)

        class Worker:
            def __init__(self, rank):
                self.rank = rank
                self.cfg = {"model_name": "teacher-snapshot"}
                self.tp_mesh = SimpleNamespace(get_local_rank=lambda: rank)
                self.cp_mesh = SimpleNamespace(get_local_rank=lambda: 0)
                self.dp_mesh = SimpleNamespace(get_local_rank=lambda: 0)
                self.exports = 0
                self.releases = 0

            def get_topk_logits_ipc(self, data, *, k):
                assert data == "unchanged" and k == 2
                self.exports += 1
                slots = range(self.rank * 16, self.rank * 16 + 8)
                for slot in slots:
                    set_value(slot, 1)
                return {
                    "dp_rank": 0,
                    "per_sample_handles": [
                        {
                            field: handle(self.rank * 16 + sample * 4 + index)
                            for index, field in enumerate(observer._FIELDS)
                        }
                        for sample in range(2)
                    ],
                }

            def release_ipc_buffer(self):
                self.releases += 1
                return "released"

        actor = ray.remote(Worker)
        metadata = actor.__ray_metadata__
        assert (
            metadata.modified_class.get_topk_logits_ipc
            is not Worker.get_topk_logits_ipc
        )
        module = SimpleNamespace(DTensorPolicyWorkerV2=actor)
        observer.patch_dtensor_worker(module)
        observer.patch_dtensor_worker(module)  # No duplicate hook installation.
        assert (
            metadata.method_meta.methods["get_topk_logits_ipc"]
            is metadata.modified_class.get_topk_logits_ipc
        )
        restored = cloudpickle.loads(cloudpickle.dumps(metadata.modified_class))
        workers = [restored(0), restored(1)]
        # A real rebuild would import CUDA memory. Fail immediately if attempted.
        with patch(
            "torch.multiprocessing.reductions.rebuild_cuda_tensor",
            side_effect=AssertionError("observer attempted an extra CUDA import"),
        ) as forbidden:
            forbidden.__signature__ = signature
            for worker in workers:
                result = metadata.method_meta.methods["get_topk_logits_ipc"](
                    worker, "unchanged", k=2
                )
                assert len(result["per_sample_handles"]) == 2
            set_value(0, -3)
            set_value(1, 0)
            workers[0].get_topk_logits_ipc("unchanged", k=2)
            set_value(0, -2)
            assert workers[0].release_ipc_buffer() == "released"
            assert workers[1].release_ipc_buffer() == "released"
            assert forbidden.call_count == 0
        events = [
            json.loads(line)
            for path in (Path(run) / "legacy-ipc-counters").glob("*.jsonl")
            for line in path.read_text().splitlines()
        ]
        selected = [event for event in events if event["controller_selects_output"]]
        discarded = [
            event for event in events if not event["controller_selects_output"]
        ]
        assert [event["stage"] for event in selected] == [
            "after_export",
            "before_next_export",
            "after_export",
            "before_release",
        ]
        assert [event["generation"] for event in selected] == [1, 1, 2, 2]
        assert [event["negative_counter_count"] for event in selected] == [0, 1, 0, 1]
        assert selected[1]["counters"][0]["value"] == -3
        assert selected[1]["counters"][1]["value"] == 0
        assert all(event["negative_counter_count"] == 0 for event in discarded)
        assert len(discarded) == 2 and all(
            len(event["counters"]) == 8 for event in events
        )
        assert all(event["read_error_count"] == 0 for event in events)
        assert [worker.exports for worker in workers] == [2, 1]
        assert [worker.releases for worker in workers] == [1, 1]
        from verify_legacy_ipc_observations import verify

        config = {
            "cluster": {"num_nodes": 1, "gpus_per_node": 2},
            "loss_fn": {"teacher_topk_ipc_k": 2},
            "teachers": [
                {
                    "model_name": "teacher-snapshot",
                    "is_cross_tokenizer": True,
                    "dtensor_cfg": {"enabled": True, "_v2": True},
                }
            ],
        }
        summary = verify(run, config, {})
        assert summary["coverage_status"] == "PASS", summary
        assert summary["counter_lifetime_status"] == "NEGATIVE_COUNTERS_CONFIRMED"
        paths = list((Path(run) / "legacy-ipc-counters").glob("*.jsonl"))
        path = paths[0]
        saved = path.read_text()
        path.write_text("\n".join(saved.splitlines()[:-1]) + "\n")
        assert verify(run, config, {})["coverage_status"] == "FAIL"
        path.unlink()
        assert verify(run, config, {})["coverage_status"] == "FAIL"

        class DenseWorker(Worker):
            def get_full_logits_ipc(self, data, *, reusable_ipc=False):
                assert data == "unchanged"
                self.exports += 1
                set_value(self.rank * 16, 1)
                payload = handle(self.rank * 16)
                if reusable_ipc:
                    from nemo_rl.utils.reusable_cuda_ipc import (
                        ReusableCudaIPCDescriptor,
                    )

                    payload = ReusableCudaIPCDescriptor(
                        bytes([2]) * 64,
                        (3, 2),
                        torch.float32,
                        24,
                        bytes([1]) * 16,
                        os.getpid(),
                        "00000000-0000-0000-0000-000000000001",
                    )
                return {
                    "dp_rank": 0,
                    "per_sample_handles": [
                        {
                            "payload_ipc": payload,
                            "tp_rank": self.rank,
                            "cp_rank": 0,
                            "sample_index_in_buf": sample,
                            "stored_seq_len": 3,
                        }
                        for sample in range(2)
                    ],
                }

        dense_actor = ray.remote(DenseWorker)
        dense_meta = dense_actor.__ray_metadata__
        observer.patch_megatron_worker(
            SimpleNamespace(MegatronPolicyWorker=dense_actor)
        )
        dense_cls = cloudpickle.loads(cloudpickle.dumps(dense_meta.modified_class))
        dense_workers = [dense_cls(0), dense_cls(1)]
        dense_run = str(Path(run) / "dense")
        with (
            patch.dict(os.environ, {"XTOKEN_NUMERICAL_CAPTURE_RUN": dense_run}),
            patch(
                "torch.multiprocessing.reductions.rebuild_cuda_tensor",
                side_effect=AssertionError("extra CUDA import"),
            ) as forbidden,
            patch(
                "nemo_rl.utils.reusable_cuda_ipc.open_reusable_cuda_ipc",
                side_effect=AssertionError("extra raw CUDA import"),
            ) as raw_forbidden,
        ):
            forbidden.__signature__ = signature
            for worker in dense_workers:
                dense_meta.method_meta.methods["get_full_logits_ipc"](
                    worker, "unchanged"
                )
                set_value(worker.rank * 16, -3)
                worker.release_ipc_buffer()
            dense_workers[0].get_full_logits_ipc("unchanged", reusable_ipc=True)
            dense_workers[0].release_ipc_buffer()
            assert forbidden.call_count == raw_forbidden.call_count == 0
        dense_config = {
            "cluster": config["cluster"],
            "loss_fn": {"teacher_topk_ipc_k": 0},
            "teachers": [
                {
                    "model_name": "teacher-snapshot",
                    "is_cross_tokenizer": True,
                    "dtensor_cfg": {"enabled": False},
                    "megatron_cfg": {"enabled": True},
                }
            ],
        }
        dense_summary = verify(dense_run, dense_config, {})
        assert dense_summary["coverage_status"] == "PASS", dense_summary
        assert dense_summary["counter_lifetime_status"] == "NEGATIVE_COUNTERS_CONFIRMED"
        assert dense_summary["totals"]["dense_all_tp"]["workers"] == 2
        assert dense_summary["totals"]["dense_all_tp"]["reusable_records"] == 4
        assert dense_summary["totals"]["discarded_tp_nonzero"]["workers"] == 0
        assert (
            dense_summary["unique_boundary_counters_by_transport"]["mcore_dense"][
                "negative"
            ]
            == 2
        )

        class RawSparseWorker(Worker):
            def get_topk_logits_ipc(self, data, *, k, bad_empty=False):
                self.exports += 1
                if self.rank > 0 or bad_empty:
                    return {
                        "dp_rank": 0,
                        "per_sample_handles": [],
                        "sparse_ipc_publisher": self.rank == 0,
                    }
                from nemo_rl.utils.reusable_cuda_ipc import ReusableCudaIPCDescriptor

                descriptor = ReusableCudaIPCDescriptor(
                    bytes([9]) * 64,
                    (2, 3, 2),
                    torch.float32,
                    48,
                    bytes([1]) * 16,
                    os.getpid(),
                    "00000000-0000-0000-0000-000000000001",
                )
                return {
                    "dp_rank": 0,
                    "sparse_ipc_publisher": True,
                    "per_sample_handles": [
                        {field: descriptor for field in observer._FIELDS[:3]}
                        for _ in range(2)
                    ],
                }

        raw_actor = ray.remote(RawSparseWorker)
        observer.patch_dtensor_worker(SimpleNamespace(DTensorPolicyWorkerV2=raw_actor))
        raw_cls = cloudpickle.loads(
            cloudpickle.dumps(raw_actor.__ray_metadata__.modified_class)
        )
        raw_workers = [raw_cls(0), raw_cls(1)]
        raw_run = str(Path(run) / "raw-sparse")
        with (
            patch.dict(os.environ, {"XTOKEN_NUMERICAL_CAPTURE_RUN": raw_run}),
            patch(
                "nemo_rl.utils.reusable_cuda_ipc.open_reusable_cuda_ipc",
                side_effect=AssertionError("extra raw CUDA import"),
            ) as raw_forbidden,
        ):
            for _ in range(2):
                for worker in raw_workers:
                    worker.get_topk_logits_ipc("unchanged", k=2)
            for worker in raw_workers:
                worker.release_ipc_buffer()
            assert raw_forbidden.call_count == 0
            try:
                raw_workers[0].get_topk_logits_ipc("unchanged", k=2, bad_empty=True)
            except ValueError as error:
                assert "unmarked empty publisher" in str(error)
            else:
                raise AssertionError("Empty TP0 publisher passed observation")
        raw_summary = verify(raw_run, config, {})
        assert raw_summary["coverage_status"] == "PASS", raw_summary
        assert raw_summary["totals"]["selected_tp0"]["reusable_records"] == 24
        assert (
            raw_summary["totals"]["discarded_tp_nonzero"][
                "intentional_nonpublisher_events"
            ]
            == 4
        )
        assert raw_summary["counter_lifetime_status"] == "NO_COUNTER_ANOMALY_OBSERVED"
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "events": len(events),
                    "extra_cuda_imports": 0,
                    "ray_version": ray.__version__,
                    "modified_class_cloudpickle_roundtrip": True,
                    "method_metadata_dispatch": True,
                    "missing_rank_and_release_rejected": True,
                    "dense_all_tp_and_reusable_metadata": True,
                    "sparse_reusable_and_intentional_nonpublisher": True,
                }
            )
        )


if __name__ == "__main__":
    main()
