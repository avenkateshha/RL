"""Check actual DTensor counter coverage separately from lifetime evidence."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml


def verify(run, config, runtime):
    world = int(runtime.get("nodes", config["cluster"]["num_nodes"])) * int(
        runtime.get("gpus_per_node_used", config["cluster"]["gpus_per_node"])
    )
    sparse_k = config["loss_fn"]["teacher_topk_ipc_k"]
    teachers = [
        teacher
        for teacher in config["teachers"]
        if teacher["is_cross_tokenizer"]
        and (
            sparse_k > 0
            and teacher["dtensor_cfg"]["enabled"]
            and teacher["dtensor_cfg"].get("_v2", False)
            or sparse_k == 0
            and teacher.get("megatron_cfg", {}).get("enabled", False)
        )
    ]
    expected = {teacher["model_name"]: teacher for teacher in teachers}
    expected_transport = {
        teacher["model_name"]: "dtensor_sparse" if sparse_k > 0 else "mcore_dense"
        for teacher in teachers
    }
    errors = []
    if not expected:
        errors.append(
            "No observed legacy sparse/dense teachers expected from supplied config"
        )
    if len(expected) != len(teachers):
        errors.append(
            "Duplicate model identities cannot be disambiguated by this observer"
        )
    events = defaultdict(list)
    for path in sorted((Path(run) / "legacy-ipc-counters").glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                event = json.loads(line)
                assert event["schema_version"] == 1
                events[event["model_name"], event["rank"]].append(event)
            except (ValueError, KeyError, AssertionError) as error:
                errors.append(f"Malformed event {path.name}:{line_number}: {error}")
    expected_keys = {(model, rank) for model in expected for rank in range(world)}
    for model, rank in sorted(expected_keys - events.keys()):
        errors.append(f"Missing worker: model={model}, rank={rank}")
    for model, rank in sorted(events.keys() - expected_keys):
        errors.append(f"Unexpected worker: model={model}, rank={rank}")

    totals = {
        selected: {
            "workers": 0,
            "events": 0,
            "after_export_counters": 0,
            "boundary_counters": 0,
            "boundary_negative": 0,
            "boundary_zero": 0,
            "boundary_positive": 0,
            "read_errors": 0,
            "reusable_records": 0,
            "empty_records": 0,
            "intentional_nonpublisher_events": 0,
        }
        for selected in ("selected_tp0", "discarded_tp_nonzero", "dense_all_tp")
    }
    workers = []
    unique_boundary_counters = {}
    for (model, rank), observed in sorted(events.items()):
        if (model, rank) not in expected_keys:
            continue
        observed.sort(key=lambda event: event["time_ns"])
        label = f"model={model}, rank={rank}"
        selections = {event["controller_selects_output"] for event in observed}
        pids = {event["pid"] for event in observed}
        if len(selections) != 1 or len(pids) != 1:
            errors.append(f"Worker identity changed: {label}")
        selected = observed[0]["controller_selects_output"]
        transport = expected_transport[model]
        bucket_name = (
            "dense_all_tp"
            if transport == "mcore_dense"
            else "selected_tp0"
            if selected
            else "discarded_tp_nonzero"
        )
        bucket = totals[bucket_name]
        bucket["workers"] += 1
        by_generation = defaultdict(list)
        for event in observed:
            by_generation[event["generation"]].append(event)
            bucket["events"] += 1
            if event.get("transport", "dtensor_sparse") != transport:
                errors.append(f"Wrong transport: {label}")
            expected_selected = (
                event["tp_rank"] == 0 if transport == "dtensor_sparse" else True
            )
            if selected != expected_selected:
                errors.append(f"Inconsistent TP selection: {label}")
            bucket["reusable_records"] += event.get("reusable_records", 0)
            bucket["empty_records"] += event.get("empty_records", 0)
            explained = (
                len(event["counters"])
                + event.get("reusable_records", 0)
                + event.get("empty_records", 0)
            )
            nonpublisher = event.get("intentional_nonpublisher", False)
            if nonpublisher:
                if (
                    transport != "dtensor_sparse"
                    or selected
                    or event["tp_rank"] <= 0
                    or event.get("sample_records") != 0
                    or explained
                ):
                    errors.append(f"Invalid intentional nonpublisher: {label}")
                bucket["intentional_nonpublisher_events"] += 1
            if not explained and not nonpublisher:
                errors.append(f"Empty export counter coverage: {label}")
            if transport == "dtensor_sparse" and event.get("reusable_records", 0):
                if (
                    explained != event.get("field_records")
                    or explained < 3 * event.get("sample_records", 0)
                    or event.get("sample_records", 0) <= 0
                ):
                    errors.append(f"Incomplete reusable sparse field coverage: {label}")
            if transport == "mcore_dense" and explained != event.get("sample_records"):
                errors.append(f"Incomplete dense sample payload coverage: {label}")
            for counter in event["counters"]:
                if "read_error" in counter or counter["value"] is None:
                    bucket["read_errors"] += 1
                elif event["stage"] == "after_export":
                    bucket["after_export_counters"] += 1
                else:
                    bucket["boundary_counters"] += 1
                    value = counter["value"]
                    category = (
                        "negative" if value < 0 else "positive" if value > 0 else "zero"
                    )
                    bucket[f"boundary_{category}"] += 1
                    unique_boundary_counters[
                        (model, rank, event["generation"], counter["descriptor_sha256"])
                    ] = (transport, category)
        generations = sorted(by_generation)
        if generations != list(range(1, len(generations) + 1)):
            errors.append(f"Missing or noncontiguous export generations: {label}")
        for generation, pair in sorted(by_generation.items()):
            stages = [event["stage"] for event in pair]
            allowed_boundaries = (
                {"before_release"}
                if generation == generations[-1]
                else {"before_release", "before_next_export"}
            )
            if (
                len(stages) != 2
                or stages[0] != "after_export"
                or stages[1] not in allowed_boundaries
            ):
                errors.append(
                    f"Incomplete stages for {label}, generation={generation}: {stages}"
                )
            elif {item["descriptor_sha256"] for item in pair[0]["counters"]} != {
                item["descriptor_sha256"] for item in pair[1]["counters"]
            }:
                errors.append(f"Counter identities changed within generation: {label}")
        workers.append(
            {
                "model_name": model,
                "rank": rank,
                "transport": transport,
                "generations": generations,
            }
        )
    read_errors = sum(bucket["read_errors"] for bucket in totals.values())
    negatives = sum(bucket["boundary_negative"] for bucket in totals.values())
    positives = sum(bucket["boundary_positive"] for bucket in totals.values())
    if read_errors:
        errors.append(f"Unreadable exported counters: {read_errors}")
    status = (
        "NEGATIVE_COUNTERS_CONFIRMED"
        if negatives
        else "OUTSTANDING_COUNTERS_OBSERVED"
        if positives
        else "NO_COUNTER_ANOMALY_OBSERVED"
    )
    return {
        "coverage_status": "FAIL" if errors else "PASS",
        "counter_lifetime_status": status if not errors or negatives else "UNPROVEN",
        "expected_worker_count": len(expected_keys),
        "observed_worker_count": len(events),
        "errors": errors,
        "totals": totals,
        "unique_boundary_counters_by_transport": {
            transport: {
                category: sum(
                    t == transport and c == category
                    for t, c in unique_boundary_counters.values()
                )
                for category in ("negative", "zero", "positive")
            }
            for transport in sorted(set(expected_transport.values()))
        },
        "workers": workers,
        "scope": "Reads counter observations only; negatives do not stop optimizer replay.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config or args.run / "resolved_config.yaml"
    runtime_path = args.runtime or args.run / "runtime.json"
    summary = verify(
        args.run,
        yaml.safe_load(config_path.read_text()),
        json.loads(runtime_path.read_text()),
    )
    output = args.output or args.run / "legacy-ipc-observations-summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["coverage_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
