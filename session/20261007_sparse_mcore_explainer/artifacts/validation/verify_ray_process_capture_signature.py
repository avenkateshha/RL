"""Reproduce Ray's isolated-driver actor signature fallback without actors."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import ray
from ray._common import signature
from ray._private.function_manager import FunctionActorManager
from ray._raylet import PythonFunctionDescriptor
from ray.actor import _ActorClassMethodMetadata, _modify_class
from ray.util.tracing.tracing_helper import _inject_tracing_into_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manager = FunctionActorManager.__new__(FunctionActorManager)
    placeholder = manager._create_fake_actor_class(
        "MissingDependencyActor", ["__ray_call__"], "missing optional driver dependency"
    )
    normal = _modify_class(type("HealthyActor", (), {}))
    reports = []
    for name, cls in (("healthy", normal), ("driver_placeholder", placeholder)):
        _inject_tracing_into_class(cls)
        metadata = _ActorClassMethodMetadata.create(
            cls, PythonFunctionDescriptor(__name__, "__init__", name)
        )
        parameters = metadata.signatures["__ray_call__"]
        positional_error = None
        try:
            signature.flatten_args(parameters, [lambda actor: None], {})
        except TypeError as error:
            positional_error = str(error)
        signature.flatten_args(parameters, [], {"fn": lambda actor: None})
        assert (positional_error is None) == (name == "healthy")
        reports.append(
            {
                "case": name,
                "ray_method_signature": str(parameters),
                "positional_error": positional_error,
                "keyword_fn": "PASS",
            }
        )
    sources = []
    for function in (
        FunctionActorManager._load_actor_class_from_gcs,
        FunctionActorManager._create_fake_actor_class,
        _ActorClassMethodMetadata.create,
        _modify_class,
        signature.extract_signature,
    ):
        path = Path(inspect.getsourcefile(function))
        line = inspect.getsourcelines(function)[1]
        relative = "python/ray/" + str(path).split("/site-packages/ray/", 1)[1]
        sources.append(
            {
                "function": function.__qualname__,
                "path": str(path),
                "line": line,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "upstream": f"https://github.com/ray-project/ray/blob/{ray.__commit__}/{relative}#L{line}",
            }
        )
    report = {
        "status": "PASS_REPRODUCTION",
        "ray_version": ray.__version__,
        "ray_commit": ray.__commit__,
        "cases": reports,
        "sources": sources,
        "scope": "Actual Ray metadata/signature code only; no actors, models or GPU. Establishes placeholder positional binding failure and compatible explicit fn keyword; actual DTensor driver import fallback still needs runtime evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
