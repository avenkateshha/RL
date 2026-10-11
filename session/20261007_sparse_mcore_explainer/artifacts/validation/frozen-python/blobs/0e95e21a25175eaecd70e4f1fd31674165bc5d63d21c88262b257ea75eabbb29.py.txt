"""Copy to a run-owned sitecustomize.py to observe real Ray worker imports."""

import importlib.abc
import importlib.machinery
import os
import sys


class _ReferenceLoader(importlib.abc.Loader):
    def __init__(self, original, callback):
        self.original = original
        self.callback = callback

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        if self.callback in ("patch_dtensor_worker", "patch_megatron_worker"):
            import legacy_ipc_counter_observer

            getattr(legacy_ipc_counter_observer, self.callback)(module)
        else:
            import controller_reference

            getattr(controller_reference, self.callback)(module)


class _ReferenceFinder(importlib.abc.MetaPathFinder):
    targets = {
        "nemo_rl.models.megatron.train": "patch_megatron",
        "nemo_rl.models.automodel.train": "patch_automodel",
        "nemo_rl.models.policy.workers.dtensor_policy_worker_v2": "patch_dtensor_worker",
        "nemo_rl.models.policy.workers.megatron_policy_worker": "patch_megatron_worker",
    }

    def find_spec(self, fullname, path=None, target=None):
        callback = self.targets.get(fullname)
        if callback is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _ReferenceLoader(spec.loader, callback)
        return spec


if os.environ.get("XTOKEN_NUMERICAL_CAPTURE_RUN"):
    sys.meta_path.insert(0, _ReferenceFinder())
