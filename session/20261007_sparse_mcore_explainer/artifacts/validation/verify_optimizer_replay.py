"""CPU selfcheck of the replay's exact pure range-coverage function.

The replay entry point imports GPU-only MCore dependencies. Extract only its
unchanged pure Python checker so this small ownership audit can run on the login
host. This does not exercise a model forward, DDP, NCCL, or an optimizer.
"""

import ast
from copy import deepcopy
from pathlib import Path


def main():
    source = Path(__file__).with_name("replay_controller_gradients.py")
    tree = ast.parse(source.read_text(), filename=str(source))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "verify_range_coverage"
    ]
    assert len(nodes) == 1
    namespace = {}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    verify = namespace["verify_range_coverage"]
    inventory = [
        {"name": "a", "full_local_shape": [7], "full_local_numel": 7},
        {"name": "b", "full_local_shape": [2, 2], "full_local_numel": 4},
    ]

    def shard(name, start, end):
        return {
            "name": name,
            "start": start,
            "end": end,
            "full_local_shape": [7] if name == "a" else [2, 2],
        }

    # Four CP+DP owners; one is empty and one owns two optimizer-group shards.
    reports = [
        [{"shards": [shard("a", 0, 3)]}],
        [{"shards": [shard("a", 3, 7)]}, {"shards": [shard("b", 0, 1)]}],
        [{"shards": []}],
        [{"shards": [shard("b", 1, 4)]}],
    ]
    assert verify(reports, inventory) == 11
    assert verify(list(reversed(reports)), inventory) == 11
    assert verify([[{"shards": []}]], []) == 0

    cases = {}
    value = deepcopy(reports)
    value[1][0]["shards"][0]["start"] = 4
    cases["gap"] = value
    value = deepcopy(reports)
    value[1][0]["shards"][0]["start"] = 2
    cases["overlap"] = value
    value = deepcopy(reports)
    value[0][0]["shards"].append(shard("a", 0, 3))
    cases["duplicate"] = value
    value = deepcopy(reports)
    value[1][1]["shards"] = []
    value[3][0]["shards"] = []
    cases["missing_parameter"] = value
    value = deepcopy(reports)
    value[0][0]["shards"][0]["full_local_shape"] = [1, 7]
    cases["wrong_shape"] = value
    value = deepcopy(reports)
    value[3][0]["shards"][0]["end"] = 5
    cases["range_past_end"] = value
    value = deepcopy(reports)
    value[3][0]["shards"][0]["end"] = 3
    cases["missing_tail"] = value
    value = deepcopy(reports)
    value[0][0]["shards"][0]["name"] = "unknown"
    cases["unknown_parameter"] = value
    for label, invalid in cases.items():
        try:
            verify(invalid, inventory)
        except (AssertionError, KeyError):
            continue
        raise AssertionError(f"Range checker accepted {label}")
    print(
        "PASS replay range selfcheck: complete/empty/reordered owners and 8 invalid layouts"
    )


if __name__ == "__main__":
    main()
