"""Exact representation audit, NOT a consciousness or biological model.

A bit-valued activity h and plastic connection w obey
    h' = (w AND h) XOR x; w' = w XOR h'.
Compare a mutable connection, a fixed transition table with state, and a
finite-horizon Boolean DAG that includes the update rule. The DAG is not an
unbounded streaming memoryless network. No physical equivalence is asserted.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random


STATES = tuple(itertools.product((0, 1), repeat=2))
# Optional clamps precede the transition; freeze/cut affect this step only.
PATCHES = tuple(itertools.product((None, 0, 1), (None, 0, 1), (False, True), (False, True)))
IDENTITY = (None, None, False, False)
TABLE = ((0, 0), (1, 1), (0, 1), (1, 0),
         (0, 0), (1, 1), (1, 0), (0, 1))  # index = 4*h + 2*w + x


class PlasticBit:
    def __init__(self, state):
        self.h, self.weight = state

    def step(self, x, patch=IDENTITY):
        h_set, w_set, freeze, cut = patch
        if h_set is not None:
            self.h = h_set
        if w_set is not None:
            self.weight = w_set
        self.h = (0 if cut else self.weight & self.h) ^ x
        if not freeze:
            self.weight ^= self.h
        return self.h, self.weight


def table_step(state, x, patch=IDENTITY):
    h_set, w_set, freeze, cut = patch
    h, w = state
    h = h if h_set is None else h_set
    w = w if w_set is None else w_set
    following_h, following_w = TABLE[4 * (0 if cut else h) + 2 * w + x]
    return following_h, w if freeze else following_w


def build_dag(horizon):
    """One immutable graph for ALL inputs/clamps at a given horizon."""
    nodes = []

    def add(op, *args):
        nodes.append((op, *args))
        return len(nodes) - 1

    zero = add("constant", 0)
    h, w = add("input", "h0"), add("input", "w0")
    outputs = []
    for t in range(horizon):
        x, hs, hv, ws, wv, freeze, cut = [
            add("input", f"{t}:{name}") for name in ("x", "hs", "hv", "ws", "wv", "freeze", "cut")]
        h = add("mux", hs, hv, h)
        w = add("mux", ws, wv, w)
        feedback = add("and", w, h)
        feedback = add("mux", cut, zero, feedback)
        h = add("xor", feedback, x)
        changed_w = add("xor", w, h)
        w = add("mux", freeze, w, changed_w)
        outputs.append((h, w))
    # All dependencies point strictly backwards: the graph is acyclic.
    for i, (op, *args) in enumerate(nodes):
        if op not in ("input", "constant"):
            assert all(isinstance(j, int) and 0 <= j < i for j in args)
    return tuple(nodes), tuple(outputs)


def run_dag(graph, initial, xs, patches):
    nodes, outputs = graph
    values = {"h0": initial[0], "w0": initial[1]}
    for t, (x, (h_set, w_set, freeze, cut)) in enumerate(zip(xs, patches, strict=True)):
        for name, value in zip(("x", "hs", "hv", "ws", "wv", "freeze", "cut"),
                               (x, h_set is not None, h_set or 0,
                                w_set is not None, w_set or 0, freeze, cut), strict=True):
            values[f"{t}:{name}"] = int(value)
    resolved = []
    for op, *args in nodes:
        if op == "input":
            value = values[args[0]]
        elif op == "constant":
            value = args[0]
        elif op == "and":
            value = resolved[args[0]] * resolved[args[1]]
        elif op == "xor":
            value = (resolved[args[0]] + resolved[args[1]]) % 2
        elif op == "mux":
            value = resolved[args[1]] if resolved[args[0]] else resolved[args[2]]
        else:
            raise ValueError(op)
        resolved.append(value)
    return [(resolved[h], resolved[w]) for h, w in outputs]


def compare(initial, xs, patches, graph):
    plastic = PlasticBit(initial)
    table_state = initial
    direct, table = [], []
    for x, patch in zip(xs, patches, strict=True):
        direct.append(plastic.step(x, patch))
        table_state = table_step(table_state, x, patch)
        table.append(table_state)
    unfolded = run_dag(graph, initial, xs, patches)
    assert direct == table == unfolded, (initial, xs, patches, direct, table, unfolded)
    return direct


def audit():
    single = build_dag(1)
    one_step = 0
    for initial, x, patch in itertools.product(STATES, (0, 1), PATCHES):
        compare(initial, [x], [patch], single)
        one_step += 1

    horizon = 4
    graph = build_dag(horizon)
    single_patch_runs = 0
    for initial, xs, t, patch in itertools.product(
            STATES, itertools.product((0, 1), repeat=horizon), range(horizon), PATCHES):
        patches = [IDENTITY] * horizon
        patches[t] = patch
        compare(initial, xs, patches, graph)
        single_patch_runs += 1

    rng = random.Random(20260914)
    for _ in range(256):
        compare(rng.choice(STATES), [rng.randrange(2) for _ in range(horizon)],
                [rng.choice(PATCHES) for _ in range(horizon)], graph)

    xs = [0] * horizon
    dynamic = compare((1, 1), xs, [IDENTITY] * horizon, graph)
    frozen = compare((1, 1), xs, [(None, None, True, False)] * horizon, graph)
    clamped = compare((1, 1), xs, [(None, 0, False, False)] + [IDENTITY] * 3, graph)
    # Hand-derived trajectories: prevent agreement on an unintended update order.
    assert dynamic == [(1, 0), (0, 0), (0, 0), (0, 0)]
    assert frozen == [(1, 1)] * 4
    assert clamped == [(0, 0)] * 4
    nodes = graph[0]
    return {
        "kind": "exact plasticity representation audit; not a consciousness measure",
        "domain": {"state_bits": 2, "input_bits": 1, "patches": len(PATCHES),
                   "finite_dag_horizon": horizon},
        "checks": {"all_one_step_cases_equal": one_step,
                   "single_patch_trajectories_equal": single_patch_runs,
                   "multiple_patch_trajectories_equal": 256, "multiple_patch_seed": 20260914},
        "dag": {"nodes": len(nodes), "operation_nodes": sum(n[0] not in ("input", "constant") for n in nodes),
                "acyclic": True, "changes_during_execution": False,
                "serialized_graph_sha256": hashlib.sha256(json.dumps(graph).encode()).hexdigest()},
        "hand_checked_examples": {"initial": [1, 1], "inputs": xs,
                                  "dynamic": dynamic, "frozen": frozen, "weight_clamped_at_start": clamped},
        "scope": ["The fixed table retains state and is not memoryless.",
                  "The DAG includes plasticity updates and has a fixed finite horizon.",
                  "Interventions are transported at a chosen computational grain.",
                  "No equivalence of physical substrates or subjective experience was tested.",
                  "The construction is not a novelty claim or an implementation of self-awareness."],
        "source_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = json.loads(json.dumps(audit()))
    if args.check:
        assert report == json.loads(args.output.read_text(encoding="utf-8")), "Stored report differs"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["checks"]))
