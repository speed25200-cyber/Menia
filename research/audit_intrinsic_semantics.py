"""Re-express and audit the published finite semantic-information model.

Model adapted from Kolchinsky's MIT-licensed notebook at commit
bc56a371130a4073062c766654ca4796110d5c8e. See the protocol and
licenses/semantic_information-MIT.txt. No downloaded code is executed.
This is a model calculation, not a measurement of subjective experience.
"""

import argparse
from collections import Counter
from itertools import product
import json
import math
from pathlib import Path

import numpy as np


def partitions(n):
    """All set partitions, uniquely represented as restricted-growth strings."""
    def extend(prefix):
        if len(prefix) == n:
            yield tuple(prefix)
        else:
            for label in range(max(prefix) + 2):
                yield from extend(prefix + [label])
    yield from extend([0])


def contiguous(partition):
    return all(b in (a, a + 1) for a, b in zip(partition, partition[1:]))


def entropy(probabilities):
    p = np.asarray(probabilities)
    p = p[p > 0]
    return float(-np.dot(p, np.log2(p)))


def build_dynamics(mode):
    """Preserve the notebook's ordering, reverse-rate assignments and clock."""
    locations = list(range(5)) + [-1]  # -1 replaces the notebook's '-' symbol.
    states = list(product(range(5), locations, range(5), locations))
    indices = {state: i for i, state in enumerate(states)}
    rates = {}

    def energy(state):
        _, _, level, food = state
        return 10 * level - (100 * math.log(2) if level == 0 else 0) + (100 if food != -1 else 0)

    for i, (loc, target, level, food) in enumerate(states):
        next_loc, next_target, next_level, next_food = loc, target, level, food
        eats = level > 0 and food != -1 and abs(loc - food) <= 1
        if eats:
            next_target, next_level, next_food = -1, 4, -1
        elif level > 0:
            next_level -= 1
            if target != -1:
                direction = (target > loc) - (target < loc)
                next_loc = min(4, max(0, loc + mode * direction))
        outcomes = [(next_food, 1.0)] if next_food == -1 else [(next_food, 0.9), (-1, 0.1)]
        for food_after, probability in outcomes:
            state_after = (next_loc, next_target, next_level, food_after)
            j = indices[state_after]
            # Assignment, not accumulation: retain the source algorithm exactly.
            rates[i, j] = probability
            rates[j, i] = probability * math.exp(energy(state_after) - energy(states[i]))

    rates = {edge: rate for edge, rate in rates.items() if edge[0] != edge[1]}
    totals = np.zeros(len(states))
    for (i, _), rate in rates.items():
        totals[i] += rate
    normalizer = float(totals.max())
    rates = {edge: rate / normalizer for edge, rate in rates.items()}
    rates.update({(i, i): 1 - total / normalizer for i, total in enumerate(totals)})
    src = np.array([i for i, _ in rates])
    dst = np.array([j for _, j in rates])
    weight = np.array(list(rates.values()))
    assert weight.min() >= 0
    np.testing.assert_allclose(np.bincount(src, weights=weight), 1, atol=1e-14, rtol=0)
    xindex = np.array([i // 6 for i in range(len(states))])
    dead = np.array([state[2] == 0 for state in states])
    return states, indices, src, dst, weight, xindex, dead


def initial_distribution(partition, indices):
    sizes = Counter(partition)
    p = np.zeros(len(indices))
    joint = np.zeros((5, 5))
    for target, food in product(range(5), repeat=2):
        if partition[target] == partition[food]:
            probability = 1 / (5 * sizes[partition[target]])
            p[indices[2, target, 4, food]] = probability
            joint[target, food] = probability
    mi = entropy(joint.sum(axis=0)) + entropy(joint.sum(axis=1)) - entropy(joint.ravel())
    expected_mi = entropy([size / 5 for size in sizes.values()])
    assert abs(mi - expected_mi) < 1e-14
    np.testing.assert_allclose(joint.sum(axis=0), 0.2, atol=1e-14, rtol=0)
    np.testing.assert_allclose(joint.sum(axis=1), 0.2, atol=1e-14, rtol=0)
    return p, mi


def run_audit():
    all6 = list(partitions(6))
    all5 = list(partitions(5))
    source6 = [p for p in all6 if contiguous(p)]
    source5 = {p[:5] for p in source6}
    assert len(all6) == 203 and len(source6) == 32
    assert len(all5) == 52 and len(source5) == 16
    assert {p[:5] for p in all6} == set(all5)
    identity = all5.index((0, 1, 2, 3, 4))
    scrambled = all5.index((0, 0, 0, 0, 0))
    source_mask = np.array([p in source5 for p in all5])
    horizons = (0, 1, 2, 3, 5, 8, 10)
    records = []
    curves = []
    diagnostics = []

    for name, mode in (("towards_food", 1), ("away_from_food", -1)):
        states, indices, src, dst, weight, xindex, dead = build_dynamics(mode)
        initial = [initial_distribution(p, indices) for p in all5]
        current = np.stack([p for p, _ in initial])
        information = np.array([mi for _, mi in initial])
        rounded_info = np.round(information, 5)
        diagnostics.append({"mode": name, "joint_states": len(states), "transition_entries": len(weight)})
        for horizon in range(max(horizons) + 1):
            np.testing.assert_allclose(current.sum(axis=1), 1, atol=1e-13, rtol=0)
            assert current.min() >= 0
            if horizon in horizons:
                macro_entropy = np.array([entropy(np.bincount(xindex, weights=p)) for p in current])
                dead_probability = current[:, dead].sum(axis=1)
                viability = -macro_entropy - 100 * dead_probability
                rounded = np.round(viability, 5)
                same_viability = rounded == rounded[identity]
                full_si = float(rounded_info[same_viability].min())
                restricted_si = float(rounded_info[same_viability & source_mask].min())
                full_curve = {float(mi): float(rounded[rounded_info == mi].max()) for mi in sorted(set(rounded_info))}
                source_curve = {float(mi): float(rounded[(rounded_info == mi) & source_mask].max())
                                for mi in sorted(set(rounded_info[source_mask]))}
                gaps = {mi: full_curve[mi] - source_curve[mi] for mi in source_curve}
                # The notebook reads S from the upper curve. That is not always
                # the same as minimizing information subject to equal viability.
                source_readout = min(mi for mi, value in source_curve.items() if value == rounded[identity])
                full_readout = min(mi for mi, value in full_curve.items() if value == rounded[identity])
                gap_mi = max(gaps, key=gaps.get)
                witness = np.flatnonzero((rounded_info == gap_mi) & (rounded == full_curve[gap_mi]))[0]
                record = {
                    "mode": name, "horizon": horizon,
                    "initial_information_bits": float(information[identity]),
                    "semantic_bits_all_partitions": full_si,
                    "semantic_bits_contiguous_partitions": restricted_si,
                    "upper_curve_readout_contiguous_bits": source_readout,
                    "upper_curve_readout_all_bits": full_readout,
                    "viability_value_bits": float(viability[identity] - viability[scrambled]),
                    "actual_viability_bits": float(viability[identity]),
                    "scrambled_viability_bits": float(viability[scrambled]),
                    "actual_alive_probability": float(1 - dead_probability[identity]),
                    "scrambled_alive_probability": float(1 - dead_probability[scrambled]),
                    "macro_entropy_only_value_bits": float(macro_entropy[scrambled] - macro_entropy[identity]),
                    "max_curve_improvement_bits": max(gaps.values()),
                    "curve_improvement_witness": {
                        "information_bits_rounded": gap_mi,
                        "contiguous_best_viability_bits_rounded": source_curve[gap_mi],
                        "all_best_viability_bits_rounded": full_curve[gap_mi],
                        "partition": list(all5[witness])},
                    "minimum_information_equal_viability_partitions": [list(p) for i, p in enumerate(all5)
                                                                       if same_viability[i] and rounded_info[i] == full_si],
                }
                records.append(record)
                if horizon in (1, 5):
                    curves.append({"mode": name, "horizon": horizon, "points": [
                        {"partition": list(p), "information_bits": float(information[i]),
                         "viability_bits": float(viability[i]), "alive_probability": float(1 - dead_probability[i]),
                         "in_source_search": bool(source_mask[i])}
                        for i, p in enumerate(all5)]})
            if horizon < max(horizons):
                current = np.stack([np.bincount(dst, weights=p[src] * weight, minlength=len(states)) for p in current])

    return {
        "schema": "intrinsic-semantics-audit-v1",
        "protocol": "docs/INTRINSIC_SEMANTICS_PROTOCOL.md",
        "source_doi": "10.1098/rsfs.2018.0041",
        "source_commit": "bc56a371130a4073062c766654ca4796110d5c8e",
        "source_sha256": {
            "model.ipynb": "55b33ca1e23468f925a04f4b6f6dc90bbfb590bdc6f032bb4ceb8744a580613c",
            "utils.py": "2d369987fb87e822989c79ff5a51c02eca59b2bdccb51651c8646f6eece656e6"},
        "external_code_executed": False,
        "intervention_counts": {"full_6_states": 203, "contiguous_6_states": 32,
                                "distinct_on_initial_support_full": 52, "distinct_on_initial_support_contiguous": 16},
        "parameters": {"locations": 5, "levels_in_code": [0, 1, 2, 3, 4], "eat_radius": 1,
                       "food_disappearance_rate": 0.1, "dead_internal_entropy_bits": 100,
                       "horizons": list(horizons), "source_rounding_decimals": 5},
        "dynamics": diagnostics, "results": records, "selected_horizons_all_partition_points": curves,
        "scope": "Finite model and deterministic coarse-grainings only; no subjective experience or novelty established.",
    }


def compare(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            compare(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            compare(a, b)
    elif isinstance(actual, float):
        assert math.isclose(actual, expected, abs_tol=1e-10, rel_tol=0), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/intrinsic-semantics-audit/report.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = run_audit()
    if args.check:
        compare(report, json.loads(args.output.read_text(encoding="utf-8")))
        print("Intrinsic semantics audit: report reproduced (absolute tolerance 1e-10).")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    for result in report["results"]:
        if result["horizon"] == 5:
            print(result["mode"], "semantic bits", result["semantic_bits_all_partitions"],
                  "viability value", result["viability_value_bits"])
    print("No consciousness conclusion.")


if __name__ == "__main__":
    main()
