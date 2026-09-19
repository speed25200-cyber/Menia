"""Local algebraic audit of a temporal-experience candidate; no consciousness test.

Independent implementation. No external source is executed or needed at replay.
See docs/TEMPORAL_BRIDGE_PROTOCOL.md for scope and fixed contrasts.
"""

import argparse
import json
import math
from pathlib import Path


def entropy(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p)


def observation(channel, belief):
    return [sum(row[j] * belief[j] for j in range(2)) for row in channel]


def components(channel, belief, preferences=(0.99, 0.01)):
    q = observation(channel, belief)
    risk = sum(qi * math.log(qi / ci) for qi, ci in zip(q, preferences))
    conditional = [entropy([channel[i][j] for i in range(2)]) for j in range(2)]
    ambiguity = sum(b * h for b, h in zip(belief, conditional))
    return {"predicted_observations": q, "risk": risk, "ambiguity": ambiguity,
            "expected_free_energy": risk + ambiguity}


def select(energies):
    # Preserve the published normalizer's epsilon, which breaks shift invariance.
    weights = [math.exp(math.log(0.99) - 4 * energy) + 1e-5 for energy in energies]
    return [w / sum(weights) for w in weights]


def symmetric_channel(diagonal):
    return [[diagonal, 1 - diagonal], [1 - diagonal, diagonal]]


def run_audit():
    diagonal = (0.75 + 1e-5) / (1 + 2e-5)
    channel = symmetric_channel(diagonal)
    channel_entropy = entropy(channel[0])
    error_energy = error_selection = error_shift = 0.0
    disagreements = 0
    risks = []
    for i in range(10001):
        p = i / 10000
        predicted_states = ([0.8 * p, 1 - 0.8 * p], [1 - p, p])
        energies, losses, risk_only = [], [], []
        for belief in predicted_states:
            c = components(channel, belief)
            q = c["predicted_observations"]
            loss = -sum(qi * math.log(ci) for qi, ci in zip(q, (0.99, 0.01)))
            loss -= entropy(q)
            loss += channel_entropy
            energy = c["expected_free_energy"]
            error_energy = max(error_energy, abs(energy - loss))
            disagreements += int((energy < 2.2) != (loss < 2.2))
            energies.append(energy)
            losses.append(loss)
            risk_only.append(c["risk"])
            risks.append(c["risk"])
        original = select(energies)
        reconstructed = select(losses)
        dropped_constant = select(risk_only)
        error_selection = max(error_selection, max(abs(a-b) for a, b in zip(original, reconstructed)))
        error_shift = max(error_shift, max(abs(a-b) for a, b in zip(original, dropped_constant)))

    matched_risk = [components(symmetric_channel(d), [0.5, 0.5]) for d in (0.9, 0.6)]
    matched_ambiguity = [components(symmetric_channel(0.75), [p, 1-p]) for p in (0.25, 0.75)]
    assert error_energy < 1e-12 and error_selection < 1e-12
    assert disagreements == 0
    assert error_shift > 1e-6  # Detect an invalid simplification of the normalizer.
    assert abs(matched_risk[0]["risk"] - matched_risk[1]["risk"]) < 1e-12
    assert abs(matched_risk[0]["ambiguity"] - matched_risk[1]["ambiguity"]) > 0.1
    assert abs(matched_ambiguity[0]["ambiguity"] - matched_ambiguity[1]["ambiguity"]) < 1e-12
    assert abs(matched_ambiguity[0]["risk"] - matched_ambiguity[1]["risk"]) > 0.1
    return {
        "schema": "temporal-bridge-audit-v1",
        "protocol": "docs/TEMPORAL_BRIDGE_PROTOCOL.md",
        "scope": "Local algebra, not a full simulation replication or a consciousness measure",
        "source": {
            "paper_doi": "10.1093/nc/niag018",
            "repository": "https://github.com/JanBellingrath/deep_parametric_generative_model_of_temporal_inference",
            "commit": "d7c01c65310e3cf932e4c5d4d1235be419a3a352",
            "file": "deep_parametric_generative_model.py",
            "sha256": "e4598a881183e330edfccb07b26bb86ab1be91216f1d7984fd2600cd05f44c82",
            "external_code_executed": False,
        },
        "local_decision_audit": {
            "beliefs": 10001, "policies": 2, "evaluations": len(risks),
            "preferences": [0.99, 0.01], "channel": channel,
            "constant_ambiguity_nats": channel_entropy,
            "max_energy_error": error_energy,
            "max_selection_probability_error": error_selection,
            "threshold_disagreements": disagreements,
            "risk_range_nats": [min(risks), max(risks)],
            "max_probability_change_if_constant_is_dropped": error_shift,
        },
        "proposed_contrasts": {
            "matched_risk": {"channel_diagonals": [0.9, 0.6], "belief": [0.5, 0.5], "cases": matched_risk},
            "matched_ambiguity": {"channel_diagonal": 0.75, "belief_first_coordinates": [0.25, 0.75], "cases": matched_ambiguity},
        },
        "limitations": [
            "No attention trajectories, skip frequencies or published plots reproduced",
            "No temporal reports generated and no human data fitted",
            "Functional equivalence does not disprove a phenomenal identity hypothesis",
            "Risk remains in the action computation even if its plot variable is removed",
            "The contrasts identify predictors, not consciousness or scientific novelty",
        ],
    }


def compare(actual, expected, path="report"):
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"Different fields at {path}")
        for k, value in actual.items():
            compare(value, expected[k], f"{path}.{k}")
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise AssertionError(f"Different lengths at {path}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f"{path}[{i}]")
    elif isinstance(actual, float):
        if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12):
            raise AssertionError(f"Numeric difference at {path}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"Difference at {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/temporal-bridge-audit/report.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = run_audit()
    if args.check:
        compare(report, json.loads(args.output.read_text(encoding="utf-8")))
        print("Temporal bridge audit reproduced (tolerance 1e-12).")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        print(args.output)
    print(json.dumps(report["local_decision_audit"], indent=2))


if __name__ == "__main__":
    main()
