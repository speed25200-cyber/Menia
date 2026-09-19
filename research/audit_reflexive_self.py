"""Exact scope checks for communication and commitment necessity arguments.

Independent small cases; no external code execution, Monte-Carlo reproduction,
agent integration, or consciousness classification. See the protocol.
"""

import argparse
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path


def success(probe_flip, final_flip, change="fixed"):
    total = F(0)
    for message, decoder, probe_noise, final_noise in product(range(2), repeat=4):
        weight = F(1, 4)
        weight *= probe_flip if probe_noise else 1 - probe_flip
        weight *= final_flip if final_noise else 1 - final_flip
        response = decoder ^ probe_noise  # Probe sent was zero.
        signal = message ^ response
        targets = [(decoder, F(1))]
        if change == "inverted":
            targets = [(1-decoder, F(1))]
        elif change == "resampled":
            targets = [(0, F(1, 2)), (1, F(1, 2))]
        elif change != "fixed":
            raise ValueError(change)
        for final_decoder, chance in targets:
            if signal ^ final_decoder ^ final_noise == message:
                total += weight * chance
    return total


def run_audit():
    truth = []
    for message, decoder in product(range(2), repeat=2):
        response = decoder
        signal = message ^ response
        delivered = signal ^ decoder
        assert delivered == message
        truth.append({"message": message, "decoder": decoder, "probe_response": response,
                      "signal": signal, "delivered": delivered})
    passive = []
    for policy in product(range(2), repeat=2):
        score = sum((policy[m] ^ t) == m for m, t in product(range(2), repeat=2))
        assert F(score, 4) == F(1, 2)
        passive.append({"signals_for_messages_0_1": list(policy), "success": "1/2"})
    noise = []
    for a, b in product((F(0), F(1, 10), F(1, 4), F(1, 2)), (F(0), F(1, 10), F(1, 2))):
        exact = success(a, b)
        formula = 1 - a - b + 2*a*b
        assert exact == formula
        noise.append({"probe_flip": str(a), "final_flip": str(b), "success": str(exact)})
    assert success(F(0), F(0), "inverted") == 0
    assert success(F(0), F(0), "resampled") == F(1, 2)

    games = []
    for cost, temptation in product((F(0), F(1, 4), F(3, 4), F(1), F(5, 4)),
                                    (F(1, 10), F(1), F(2))):
        bind = cost < 1
        chosen_payoff = 1-cost if bind else F(0)
        alternate_payoff = F(0) if bind else 1-cost
        # Check each information set, including paths not reached in equilibrium.
        deviations = {
            "initial_binding_choice": alternate_payoff-chosen_payoff,
            "trustor_after_binding": F(0)-F(1),
            "trustor_without_binding": F(-1)-F(0),
            "trustee_honor_instead_of_exploit_when_unbound": -temptation,
        }
        assert max(deviations.values()) <= 0
        games.append({"cost": str(cost), "temptation": str(temptation), "bind": bind,
                      "outcome": [str(chosen_payoff), "1" if bind else "0"],
                      "deviation_gains": {k: str(v) for k, v in deviations.items()}})
    return {
        "schema": "reflexive-self-scope-v1",
        "protocol": "docs/REFLEXIVE_SELF_PROTOCOL.md",
        "source_doi": "10.1609/aaaiss.v8i1.42546",
        "source_code_commit": "259e0ec398dbbd3b5b0a0312a0ea8de305d4bb2c",
        "source_code_sha256": "f652f3bc484311bce51a974420811aa29fe6ba68a3eb327a17fe61380cad1ca4",
        "external_code_executed": False,
        "perfect_probe_truth_table": truth,
        "passive_policies": passive,
        "independent_bit_flip_cases": noise,
        "decoder_change_controls": {"inverted_success": "0", "resampled_success": "1/2"},
        "commitment_cases": games,
        "scope": [
            "Response information is necessary in the specified communication game",
            "One stored bit and XOR implement the perfect-probe strategy",
            "A direct strategy profile implements the specified trust equilibrium",
            "A functional definition may call these implementations selves",
            "No recursively explicit architecture or phenomenal state is identified by these outcomes",
            "The published Monte-Carlo benchmarks were not reproduced",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reflexive-self-audit/report.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = run_audit()
    if args.check:
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        if report != expected:
            raise AssertionError("The exact report changed")
        print("Reflexive self audit: exact report reproduced.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2)+"\n").encode("utf-8"))
        print(args.output)
    print("Perfect probe 1; passive 1/2; 12 noise cases; 15 equilibrium cases.")
    print("Changed decoder controls: inverted 0; resampled 1/2. No consciousness conclusion.")


if __name__ == "__main__":
    main()
