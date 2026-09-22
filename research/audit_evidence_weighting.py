"""Exact audit of unequal exponentiation of raw model evidence.

This evaluates a literal scoring rule, not a released INTREPID analysis.
No human data, theory probabilities, or consciousness scores are estimated.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path


def normalized(values):
    return [value / sum(values) for value in values]


def audit():
    evidence = (F(4, 5), F(3, 10))
    confidence = (2, 1)
    rows = []
    for common in (F(1), F(1, 10)):
        ordinary = [common * value for value in evidence]
        weighted = [value ** power for value, power in zip(ordinary, confidence, strict=True)]
        rows.append({
            "common_factor": str(common),
            "ordinary_odds_A_B": str(ordinary[0] / ordinary[1]),
            "ordinary_share_A_equal_prior": str(normalized(ordinary)[0]),
            "weighted_odds_A_B": str(weighted[0] / weighted[1]),
            "weighted_share_A": str(normalized(weighted)[0]),
            "weighted_winner": "A" if weighted[0] > weighted[1] else "B",
        })
    # Hand-derived fractions, separate from the computed normalization.
    assert [r["ordinary_share_A_equal_prior"] for r in rows] == ["8/11", "8/11"]
    assert [r["weighted_odds_A_B"] for r in rows] == ["32/15", "16/75"]
    assert [r["weighted_share_A"] for r in rows] == ["32/47", "16/91"]
    assert [r["weighted_winner"] for r in rows] == ["A", "B"]

    checked = 0
    for denominator in range(1, 101):
        common = F(1, denominator)
        after = (common * evidence[0]) ** 2 / (common * evidence[1])
        before = evidence[0] ** 2 / evidence[1]
        assert after / before == common  # c ** (w_A - w_B)
        for common_power in (1, 2, 3):
            assert ((common * evidence[0]) ** common_power /
                    (common * evidence[1]) ** common_power ==
                    (evidence[0] / evidence[1]) ** common_power)
        checked += 1

    # Confidence encoded in proper predictive models preserves neutral factors.
    # Each Beta(a,b) prior gives predictive P(success)=a/(a+b).
    # Both means equal 4/5, with different concentrations (5 versus 50).
    predictive = [F(a, a + b) for a, b in ((4, 1), (40, 10))]
    assert normalized(predictive) == normalized([F(1, 10) * p for p in predictive])
    return {
        "scope": "Literal unequal powers of raw evidence; not an audit of unpublished consortium results",
        "base_evidence_A_B": [str(p) for p in evidence],
        "powers_A_B": list(confidence), "examples": rows,
        "checks": {"common_factors": checked, "equal_power_controls": 3 * checked,
                   "proper_prior_neutral_factor_invariance": True},
        "interpretation": [
            "An independent uniform draw with ten outcomes supplies common factor 1/10.",
            "It supplies no likelihood-ratio evidence between the hypotheses.",
            "Unequal exponents of raw evidence can nevertheless reverse their ranking.",
            "Equal exponents preserve this invariance, but need not be ordinary Bayesian updating.",
            "Proper prior precision and posterior evidence exponentiation are different operations.",
            "No novelty, empirical consciousness finding, or actual INTREPID implementation error is claimed."],
        "source_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = audit()
    if args.check:
        assert result == json.loads(args.output.read_text(encoding="utf-8")), "Stored report differs"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["checks"]))
