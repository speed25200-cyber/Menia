"""Exact causal audit of a proposed experience bridge; no consciousness test.

Two hand-specified controllers agree in ordinary operation. Transplanting a
source-belief state distinguishes integrated control from report-only control.
An unobserved experience label with no additional measurement assumptions is
not identified by these outcomes. No model is labelled actually unconscious.
"""
from fractions import Fraction as F
from itertools import product
import argparse
import json
from pathlib import Path


BELIEFS = (F(1, 10), F(4, 10), F(6, 10), F(9, 10))
COSTS = (F(15, 100), F(35, 100))


def controller(kind, content, belief, cost, transplanted=None):
    """Content is held fixed. The belief concerns external source, not accuracy.

    A verification reveals the source perfectly at cost `cost`; accepting an
    internally generated item incurs unit loss. Thus verify iff 1 - q > cost.
    This utility and the memory threshold are design assumptions, not findings
    about human behaviour. All beliefs and intervention states are on-support.
    """
    if kind not in ("integrated", "report_only"):
        raise ValueError(kind)
    if content not in (0, 1) or belief not in BELIEFS or cost not in COSTS:
        raise ValueError("Outside the declared finite audit domain")
    if transplanted is not None and transplanted not in BELIEFS:
        raise ValueError("Intervention must use a declared donor belief")
    state = belief if transplanted is None else transplanted
    control_belief = state if kind == "integrated" else belief
    return {
        "content_choice": content,
        "source_report": state,
        "verify": 1 - control_belief > cost,
        "store_as_external": control_belief >= F(1, 2),
    }


def serial(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serial(v) for v in value]
    return value


def audit():
    observational = []
    interventions = []
    for content, belief, cost in product((0, 1), BELIEFS, COSTS):
        left = controller("integrated", content, belief, cost)
        right = controller("report_only", content, belief, cost)
        observational.append({"content": content, "belief": belief,
                              "cost": cost, "equal": left == right})
        for donor in BELIEFS:
            left_do = controller("integrated", content, belief, cost, donor)
            right_do = controller("report_only", content, belief, cost, donor)
            interventions.append({
                "content": content, "belief": belief, "cost": cost,
                "donor": donor, "integrated": left_do, "report_only": right_do,
                "action_distinguishes": left_do["verify"] != right_do["verify"],
                "memory_distinguishes": left_do["store_as_external"] != right_do["store_as_external"],
                "report_distinguishes": left_do["source_report"] != right_do["source_report"],
            })

    # Independent hand-calculated boundary cases, including a joint dissociation.
    assert controller("integrated", 1, F(9, 10), F(15, 100)) == {
        "content_choice": 1, "source_report": F(9, 10),
        "verify": False, "store_as_external": True}
    assert controller("integrated", 1, F(9, 10), F(15, 100), F(1, 10)) == {
        "content_choice": 1, "source_report": F(1, 10),
        "verify": True, "store_as_external": False}
    assert controller("report_only", 1, F(9, 10), F(15, 100), F(1, 10)) == {
        "content_choice": 1, "source_report": F(1, 10),
        "verify": False, "store_as_external": True}
    assert all(r["equal"] for r in observational)
    assert all(not r["report_distinguishes"] for r in interventions)
    assert all(r["integrated"]["content_choice"] == r["report_only"]["content_choice"]
               for r in interventions)
    assert all(r["integrated"] == r["report_only"]
               for r in interventions if r["belief"] == r["donor"])

    summary = {
        "observational_conditions": len(observational),
        "observational_equal": sum(r["equal"] for r in observational),
        "intervention_conditions_including_identity": len(interventions),
        "action_distinguishes": sum(r["action_distinguishes"] for r in interventions),
        "memory_distinguishes": sum(r["memory_distinguishes"] for r in interventions),
        "either_nonverbal_distinguishes": sum(r["action_distinguishes"] or r["memory_distinguishes"] for r in interventions),
        "report_distinguishes": sum(r["report_distinguishes"] for r in interventions),
    }
    return serial({
        "schema": 1,
        "scope": "Exact finite structural-model audit, not learned Menia behaviour or human data",
        "summary": summary,
        "observational": observational,
        "interventions": interventions,
        "experience_identification": {
            "identified": False,
            "assumption": "Competing experience assignments leave all observable structural equations unchanged",
            "likelihood_ratio_on_supported_data": "1",
            "limitation": "Conditional non-identifiability, not proof that experience is causally inert or scientifically unknowable",
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
