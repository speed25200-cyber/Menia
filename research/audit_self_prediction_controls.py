"""Analytic counterexamples for self/cross-prediction metrics, NOT LLM evidence.

Enumerate binary cases exactly. No training, model inference or device export.
Show why both a diagonal advantage and an input-matched observer are needed,
and why even both together do not identify second-order monitoring.
"""
import argparse
import itertools
import json
from pathlib import Path

MODELS = ("A", "B")


def mean(values):
    return sum(values)/len(values)


def summarize(rows):
    if not rows:
        raise ValueError("Empty matrix")
    losses = {predictor: {} for predictor in MODELS}
    observer = {}
    for target in MODELS:
        for predictor in MODELS:
            losses[predictor][target] = mean([
                (r["predictions"][predictor][target]-r["outcomes"][target])**2 for r in rows])
        observer[target] = mean([(r["observer"][target]-r["outcomes"][target])**2 for r in rows])
    self_gain = {t: losses[MODELS[1-MODELS.index(t)]][t]-losses[t][t] for t in MODELS}
    observer_gain = {t: observer[t]-losses[t][t] for t in MODELS}
    return dict(items=len(rows), brierByPredictorAndTarget=losses, inputObserverBrier=observer,
                otherMinusSelfByTarget=self_gain,
                diagonalAdvantage=mean(list(self_gain.values())),
                inputObserverMinusSelfByTarget=observer_gain,
                bothTargetsBeatOther=all(v > 1e-12 for v in self_gain.values()),
                bothTargetsBeatInputObserver=all(v > 1e-12 for v in observer_gain.values()))


def confident(bit):
    return .9 if bit else .1


def visible_rule_case():
    rows = []
    # Public input gives x and the target rules. Each predictor ignores the target
    # identifier and applies its own fixed rule to every prediction request.
    for x in (0, 1):
        public = dict(x=x, ruleA="x", ruleB="1-x")
        visible = dict(A=public["x"], B=1-public["x"])
        forecasts = {p: {t: confident(visible[p]) for t in MODELS} for p in MODELS}
        observer = {t: confident(visible[t]) for t in MODELS}
        rows.append(dict(outcomes=visible, predictions=forecasts, observer=observer))
    return dict(name="visible_rule_similarity", origin="exact synthetic enumeration, public rules only",
                secondOrderMechanism="none: fixed input rule", **summarize(rows),
                interpretation="A diagonal advantage occurs without privileged information; a matched input observer ties the diagonal.")


def stronger_predictor_case():
    rows = []
    for x in (0, 1):
        outcomes = dict(A=x, B=1-x)
        forecasts = dict(A={t: confident(outcomes[t]) for t in MODELS}, B={t:.5 for t in MODELS})
        rows.append(dict(outcomes=outcomes, predictions=forecasts,
                         observer={t:confident(outcomes[t]) for t in MODELS}))
    return dict(name="general_predictor_strength", origin="exact synthetic enumeration, public rules only",
                secondOrderMechanism="none: A is a better external predictor for both targets", **summarize(rows),
                interpretation="A-only evaluation has a self advantage; the reversed comparison and full matrix reveal general predictor strength.")


def direct_signal_case():
    rows = []
    # The visible prompt is identical in all four cases. The owner has a private
    # bit. The report and behavior directly read that bit through two fixed
    # output mappings; there is no learned monitor or representation of knowing.
    for a, b in itertools.product((0, 1), repeat=2):
        private = dict(A=a, B=b)
        forecasts = {p: {t: confident(private[p]) if p == t else .5 for t in MODELS} for p in MODELS}
        rows.append(dict(outcomes=private, predictions=forecasts, observer={t:.5 for t in MODELS}))
    return dict(name="direct_private_signal", origin="exact synthetic enumeration, direct private bit readout",
                secondOrderMechanism="not modeled: parallel first-order readouts of the same bit", **summarize(rows),
                interpretation="Both behavioral comparisons favor self, yet the construction uses a direct signal readout. These scores alone do not identify second-order monitoring.")


def report():
    return dict(schema="menia-self-prediction-controls-v1", llmExecuted=False,
                origin="Analytic counterexamples; not Menia measurements or a new empirical discovery",
                cases=[visible_rule_case(), stronger_predictor_case(), direct_signal_case()],
                conclusion="Behavioral screens constrain explanations; none is a consciousness classifier.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/self-prediction-controls/report.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = report()
    if args.check:
        if json.loads(args.output.read_text(encoding="utf-8")) != result:
            raise SystemExit("Counterexample artifact mismatch")
        print("Exact synthetic counterexamples replayed; no LLM executed.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
