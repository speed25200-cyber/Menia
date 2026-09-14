"""Paired simulation of binding-state and report interventions in reaching."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from research.binding_control import (BindingControlAgent, BindingController,
                                     Observation, ordinary_conditional_mean)
from research.causal_binding import logistic, logit

POLICIES = ("calibrated", "fixed_prior", "always_common", "always_displaced",
            "anchor_only", "linear_regression", "ordinary_regression",
            "report_shift", "report_removed", "prior_compensated", "state_graft")


def simulate_batch(prior, visual_sigma, seed, trials=512):
    rng = np.random.default_rng(seed)
    common = rng.random(trials) < prior
    positions = rng.normal(size=trials)
    offset = rng.normal(0., 4., size=trials)
    visual = positions+np.where(common, 0., offset)+rng.normal(0., visual_sigma, trials)
    controller = BindingController(prior, visual_sigma=visual_sigma)
    shifted = BindingController(logistic(logit(prior)+math.log(4)),
                                visual_sigma=visual_sigma)
    observations = [Observation(0., float(y), 3.) for y in visual]
    original = [controller.decide(o) for o in observations]
    q = np.array([d.used_binding for d in original])
    donor = np.roll(q, -1)
    if not np.array_equal(np.sort(q), np.sort(donor)):
        raise AssertionError("Graft must preserve the marginal binding values")
    decisions = {
        "calibrated": original,
        "fixed_prior": [controller.decide(o, binding_override=prior) for o in observations],
        "always_common": [controller.decide(o, binding_override=1.) for o in observations],
        "always_displaced": [controller.decide(o, binding_override=0.) for o in observations],
        "report_shift": [controller.decide(o, report_threshold=math.log(4)) for o in observations],
        "report_removed": [controller.decide(o, report_enabled=False) for o in observations],
        "prior_compensated": [shifted.decide(o, report_threshold=math.log(4)) for o in observations],
        "state_graft": [controller.decide(o, binding_override=float(d))
                        for o, d in zip(observations, donor)],
    }
    actions = {name: np.array([d.displacement for d in ds]) for name, ds in decisions.items()}
    actions["anchor_only"] = np.full(trials, 3.)
    actions["linear_regression"] = 3.-visual/(1+visual_sigma**2+(1-prior)*16)
    actions["ordinary_regression"] = np.array([
        3.-ordinary_conditional_mean(o, prior, 1., visual_sigma, 4.) for o in observations])
    baseline = actions["calibrated"]
    reports = {name: np.array([d.report for d in ds]) for name, ds in decisions.items()}
    for name in ("report_shift", "report_removed"):
        if not np.array_equal(actions[name], baseline):
            raise AssertionError("The report path changed action")
    if not np.array_equal(reports["calibrated"], reports["prior_compensated"]):
        raise AssertionError("Prior/threshold compensation changed reports")
    ordinary_error = float(np.max(np.abs(actions["ordinary_regression"]-baseline)))
    if ordinary_error > 1e-12:
        raise AssertionError("Ordinary regression no longer matches the Bayesian action")
    # Hidden x is used here, after all decisions, solely by the world/evaluator.
    costs = {name: (positions+action-3.)**2 for name, action in actions.items()}
    metrics = {}
    for name in POLICIES:
        ds = decisions.get(name)
        metrics[name] = {
            "mse": float(np.mean(costs[name])),
            "cost_difference": float(np.mean(costs[name]-costs["calibrated"])),
            "action_change_rate": float(np.mean(np.abs(actions[name]-baseline) > 1e-12)),
            "action_rms_difference": float(np.sqrt(np.mean((actions[name]-baseline)**2))),
            "report_disagreement": (float(np.mean(reports[name] != reports["calibrated"]))
                                    if ds is not None and name != "report_removed" else None),
            "predicted_error_variance": (float(np.mean([d.predicted_error_variance for d in ds]))
                                         if ds is not None else None),
        }
    return metrics, ordinary_error


def paired_summary(values, indices):
    values = np.asarray(values, dtype=float)
    interval = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return {"mean": float(values.mean()), "bootstrap_95": interval.tolist()}


def run_audit():
    conditions = []
    maximum_equivalence_error = 0.
    for index, (prior, visual_sigma) in enumerate(
            (p, s) for p in (.1, .5, .9) for s in (.25, 1., 2.)):
        batches = []
        for repeat in range(32):
            metrics, error = simulate_batch(prior, visual_sigma, 716000+100*index+repeat)
            batches.append(metrics)
            maximum_equivalence_error = max(maximum_equivalence_error, error)
        indices = np.random.default_rng(717000+index).integers(0, 32, (5000, 32))
        summary = {}
        for name in POLICIES:
            entry = {}
            for metric in batches[0][name]:
                values = [b[name][metric] for b in batches]
                entry[metric] = None if values[0] is None else paired_summary(values, indices)
            entry["mse_by_batch"] = [b[name]["mse"] for b in batches]
            # Paired trial differences, retained without subtracting rounded means.
            entry["cost_difference_by_batch"] = [b[name]["cost_difference"] for b in batches]
            summary[name] = entry
        conditions.append({"index": index, "prior": prior, "visual_sigma": visual_sigma,
                           "first_seed": 716000+100*index, "policies": summary})
    agent = BindingControlAgent(BindingController(.5))
    observation = Observation(0., 1.3, 3.)
    # Example world state is private to the actuator, never an observation field.
    x = .7
    agent.step(observation, lambda action: {"position": x+action,
                                            "squared_error": (x+action-3)**2})
    return {
        "schema": "binding-control-v1",
        "protocol": "docs/BINDING_CONTROL_PROTOCOL.md",
        "scope": "Independent Gaussian trials; known parameters and dynamics; no consciousness measure",
        "settings": {"proprio_sigma": 1., "source_sigma": 4., "anchor": 0., "target": 3.,
                     "conditions": 9, "batches_per_condition": 32, "trials_per_batch": 512,
                     "unique_world_trials": 147456, "policies": len(POLICIES),
                     "paired_policy_evaluations": 147456*len(POLICIES), "bootstrap_samples": 5000},
        "invariants": {"report_only_action_changes": 0,
                       "compensated_prior_report_changes": 0,
                       "ordinary_regression_max_action_difference": maximum_equivalence_error,
                       "graft_preserves_marginal_binding": True},
        "conditions": conditions,
        "example_trace": agent.events,
    }


def assert_report_equal(actual, expected, path="report"):
    """Compare all fields; small numerical tolerance allows platform libm drift."""
    if type(actual) is not type(expected):
        raise AssertionError(f"Type mismatch at {path}")
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"Keys changed at {path}")
        for key in actual:
            assert_report_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise AssertionError(f"Length changed at {path}")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_report_equal(a, e, f"{path}[{i}]")
    elif isinstance(actual, float):
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11):
            raise AssertionError(f"Numeric mismatch at {path}: {actual} vs {expected}")
    elif actual != expected:
        raise AssertionError(f"Value mismatch at {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/binding-control/report.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = run_audit()
    if args.check:
        assert_report_equal(report, json.loads(args.output.read_text(encoding="utf-8")))
        print("Binding control: full report reproduced (absolute/relative tolerance 1e-11).")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))
        print(args.output)
    for condition in report["conditions"]:
        policies = condition["policies"]
        print(f"p={condition['prior']}, sigma_v={condition['visual_sigma']}: "
              f"MSE={policies['calibrated']['mse']['mean']:.6f}; "
              f"fixed-prior excess={policies['fixed_prior']['cost_difference']['mean']:.6f}; "
              f"graft excess={policies['state_graft']['cost_difference']['mean']:.6f}")


if __name__ == "__main__":
    main()
