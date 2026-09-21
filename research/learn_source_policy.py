"""Logged-outcome learning and closed-loop tests of Menia's verification policy."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
from menia.source_environment import CONDITIONS, streams
from menia.source_monitor import SourceMonitor
from menia.source_policy import SourcePolicy

SEEDS = (11, 23, 37)
COSTS = (.10, .25, .40)
SOURCES = ("menia/source_environment.py", "menia/source_monitor.py", "menia/source_policy.py",
           "menia/source_agent.py", "research/learn_source_policy.py", "docs/LEARNED_VERIFICATION_PROTOCOL.md")
ROOT = Path(__file__).resolve().parents[1]


def exploratory_log(model, frames, truth, actions, costs):
    """The learner receives selected-action costs, never unchosen-action targets."""
    if frames.shape != (*truth.shape, 3) or actions.shape != truth.shape or costs.shape != truth.shape:
        raise ValueError("Exploration arrays must align")
    state, feedback = model.zero(truth.shape[1]), np.zeros((truth.shape[1], 2))
    qs, outcomes = [], []
    for frame, y, action, cost in zip(frames, truth, actions, costs):
        state, q = model.step(np.column_stack((frame, feedback)), state)
        # Outcome is received after the chosen action; only verification is fed back.
        outcome = np.where(action, cost, (q >= .5) != y)
        qs.append(q)
        outcomes.append(outcome)
        feedback = np.column_stack((action, np.where(action, y, 0.)))
    return tuple(np.asarray(v).ravel() for v in (qs, costs, actions, outcomes))


def select(policy, q, cost):
    if isinstance(policy, SourcePolicy):
        return policy.choose(q, cost)
    if policy == "analytic":
        return np.minimum(q, 1-q) > cost
    if policy == "always":
        return np.ones_like(q, dtype=bool)
    if policy == "never":
        return np.zeros_like(q, dtype=bool)
    raise ValueError("Unknown controller")


def matched_donor(q):
    """Cyclic donor within predicted class; singleton classes retain themselves."""
    donor = q.copy()
    for predicted in (False, True):
        indices = np.flatnonzero((q >= .5) == predicted)
        donor[indices] = q[np.roll(indices, 1)]
    return donor


def evaluate(model, policy, frames, truth, cost, *, probe=False):
    batch = truth.shape[1]
    state, feedback = model.zero(batch), np.zeros((batch, 2))
    accumulated = {key: np.zeros(batch) for key in
                   ("loss", "verification_rate", "memory_error_rate", "brier")}
    changes = delta = 0.
    for frame, y in zip(frames, truth):
        state, q = model.step(np.column_stack((frame, feedback)), state)
        verify = select(policy, q, cost)
        first_prediction = q >= .5
        # Hidden truth is used only after choosing, by the environment/scorer.
        error = (~verify) & (first_prediction != y)
        loss = error.astype(float) + cost*verify
        for key, values in (("loss", loss), ("verification_rate", verify),
                            ("memory_error_rate", error), ("brier", (q-y)**2)):
            accumulated[key] += values
        if probe:
            grafted = select(policy, matched_donor(q), cost)
            grafted_loss = ((~grafted) & (first_prediction != y)).astype(float) + cost*grafted
            changes += np.count_nonzero(grafted != verify)
            delta += float((grafted_loss-loss).sum())
        feedback = np.column_stack((verify, np.where(verify, y, 0.)))
    episode_loss = accumulated["loss"] / len(truth)
    metrics = {key: float(values.mean()/len(truth)) for key, values in accumulated.items()}
    metrics["episode_loss_sd"] = float(episode_loss.std(ddof=1))
    if probe:
        metrics.update(probe_changed_rate=changes/truth.size, probe_loss_delta=delta/truth.size)
    return metrics, episode_loss


def evaluation_suite(model, policy, blind):
    rows = []
    for i, condition in enumerate(CONDITIONS):
        frames, truth = streams(81000+1000*i, batch=512, length=48, condition=condition)
        for cost in COSTS:
            outputs = {name: evaluate(model, p, frames, truth, cost, probe=name == "learned")
                       for name, p in (("learned", policy), ("blind", blind),
                                       ("analytic", "analytic"), ("always", "always"), ("never", "never"))}
            paired = {}
            for reference in ("blind", "analytic"):
                diff = outputs["learned"][1] - outputs[reference][1]
                paired[reference] = {"mean_loss_difference": float(diff.mean()),
                                     "episode_difference_sd": float(diff.std(ddof=1))}
            rows.append({"condition": condition, "cost": cost, "seed": 81000+1000*i,
                         "episodes": 512, "steps_per_episode": 48,
                         "controllers": {k: v[0] for k, v in outputs.items()}, "paired": paired})
    return rows


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_text(encoding="utf-8").encode()).hexdigest()
            for name in SOURCES}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n",
                    encoding="utf-8", newline="\n")


def criteria(evaluations):
    central = next(row for row in evaluations if row["condition"] == "standard" and row["cost"] == .25)
    return {"better_than_blind": central["paired"]["blind"]["mean_loss_difference"] < 0,
            "within_001_of_analytic": central["paired"]["analytic"]["mean_loss_difference"] <= .01,
            "graft_changes_choices": central["controllers"]["learned"]["probe_changed_rate"] > 0,
            "graft_increases_immediate_cost": central["controllers"]["learned"]["probe_loss_delta"] > 0}


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    report = {"format": "menia-learned-verification", "version": 1, "status": "running",
              "scope": "functional synthetic pilot; subjective experience and novelty unestablished",
              "versions": {"python": platform.python_version(), "numpy": np.__version__},
              "source_sha256": source_hashes(), "models": []}
    for seed in SEEDS:
        monitor_path = ROOT/f"artifacts/learned-source-monitor/recurrent-{seed}.json"
        model = SourceMonitor.load(monitor_path)
        frames, truth = streams(71000+seed, batch=512, length=48)
        rng = np.random.default_rng(72000+seed)
        actions = rng.random(truth.shape) < .5
        costs = rng.uniform(.05, .45, truth.shape)
        log = exploratory_log(model, frames, truth, actions, costs)
        policies = {"learned": SourcePolicy.fit(*log), "blind": SourcePolicy.fit(*log, blind=True)}
        files = {}
        for name, policy in policies.items():
            path = out/f"{name}-{seed}.json"
            policy.save(path)
            files[name] = {"file": path.name, "sha256": digest(path)}
        training = {"environment_seed": 71000+seed, "exploration_seed": 72000+seed,
                    "episodes": 512, "steps_per_episode": 48, "verified": int(actions.sum()),
                    "accepted": int((~actions).sum()), "parameters_per_policy": 20}
        evaluations = evaluation_suite(model, policies["learned"], policies["blind"])
        central = next(row for row in evaluations if row["condition"] == "standard" and row["cost"] == .25)
        report["models"].append({"monitor_seed": seed, "monitor_sha256": digest(monitor_path),
                                 "policies": files, "training": training, "evaluation": evaluations,
                                 "exploratory_criteria": criteria(evaluations)})
        save_json(out/"report.json", report)
        print(f"monitor {seed}: {json.dumps(central['controllers'])}", flush=True)
    report["status"] = "completed"
    save_json(out/"report.json", report)
    return report


def compare(actual, expected, path="report"):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"Keys differ at {path}")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"Lengths differ at {path}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{path}[{i}]")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=1e-9, atol=1e-12):
            raise AssertionError(f"Numerical mismatch at {path}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"Mismatch at {path}")


def check(out):
    report = json.loads((out/"report.json").read_text(encoding="utf-8"))
    if report["format"] != "menia-learned-verification" or report["version"] != 1 or report["status"] != "completed":
        raise AssertionError("Incomplete or unknown report")
    compare(source_hashes(), report["source_sha256"])
    if [row["monitor_seed"] for row in report["models"]] != list(SEEDS):
        raise AssertionError("Expected all three monitors")
    for row in report["models"]:
        monitor_path = ROOT/f"artifacts/learned-source-monitor/recurrent-{row['monitor_seed']}.json"
        compare(digest(monitor_path), row["monitor_sha256"])
        model = SourceMonitor.load(monitor_path)
        policies = {}
        for name, info in row["policies"].items():
            path = out/info["file"]
            compare(digest(path), info["sha256"])
            policies[name] = SourcePolicy.load(path)
        evaluations = evaluation_suite(model, policies["learned"], policies["blind"])
        compare(evaluations, row["evaluation"])
        compare(criteria(evaluations), row["exploratory_criteria"])
    print("All 225 controller evaluations reproduced from saved policies and monitors.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check(args.out) if args.check else run(args.out)


if __name__ == "__main__":
    main()
