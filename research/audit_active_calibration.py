"""Reproducible paired audit of learning actuator and sensor properties."""
import argparse
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

import numpy as np

from research.active_calibration import AdaptiveBindingAgent, SensorReading
from research.binding_control import Observation


@dataclass(frozen=True)
class Regime:
    gain: float = 1.
    reference_scale: float = 1.
    visual_scale: float = 1.
    visual_bias: float = 0.
    visual_noise: float = .5
    quadratic: float = 0.

    def calibration(self, command, noise):
        position = self.gain*command
        return SensorReading(self.reference_scale*position+.15*noise[0],
                             self.visual_value(position)+self.visual_noise*noise[1])

    def visual_value(self, position):
        return self.visual_scale*position+self.visual_bias+self.quadratic*position**2


CONDITIONS = {
    "unchanged": Regime(), "motor_reversal": Regime(gain=-1.),
    "motor_loss": Regime(gain=.5), "visual_bias": Regime(visual_bias=1.5),
    "visual_scale": Regime(visual_scale=.5), "visual_noise": Regime(visual_noise=2.),
    "combined": Regime(gain=-.5, visual_bias=1., visual_noise=1.),
    "misleading_reference": Regime(reference_scale=.5, visual_scale=.5),
    "nonlinear_vision": Regime(quadratic=.15),
}
POLICIES = ("adaptive", "frozen", "passive", "ordinary")
PHASES = {"initial": slice(0, 32), "transition": slice(32, 44), "final": slice(72, 96)}


def run_episode(regime, seed, policy, keep_trace=False):
    rng = np.random.default_rng(seed)
    calibration_noise = rng.normal(size=(96, 2))
    positions = rng.normal(size=(96, 8))
    offsets = np.where(rng.random((96, 8)) < .5, 0., rng.normal(0., 4., (96, 8)))
    visual_noise = rng.normal(size=(96, 8))
    agent = AdaptiveBindingAgent(method="ols" if policy == "ordinary" else "bayes",
                                 record=keep_trace)
    history = []
    for tick in range(96):
        world = Regime() if tick < 32 else regime
        action = (None if policy == "adaptive" else
                  0. if policy == "passive" else (-1. if tick % 2 == 0 else 1.))
        result = agent.probe(lambda u: world.calibration(u, calibration_noise[tick]),
                             action=action, learn=policy != "frozen" or tick < 32)
        state = result['parameters']
        costs, saturated = [], []
        for j in range(8):
            x = positions[tick, j]
            y = world.visual_value(x+offsets[tick, j])+world.visual_noise*visual_noise[tick, j]
            decision = agent.reach(Observation(0., float(y), 1.5), parameters=state)
            # The true state and true cost remain exclusively with the evaluator.
            costs.append(float((x+world.gain*decision['command']-1.5)**2))
            saturated.append(decision['saturated'])
        history.append({"cost": float(np.mean(costs)), "probe_effort": result['action']**2,
                        "probe": result['action'], "visual_nll": result['nll']['visual'],
                        "reference_nll": result['nll']['reference'],
                        "saturation_rate": float(np.mean(saturated)), **state})
    summary = {phase: {metric: float(np.mean([row[metric] for row in history[interval]]))
                       for metric in ("cost", "probe_effort", "visual_nll", "reference_nll",
                                      "gain", "gain_variance", "visual_bias", "visual_noise",
                                      "saturation_rate", "identified_given_reference_assumption")}
               for phase, interval in PHASES.items()}
    summary['final_parameters'] = history[-1]
    return {"phases": summary, "history": history,
            "trace": agent.events[30*3:35*3] if keep_trace else []}


def ambiguity_audit():
    motor = Regime(gain=.5)
    sensors = Regime(reference_scale=.5, visual_scale=.5)
    a, b = AdaptiveBindingAgent(record=False), AdaptiveBindingAgent(record=False)
    rng = np.random.default_rng(719999)
    max_observed_difference = 0.
    max_parameter_difference = 0.
    physical_difference = 0.
    for _ in range(48):
        noise = rng.normal(size=2)
        action_a, action_b = a.visual.choose_probe(), b.visual.choose_probe()
        if action_a != action_b:
            raise AssertionError("Observationally equivalent calibrators chose different probes")
        first, second = motor.calibration(action_a, noise), sensors.calibration(action_b, noise)
        max_observed_difference = max(max_observed_difference,
                                     abs(first.reference-second.reference), abs(first.visual-second.visual))
        a.probe(lambda u: first, action=action_a)
        b.probe(lambda u: second, action=action_b)
        physical_difference = max(physical_difference, abs(motor.gain*action_a-sensors.gain*action_b))
    for key in ('gain', 'gain_variance', 'visual_slope', 'visual_bias', 'visual_noise'):
        max_parameter_difference = max(max_parameter_difference, abs(a.parameters()[key]-b.parameters()[key]))
    absent = AdaptiveBindingAgent(record=False)
    for i in range(48):
        absent.probe(lambda u: SensorReading(None, .5*u), action=-1. if i % 2 == 0 else 1.)
    nonlinear, biased = Regime(quadratic=.15), Regime(visual_bias=.15)
    support = {str(u): abs(nonlinear.calibration(u, (0., 0.)).visual
                           -biased.calibration(u, (0., 0.)).visual)
               for u in (-1., 0., 1.)}
    designer = AdaptiveBindingAgent(record=False)
    sequence = []
    for _ in range(96):
        result = designer.probe(lambda u: SensorReading(u, u))
        sequence.append(result['action'])
    return {"scope": "Reset-to-zero calibration observations only",
            "max_observation_difference": max_observed_difference,
            "max_learned_parameter_difference": max_parameter_difference,
            "max_true_displacement_difference": physical_difference,
            "motor_world_parameters": a.parameters(), "sensor_world_parameters": b.parameters(),
            "reference_absent_parameters": absent.parameters(),
            "nonlinear_vs_bias_mean_difference_by_probe": support,
            "geometric_probe_sequence": sequence,
            "geometric_equals_alternating": sequence == [-1., 1.]*48}


def run_audit():
    conditions = []
    example_trace = None
    for index, (name, regime) in enumerate(CONDITIONS.items()):
        episodes = {policy: [run_episode(regime, 718000+100*index+repeat, policy,
                                        keep_trace=(index == 1 and repeat == 0 and policy == 'adaptive'))
                             for repeat in range(32)] for policy in POLICIES}
        if index == 1:
            example_trace = episodes['adaptive'][0]['trace']
        baseline = np.array([e['phases']['final']['cost'] for e in episodes['adaptive']])
        boot = np.random.default_rng(719000+index).integers(0, 32, (4000, 32))
        summaries = {}
        for policy, records in episodes.items():
            costs = np.array([e['phases']['final']['cost'] for e in records])
            delta = costs-baseline
            phases = {phase: {metric: float(np.mean([e['phases'][phase][metric] for e in records]))
                              for metric in records[0]['phases'][phase]} for phase in PHASES}
            curve = {metric: np.mean([[row[metric] for row in e['history']] for e in records], axis=0).tolist()
                     for metric in ('cost', 'gain', 'gain_variance', 'visual_bias', 'visual_noise',
                                    'visual_nll', 'reference_nll')}
            parameters = {}
            for metric in ('gain', 'gain_variance', 'visual_scale', 'visual_bias', 'visual_noise',
                           'identified_given_reference_assumption'):
                values = [e['phases']['final_parameters'][metric] for e in records]
                valid = [v for v in values if v is not None]
                parameters[metric] = {"mean": float(np.mean(valid)) if valid else None,
                                      "valid_episodes": len(valid)}
            summaries[policy] = {"phases": phases, "final_parameters": parameters,
                                 "final_cost_by_episode": costs.tolist(),
                                 "final_minus_adaptive": {"mean": float(delta.mean()),
                                     "bootstrap_95": np.quantile(delta[boot].mean(axis=1), [.025, .975]).tolist()},
                                 "mean_curves": curve}
        conditions.append({"name": name, "world": asdict(regime), "first_seed": 718000+100*index,
                           "policies": summaries})
    return {"schema": "active-calibration-v1", "protocol": "docs/ACTIVE_CALIBRATION_PROTOCOL.md",
            "settings": {"conditions": 9, "episodes_per_condition": 32, "policies": 4,
                         "cycles_per_episode": 96, "reaches_per_cycle": 8,
                         "policy_episodes": 1152, "calibration_commands": 110592,
                         "reach_evaluations": 884736, "bootstrap_resamples": 4000,
                         "window": 24, "change_cycle": 32},
            "conditions": conditions, "ambiguity": ambiguity_audit(), "example_trace": example_trace}


def compare(actual, expected, path="report"):
    if type(actual) is not type(expected):
        raise AssertionError(f"Type mismatch at {path}")
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"Keys changed at {path}")
        for key in actual:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise AssertionError(f"Length changed at {path}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{path}[{i}]")
    elif isinstance(actual, float):
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise AssertionError(f"Numeric difference at {path}: {actual} vs {expected}")
    elif actual != expected:
        raise AssertionError(f"Changed value at {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/active-calibration/report.json'))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = run_audit()
    if args.check:
        compare(report, json.loads(args.output.read_text(encoding='utf-8')))
        print('Active calibration: complete report reproduced (tolerance 1e-10).')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8'))
        print(args.output)
    for c in report['conditions']:
        print(c['name'], {p: round(c['policies'][p]['phases']['final']['cost'], 6) for p in POLICIES})


if __name__ == '__main__':
    main()
