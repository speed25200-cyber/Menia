"""Paired factorial evaluation of a simulated sensor-maintenance controller."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np

from research.capacity_planning import ACTIONS, Model, CompiledPlanner, CapacityAgent, recursive_costs


N_EPISODES = 128
N_STEPS = 64
FIRST_SEED = 20260915
POLICIES = [(False, 1), (True, 1), (False, 4), (True, 4)]


def conditions():
    base = Model()
    return {
        'nominal': (base, base),
        'slow_wear': (replace(base, wear=.02), replace(base, wear=.02)),
        'fast_wear': (replace(base, wear=.4), replace(base, wear=.4)),
        'uninformative_diagnostic': (replace(base, diagnostic_accuracy=.5), base),
        'reversed_diagnostic': (replace(base, diagnostic_accuracy=.15), base),
        'known_ineffective_maintenance': (replace(base, effective_maintenance=False),
                                        replace(base, effective_maintenance=False)),
        'hidden_ineffective_maintenance': (replace(base, effective_maintenance=False), base),
        'low_sensor_contrast': (replace(base, good_accuracy=.65), replace(base, good_accuracy=.65)),
        'expensive_maintenance': (replace(base, maintenance_cost=1.2), replace(base, maintenance_cost=1.2)),
    }


def episode(world, planner, seed, trace=False):
    rng = np.random.default_rng(seed)
    good = int(rng.random() < .5)
    draws = rng.random((N_STEPS, 4))
    agent = CapacityAgent(planner)
    totals = np.zeros(7)  # loss, good state, belief Brier, answer/abstain/maintain, online dot products
    records = []
    transitions = [world.transition(a) for a in range(3)]
    for tick, (target_draw, sensor_draw, diagnostic_draw, transition_draw) in enumerate(draws):
        target = int(target_draw < .5)
        sensor_accuracy = world.good_accuracy if good else world.bad_accuracy
        reading = target if sensor_draw < sensor_accuracy else 1-target
        diagnostic = good if diagnostic_draw < world.diagnostic_accuracy else 1-good
        prior = agent.belief
        agent.observe(diagnostic, reading)
        action, prediction = agent.decide(N_STEPS-tick)
        # Outcomes are evaluated only after observe -> predict -> choose. They
        # never enter the agent, including on the next tick.
        loss = (float(prediction['response'] != target) if action == 0 else
                world.abstain_cost if action == 1 else world.maintenance_cost)
        next_good = int(transition_draw < transitions[action][good, 1])
        totals[0] += loss
        totals[1] += good
        totals[2] += (prediction['belief_good']-good)**2
        totals[3+action] += 1
        totals[6] += prediction['online_dot_products']
        if trace:
            records.append({'tick': tick, 'observed': {'diagnostic': diagnostic, 'reading': reading},
                            'prior_good': prior, 'prediction_before_outcome': prediction,
                            'evaluator_only_after_decision': {'target': target, 'good': good,
                                                              'next_good': next_good, 'loss': loss}})
        good = next_good
    return totals/N_STEPS, records


def cross_check(model):
    max_error, comparisons, disagreements, tied_differences = 0., 0, 0, 0
    recursive_nodes = {False: {}, True: {}}
    for use in (False, True):
        planner = CompiledPlanner(model, 4, use)
        for depth in range(1, 5):
            counts = []
            for q in np.linspace(0, 1, 1001):
                expected, nodes = recursive_costs(model, q, depth, use)
                choice, actual, _ = planner.evaluate(q, depth)
                max_error = max(max_error, float(np.max(np.abs(expected-actual))))
                reference_choice = int(np.flatnonzero(expected <= np.min(expected)+1e-12)[0])
                if choice != reference_choice:
                    if abs(expected[choice]-min(expected)) <= 1e-10:
                        tied_differences += 1
                    else:
                        disagreements += 1
                counts.append(nodes)
                comparisons += 1
            recursive_nodes[use][depth] = float(np.mean(counts))
    assert max_error < 1e-10 and disagreements == 0
    return {'belief_horizon_factor_comparisons': comparisons,
            'action_values_compared': 3*comparisons, 'max_absolute_value_error': max_error,
            'choice_disagreements_outside_numerical_ties': disagreements,
            'choice_differences_at_numerical_ties': tied_differences,
            'mean_recursive_value_cache_misses_on_grid': recursive_nodes}


def run_audit():
    bootstrap = np.random.default_rng(20260916).integers(0, N_EPISODES, (10000, N_EPISODES))
    results, validations, all_traces = [], {}, {}
    for name, (world, model) in conditions().items():
        # Several misspecified worlds share an agent model; validate each unique
        # model once and keep the mapping explicit.
        key = json.dumps(asdict(model), sort_keys=True)
        if key not in validations:
            validations[key] = {'model': asdict(model), 'result': cross_check(model)}
        per_policy, episode_costs, episode_operations = [], [], []
        for use, horizon in POLICIES:
            planner = CompiledPlanner(model, horizon, use)
            stats = []
            for i in range(N_EPISODES):
                values, trace = episode(world, planner, FIRST_SEED+i, trace=(name == 'nominal' and i == 0))
                stats.append(values)
                if trace:
                    all_traces[f'diagnostic_{use}_horizon_{horizon}'] = trace
            stats = np.stack(stats)
            mean = np.mean(stats, axis=0)
            samples = np.mean(stats[bootstrap, 0], axis=1)
            per_policy.append({'use_diagnostic': use, 'horizon': horizon,
                               'loss_per_step': float(mean[0]),
                               'loss_per_step_ci95': np.quantile(samples, [.025, .975]).tolist(),
                               'good_state_fraction': float(mean[1]), 'belief_brier': float(mean[2]),
                               'action_fractions': dict(zip(ACTIONS, mean[3:6].tolist())),
                               'online_dot_products_per_step': float(mean[6]),
                               'compilation': planner.compilation})
            episode_costs.append(stats[:, 0])
            episode_operations.append(stats[:, 6])
        costs = np.stack(episode_costs)
        operations = np.stack(episode_operations)
        contrasts = {
            'diagnostic_gain_h1': costs[0]-costs[1],
            'diagnostic_gain_h4': costs[2]-costs[3],
            'horizon_gain_without_diagnostic': costs[0]-costs[2],
            'horizon_gain_with_diagnostic': costs[1]-costs[3],
            'interaction_gain': (costs[2]-costs[3])-(costs[0]-costs[1]),
        }
        measured = {}
        for label, contrast in contrasts.items():
            measured[label] = {'mean': float(np.mean(contrast)),
                               'ci95': np.quantile(np.mean(contrast[bootstrap], axis=1), [.025, .975]).tolist()}
        sensitivity = []
        for penalty in (0., .0001, .001, .01):
            adjusted = np.mean(costs+penalty*operations, axis=1)
            sensitivity.append({'penalty_per_online_dot_product': penalty,
                                'mean_adjusted_costs_in_policy_order': adjusted.tolist(),
                                'lowest_mean_policy_index': int(np.argmin(adjusted))})
        if not model.effective_maintenance:
            assert all(p['action_fractions']['maintain'] == 0 for p in per_policy)
        assert all(p['action_fractions']['maintain'] == 0 for p in per_policy if p['horizon'] == 1)
        results.append({'condition': name, 'world': asdict(world), 'agent_model': asdict(model),
                        'policies': per_policy, 'paired_contrasts': measured,
                        'online_compute_sensitivity_excluding_compilation': sensitivity})
    return {'schema': 'capacity-planning-v1', 'protocol': 'docs/CAPACITY_PLANNING_PROTOCOL.md',
            'parameters': {'episodes_per_policy_condition': N_EPISODES, 'steps_per_episode': N_STEPS,
                           'first_seed': FIRST_SEED, 'bootstrap_seed': 20260916, 'bootstrap_samples': 10000,
                           'policy_order': [{'use_diagnostic': use, 'horizon': h} for use, h in POLICIES]},
            'decisions_evaluated': len(results)*len(POLICIES)*N_EPISODES*N_STEPS,
            'independent_planner_checks': list(validations.values()), 'results': results,
            'nominal_first_episode_traces': all_traces,
            'scope': 'Functional state inference and maintenance only; no parameter learning, subjective experience, or novelty established.'}


def compare(actual, expected):
    if isinstance(actual, dict):
        # JSON serialises boolean and integer keys; normalise both sides.
        actual = json.loads(json.dumps(actual))
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
    parser.add_argument('--output', type=Path, default=Path('artifacts/capacity-planning/report.json'))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    report = run_audit()
    if args.check:
        compare(report, json.loads(args.output.read_text(encoding='utf-8')))
        print('Capacity planning report reproduced (absolute tolerance 1e-10).')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2)+'\n').encode('utf-8'))
    for result in report['results']:
        print(result['condition'], [round(p['loss_per_step'], 6) for p in result['policies']],
              'interaction', result['paired_contrasts']['interaction_gain'])
    print('No subjective experience or novelty established.')


if __name__ == '__main__':
    main()
