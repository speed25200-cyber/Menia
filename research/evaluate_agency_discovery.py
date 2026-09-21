"""Reproducible exploration benchmark and explicit causal counterexample."""
import argparse
import hashlib
import json
from pathlib import Path
from menia.agency_discovery import ConfoundedChannels, ProbeAgent

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('menia/agency_discovery.py', 'research/evaluate_agency_discovery.py',
           'docs/AGENCY_DISCOVERY_PROTOCOL.md')


def counterexample():
    rows = []
    for cause in ('action', 'cue'):
        observations = []
        for cue in (-1, 1):
            action = cue
            y = action if cause == 'action' else cue
            observations.append({'cue': cue, 'action': action, 'y': y,
                                 'prediction': action, 'ablated_prediction': 0})
        rows.append({'cause': cause, 'observations': observations,
                     'observational_mse': 0.0, 'action_input_ablated_mse': 1.0,
                     'intervention_action_opposes_cue': [
                         {'cue': cue, 'action': -cue, 'y': -cue if cause == 'action' else cue}
                         for cue in (-1, 1)]})
    assert rows[0]['observations'] == rows[1]['observations']
    return {'worlds': rows, 'conclusion': 'Predictor input dependence alone does not identify an action effect.',
            'fair_cue_aware_baseline_mse': 0.0,
            'qualification': 'A baseline that sees the public cue also predicts perfectly; this is not a reproduction of another paper.'}


def trial(seed, condition, policy):
    env = ConfoundedChannels(seed, condition=condition)
    agent = ProbeAgent(seed=seed+100000, policy=policy)
    for tick in range(36):
        decision = agent.choose(env.cue(), warmup=tick < 24)
        agent.receive(env.read(decision['action'], decision['channel']))
    assessment = agent.model.assessment()
    truth = [cause == 'action' for cause in env.causes]
    labels = ['controlled' if value else 'external' for value in truth]
    unique_truth = truth.index(True) if sum(truth) == 1 else None
    unique = assessment['unique_controlled_channel']
    return {'seed': seed, 'condition': condition, 'policy': policy, 'truth': truth,
            **assessment, 'exact_classification': assessment['labels'] == labels,
            'unique_correct': unique is not None and unique == unique_truth,
            'unique_false': unique is not None and unique != unique_truth,
            'abstains_unique': unique is None,
            'brier': sum((p-int(y))**2 for p, y in zip(assessment['control_probabilities'], truth))/2,
            'readings_received': sum(agent.model.received),
            'interventions': sum(row['decision']['action'] != row['decision']['cue'] for row in agent.trace[24:])}, agent.trace


def evaluate():
    rows, examples, summaries = [], {}, []
    for condition in ('single', 'missing', 'masked', 'none', 'twins', 'twins_missing'):
        for policy in ('active', 'random', 'passive'):
            group = []
            for seed in range(1000, 1100):
                row, trace = trial(seed, condition, policy)
                group.append(row)
                if seed == 1000 and policy == 'active':
                    examples[condition] = trace
            rows.extend(group)
            summaries.append({'condition': condition, 'policy': policy, 'n': len(group),
                **{key: sum(row[key] for row in group) for key in
                   ('exact_classification', 'unique_correct', 'unique_false', 'abstains_unique')},
                **{'mean_'+key: sum(row[key] for row in group)/len(group) for key in
                   ('brier', 'readings_received', 'interventions')}})
    return {'schema': 1, 'scope': 'exploratory causal-attribution reference, not consciousness or verified novelty',
            'source_sha256': {p: hashlib.sha256((ROOT/p).read_text(encoding='utf-8').encode()).hexdigest() for p in SOURCES},
            'counterexample': counterexample(), 'summary': summaries, 'trials': rows, 'examples': examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report = evaluate()
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
