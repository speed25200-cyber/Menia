"""Post-pilot strong fixed-policy control on new seeds; no changes to the pilot."""
import argparse
import hashlib
import json
from pathlib import Path
from menia.agency_discovery import AttributionModel, ConfoundedChannels
from research.evaluate_agency_discovery import trial

ROOT = Path(__file__).resolve().parents[1]


def fixed_trial(seed, condition):
    env, model = ConfoundedChannels(seed, condition=condition), AttributionModel()
    for tick in range(36):
        cue = env.cue()
        action = cue if tick < 24 else -cue
        channel = tick % 2
        model.update(channel, action, cue, env.read(action, channel))
    state = model.assessment()
    truth = [cause == 'action' for cause in env.causes]
    unique_truth = truth.index(True) if sum(truth) == 1 else None
    unique = state['unique_controlled_channel']
    return {'seed': seed, 'condition': condition, 'policy': 'fixed_opposed',
        'exact_classification': state['labels'] == ['controlled' if y else 'external' for y in truth],
        'unique_correct': unique is not None and unique == unique_truth,
        'unique_false': unique is not None and unique != unique_truth,
        'abstains_unique': unique is None,
        'brier': sum((p-int(y))**2 for p, y in zip(state['control_probabilities'], truth))/2,
        'readings_received': sum(model.received), 'interventions': 12}


def evaluate():
    rows, summary = [], []
    for condition in ('single', 'missing', 'masked', 'none', 'twins', 'twins_missing'):
        for policy in ('active', 'random', 'passive', 'fixed_opposed'):
            group = [fixed_trial(seed, condition) if policy == 'fixed_opposed' else trial(seed, condition, policy)[0]
                     for seed in range(2000, 2100)]
            rows.extend(group)
            summary.append({'condition': condition, 'policy': policy, 'n': 100,
                **{key: sum(r[key] for r in group) for key in
                   ('exact_classification', 'unique_correct', 'unique_false', 'abstains_unique')},
                **{'mean_'+key: sum(r[key] for r in group)/100 for key in
                   ('brier', 'readings_received', 'interventions')}})
    sources = ('menia/agency_discovery.py', 'research/evaluate_agency_discovery.py',
               'research/evaluate_agency_fixed_probe.py', 'docs/AGENCY_DISCOVERY_ADDENDUM.md')
    return {'scope': 'post-pilot stronger fixed-policy control on fresh seeds',
            'source_sha256': {p: hashlib.sha256((ROOT/p).read_text(encoding='utf-8').encode()).hexdigest() for p in sources},
            'summary': summary, 'trials': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report = evaluate()
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    for row in report['summary']:
        print(row['condition'], row['policy'], row['exact_classification'], row['unique_false'])
