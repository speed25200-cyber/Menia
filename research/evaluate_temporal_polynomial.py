"""Stronger post-pilot control; intentionally kept separate from the first protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .temporal_effects import dataset, features, outcomes, fit_ridge, predict, world_parameters, FAMILIES

ROOT = Path(__file__).resolve().parents[1]


def polynomial(history):
    x = features(history)
    pairs = [x[:, i]*x[:, j] for i in range(14) for j in range(i, 14)]
    return np.column_stack([x, *pairs])


def evaluate():
    rows, checkpoints = [], {}
    for family in FAMILIES:
        for seed in range(300, 303):
            for policy in ('coupled', 'intervened'):
                parameters = world_parameters(seed, family)
                history, y, mask = dataset(parameters, policy)
                weights = fit_ridge(polynomial(history), np.where(mask, y, 0), mask)
                test, y, _ = dataset(parameters, policy, split='test', size=512)
                probe, _, _ = dataset(parameters, policy, split='probe', size=32)
                truth, estimate = [], []
                for delay in range(11):
                    for a, b in ((1, -1), (1, 0), (-1, 0)):
                        first, second = probe.copy(), probe.copy()
                        first[:, delay], second[:, delay] = a, b
                        truth.append(outcomes(first, parameters)-outcomes(second, parameters))
                        estimate.append(predict(weights, polynomial(first))-predict(weights, polynomial(second)))
                truth, estimate = np.array(truth), np.array(estimate)
                nonzero = abs(truth) > 1e-10
                error = (truth-estimate)**2
                key = f'{family}-{seed}-{policy}'
                checkpoints[key] = weights['linear']
                rows.append({'world': parameters, 'policy': policy, 'checkpoint_key': key,
                    'prediction_mse': float(np.mean((predict(weights, polynomial(test))-y)**2)),
                    'causal_mse_all': float(np.mean(error)),
                    'causal_mse_nonzero': float(np.mean(error[nonzero])) if nonzero.any() else None,
                    'spurious_effect_rms': float(np.sqrt(np.mean(estimate[~nonzero]**2)))})
    return rows, checkpoints


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rows, checkpoints = evaluate()
    np.savez_compressed(out/'weights.npz', **checkpoints)
    sources = ('research/temporal_effects.py', 'research/evaluate_temporal_polynomial.py',
               'docs/TEMPORAL_POLYNOMIAL_ADDENDUM.md')
    report = {'scope': 'post-pilot degree-two polynomial control on the same worlds',
              'source_sha256': {p: hashlib.sha256((ROOT/p).read_text(encoding='utf-8').encode()).hexdigest() for p in sources},
              'checkpoint_sha256': hashlib.sha256((out/'weights.npz').read_bytes()).hexdigest(), 'rows': rows}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    for family in FAMILIES:
        for policy in ('coupled', 'intervened'):
            group = [r for r in rows if r['world']['family'] == family and r['policy'] == policy]
            print(family, policy, 'prediction', np.mean([r['prediction_mse'] for r in group]),
                  'nonzero causal', np.mean([r['causal_mse_nonzero'] for r in group]) if family != 'null' else None, flush=True)
