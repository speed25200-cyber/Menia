"""Calibrate recall limits and measure them on held-out symbolic histories."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from .recurrent import RecurrentMemory, episodes
from .reliability import calibrate

CALIBRATION_SEED = 720001
# 830001 was inspected during development; the final test uses this fresh seed.
TEST_SEED = 940001
MAX_CALIBRATION_AGE = 128
TEST_AGES = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def delayed_predictions(model, seed, *, batch=512, max_age=512, prefix_length=32,
                        intervention='none'):
    """Random prefixes, balanced final observation, then an uninterrupted gap.

    One trajectory per episode, one decision per delay. Labels never enter step.
    The shuffled/reset interventions occur AFTER the final observation.
    """
    if batch < 4 or batch % 4 or max_age < 0:
        raise ValueError("Use a positive batch divisible by four and nonnegative age")
    if intervention not in {'none', 'reset', 'shuffle'}:
        raise ValueError("Unknown state intervention")
    x, _, _ = episodes(seed, batch=batch, length=prefix_length)
    labels = np.random.default_rng(seed+1).permutation(np.arange(batch) % 4)
    x[-1, :, :4] = np.eye(4)[labels]
    x[-1, :, 4] = 1
    _, cache = model.forward(x)
    state = cache[-1][0].copy()
    if intervention == 'reset':
        state[:] = 0
    elif intervention == 'shuffle':
        state = state[np.random.default_rng(seed+2).permutation(batch)]
    # Read the intervened state without an extra recurrent step.
    logits = state @ model.p['V'] + model.p['bo']
    p = np.exp(logits-logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    outputs = [p]
    blank = np.zeros((batch, 5))
    for _ in range(max_age):
        state, p, _ = model.step(blank, state)
        outputs.append(p)
    return np.asarray(outputs), labels


def selective_metrics(p, labels, accepted):
    correct = p.argmax(-1) == labels
    count = int(accepted.sum())
    return {'episodes': int(len(labels)), 'answered': count,
            'coverage': float(accepted.mean()),
            'accuracy_when_answered': float(correct[accepted].mean()) if count else None,
            'wrong_answers_per_episode': float((accepted & ~correct).mean())}


def evaluate_policy(probabilities, labels, policy, ages):
    rows = []
    for age in ages:
        p = probabilities[age]
        high_probability = p.max(-1) >= policy.min_probability
        accepted = high_probability & (1 <= age <= policy.max_age)
        rows.append({'age': age,
                     'raw_accuracy': float((p.argmax(-1) == labels).mean()),
                     'mean_max_probability': float(p.max(-1).mean()),
                     'brier_multiclass': float(((p-np.eye(4)[labels])**2).sum(-1).mean()),
                     'probability_only': selective_metrics(p, labels, high_probability),
                     'calibrated_policy': selective_metrics(p, labels, accepted),
                     'exact_symbol_storage_accuracy_analytical': 1.0})
    return rows


def run(models, out, batch=512):
    models, out = Path(models), Path(out)
    checkpoints = sorted(models.glob('memory-seed-*.json'))
    if not checkpoints:
        raise ValueError("No memory checkpoints found")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty destination to preserve previous experiments")
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in checkpoints:
        model = RecurrentMemory.load(path)
        p, y = delayed_predictions(model, CALIBRATION_SEED, batch=batch,
                                   max_age=MAX_CALIBRATION_AGE)
        policy, calibration_rows = calibrate(model, p, y)
        policy_path = out/f'{path.stem}-policy.json'
        policy.save(policy_path)
        ages = sorted(set(TEST_AGES) | set(range(max(0, policy.max_age)+2)))
        p, y = delayed_predictions(model, TEST_SEED, batch=batch)
        test_rows = evaluate_policy(p, y, policy, ages)
        shifted_histories = {}
        for prefix_length in (2, 128):
            shifted, labels = delayed_predictions(model, TEST_SEED+prefix_length,
                                                  batch=batch, prefix_length=prefix_length)
            shifted_histories[str(prefix_length)] = evaluate_policy(shifted, labels, policy, ages)
        interventions = {}
        # Internal corruption is deliberately outside the calibration family.
        for intervention in ('reset', 'shuffle'):
            altered, labels = delayed_predictions(model, TEST_SEED, batch=batch,
                                                  intervention=intervention)
            interventions[intervention] = evaluate_policy(altered, labels, policy, ages)
        result = {'checkpoint': path.name,
                  'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'policy': policy_path.name,
                  'policy_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                  'calibrated_max_age': policy.max_age,
                  'calibration': calibration_rows, 'test': test_rows,
                  'shifted_prefix_lengths': shifted_histories,
                  'internal_state_interventions': interventions}
        runs.append(result)
        print(json.dumps({'checkpoint': path.name, 'max_age': policy.max_age,
                          'long_gap_test': test_rows[-1]}), flush=True)
    source_root = Path(__file__).resolve().parents[1]
    report = {'status': 'executed', 'python': platform.python_version(), 'numpy': np.__version__,
              'calibration_seed': CALIBRATION_SEED, 'test_seed': TEST_SEED,
              'episodes_per_split': batch, 'prefix_length': 32,
              'max_calibration_age': MAX_CALIBRATION_AGE,
              'criterion': {'min_probability': 0.9, 'min_accepted_per_symbol': 64,
                            'per_cell_wilson_lower_target': 0.95, 'first_recall_age': 1},
              'source_sha256': {name: hashlib.sha256((source_root/name).read_bytes()).hexdigest()
                  for name in ('research/recurrent.py', 'research/reliability.py',
                               'research/evaluate_reliability.py')},
              'limitations': ['four-symbol synthetic recall only; not consciousness evidence',
                              'no general reliability or subjective self-awareness established',
                              'Wilson intervals are per cell, not simultaneous guarantees',
                              'ages within an episode are correlated; no pooled step significance',
                              'policy is conditional on the calibration family',
                              'a shuffled valid state can fool the policy; this is measured',
                              'deterministic last-observation memory remains a perfect reference',
                              'no iPhone or language-model integration'],
              'runs': runs}
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default='artifacts/recurrent-memory')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.models, args.out)


if __name__ == '__main__':
    main()
