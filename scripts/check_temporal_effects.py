"""Verify saved weights and re-evaluate held-out outcomes and simulator interventions."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.temporal_effects import metrics
from research.train_temporal_effects import summarize
from research.evaluate_temporal_polynomial import evaluate

ROOT = Path(__file__).resolve().parents[1]


def close(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected: close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected): close(a, e)
    elif isinstance(expected, float):
        assert np.isclose(actual, expected, rtol=2e-5, atol=1e-9), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def sources(report):
    for path, digest in report['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_text(encoding='utf-8').encode()).hexdigest() == digest, path


if __name__ == '__main__':
    folder = ROOT/'artifacts/temporal-effects'
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'completed'
    assert len(report['rows']) == 168
    expected = {(model, seed, family, world, policy)
                for model, seeds in [('temporal_mlp', (11, 23, 37)), ('instant_mlp', (11, 23, 37)),
                                     ('temporal_ridge', (None,))]
                for seed in seeds for family in ('affine', 'interaction', 'null', 'beyond_window')
                for world in range(300, 303) for policy in ('coupled', 'intervened')}
    assert {(r['model'], r['initial_seed'], r['world']['family'], r['world']['seed'], r['policy'])
            for r in report['rows']} == expected
    sources(report)
    for row in report['rows']:
        path = folder/row['checkpoint']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['checkpoint_sha256']
        with np.load(path, allow_pickle=False) as checkpoint:
            close(metrics(dict(checkpoint), row['world'], row['policy'], row['features']), row['metrics'])
    close(summarize(report['rows']), report['summary'])
    print(f"{len(report['rows'])} saved predictors and their intervention metrics verified (no neural retraining)")
    folder = ROOT/'artifacts/temporal-polynomial'
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    assert len(report['rows']) == 24
    sources(report)
    assert hashlib.sha256((folder/'weights.npz').read_bytes()).hexdigest() == report['checkpoint_sha256']
    rows, weights = evaluate()
    close(rows, report['rows'])
    with np.load(folder/'weights.npz', allow_pickle=False) as checkpoint:
        for key in weights: np.testing.assert_allclose(weights[key], checkpoint[key], rtol=1e-7, atol=1e-9)
    print('24 polynomial controls refitted and re-evaluated')
