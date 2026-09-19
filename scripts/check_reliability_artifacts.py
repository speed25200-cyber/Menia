"""Reproduce published calibration, held-out results and state interventions."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.evaluate_reliability import run


def compare(expected, actual, location='report'):
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f'{location}: keys differ')
        for key in expected:
            compare(expected[key], actual[key], f'{location}.{key}')
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            raise AssertionError(f'{location}: lengths differ')
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f'{location}[{index}]')
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9, err_msg=location)
    elif expected != actual:
        raise AssertionError(f'{location}: {expected!r} != {actual!r}')


def main():
    directory = ROOT/'artifacts/recall-reliability'
    expected = json.loads((directory/'report.json').read_text(encoding='utf-8'))
    for name, digest in expected['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f'Source changed since published experiment: {name}')
    for item in expected['runs']:
        if hashlib.sha256((directory/item['policy']).read_bytes()).hexdigest() != item['policy_sha256']:
            raise AssertionError(f"Policy checksum differs: {item['policy']}")
    with tempfile.TemporaryDirectory() as temporary:
        actual = run(ROOT/'artifacts/recurrent-memory', temporary,
                     batch=expected['episodes_per_split'])
    for report in (expected, actual):
        # Record each runtime, but allow numerical reproduction on another one.
        report.pop('python')
        report.pop('numpy')
    compare(expected, actual)
    print('Calibration, policy hashes, all held-out delays and interventions reproduced')


if __name__ == '__main__':
    main()
