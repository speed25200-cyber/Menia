"""Recompute all 240 trials; compare stored functional results and source hashes."""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.evaluate_agent import evaluate, fingerprints


def compare(actual, expected, path='report'):
    if type(actual) is float and type(expected) is float:
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise AssertionError(f'{path}: numeric mismatch')
    elif isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f'{path}: different keys')
        for key in expected:
            compare(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f'{path}: different lengths')
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f'{path}[{index}]')
    elif actual != expected:
        raise AssertionError(f'{path}: mismatch {actual!r} != {expected!r}')


if __name__ == '__main__':
    path = Path(__file__).resolve().parents[1]/'artifacts/integrated-agent/report.json'
    expected = json.loads(path.read_text(encoding='utf-8'))
    if expected['source_sha256'] != fingerprints():
        raise AssertionError('Source changed since recorded experiment; rerun and review it')
    actual = evaluate()
    # Interpreter patch versions are provenance, not functional output.
    actual.pop('python')
    expected.pop('python')
    compare(actual, expected)
    print('240 functional trials and 20 learned action models reproduced; source fingerprints match.')
