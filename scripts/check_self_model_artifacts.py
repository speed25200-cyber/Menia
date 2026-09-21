"""Reload monitors and independently regenerate every reported evaluation."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.recurrent import RecurrentMemory
from research.self_model import SelfMonitor
from research.evaluate_self_model import worlds, evaluate, FAMILIES


def check(actual, expected, path='report'):
    if isinstance(expected,dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            check(actual[key],expected[key],path+'/'+key)
    elif isinstance(expected,(float,list)):
        np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=2e-7,err_msg=path)
    else:
        assert actual == expected, path


def main():
    root = Path(__file__).resolve().parents[1]
    artifacts = root/'artifacts/self-model'
    report = json.loads((artifacts/'report.json').read_text())
    for filename, expected in report['source_sha256'].items():
        assert hashlib.sha256((root/filename).read_bytes()).hexdigest() == expected, filename
    assert hashlib.sha256((root/'docs/SELF_MODEL_PILOT.md').read_bytes()).hexdigest() == report['plan_sha256']
    for run in report['runs']:
        seed = run['seed']
        model = RecurrentMemory.load(root/f'artifacts/recurrent-memory/memory-seed-{seed}.json')
        monitors = {}
        for role, record in run['checkpoints'].items():
            path = artifacts/record['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
            monitors[role] = SelfMonitor.load(path,model)
        assert monitors['own'].model_sha256 == run['model_sha256']
        for family in FAMILIES:
            data = worlds(model,run['data_seeds']['evaluation'],report['evaluation_episodes_per_family'],family)
            check(evaluate(monitors['own'],monitors['observer'],data),run['families'][family])
        print(f'memory {seed}: all five family evaluations reproduced')


if __name__ == '__main__':
    main()
