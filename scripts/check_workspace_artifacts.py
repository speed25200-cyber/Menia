"""Reload every published workspace/control and reproduce its held-out evaluation."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.workspace import SharedWorkspace, DirectLookup
from research.train_workspace import evaluate


def compare(expected, actual, location='report'):
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f'{location}: keys differ')
        for key in expected:
            compare(expected[key], actual[key], f'{location}.{key}')
    elif isinstance(expected, float):
        if abs(expected-actual) > 1e-5:
            raise AssertionError(f'{location}: {expected} != {actual}')
    elif expected != actual:
        raise AssertionError(f'{location}: {expected!r} != {actual!r}')


def main():
    torch.set_num_threads(1)
    directory = ROOT/'artifacts/shared-workspace'
    report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
    for name, digest in report['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f'Source changed since experiment: {name}')
    for item in report['runs']:
        checkpoint = directory/item['checkpoint']
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != item['sha256']:
            raise AssertionError(f'Checkpoint changed: {checkpoint.name}')
        model = DirectLookup() if item['variant'] == 'direct' else SharedWorkspace()
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
        model.eval()
        compare(item['evaluation'], evaluate(model,
                    trained_without_broadcast=item['variant'] == 'without_broadcast'))
        if 'rounds_transfer' in item:
            for rounds, expected in item['rounds_transfer'].items():
                compare(expected, evaluate(model, rounds=int(rounds))['none'])
        print(f'{checkpoint.name}: held-out compositions, interventions and round transfer OK')


if __name__ == '__main__':
    main()
