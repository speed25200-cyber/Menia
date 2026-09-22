"""Enumerate indistinguishable recall worlds; no training or consciousness score.

For a fixed donor state and delay, vary the original target over all four
symbols. The monitor sees identical inputs, while one of four recalls is right.
This is an information-availability audit, not a statistical significance test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.recurrent import RecurrentMemory
from research.self_model import SelfMonitor, features
from research.reliability import model_fingerprint

AGES = (1, 2, 4, 8, 16, 32, 64, 128, 512)


def audit():
    runs = []
    for seed in (17, 29, 43):
        model = RecurrentMemory.load(ROOT/f'artifacts/recurrent-memory/memory-seed-{seed}.json')
        monitor_path = ROOT/f'artifacts/self-model/own-{seed}.json'
        monitor = SelfMonitor.load(monitor_path, model)
        # One deterministic trajectory for each of the four donor observations.
        observation = np.concatenate((np.eye(4), np.ones((4, 1))), axis=1)
        state, _, _ = model.step(observation, model.zero(4))
        blank = np.zeros((4, 5))
        groups = []
        for age in range(1, max(AGES)+1):
            state, probability, _ = model.step(blank, state)
            if age not in AGES:
                continue
            for donor in range(4):
                # A foreign replacement followed by a blank produces exactly this
                # state irrespective of the overwritten original memory target.
                states = np.repeat(state[donor:donor+1], 4, axis=0)
                probabilities = np.repeat(probability[donor:donor+1], 4, axis=0)
                x = features(states, probabilities, age)
                targets = np.arange(4)
                candidate = int(probability[donor].argmax())
                outcomes = (targets == candidate).astype(float)
                forecasts = monitor.predict_features(x)
                assert np.array_equal(x, np.repeat(x[:1], 4, axis=0))
                assert np.array_equal(forecasts, np.repeat(forecasts[:1], 4))
                assert outcomes.sum() == 1
                observed_loss = float(np.mean((forecasts-outcomes)**2))
                # Analytic identity for ANY shared forecast q, including this one:
                # ((q-1)^2 + 3*q^2)/4 = (q-.25)^2 + 3/16.
                formula_loss = float((forecasts[0]-.25)**2 + 3/16)
                np.testing.assert_allclose(observed_loss, formula_loss, atol=1e-14, rtol=0)
                groups.append({'age':age, 'donor_symbol':donor,
                    'identical_feature_rows':4, 'correct_outcomes':1,
                    'candidate':candidate, 'monitor_forecast':float(forecasts[0]),
                    'balanced_brier':observed_loss})
        runs.append({'memory_seed':seed,'model_sha256':model_fingerprint(model),
            'monitor_sha256':hashlib.sha256(monitor_path.read_bytes()).hexdigest(),
            'groups':groups})
    return {'scope':'Exhaustive within four donor symbols, four original targets and nine delays; no consciousness inference',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'groups':108, 'counterfactual_worlds':432,
            'minimum_balanced_brier_from_these_features':3/16,
            'optimal_balanced_success_forecast':.25,
            'runs':runs}


def compare(actual, expected):
    if isinstance(expected,dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected,list):
        assert len(actual) == len(expected)
        for a,b in zip(actual,expected):
            compare(a,b)
    elif isinstance(expected,float):
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
    else:
        assert actual == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true',help='Write a newly computed audit; default verifies existing artifact')
    args = parser.parse_args()
    result = audit()
    path = ROOT/'artifacts/self-model/information-audit.json'
    if args.write:
        path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    else:
        compare(result,json.loads(path.read_text(encoding='utf-8')))
    print('108 groups / 432 counterfactual worlds: identical monitor inputs, one correct recall per four targets.')
    print('Balanced conditional optimum: success forecast 0.25, Brier 0.1875. This is not a consciousness bound.')


if __name__ == '__main__':
    main()
