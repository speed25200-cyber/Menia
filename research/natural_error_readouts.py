"""Train-only ridge fits and validation-only selection for prospective error readouts.

External readouts with an explicit final-state control; not native introspection.
The collection and final analysis protocol remain separate.
"""
import copy

import numpy as np

from research.activation_monitor import CELLS, predict, ridge_fit
from research.cross_model_prediction import digest, grade
from research.iphone_coupling_report import require
from research.natural_error_questions import COUNTS, SEED
from research.prospective_confidence_features import NAMES, features, validate_capture

ALPHAS = (.001, .01, .1, 1., 10., 100.)
FITTED = NAMES + ('shuffledMiddle',)
FORECASTS = ('betaCell',) + FITTED + ('donorMiddle',)


def cell(row):
    t = row['task']
    return t['family'], t['level']


def fitting_rows(rows, replication, split):
    # Test rows, including their labels and states, are ignored by construction.
    selected = [r for r in rows if r['task']['replication'] == replication and r['task']['split'] == split]
    require(len({r['task']['id'] for r in selected}) == len(selected), 'Duplicate fit rows')
    require(len(selected) == COUNTS[split]*len(CELLS) and
            all(sum(cell(r) == c for r in selected) == COUNTS[split] for c in CELLS), 'Incomplete fit partition')
    require(all(r['result']['status'] == 'ok' for r in selected), 'Technical error in fit partition; retain failed attempt')
    for row in selected:
        validate_capture(row['capture'])
    return selected


def donor_capture(training, task, capture):
    donors = [r for r in training if cell(r) == (task['family'], task['level'])]
    require(bool(donors), 'Missing donor category')
    index = int(digest([SEED, task['replication'], task['id'], 'middle-donor'])[:8], 16) % len(donors)
    replaced = copy.deepcopy(capture)
    replaced['state']['middle'] = list(donors[index]['capture']['state']['middle'])
    return replaced


def fit_bundle(rows, replication):
    training = fitting_rows(rows, replication, 'train')
    validation = fitting_rows(rows, replication, 'validation')
    y = np.asarray([grade(r['result'], r['task']) for r in training], dtype=np.float64)
    vy = np.asarray([grade(r['result'], r['task']) for r in validation], dtype=np.float64)
    permutation = np.arange(len(training))
    rng = np.random.default_rng(SEED + 31 * replication + 1)
    for c in CELLS:
        ids = [i for i, row in enumerate(training) if cell(row) == c]
        permutation[ids] = rng.permutation(ids)
    shuffled_captures = []
    for i, row in enumerate(training):
        capture = copy.deepcopy(row['capture'])
        capture['state']['middle'] = list(training[permutation[i]]['capture']['state']['middle'])
        shuffled_captures.append(capture)
    models, selection = {}, {}
    for name in FITTED:
        feature_name = 'internal' if name == 'shuffledMiddle' else name
        x = np.asarray([features(r['task'], shuffled_captures[i] if name == 'shuffledMiddle' else r['capture'], feature_name)
                        for i, r in enumerate(training)])
        vx = np.asarray([features(r['task'], donor_capture(training, r['task'], r['capture'])
                                 if name == 'shuffledMiddle' else r['capture'], feature_name) for r in validation])
        best = None
        candidates = []
        for alpha in ALPHAS:
            model = ridge_fit(x, y, alpha)
            brier = float(np.mean((predict(model, vx) - vy)**2))
            require(np.isfinite(brier), 'Invalid validation loss')
            candidates.append(dict(alpha=alpha, validationBrier=brier))
            if best is None or brier < best[0]:
                best = brier, model
        models[name], selection[name] = best[1], candidates
    beta = {}
    for c in CELLS:
        indexes = [i for i, row in enumerate(training) if cell(row) == c]
        beta[f'{c[0]}/{c[1]}'] = float((y[indexes].sum()+1)/(len(indexes)+2))
    return dict(schema='menia-natural-error-readouts-v1', replication=replication,
                trainCount=len(training), validationCount=len(validation),
                fitDataHash=digest(training+validation), models=models, selection=selection, beta=beta,
                shuffledTrainingDonorIds=[training[i]['task']['id'] for i in permutation],
                scope='Training labels fit coefficients; validation labels select alpha. No test labels or test states used. Foreign middle states change readout inputs only, not the LLM.')


def forecast(bundle, rows, task, capture):
    require(task['replication'] == bundle['replication'], 'Wrong repetition for predictor')
    validate_capture(capture)
    training = fitting_rows(rows, task['replication'], 'train')
    foreign = donor_capture(training, task, capture)
    values = dict(betaCell=bundle['beta'][f"{task['family']}/{task['level']}"])
    for name in FITTED:
        x = features(task, foreign if name == 'shuffledMiddle' else capture,
                     'internal' if name == 'shuffledMiddle' else name)
        values[name] = float(predict(bundle['models'][name], x))
    values['donorMiddle'] = float(predict(bundle['models']['internal'], features(task, foreign, 'internal')))
    require(set(values) == set(FORECASTS) and all(np.isfinite(v) and 0 <= v <= 1 for v in values.values()),
            'Invalid prospective forecast')
    return values
