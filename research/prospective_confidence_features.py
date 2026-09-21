"""Nested, explicitly timed feature sets for a prospective external error predictor."""
import math

import numpy as np

from research.activation_monitor import CELLS, public_features, validate_state
from research.iphone_coupling_report import require

NAMES = ('inputOnly', 'outputConfidence', 'inputConfidence', 'finalControl', 'internal')
CONFIDENCE_KEYS = {'vocabularySize', 'topTokenId', 'maxProbability', 'topTwoMargin',
                   'entropyNats', 'normalizedEntropy'}


def validate_capture(capture):
    require(type(capture) is dict and set(capture) == {'state', 'preAnswer'},
            'Only pre-answer state and confidence are allowed; no completion or outcome')
    validate_state(capture['state'])
    c = capture['preAnswer']
    require(type(c) is dict and set(c) == CONFIDENCE_KEYS, 'Unexpected pre-answer confidence fields')
    n = c['vocabularySize']
    require(type(n) is int and n >= 2, 'Invalid vocabulary size')
    require(type(c['topTokenId']) is int and 0 <= c['topTokenId'] < n, 'Invalid top-token identifier')
    for key in ('maxProbability', 'topTwoMargin', 'entropyNats', 'normalizedEntropy'):
        require(type(c[key]) in (float, int) and math.isfinite(c[key]), 'Non-finite confidence')
    require(1/n - 1e-12 <= c['maxProbability'] <= 1 and 0 <= c['topTwoMargin'] <= c['maxProbability'],
            'Invalid probability or margin')
    require(0 <= c['entropyNats'] <= math.log(n) + 1e-10 and 0 <= c['normalizedEntropy'] <= 1 + 1e-10,
            'Invalid entropy')
    require(abs(c['normalizedEntropy'] - c['entropyNats']/math.log(n)) <= 1e-10,
            'Entropy normalization mismatch')


def features(task, capture, name):
    """No sampled token, future sequence likelihood, grading function or label."""
    require(name in NAMES, 'Unknown prospective feature set')
    validate_capture(capture)
    c = capture['preAnswer']
    confidence = np.asarray([c['maxProbability'], c['topTwoMargin'], c['normalizedEntropy']], dtype=np.float64)
    public = public_features(task)
    input_features = np.concatenate([public, capture['state']['input']])
    if name == 'inputOnly':
        return input_features
    if name == 'outputConfidence':
        return np.concatenate([public[-len(CELLS):], confidence])
    combined = np.concatenate([input_features, confidence])
    if name == 'inputConfidence':
        return combined
    combined = np.concatenate([combined, capture['state']['final']])
    if name == 'finalControl':
        return combined
    return np.concatenate([combined, capture['state']['middle']])
