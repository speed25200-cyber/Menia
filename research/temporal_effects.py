"""Temporal effect benchmark and NumPy checkpoint inference.

Only feature histories reach predictors. Hidden world parameters belong to the
simulator/evaluator. The feature window is designed, not a learned recurrent state.
"""
import numpy as np


FAMILIES = ('affine', 'interaction', 'null', 'beyond_window')


def world_parameters(seed, family):
    if family not in FAMILIES:
        raise ValueError('Unknown mechanism family')
    rng = np.random.default_rng(seed)
    return {'seed': seed, 'family': family, 'channel': int(rng.integers(2)),
            'gain': float(rng.choice([-1, 1])*rng.uniform(.8, 1.2)),
            'delay': 10 if family == 'beyond_window' else int(rng.integers(1, 7))}


def outcomes(history, parameters):
    """Structural simulator: history stores A[0:11] followed by U[0:11]."""
    action, cue = history[:, :11], history[:, 11:]
    selected = parameters['channel']
    y = np.empty((len(history), 2))
    y[:, 1-selected] = .8*cue[:, 1]+.2*cue[:, 3]
    base = .4*cue[:, 2]
    past = action[:, parameters['delay']]
    if parameters['family'] == 'null':
        effect = 0
    elif parameters['family'] == 'interaction':
        effect = parameters['gain']*past*cue[:, 1]+.2*past**2
    else:
        effect = parameters['gain']*past
    y[:, selected] = base+effect
    return y


def features(history, kind='temporal'):
    if kind == 'instant':
        return history[:, [0, 11]]
    if kind == 'temporal':
        return np.concatenate((history[:, :7], history[:, 11:18]), axis=1)
    raise ValueError('Unknown feature set')


def dataset(parameters, policy, *, split='train', size=1024):
    if policy not in ('coupled', 'intervened') or split not in ('train', 'test', 'probe'):
        raise ValueError('Invalid data selection')
    rng = np.random.default_rng(parameters['seed']+{'train': 10000, 'test': 20000, 'probe': 30000}[split])
    cue = rng.uniform(-1, 1, size+10)
    action = rng.uniform(-1, 1, size+10)
    if split == 'train':
        boundary = size+10 if policy == 'coupled' else min(522, size+10)
        action[:boundary] = cue[:boundary]
    indices = np.arange(10, size+10)[:, None]-np.arange(11)[None, :]
    history = np.concatenate((action[indices], cue[indices]), axis=1)
    y = outcomes(history, parameters)+rng.normal(0, .03, (size, 2))
    selected = rng.integers(2, size=size)
    mask = np.zeros((size, 2), dtype=bool)
    mask[np.arange(size), selected] = rng.random(size) >= .2
    return history, y, mask


def predict(weights, x):
    """One independent predictor, with no access to the environment."""
    if 'linear' in weights:
        return np.column_stack((x, np.ones(len(x)))) @ weights['linear']
    return np.tanh(x @ weights['w1']+weights['b1']) @ weights['w2']+weights['b2']


def fit_ridge(x, visible_y, mask):
    design = np.column_stack((x, np.ones(len(x))))
    coefficients = []
    for channel in (0, 1):
        selected = mask[:, channel]
        a, b = design[selected], visible_y[selected, channel]
        coefficients.append(np.linalg.solve(a.T@a+.1*np.eye(design.shape[1]), a.T@b))
    return {'linear': np.column_stack(coefficients)}


def metrics(weights, parameters, policy, kind):
    history, observed, _ = dataset(parameters, policy, split='test', size=512)
    forecast = predict(weights, features(history, kind))
    probe_history, _, _ = dataset(parameters, policy, split='probe', size=32)
    truth, estimated = [], []
    for delay in range(11):
        for first_value, second_value in ((1, -1), (1, 0), (-1, 0)):
            positive, negative = probe_history.copy(), probe_history.copy()
            positive[:, delay], negative[:, delay] = first_value, second_value
            truth.append(outcomes(positive, parameters)-outcomes(negative, parameters))
            estimated.append(predict(weights, features(positive, kind))-predict(weights, features(negative, kind)))
    truth, estimated = np.array(truth), np.array(estimated)
    nonzero = abs(truth) > 1e-10
    squared_error = (truth-estimated)**2
    return {'prediction_mse': float(np.mean((forecast-observed)**2)),
            'causal_mse_all': float(np.mean(squared_error)),
            'causal_mse_nonzero': float(np.mean(squared_error[nonzero])) if nonzero.any() else None,
            'spurious_effect_rms': float(np.sqrt(np.mean(estimated[~nonzero]**2))),
            'true_effect_rms': float(np.sqrt(np.mean(truth**2)))}
