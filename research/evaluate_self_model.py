"""Reproducible, paired counterfactual recall worlds and monitor evaluation."""
import numpy as np
from .self_model import features

FAMILIES = ('intact', 'noise', 'dropout', 'foreign', 'long_delay')


def worlds(model, seed, n, family='mixed'):
    if family not in FAMILIES + ('mixed',) or n < 1:
        raise ValueError('Invalid experiment')
    # Public RNG is independent of condition: paired worlds have identical input.
    public = np.random.default_rng(seed)
    target = public.integers(0, 4, n)
    ages = public.choice([128, 512] if family == 'long_delay' else [1, 2, 4, 8, 16, 32, 64], n)
    private = np.random.default_rng(seed + 1_000_000)
    condition = private.integers(0, 3, n) if family == 'mixed' else np.full(n, {'intact':0, 'noise':1, 'dropout':2, 'foreign':3, 'long_delay':0}[family])
    # Strength and alternate observations remain evaluator-private.
    strength = private.uniform(.1, 1, n)
    drop = private.uniform(.25, .75, n)
    alternate = private.integers(0, 4, n)
    own = np.empty((n, model.hidden+5))
    observer = np.empty_like(own)
    correct = np.empty(n, dtype=float)
    candidates = np.empty(n, dtype=int)
    for age in np.unique(ages):
        indices = np.flatnonzero(ages == age)
        x = np.zeros((len(indices), 5))
        x[np.arange(len(indices)), target[indices]] = 1
        x[:, 4] = 1
        state, _, _ = model.step(x, model.zero(len(indices)))
        donor_x = np.zeros_like(x)
        donor_x[np.arange(len(indices)), alternate[indices]] = 1
        donor_x[:, 4] = 1
        donor, _, _ = model.step(donor_x, model.zero(len(indices)))
        blank = np.zeros_like(x)
        for _ in range(int(age)-1):
            state, _, _ = model.step(blank, state)
            donor, _, _ = model.step(blank, donor)
        damaged = state.copy()
        noise = private.normal(size=state.shape) * strength[indices, None]
        mask = private.random(state.shape) >= drop[indices, None]
        local = condition[indices]
        damaged[local == 1] += noise[local == 1]
        damaged[local == 2] *= mask[local == 2]
        damaged[local == 3] = donor[local == 3]
        actual, p, _ = model.step(blank, damaged)
        nominal, nominal_p, _ = model.step(blank, state)
        own[indices] = features(actual, p, age)
        observer[indices] = features(nominal, nominal_p, age)
        candidates[indices] = p.argmax(1)
        correct[indices] = candidates[indices] == target[indices]
    # Labels and condition are kept in a distinct evaluator record.
    return {'own': own, 'observer': observer,
            'outcome': correct, 'candidate': candidates, 'target': target,
            'age': ages, 'condition': condition}


def evaluate(own, observer, data, seed=991):
    y = data['outcome']
    p = own.predict_features(data['own'])
    q = observer.predict_features(data['observer'])
    rng = np.random.default_rng(seed)
    shuffled = p[rng.permutation(len(p))]
    raw = data['own'][:, -5:-1].max(1)
    def scores(forecast):
        answer = forecast >= .8
        return {'brier':float(np.mean((forecast-y)**2)),
                'mean_forecast':float(forecast.mean()),
                'answer_fraction':float(answer.mean()),
                'decision_cost':float(np.where(answer, 1-y, .2).mean())}
    diff = (q-y)**2 - (p-y)**2
    boot = np.array([diff[rng.integers(len(y), size=len(y))].mean() for _ in range(1000)])
    return {'episodes':len(y), 'recall_accuracy':float(y.mean()),
            'own':scores(p), 'observer':scores(q), 'shuffled_own':scores(shuffled),
            'raw_max_probability':scores(raw), 'always_verify_cost':.2,
            'brier_advantage':float(diff.mean()),
            'brier_advantage_ci95_descriptive':np.quantile(boot,[.025,.975]).tolist()}
