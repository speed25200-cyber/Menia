"""Learned recall-success monitor. Functional self-information, not consciousness.

Inference is NumPy-only; the offline training script uses PyTorch. Inputs never
include the ground-truth symbol, intervention label, or an external assessment.
"""
import json
from pathlib import Path
import numpy as np
from .reliability import model_fingerprint


def features(state, probabilities, age):
    h = np.asarray(state, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if h.ndim != 2 or p.shape != (len(h), 4):
        raise ValueError('Expected batched state and four probabilities')
    a = np.broadcast_to(np.asarray(age, dtype=float), (len(h),))
    if (not np.isfinite(h).all() or not np.isfinite(p).all()
            or not np.isfinite(a).all() or (a < 1).any()
            or (p < 0).any() or not np.allclose(p.sum(1), 1)):
        raise ValueError('Invalid forecast inputs or recall age')
    return np.concatenate((h, p, np.log1p(a)[:, None]), axis=1)


class SelfMonitor:
    def __init__(self, model, mean, scale, w1, b1, w2, b2):
        self.model_sha256 = model_fingerprint(model)
        width = model.hidden + 5
        self.mean, self.scale, self.w1, self.b1, self.w2, self.b2 = (
            np.asarray(x, dtype=float) for x in (mean, scale, w1, b1, w2, b2))
        shapes = ((width,), (width,), (width, 32), (32,), (32,), ())
        for value, shape in zip(self.arrays(), shapes):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid monitor parameters')
        if (self.scale <= 0).any():
            raise ValueError('Invalid monitor scale')

    def arrays(self):
        return self.mean, self.scale, self.w1, self.b1, self.w2, self.b2

    def check_model(self, model):
        if model_fingerprint(model) != self.model_sha256:
            raise ValueError('Self-monitor belongs to different memory weights')

    def predict_features(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Invalid monitor features')
        hidden = np.maximum(0, ((x-self.mean)/self.scale) @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))

    def forecast(self, model, state, probabilities, age):
        self.check_model(model)
        return self.predict_features(features(state, probabilities, age))

    def save(self, path, metadata):
        names = ('mean', 'scale', 'w1', 'b1', 'w2', 'b2')
        value = {'format': 'menia-recall-self-monitor', 'version': 1,
                 'model_sha256': self.model_sha256, 'metadata': metadata,
                 'parameters': {k: v.tolist() for k, v in zip(names, self.arrays())}}
        Path(path).write_text(json.dumps(value, separators=(',', ':'))+'\n', encoding='utf-8', newline='\n')

    @classmethod
    def load(cls, path, model):
        path = Path(path)
        if path.stat().st_size > 1_000_000:
            raise ValueError('Monitor checkpoint too large')
        value = json.loads(path.read_text(encoding='utf-8'))
        if (value['format'] != 'menia-recall-self-monitor' or value['version'] != 1
                or value['model_sha256'] != model_fingerprint(model)):
            raise ValueError('Incompatible monitor checkpoint')
        return cls(model, **value['parameters'])
