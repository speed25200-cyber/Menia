"""Empirical recall limits, not a consciousness or general confidence score.

The policy learns a usable delay from calibration episodes. A high softmax alone
cannot override that limit. Labels are only used by the offline calibrator.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def model_fingerprint(model):
    digest = hashlib.sha256(f"menia-memory:{model.VERSION}:{model.hidden}".encode())
    for key in sorted(model.p):
        digest.update(key.encode())
        digest.update(np.asarray(model.p[key], dtype="<f8").tobytes())
    return digest.hexdigest()


def wilson_lower(correct, count):
    """Two-sided 95% Wilson interval's lower endpoint for one binomial cell.

    This is not a simultaneous bound across delays or a deployment guarantee.
    Each cell must contain one decision per independently generated episode.
    """
    if count == 0:
        return 0.0
    z = 1.959963984540054
    fraction = correct / count
    return float((fraction + z*z/(2*count) - z*np.sqrt(
        fraction*(1-fraction)/count + z*z/(4*count*count))) / (1+z*z/count))


@dataclass(frozen=True)
class RecallPolicy:
    model_sha256: str
    max_age: int
    min_probability: float = 0.9

    def __post_init__(self):
        if (not isinstance(self.model_sha256, str) or len(self.model_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.model_sha256)):
            raise ValueError("Invalid model fingerprint")
        if type(self.max_age) is not int or not -1 <= self.max_age <= 4096:
            raise ValueError("Invalid calibrated delay")
        if (isinstance(self.min_probability, bool)
                or not np.isfinite(self.min_probability)
                or not 0 < self.min_probability <= 1):
            raise ValueError("Invalid probability threshold")

    def check_model(self, model):
        if model_fingerprint(model) != self.model_sha256:
            raise ValueError("Recall policy belongs to different model weights")

    def reason(self, probabilities, age):
        if age is None:
            return 'no_observation'
        if age == 0:
            return 'current_observation_not_recall'
        if age > self.max_age:
            return 'outside_calibrated_delay'
        if float(np.max(probabilities)) < self.min_probability:
            return 'low_model_probability'
        return None

    def save(self, path):
        value = {'format': 'menia-recall-policy', 'version': 1, **asdict(self)}
        Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')

    @classmethod
    def load(cls, path, model):
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        if value.pop('format') != 'menia-recall-policy' or value.pop('version') != 1:
            raise ValueError("Unknown recall policy format")
        result = cls(**value)
        result.check_model(model)
        return result


def calibrate(model, probabilities, labels, *, min_probability=0.9,
              min_count=64, target_lower=0.95):
    """Fit a contiguous usable horizon on [age, episode, class] predictions.

    The criterion must pass separately for every true symbol at every recall age
    (1 onward) up to the horizon. Age 0 is a visible input, reported diagnostically
    but answered directly by the session. Never fit on the evaluation split.
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    if (probabilities.ndim != 3 or probabilities.shape[2] != 4
            or labels.shape != (probabilities.shape[1],)
            or not np.isfinite(probabilities).all()
            or not np.issubdtype(labels.dtype, np.integer)
            or not np.isin(labels, np.arange(4)).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(-1), 1)):
        raise ValueError("Invalid calibration samples")
    if type(min_count) is not int or min_count < 1 or not 0 < target_lower < 1:
        raise ValueError("Invalid calibration criterion")
    # Also validate threshold before using it.
    fingerprint = model_fingerprint(model)
    RecallPolicy(fingerprint, -1, min_probability)
    horizon, contiguous, rows = -1, True, []
    for age, p in enumerate(probabilities):
        candidate = p.argmax(-1)
        selected = p.max(-1) >= min_probability
        classes = []
        for symbol in range(4):
            keep = selected & (labels == symbol)
            count = int(keep.sum())
            correct = int(((candidate == labels) & keep).sum())
            lower = wilson_lower(correct, count)
            classes.append({'symbol': symbol, 'answered': count, 'correct': correct,
                            'wilson_lower_95': lower})
        passed = all(c['answered'] >= min_count and
                     c['wilson_lower_95'] >= target_lower for c in classes)
        if age > 0:
            contiguous = contiguous and passed
            if contiguous:
                horizon = age
        rows.append({'age': age, 'passes_criterion': passed, 'classes': classes})
    return RecallPolicy(fingerprint, horizon, min_probability), rows
