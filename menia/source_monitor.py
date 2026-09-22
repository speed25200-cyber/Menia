"""Learned source probability, with NumPy inference and explicit state.

The output concerns an externally verifiable source contribution. It is not
a probability of consciousness or an estimate of subjective intensity.
"""
import json
from pathlib import Path
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -40, 40)))


class SourceMonitor:
    def __init__(self, kind="recurrent", seed=11):
        if kind not in ("recurrent", "reset", "window"):
            raise ValueError("Unknown source monitor")
        self.kind = kind
        rng = np.random.default_rng(seed)
        if kind == "window":
            self.p = {"W1": rng.normal(0, .15, (40, 12)), "b1": np.zeros(12),
                      "W2": rng.normal(0, .15, (12,)), "b2": np.zeros(1)}
        else:
            self.p = {"W": rng.normal(0, .2, (5, 16)), "U": rng.normal(0, .1, (16, 16)),
                      "G": rng.normal(0, .1, (5, 16)), "b": np.zeros(16),
                      "bg": np.full(16, -1.), "V": rng.normal(0, .2, 16), "bo": np.zeros(1)}

    def zero(self, batch=1):
        return np.zeros((batch, 40 if self.kind == "window" else 16))

    def step(self, x, state):
        x, state = np.asarray(x, dtype=float), np.asarray(state, dtype=float)
        if x.ndim != 2 or x.shape[1] != 5 or state.shape != self.zero(len(x)).shape:
            raise ValueError("Expected [batch,5] features and compatible state")
        if not np.isfinite(x).all() or not np.isfinite(state).all():
            raise ValueError("Non-finite monitor input")
        p = self.p
        if self.kind == "window":
            state = np.concatenate((state[:, 5:], x), axis=1)
            logit = np.tanh(state @ p["W1"]+p["b1"]) @ p["W2"]+p["b2"]
        else:
            if self.kind == "reset":
                state = np.zeros_like(state)
            gate = sigmoid(x @ p["G"]+p["bg"])
            candidate = np.tanh(x @ p["W"]+state @ p["U"]+p["b"])
            state = (1-gate)*state+gate*candidate
            logit = state @ p["V"]+p["bo"]
        return state, sigmoid(logit)

    def payload(self):
        return {"format": "menia-source-monitor", "version": 1, "kind": self.kind,
                "parameters": {k: v.tolist() for k, v in self.p.items()}}

    def save(self, path):
        Path(path).write_text(json.dumps(self.payload(), allow_nan=False)+"\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 1_000_000:
            raise ValueError("Checkpoint too large")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["format"] != "menia-source-monitor" or raw["version"] != 1:
            raise ValueError("Unknown checkpoint format")
        model = cls(raw["kind"])
        if set(raw["parameters"]) != set(model.p):
            raise ValueError("Parameter keys differ")
        for key, reference in model.p.items():
            value = np.asarray(raw["parameters"][key], dtype=float)
            if value.shape != reference.shape or not np.isfinite(value).all():
                raise ValueError("Invalid parameter "+key)
            model.p[key] = value
        return model
