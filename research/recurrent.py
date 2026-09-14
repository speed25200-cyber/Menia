"""A small learned gated recurrent memory, trained with BPTT in NumPy.

This is a task-specific neural module, NOT a language model or a consciousness
model. Keeping it separate makes causal ablations and mobile export tractable.
"""
import json
from pathlib import Path
import numpy as np


class RecurrentMemory:
    INPUT = 5  # four symbol channels and an observation-valid flag
    OUTPUT = 4
    VERSION = 1

    def __init__(self, hidden=32, seed=17):
        if not isinstance(hidden, int) or not 4 <= hidden <= 128:
            raise ValueError("hidden must be between 4 and 128")
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        self.p = {
            "W": rng.normal(0, .2, (5, hidden)),
            "U": rng.normal(0, .1 / np.sqrt(hidden), (hidden, hidden)),
            "G": rng.normal(0, .1, (5, hidden)),
            "b": np.zeros(hidden),
            "bg": np.full(hidden, -1.0),
            "V": rng.normal(0, .2, (hidden, 4)),
            "bo": np.zeros(4),
        }

    @property
    def parameter_count(self):
        return sum(p.size for p in self.p.values())

    def zero(self, batch=1):
        return np.zeros((batch, self.hidden))

    def step(self, x, state):
        x, state = np.asarray(x, dtype=float), np.asarray(state, dtype=float)
        if x.ndim != 2 or x.shape[1] != 5 or state.shape != (len(x), self.hidden):
            raise ValueError("Expected [batch,5] observation and [batch,hidden] state")
        if not np.isfinite(x).all() or not np.isfinite(state).all():
            raise ValueError("Non-finite input")
        p = self.p
        gate = 1 / (1 + np.exp(-np.clip(x @ p["G"] + p["bg"], -30, 30)))
        candidate = np.tanh(x @ p["W"] + state @ p["U"] + p["b"])
        h = (1-gate)*state + gate*candidate
        logits = h @ p["V"] + p["bo"]
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return h, probabilities, (x, state, gate, candidate)

    def forward(self, x, *, reset_each_step=False):
        if x.ndim != 3 or x.shape[2] != 5:
            raise ValueError("Expected [time,batch,5]")
        state = self.zero(x.shape[1])
        outputs, cache = [], []
        for observation in x:
            if reset_each_step:
                state = self.zero(x.shape[1])
            state, prob, inner = self.step(observation, state)
            outputs.append(prob)
            cache.append((state, prob, inner))
        return np.asarray(outputs), cache

    def loss_and_grad(self, x, labels, mask):
        probs, cache = self.forward(x)
        if labels.shape != x.shape[:2] or mask.shape != labels.shape or mask.sum() <= 0:
            raise ValueError("Invalid labels/mask")
        total = mask.sum()
        truth = np.eye(4)[labels]
        loss = -(truth * np.log(np.maximum(probs, 1e-12)) * mask[..., None]).sum()/total
        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        carry = self.zero(x.shape[1])
        for t in reversed(range(len(cache))):
            h, prob, (observation, previous, gate, candidate) = cache[t]
            dz = (prob-truth[t]) * (mask[t, :, None]/total)
            grads["V"] += h.T @ dz
            grads["bo"] += dz.sum(axis=0)
            dh = dz @ self.p["V"].T + carry
            da = dh * gate * (1-candidate*candidate)
            dg = dh * (candidate-previous) * gate*(1-gate)
            grads["W"] += observation.T @ da
            grads["U"] += previous.T @ da
            grads["b"] += da.sum(axis=0)
            grads["G"] += observation.T @ dg
            grads["bg"] += dg.sum(axis=0)
            carry = dh*(1-gate) + da @ self.p["U"].T
        return float(loss), grads

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"format": "menia-recurrent-memory", "version": self.VERSION,
                 "hidden": self.hidden, "parameters": {k: v.tolist() for k, v in self.p.items()},
                 "metadata": metadata}
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"))+'\n')

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 4_000_000:
            raise ValueError("Checkpoint too large")
        value = json.loads(path.read_text())
        if value["format"] != "menia-recurrent-memory" or value["version"] != cls.VERSION:
            raise ValueError("Unknown format")
        model = cls(hidden=value["hidden"])
        if set(value["parameters"]) != set(model.p):
            raise ValueError("Parameter keys differ")
        for k, old in model.p.items():
            new = np.asarray(value["parameters"][k], dtype=float)
            if new.shape != old.shape or not np.isfinite(new).all():
                raise ValueError("Invalid parameter: " + k)
            model.p[k] = new
        return model


def episodes(seed, batch=64, length=12):
    """Partially observed four-symbol world. Targets are never model inputs.

    The first symbol is always observed. Later observations occasionally replace
    it. Evaluate only unobserved timesteps: success requires carrying information.
    No action, survival reward, self-report or subjective state is involved.
    """
    if batch < 1 or length < 2:
        raise ValueError("Invalid episode dimensions")
    rng = np.random.default_rng(seed)
    symbols = rng.integers(0, 4, size=(length, batch))
    seen = rng.random((length, batch)) < .18
    seen[0] = True
    seen[-1] = False
    x = np.zeros((length, batch, 5))
    labels = np.zeros((length, batch), dtype=int)
    last = symbols[0].copy()
    for t in range(length):
        last = np.where(seen[t], symbols[t], last)
        x[t, :, :4] = np.eye(4)[symbols[t]] * seen[t, :, None]
        x[t, :, 4] = seen[t]
        labels[t] = last
    return x, labels, (~seen).astype(float)


def metrics(prob, labels, mask):
    selected = mask.astype(bool)
    p, y = prob[selected], labels[selected]
    if not len(y):
        raise ValueError("No evaluated samples")
    return {"evaluated_steps": int(len(y)), "accuracy": float((p.argmax(-1) == y).mean()),
            "brier_multiclass": float(((p-np.eye(4)[y])**2).sum(axis=-1).mean())}
