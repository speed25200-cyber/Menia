"""Learn action costs from observed outcomes; no consciousness score."""
import json
from pathlib import Path
import numpy as np


class SourcePolicy:
    def __init__(self, coefficients=None, *, blind=False):
        self.blind = bool(blind)
        self.coefficients = (np.zeros((10, 2)) if coefficients is None
                             else np.asarray(coefficients, dtype=float).copy())
        if self.coefficients.shape != (10, 2) or not np.isfinite(self.coefficients).all():
            raise ValueError("Expected finite [10,2] action-cost coefficients")

    def features(self, q, cost):
        q, cost = np.broadcast_arrays(np.asarray(q, dtype=float), np.asarray(cost, dtype=float))
        if not np.isfinite(q).all() or not np.isfinite(cost).all():
            raise ValueError("Non-finite policy input")
        if np.any((q < 0) | (q > 1)) or np.any((cost <= 0) | (cost >= .5)):
            raise ValueError("Expected probability in [0,1] and cost in (0,.5)")
        if self.blind:
            q = np.full_like(q, .5)
        basis = np.maximum(1 - np.abs(q[..., None] - np.linspace(0, 1, 9))*8, 0)
        return np.concatenate((basis, cost[..., None]), axis=-1)

    def values(self, q, cost):
        return self.features(q, cost) @ self.coefficients

    def choose(self, q, cost):
        values = self.values(q, cost)
        return values[..., 1] < values[..., 0]

    @classmethod
    def fit(cls, q, cost, action, outcome, *, blind=False):
        """Only the realized outcome of each sampled action enters its regression."""
        model = cls(blind=blind)
        x = model.features(q, cost)
        action, outcome = np.asarray(action), np.asarray(outcome, dtype=float)
        if x.ndim != 2 or action.shape != (len(x),) or outcome.shape != (len(x),):
            raise ValueError("Expected aligned vectors of logged decisions")
        if not np.isin(action, [0, 1]).all() or not np.isfinite(outcome).all():
            raise ValueError("Invalid action or outcome")
        if np.any((outcome < 0) | (outcome > 1)):
            raise ValueError("Costs must be in [0,1]")
        for a in (0, 1):
            selected = action == a
            if not selected.any():
                raise ValueError("Both actions must have observed outcomes")
            xa, ya = x[selected], outcome[selected]
            model.coefficients[:, a] = np.linalg.solve(xa.T @ xa + .001*np.eye(10), xa.T @ ya)
        return model

    def payload(self):
        return {"format": "menia-source-policy", "version": 1, "blind": self.blind,
                "coefficients": self.coefficients.tolist()}

    def save(self, path):
        Path(path).write_text(json.dumps(self.payload(), allow_nan=False)+"\n", encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 100_000:
            raise ValueError("Policy checkpoint too large")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (set(raw) != {"format", "version", "blind", "coefficients"}
                or raw["format"] != "menia-source-policy" or raw["version"] != 1
                or type(raw["blind"]) is not bool):
            raise ValueError("Unknown policy format")
        return cls(raw["coefficients"], blind=raw["blind"])
