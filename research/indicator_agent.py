"""An agent built to carry the fourteen indicator properties of Butlin, Long et al. at once.

Four specialist modules run in parallel on their own senses (Pos, Body, Vision, Intero). A
limited-capacity workspace receives the content of one module per step; the policy reads only
the workspace, and Pos reads the body belief from it. A learned controller chooses which module
writes. A learned metacognitive monitor sets how much Pos trusts each reading. A learned
attention schema estimates where the spotlight really landed, so that Vision binds a hue to the
right square. A learned sparse, smooth hue code gives Vision a quality space. Goals compete:
eat the best object or recharge. Protocol: docs/INDICATOR_AGENT_PROTOCOL.md.

Variants are lesions applied to a trained agent (see VARIANTS).
"""
import json
import math
from pathlib import Path
import numpy as np
from .sense_atelier import (RING, DELTAS, N_MOVE, STAY, N_ACTIONS, FELT_OK, ENERGY_STEP, SATIETY_STEP, BAND, value,
                            motor_delta, ring_distance)

MODULES = ("pos", "body", "vis", "intero")
K = 12
HAZARD = 0.02
LOG8 = math.log(RING)
ENTROPY_BONUS = 0.01
INTERO_RESET = 0.15
EPISTEMIC = 1.0  # distance units a move is worth, per unit of normalized uncertainty about the body
ATTENTION_PRIOR = {1: 1.0, 2: 2.0}  # innate pull of attention towards stale (age) and news (salience)
VARIANTS = ("agent", "unlimited", "random", "round_robin", "no_recurrence", "bag", "no_broadcast", "no_prediction",
            "constant_gain", "random_code", "no_schema", "single_goal", "frozen_body", "lesion_intero", "lesion_vis")
DIST = np.array([[ring_distance(a, b) for b in range(RING)] for a in range(RING)], dtype=float)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def hue_features(theta):
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    return np.stack([np.cos(2 * np.pi * theta), np.sin(2 * np.pi * theta), np.ones_like(theta)], axis=1)


# ---------------------------------------------------------------------------------------------- hue codes (HOT-4)

class HueCode:
    """code(θ) = relu([cos, sin, 1] W1), learned without reward to rebuild (cos, sin) under an L1 penalty;
    a value head is then fitted on the tasted values."""

    def __init__(self, W1, W2, b2, wv, bv):
        self.W1, self.W2, self.b2, self.wv, self.bv = (np.asarray(a, dtype=float) for a in (W1, W2, b2, wv, bv))

    def code(self, theta):
        return np.maximum(hue_features(theta) @ self.W1, 0.0)

    def value(self, codes):
        return np.atleast_2d(codes) @ self.wv + self.bv

    def to_json(self):
        return {"kind": "learned", **{k: getattr(self, k).tolist() for k in ("W1", "W2", "b2", "wv", "bv")}}


def ridge(C, values, l2=1e-3):
    A = np.hstack([C, np.ones((len(C), 1))])
    sol = np.linalg.solve(A.T @ A + l2 * np.eye(A.shape[1]), A.T @ np.asarray(values, dtype=float))
    return sol[:-1], sol[-1:]


def train_hue_code(thetas, values, seed, steps=3000, lr=0.01, l1=0.01):
    """Sparse autoencoder of the seen hues (units initialised as an evenly spaced fan), then a ridge value head."""
    rng = np.random.default_rng(seed)
    X = hue_features(thetas)
    Y = X[:, :2]
    phi = 2 * np.pi * (np.arange(K) + rng.random()) / K
    p = {"W1": np.stack([np.cos(phi), np.sin(phi), np.full(K, -0.3)]) + rng.normal(0, 0.1, (3, K)),
         "W2": rng.normal(0, 0.1, (K, 2)), "b2": np.zeros(2)}
    m = {k: np.zeros_like(a) for k, a in p.items()}
    s = {k: np.zeros_like(a) for k, a in p.items()}
    n = len(X)
    history = []
    for it in range(1, steps + 1):
        pre = X @ p["W1"]
        h = np.maximum(pre, 0.0)
        er = h @ p["W2"] + p["b2"] - Y
        loss = (er ** 2).sum() / n + l1 * h.sum() / n
        dpre = ((2 * er @ p["W2"].T + l1) / n) * (pre > 0)
        g = {"W1": X.T @ dpre, "W2": 2 * h.T @ er / n, "b2": 2 * er.sum(0) / n}
        for k in p:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            s[k] = 0.999 * s[k] + 0.001 * g[k] ** 2
            p[k] -= lr * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(s[k] / (1 - 0.999 ** it)) + 1e-8)
        if it == 1 or it % 500 == 0:
            history.append({"step": it, "loss": round(float(loss), 6)})
    wv, bv = ridge(np.maximum(X @ p["W1"], 0.0), values)
    return HueCode(p["W1"], p["W2"], p["b2"], wv, bv), history


class RandomHueCode:
    """Ablation: one random sparse code per hue interval of width 0.05, value head refitted on seen hues."""

    BINS = 20

    def __init__(self, table, wv, bv):
        self.table, self.wv, self.bv = np.asarray(table, dtype=float), np.asarray(wv, dtype=float), np.asarray(bv, dtype=float)

    def code(self, theta):
        bins = (np.atleast_1d(np.asarray(theta, dtype=float)) % 1.0 * self.BINS).astype(int) % self.BINS
        return self.table[bins]

    def value(self, codes):
        return np.atleast_2d(codes) @ self.wv + self.bv

    def to_json(self):
        return {"kind": "random", "table": self.table.tolist(), "wv": self.wv.tolist(), "bv": self.bv.tolist()}


def random_hue_code(active, thetas, values, seed):
    rng = np.random.default_rng(seed)
    n_active = max(1, int(round(active * K)))
    table = np.zeros((RandomHueCode.BINS, K))
    for b in range(RandomHueCode.BINS):
        units = rng.choice(K, n_active, replace=False)
        table[b, units] = rng.uniform(0.5, 1.5, n_active)
    code = RandomHueCode(table, np.zeros(K), np.zeros(1))
    wv, bv = ridge(code.code(thetas), values)
    return RandomHueCode(table, wv, bv)


def load_hue_code(value):
    if value["kind"] == "learned":
        return HueCode(value["W1"], value["W2"], value["b2"], value["wv"], value["bv"])
    return RandomHueCode(value["table"], value["wv"], value["bv"])


# ---------------------------------------------------------------------------------------------- learned readers

MONITOR_FEATURES = 7


def monitor_features(prior, read, previous, trace):
    """Surprise, prediction entropy, agreement and peak, then the previous surprise and a decaying trace of surprises."""
    entropy = -(prior * np.log(prior + 1e-12)).sum() / LOG8
    surprise = -math.log(prior[read] + 1e-9)
    return np.array([1.0, surprise / LOG8, entropy, float(read == int(np.argmax(prior))), float(prior.max()),
                     previous, trace]), surprise


def schema_candidates(intent, onsets, presence, hue_seen, cue):
    """Candidate squares for where the spotlight landed, with one feature row each (conditional logit)."""
    others = sorted({j for j in onsets if j != intent})
    squares = [intent] + others
    n = len(others)
    rows = []
    for k, square in enumerate(squares):
        is_intent = float(k == 0)
        consistent = float(bool(presence[square]) == hue_seen)
        rows.append([is_intent, 1 - is_intent, cue * is_intent, cue * (1 - is_intent),
                     (1 - is_intent) * math.log(max(n, 1)), consistent])
    return squares, np.array(rows)


def fit_logistic(X, y, l2=1e-3, iterations=60):
    """Newton's method for a logistic regression."""
    w = np.zeros(X.shape[1])
    for _ in range(iterations):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y) + l2 * w
        H = (X.T * (p * (1 - p))) @ X / len(y) + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, grad)
    return w


def fit_conditional_logit(groups, labels, dims, lr=0.5, steps=600, l2=1e-3):
    """Softmax over the candidates of each group, shared weights; groups padded and masked."""
    width = max(len(rows) for rows in groups)
    R = np.zeros((len(groups), width, dims))
    mask = np.zeros((len(groups), width), dtype=bool)
    for n, rows in enumerate(groups):
        R[n, :len(rows)] = rows
        mask[n, :len(rows)] = True
    target = R[np.arange(len(groups)), np.asarray(labels)]
    w = np.zeros(dims)
    for _ in range(steps):
        z = np.where(mask, R @ w, -np.inf)
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        prob = e / e.sum(1, keepdims=True)
        g = ((prob[:, :, None] * R).sum(1) - target).mean(0) + l2 * w
        w -= lr * g
    return w


# ---------------------------------------------------------------------------------------------- the agent

ATT_FEATURES = 5
GOAL_FEATURES = 6


class Params:
    """Everything the agent learns."""

    def __init__(self, hue_code, monitor, base_rate, schema, attention=None, goal=None, random_code=None, q=None):
        self.hue_code = hue_code
        self.monitor = np.asarray(monitor, dtype=float)
        self.base_rate = float(base_rate)
        self.schema = np.asarray(schema, dtype=float)
        self.attention = np.zeros((len(MODULES), ATT_FEATURES)) if attention is None else np.asarray(attention, dtype=float)
        self.goal = np.zeros(GOAL_FEATURES) if goal is None else np.asarray(goal, dtype=float)
        self.random_code = random_code
        self.q = None if q is None else np.asarray(q, dtype=float)

    def to_json(self):
        return {"format": "menia-indicator-agent", "version": 1, "hue_code": self.hue_code.to_json(),
                "random_code": self.random_code.to_json() if self.random_code else None,
                "monitor": self.monitor.tolist(), "base_rate": self.base_rate, "schema": self.schema.tolist(),
                "attention": self.attention.tolist(), "goal": self.goal.tolist(),
                **({"q": self.q.tolist()} if self.q is not None else {})}

    @classmethod
    def from_json(cls, value):
        if value.get("format") != "menia-indicator-agent":
            raise ValueError("unknown format")
        return cls(load_hue_code(value["hue_code"]), value["monitor"], value["base_rate"], value["schema"],
                   value["attention"], value["goal"], load_hue_code(value["random_code"]) if value["random_code"] else None,
                   value.get("q"))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_json()) + "\n")

    @classmethod
    def load(cls, path):
        return cls.from_json(json.loads(Path(path).read_text()))


def total_variation(a, b):
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


class Agent:
    """One life. step(obs) -> (action, intent, record). learn=True keeps the gradients REINFORCE needs.
    phase="childhood": random actions and attention, spotlight bound to the intent, gain = base rate."""

    def __init__(self, params, variant="agent", seed=0, learn=False, phase="adult"):
        if any(part not in VARIANTS for part in variant.split("+")):
            raise ValueError("unknown variant")
        self.P, self.variant, self.learn, self.phase = params, variant, learn, phase
        self.flags = set(variant.split("+"))  # a lesion, or several joined by "+" (exploratory analyses only)
        self.rng = np.random.default_rng(seed)
        self.code = params.random_code if "random_code" in self.flags else params.hue_code
        self.b = np.full(RING, 1.0 / RING)
        self.beta = np.full(4, 0.25)
        self.map = {}
        self.last_read = {}
        self.bag = []
        self.presence = [0] * RING
        self.e_hat = 1.0
        self.f_hat = 1.0
        self.previous_surprise = 0.0
        self.surprise_trace = 0.0
        self.intent = None
        self.goal = "stay"
        self.decide = True
        self.last_action = None
        self.W = {"pos": np.full(RING, 1.0 / RING), "body": np.full(4, 0.25),
                  "vis": {"values": [None] * RING, "presence": [0] * RING}, "intero": np.ones(2)}
        self.age = {m: 8 for m in MODULES}
        self.grads = []

    def _is(self, lesion):
        return lesion in self.flags

    # -- modules ------------------------------------------------------------------------------------------------

    def _body(self, obs):
        frozen = self._is("frozen_body") and obs["t"] > 8
        if frozen:
            return
        a = self.last_action
        if a is not None and a < N_MOVE and obs["felt"] is not None:
            lik = np.array([FELT_OK + (1 - FELT_OK) / 4 if motor_delta(d, a) == obs["felt"] else (1 - FELT_OK) / 4
                            for d in range(4)])
            self.beta = self.beta * lik
            self.beta /= self.beta.sum()
        self.beta = (1 - HAZARD) * self.beta + HAZARD * (1 - self.beta) / 3

    def _pos(self, obs, rec):
        a = self.last_action
        body = np.full(4, 0.25) if self._is("no_broadcast") else self.W["body"]
        if self._is("no_prediction"):
            prior = np.full(RING, 1.0 / RING)
        elif a is not None and a < N_MOVE:
            prior = np.zeros(RING)
            for d in range(4):
                prior += body[d] * np.roll(self.b, motor_delta(d, a))
        else:
            prior = self.b.copy()
        prior = 0.99 * prior + 0.01 / RING
        read = obs["pos"]
        rec["prior_argmax"] = int(np.argmax(prior))
        if read is None:
            self.b = prior
            rec.update(surprise=None, reliability=None)
            return
        features, surprise = monitor_features(prior, read, self.previous_surprise, self.surprise_trace)
        self.previous_surprise = surprise / LOG8
        self.surprise_trace = 0.5 * self.surprise_trace + 0.5 * surprise / LOG8
        if self._is("constant_gain") or self.phase == "childhood":
            rho = self.P.base_rate
        else:
            rho = float(sigmoid(features @ self.P.monitor))
        lik = np.full(RING, (1 - rho) / RING)
        lik[read] += rho
        self.b = prior * lik
        self.b /= self.b.sum()
        rec.update(surprise=round(surprise, 6), reliability=round(rho, 6), monitor_features=features[1:].round(6).tolist())

    def _schema(self, obs, rec):
        """Where did the spotlight land? Returns the square the hue reading is bound to."""
        intent = self.intent
        if intent is None:
            return None
        squares, rows = schema_candidates(intent, obs["onsets"], obs["presence"], obs["hue"] is not None, obs["cue"])
        rec["schema_squares"] = squares
        if self.phase == "childhood":
            rec["schema_rows"] = rows.tolist()
        if self._is("no_schema") or self.phase == "childhood":
            rec["schema_estimate"] = intent
            return intent
        probs = softmax(rows @ self.P.schema)
        estimate = squares[int(np.argmax(probs))]
        rec["schema_estimate"] = estimate
        rec["schema_confidence"] = round(float(probs.max()), 6)
        return estimate

    def _vision(self, obs, bound_to, rec):
        presence = obs["presence"]
        for square in list(self.map):
            if not presence[square]:
                del self.map[square]
        for square in obs["onsets"]:
            if presence[square] and not self.presence[square]:
                self.map.pop(square, None)
        self.presence = list(presence)
        rec["intent_unknown"] = self.intent is not None and self.intent not in self.map
        rec["bound"] = None
        if obs["hue"] is not None and bound_to is not None:
            code = self.code.code(obs["hue"])[0]
            if self._is("bag"):
                self.bag = (self.bag + [code])[-2:]
            elif presence[bound_to]:
                if self._is("no_recurrence"):
                    self.map = {}
                self.map[bound_to] = code
                self.last_read[bound_to] = obs["t"]
                rec["bound"] = bound_to
        elif self._is("no_recurrence"):
            self.map = {}
        values = [None] * RING
        if self._is("lesion_vis"):
            values = [0.5 if presence[x] else None for x in range(RING)]
        elif self._is("bag"):
            mean = float(np.mean(self.code.value(np.array(self.bag)))) if self.bag else None
            values = [mean if presence[x] else None for x in range(RING)]
        else:
            for square, code in self.map.items():
                values[square] = float(self.code.value(code)[0])
        rec["vis_values"] = [None if v is None else round(v, 6) for v in values]
        rec["last_read"] = {int(x): int(self.last_read.get(x, -1)) for x in self.map}
        rec["hue_read"] = obs["hue"] is not None
        return {"values": values, "presence": list(presence)}

    @staticmethod
    def _track(estimate, step, reading):
        """Predict the level, then trust the reading alone when it contradicts the prediction."""
        predicted = max(estimate - step, 0.0)
        if abs(reading - predicted) > INTERO_RESET:
            return float(np.clip(reading, 0.0, 1.0))
        return float(np.clip(0.5 * predicted + 0.5 * reading, 0.0, 1.0))

    def _intero(self, obs):
        self.e_hat = self._track(self.e_hat, ENERGY_STEP, obs["energy"])
        self.f_hat = self._track(self.f_hat, SATIETY_STEP, obs["satiety"])
        return np.ones(2) if self._is("lesion_intero") else np.array([self.e_hat, self.f_hat])

    # -- workspace ----------------------------------------------------------------------------------------------

    def _salience(self, contents):
        vis_old, vis_new = self.W["vis"], contents["vis"]
        news = 0.0
        for x in range(RING):
            a, b = vis_old["values"][x], vis_new["values"][x]
            if vis_old["presence"][x] != vis_new["presence"][x] or (a is None) != (b is None):
                news += 1
            elif a is not None and abs(a - b) > 0.2:
                news += 1
        return {"pos": total_variation(contents["pos"], self.W["pos"]), "body": total_variation(contents["body"], self.W["body"]),
                "vis": min(1.0, news / 2), "intero": min(1.0, 4 * float(np.abs(contents["intero"] - self.W["intero"]).max()))}

    def _attend(self, contents, obs, rec):
        salience = self._salience(contents)
        X = np.array([[1.0, self.age[m] / 8, salience[m], float(self.W["intero"].min()), float(self.goal == "charger")]
                      for m in MODULES])
        probs = softmax((self.P.attention * X).sum(1))
        if self._is("unlimited"):
            chosen = list(MODULES)
        elif self._is("random") or self.phase == "childhood":
            chosen = [MODULES[int(self.rng.integers(len(MODULES)))]]
        elif self._is("round_robin"):
            chosen = [MODULES[obs["t"] % len(MODULES)]]
        else:
            k = int(self.rng.choice(len(MODULES), p=probs))
            chosen = [MODULES[k]]
            if self.learn:
                grad = -probs[:, None] * X
                grad[k] += X[k]
                self.grads.append(("attention", obs["t"], grad))
                logp = np.log(probs + 1e-12)
                entropy = -(probs * logp).sum()
                self.grads.append(("attention_entropy", obs["t"], (-probs * (logp + entropy))[:, None] * X))
        for m in MODULES:
            self.age[m] = min(self.age[m] + 1, 8)
        for m in chosen:
            content = contents[m]
            self.W[m] = {"values": list(content["values"]), "presence": list(content["presence"])} if m == "vis" else content.copy()
            self.age[m] = 0
        rec["writers"] = chosen
        if "vis" in chosen or "intero" in chosen:
            self.decide = True
        rec["attention_probs"] = probs.round(6).tolist()
        rec["salience"] = {m: round(v, 6) for m, v in salience.items()}

    # -- policy ---------------------------------------------------------------------------------------------------

    def _expected_distance(self, target, action):
        b, beta = self.W["pos"], self.W["body"]
        if action == STAY:
            return float(b @ DIST[:, target])
        total = 0.0
        for d in range(4):
            total += beta[d] * float(np.roll(b, motor_delta(d, action)) @ DIST[:, target])
        return total

    def _policy(self, obs, rec):
        vis = self.W["vis"]
        known = [x for x in range(RING) if vis["presence"][x] and vis["values"][x] is not None]
        best = max(known, key=lambda x: vis["values"][x]) if known else None
        best_value = vis["values"][best] if best is not None else 0.0
        candidate = best if best is not None and best_value > 0 else None
        charger = obs["charger"]
        b = self.W["pos"]
        g = np.array([1.0, self.W["intero"][0], self.W["intero"][1], float(b @ DIST[:, charger]) / 4,
                      best_value if candidate is not None else 0.0, float(b @ DIST[:, candidate]) / 4 if candidate is not None else 0.0])
        p_charge = float(sigmoid(g @ self.P.goal))
        believed = int(np.argmax(b))
        current = self.goal
        if isinstance(current, int) and (not vis["presence"][current] or believed == current):
            self.decide = True
        if current == "charger" and believed == charger:
            self.decide = True
        if current in ("stay", "random"):
            self.decide = True
        decided = self.decide
        if self.phase == "childhood":
            goal = "random"
        elif self._is("single_goal"):
            goal = candidate if candidate is not None else "stay"
        elif not self.decide:
            goal = current
        else:
            charge = self.rng.random() < p_charge
            goal = "charger" if charge else (candidate if candidate is not None else "stay")
            if self.learn:
                self.grads.append(("goal", obs["t"], (float(charge) - p_charge) * g))
        self.decide = False
        self.goal = goal
        if goal == "random":
            action = int(self.rng.integers(N_ACTIONS))
        elif goal == "stay":
            action = STAY
        else:
            target = charger if goal == "charger" else goal
            beta = self.W["body"]
            uncertainty = float(-(beta * np.log(beta + 1e-12)).sum() / math.log(4))
            costs = [self._expected_distance(target, a) - (EPISTEMIC * uncertainty if a < N_MOVE else 0.0)
                     for a in range(N_ACTIONS)]
            order = [STAY] + list(range(N_MOVE))
            action = min(order, key=lambda a: (round(costs[a], 9), order.index(a)))
        rec.update(goal=goal, p_charge=round(p_charge, 6), best=best, candidate=candidate, decided=decided)
        return action

    def _spotlight(self, rec):
        present = [x for x in range(RING) if self.presence[x]]
        if self._is("bag"):
            return int(present[int(self.rng.integers(len(present)))]) if present else None
        unknown = [x for x in present if x not in self.map]
        if self.intent is not None and self.intent in unknown:
            return self.intent
        if unknown:
            return int(unknown[int(self.rng.integers(len(unknown)))])
        if isinstance(self.goal, int):
            return self.goal
        return int(present[int(self.rng.integers(len(present)))]) if present else None

    # -- one step ------------------------------------------------------------------------------------------------

    def step(self, obs):
        rec = {"t": obs["t"], "intent": self.intent}
        self._body(obs)
        self._pos(obs, rec)
        bound_to = self._schema(obs, rec)
        contents = {"pos": self.b.copy(), "body": self.beta.copy(), "vis": self._vision(obs, bound_to, rec),
                    "intero": self._intero(obs)}
        self._attend(contents, obs, rec)
        action = self._policy(obs, rec)
        intent = self._spotlight(rec)
        rec.update(action=action, next_intent=intent, pos_belief=int(np.argmax(self.b)), pos_confidence=round(float(self.b.max()), 6),
                   body_belief=int(np.argmax(self.beta)), body_confidence=round(float(self.beta.max()), 6),
                   workspace_body=int(np.argmax(self.W["body"])), energy_estimate=round(self.e_hat, 6),
                   satiety_estimate=round(self.f_hat, 6))
        self.intent = intent
        self.last_action = action
        return action, intent, rec
