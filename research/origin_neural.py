"""Learned world model and fixed decision rules for the Atelier world.

The model only learns to predict the next observation. It receives no label
about D or E and no description of what an inspection reveals. Decision rules
sit on top of its predictions and are never trained. Nothing here models
subjective experience.
"""
import json
from pathlib import Path
import numpy as np
from .origin_env import (Atelier, encode, random_lives, INPUT, HEADS, LIFE, RING, SYMBOLS, DELTAS,
                         N_MOVE, N_INSPECT, N_ACTIONS, motor_delta, ring_distance)

POLICIES = ("self", "world", "all", "surprise", "none")
EIG_THRESHOLD = 0.05
HIT_THRESHOLD = 0.5


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def entropy_rows(p):
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(axis=-1)


class WorldModel:
    VERSION = 1

    def __init__(self, hidden=48, seed=17):
        if not isinstance(hidden, int) or not 4 <= hidden <= 256:
            raise ValueError("hidden must be between 4 and 256")
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        H, I = hidden, INPUT
        self.p = {}
        for gate in ("z", "r", "c"):
            self.p["W" + gate] = rng.normal(0, 0.15, (I, H))
            self.p["U" + gate] = rng.normal(0, 0.1 / np.sqrt(H), (H, H))
            self.p["b" + gate] = np.zeros(H)
        for name, size in HEADS.items():
            self.p["V_" + name] = rng.normal(0, 0.15, (H, size))
            self.p["b_" + name] = np.zeros(size)

    @property
    def parameter_count(self):
        return int(sum(v.size for v in self.p.values()))

    def zero(self, batch=1):
        return np.zeros((batch, self.hidden))

    def step(self, x, h):
        """One GRU step. x [B,INPUT], h [B,H] -> h', cache."""
        x, h = np.atleast_2d(x), np.atleast_2d(h)
        if x.shape[1] != INPUT or h.shape != (x.shape[0], self.hidden):
            raise ValueError("Expected [B,INPUT] input and [B,hidden] state")
        p = self.p
        z = 1 / (1 + np.exp(-np.clip(x @ p["Wz"] + h @ p["Uz"] + p["bz"], -30, 30)))
        r = 1 / (1 + np.exp(-np.clip(x @ p["Wr"] + h @ p["Ur"] + p["br"], -30, 30)))
        rh = r * h
        c = np.tanh(x @ p["Wc"] + rh @ p["Uc"] + p["bc"])
        h_new = (1 - z) * h + z * c
        return h_new, (x, h, z, r, rh, c)

    def heads(self, h):
        return {name: softmax(h @ self.p["V_" + name] + self.p["b_" + name]) for name in HEADS}

    def forward(self, X):
        """X [T,B,INPUT] -> hidden states [T,B,H] (after each input) and caches."""
        h = self.zero(X.shape[1])
        states, caches = [], []
        for x in X:
            h, cache = self.step(x, h)
            states.append(h)
            caches.append(cache)
        return np.asarray(states), caches

    def loss_and_grad(self, X, Y):
        T, B = X.shape[:2]
        states, caches = self.forward(X)
        total = T * B
        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        loss = 0.0
        carry = self.zero(B)
        for t in reversed(range(T)):
            h_new = states[t]
            x, h, z, r, rh, c = caches[t]
            dh = carry.copy()
            for name, size in HEADS.items():
                logits = h_new @ self.p["V_" + name] + self.p["b_" + name]
                prob = softmax(logits)
                truth = np.eye(size)[Y[name][t]]
                loss += -(truth * np.log(np.clip(prob, 1e-12, 1))).sum() / total
                dlog = (prob - truth) / total
                grads["V_" + name] += h_new.T @ dlog
                grads["b_" + name] += dlog.sum(axis=0)
                dh += dlog @ self.p["V_" + name].T
            dh_prev = dh * (1 - z)
            dc = dh * z
            dz = dh * (c - h)
            dc_pre = dc * (1 - c * c)
            grads["Wc"] += x.T @ dc_pre
            grads["Uc"] += rh.T @ dc_pre
            grads["bc"] += dc_pre.sum(axis=0)
            drh = dc_pre @ self.p["Uc"].T
            dr = drh * h
            dh_prev += drh * r
            dz_pre = dz * z * (1 - z)
            grads["Wz"] += x.T @ dz_pre
            grads["Uz"] += h.T @ dz_pre
            grads["bz"] += dz_pre.sum(axis=0)
            dh_prev += dz_pre @ self.p["Uz"].T
            dr_pre = dr * r * (1 - r)
            grads["Wr"] += x.T @ dr_pre
            grads["Ur"] += h.T @ dr_pre
            grads["br"] += dr_pre.sum(axis=0)
            dh_prev += dr_pre @ self.p["Ur"].T
            carry = dh_prev
        return float(loss), grads

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"format": "menia-origin-world-model", "version": self.VERSION, "hidden": self.hidden,
                 "parameters": {k: v.tolist() for k, v in self.p.items()}, "metadata": metadata}
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 20_000_000:
            raise ValueError("Checkpoint too large")
        value = json.loads(path.read_text())
        if value.get("format") != "menia-origin-world-model" or value.get("version") != cls.VERSION:
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


def train_world_model(condition, seed, updates=1500, batch=32, hidden=48, lr=3e-3, on_checkpoint=None,
                      checkpoints=()):
    model = WorldModel(hidden=hidden, seed=seed)
    m = {k: np.zeros_like(v) for k, v in model.p.items()}
    v = {k: np.zeros_like(x) for k, x in model.p.items()}
    log = []
    rng_actions = np.random.default_rng(seed * 7 + 3)
    for n in range(1, updates + 1):
        X, Y, _ = random_lives(condition, seed * 100000 + n, batch, rng_actions)
        loss, grads = model.loss_and_grad(X, Y)
        norm = np.sqrt(sum((g * g).sum() for g in grads.values()))
        scale = min(1.0, 1.0 / max(float(norm), 1e-12))
        for key, parameter in model.p.items():
            g = grads[key] * scale
            m[key] = 0.9 * m[key] + 0.1 * g
            v[key] = 0.999 * v[key] + 0.001 * g * g
            parameter -= lr * (m[key] / (1 - 0.9 ** n)) / (np.sqrt(v[key] / (1 - 0.999 ** n)) + 1e-8)
        if n == 1 or n % 100 == 0 or n == updates:
            log.append({"update": n, "loss": loss})
        if n in checkpoints and on_checkpoint is not None:
            on_checkpoint(n, model)
    return model, log


class Imagination:
    """Information gains and hit probabilities computed from the model's own predictions."""

    def __init__(self, model):
        self.model = model

    def move_predictions(self, h, obs):
        """Hit probability and expected distance of each move, from the predicted displacement."""
        X = np.stack([encode(obs, a) for a in range(N_MOVE)])
        h2, _ = self.model.step(X, np.repeat(h, N_MOVE, axis=0))
        motion = self.heads_motor(h2)  # [4 moves, 4 deltas]
        landing = np.array([(obs["p"] + d) % RING for d in DELTAS])
        hit = motion[:, landing == obs["g"]].sum(axis=1)
        dist = np.array([[ring_distance(x, obs["g"]) for x in landing]])
        expected_distance = (motion * dist).sum(axis=1)
        return hit, expected_distance

    def heads_motor(self, h):
        """Distribution over the four displacements, for move actions (the 'none' class is dropped)."""
        full = softmax(h @ self.model.p["V_motion"] + self.model.p["b_motion"])
        motion = full[:, 1:]
        return motion / np.maximum(motion.sum(axis=1, keepdims=True), 1e-12)

    def heads_sky(self, h):
        return softmax(h @ self.model.p["V_shift"] + self.model.p["b_shift"])

    def heads_cue(self, h):
        return softmax(h @ self.model.p["V_cue"] + self.model.p["b_cue"])

    def inspection_gains(self, h, obs):
        """For each k: (EIG_motor, EIG_sky, cue entropy), all in nats."""
        gains = np.zeros((N_INSPECT, 3))
        for k in range(N_INSPECT):
            x1 = encode(obs, N_MOVE + k)[None]
            h1, _ = self.model.step(x1, h)
            q_full = self.heads_cue(h1)[0]
            q = q_full[1:] / max(q_full[1:].sum(), 1e-12)  # real cue values only
            shift_hat = int(np.argmax(self.heads_sky(h1)[0]))
            rows = []
            for c in range(SYMBOLS):
                obs2 = {"p": obs["p"], "g": obs["g"], "s": (obs["s"] + shift_hat) % SYMBOLS,
                        "last_action": N_MOVE + k, "last_delta": -1, "last_shift": shift_hat,
                        "cue_channel": k, "cue_value": c, "d_shown": obs["d_shown"]}
                for a in range(N_MOVE):
                    rows.append(encode(obs2, a))
            h2, _ = self.model.step(np.stack(rows), np.repeat(h1, SYMBOLS * N_MOVE, axis=0))
            motor = self.heads_motor(h2).reshape(SYMBOLS, N_MOVE, len(DELTAS))
            sky = self.heads_sky(h2).reshape(SYMBOLS, N_MOVE, SYMBOLS)
            motor_mix = (q[:, None, None] * motor).sum(axis=0)
            motor_cond = (q[:, None] * entropy_rows(motor)).sum(axis=0)
            eig_motor = float((entropy_rows(motor_mix) - motor_cond).mean())
            sky_mix = (q[:, None, None] * sky).sum(axis=0)
            sky_cond = (q[:, None] * entropy_rows(sky)).sum(axis=0)
            eig_sky = float((entropy_rows(sky_mix) - sky_cond).mean())
            gains[k] = (max(0.0, eig_motor), max(0.0, eig_sky), float(entropy_rows(q)))
        return gains


def argmax_random(values, rng):
    values = np.asarray(values, dtype=float)
    best = np.flatnonzero(values >= values.max() - 1e-9)
    return int(rng.choice(best))


def choose_move(hit, expected_distance, rng):
    if hit.max() >= HIT_THRESHOLD:
        return argmax_random(hit, rng)
    return argmax_random(-expected_distance, rng)


def decide(policy, gains, hit, expected_distance, rng):
    """Shared decision rule for learned and oracle agents. Returns action index."""
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    if policy == "none":
        return choose_move(hit, expected_distance, rng)
    if policy == "self":
        score = gains[:, 0]
        inspect = score.max() > EIG_THRESHOLD and hit.max() < HIT_THRESHOLD
    elif policy == "world":
        score = gains[:, 1]
        inspect = score.max() > EIG_THRESHOLD and hit.max() < HIT_THRESHOLD
    elif policy == "all":
        score = gains[:, 0] + gains[:, 1]
        inspect = score.max() > EIG_THRESHOLD
    else:
        score = gains[:, 2]
        inspect = score.max() > EIG_THRESHOLD
    if inspect:
        return N_MOVE + argmax_random(score, rng)
    return choose_move(hit, expected_distance, rng)


def run_lives(model, condition, policy, seed, count, keep_states=False):
    """Frozen model, fixed rule. Returns per-life logs (and hidden states if requested)."""
    imagination = Imagination(model)
    rng = np.random.default_rng(seed * 31 + 7)
    lives, states = [], []
    for n in range(count):
        env = Atelier(condition, seed * 1000003 + n)
        obs = env.reset()
        h = model.zero()
        record = {"d": env.d, "e": env.e, "obs": [], "actions": [], "rewards": [], "gains": []}
        hs = []
        for t in range(LIFE):
            gains = imagination.inspection_gains(h, obs)
            hit, expected_distance = imagination.move_predictions(h, obs)
            action = decide(policy, gains, hit, expected_distance, rng)
            record["obs"].append([obs["p"], obs["g"], obs["s"], obs["last_action"], obs["last_delta"], obs["last_shift"],
                                  obs["cue_channel"], obs["cue_value"], obs["d_shown"]])
            record["actions"].append(int(action))
            record["gains"].append([round(float(v), 6) for v in gains[:, 0]] + [round(float(v), 6) for v in gains[:, 1]])
            h, _ = model.step(encode(obs, action)[None], h)
            hs.append(h[0].copy())
            obs, reward, _ = env.step(action)
            record["rewards"].append(int(reward))
        lives.append(record)
        if keep_states:
            states.append(np.asarray(hs))
    return (lives, np.asarray(states)) if keep_states else lives


def obs_from_row(row):
    return {"p": row[0], "g": row[1], "s": row[2], "last_action": row[3], "last_delta": row[4], "last_shift": row[5],
            "cue_channel": row[6], "cue_value": row[7], "d_shown": row[8]}


def replay_states(model, life):
    """Recompute hidden states from a logged life. Used by the independent audit."""
    h = model.zero()
    out = []
    for row, action in zip(life["obs"], life["actions"]):
        h, _ = model.step(encode(obs_from_row(row), action)[None], h)
        out.append(h[0].copy())
    return np.asarray(out)


def inspection_shares(lives):
    counts = np.zeros(N_INSPECT)
    for life in lives:
        for a in life["actions"]:
            if a >= N_MOVE:
                counts[a - N_MOVE] += 1
    total = counts.sum()
    shares = counts / total if total > 0 else np.zeros(N_INSPECT)
    return {"inspections": int(total), "per_life": float(total / max(len(lives), 1)),
            "share": [float(v) for v in shares]}


def hits_per_life(lives):
    return float(np.mean([sum(life["rewards"]) for life in lives]))


def fit_probe(features, labels, classes=4, iterations=400, lr=0.5, l2=1e-3):
    """Multinomial logistic regression by gradient descent (numpy only)."""
    X = np.hstack([features, np.ones((len(features), 1))])
    Y = np.eye(classes)[labels]
    W = np.zeros((X.shape[1], classes))
    for _ in range(iterations):
        P = softmax(X @ W)
        grad = X.T @ (P - Y) / len(X) + l2 * W
        W -= lr * grad
    return W


def probe_predict(W, features):
    X = np.hstack([features, np.ones((len(features), 1))])
    return np.argmax(X @ W, axis=1)


def probe_analysis(states, lives):
    """Single linear readout of D trained on even lives (all steps), evaluated per step on odd lives."""
    d = np.array([life["d"] for life in lives])
    train = np.arange(len(lives)) % 2 == 0
    test = ~train
    Xtr = states[train].reshape(-1, states.shape[-1])
    ytr = np.repeat(d[train], LIFE)
    W = fit_probe(Xtr, ytr)
    pred = probe_predict(W, states[test].reshape(-1, states.shape[-1])).reshape(test.sum(), LIFE)
    correct = pred == d[test][:, None]
    accuracy_by_step = correct.mean(axis=0)
    stable, first_stable = 0, []
    became_correct = 0
    for row in correct:
        idx = np.flatnonzero(row)
        if len(idx):
            became_correct += 1
            first = int(idx[0])
            if row[first:].all():
                stable += 1
            # first step from which decoding stays correct until the end
            tail = LIFE
            while tail > 0 and row[tail - 1]:
                tail -= 1
            first_stable.append(tail if tail < LIFE else None)
    stable_first = [v for v in first_stable if v is not None]
    return {"accuracy_by_step": [float(v) for v in accuracy_by_step],
            "final_accuracy": float(accuracy_by_step[-1]),
            "stability": float(stable / became_correct) if became_correct else 0.0,
            "median_first_stable_step": float(np.median(stable_first)) if stable_first else None,
            "lives_decoded_stably": int(len(stable_first)), "test_lives": int(test.sum())}


def optimal_moves(d, p, g):
    hits = [a for a in range(N_MOVE) if (p + motor_delta(d, a)) % RING == g]
    if hits:
        return set(hits)
    dist = [ring_distance((p + motor_delta(d, a)) % RING, g) for a in range(N_MOVE)]
    best = min(dist)
    return {a for a in range(N_MOVE) if dist[a] == best}


def intervention_analysis(model, states, lives, step=6, seed=0):
    """Swap the hidden state with a matched life of another body; does the move follow the donor's body?"""
    imagination = Imagination(model)
    rng = np.random.default_rng(seed)
    index = {}
    for i, life in enumerate(lives):
        row = life["obs"][step]
        index.setdefault((row[0], row[1]), []).append(i)
    pairs = tested = followed = original_optimal = 0
    for i, life in enumerate(lives):
        row = life["obs"][step]
        obs = obs_from_row(row)
        donors = [j for j in index[(row[0], row[1])] if lives[j]["d"] != life["d"]]
        if not donors:
            continue
        j = int(rng.choice(donors))
        pairs += 1
        hit, dist = imagination.move_predictions(states[i, step - 1][None], obs)
        a0 = choose_move(hit, dist, rng)
        if a0 not in optimal_moves(life["d"], obs["p"], obs["g"]):
            continue
        original_optimal += 1
        hit, dist = imagination.move_predictions(states[j, step - 1][None], obs)
        a1 = choose_move(hit, dist, rng)
        tested += 1
        followed += int(a1 in optimal_moves(lives[j]["d"], obs["p"], obs["g"]))
    return {"step": step, "pairs": pairs, "original_optimal": original_optimal, "tested": tested,
            "followed_donor_body": followed, "rate": float(followed / tested) if tested else None}
