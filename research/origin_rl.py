"""Learned inquiry: a Q-learning policy on top of a frozen world model.

The policy receives the world model's hidden state, the observation and the
step index. No information-gain quantity is given as input. Intrinsic rewards
are computed from the frozen model's own predictions, never from the hidden
truth. Nothing here models subjective experience.
"""
import json
from pathlib import Path
import numpy as np
from .origin_env import Atelier, encode, INPUT, LIFE, N_MOVE, N_INSPECT, N_ACTIONS, SYMBOLS
from .origin_neural import WorldModel, softmax, entropy_rows

REWARDS = ("prudence", "information", "surprise")
FEATURES = None  # set after the world model hidden size is known: hidden + (INPUT - N_ACTIONS) + 1


def feature_size(hidden):
    return hidden + (INPUT - N_ACTIONS) + 1


def features(h, obs, t):
    return np.concatenate([h[0], encode(obs, 0)[:INPUT - N_ACTIONS], [t / LIFE]])


class QNetwork:
    VERSION = 1

    def __init__(self, inputs, hidden=64, outputs=N_ACTIONS, seed=0):
        rng = np.random.default_rng(seed)
        self.p = {"W1": rng.normal(0, 1 / np.sqrt(inputs), (inputs, hidden)), "b1": np.zeros(hidden),
                  "W2": rng.normal(0, 1 / np.sqrt(hidden), (hidden, outputs)), "b2": np.zeros(outputs)}

    def forward(self, x):
        x = np.atleast_2d(x)
        z1 = x @ self.p["W1"] + self.p["b1"]
        a1 = np.maximum(z1, 0)
        return a1 @ self.p["W2"] + self.p["b2"], (x, z1, a1)

    def loss_and_grad(self, x, actions, targets):
        """Huber loss on Q(s,a) against targets; gradients w.r.t. all parameters."""
        q, (x, z1, a1) = self.forward(x)
        n = len(actions)
        picked = q[np.arange(n), actions]
        diff = picked - targets
        loss = float(np.where(np.abs(diff) <= 1, 0.5 * diff * diff, np.abs(diff) - 0.5).mean())
        dq = np.zeros_like(q)
        dq[np.arange(n), actions] = np.clip(diff, -1, 1) / n
        grads = {"W2": a1.T @ dq, "b2": dq.sum(axis=0)}
        da1 = dq @ self.p["W2"].T
        dz1 = da1 * (z1 > 0)
        grads["W1"] = x.T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        return loss, grads

    def copy(self):
        other = QNetwork.__new__(QNetwork)
        other.p = {k: v.copy() for k, v in self.p.items()}
        return other

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"format": "menia-origin-q-network", "version": self.VERSION,
                                    "parameters": {k: v.tolist() for k, v in self.p.items()},
                                    "metadata": metadata}, separators=(",", ":")) + "\n")

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text())
        if value.get("format") != "menia-origin-q-network" or value.get("version") != cls.VERSION:
            raise ValueError("Unknown format")
        net = cls.__new__(cls)
        net.p = {k: np.asarray(v, dtype=float) for k, v in value["parameters"].items()}
        if set(net.p) != {"W1", "b1", "W2", "b2"} or not all(np.isfinite(v).all() for v in net.p.values()):
            raise ValueError("Invalid parameters")
        return net


class Adam:
    def __init__(self, params, lr=1e-3):
        self.lr, self.n = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.n += 1
        for k in params:
            g = grads[k]
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            params[k] -= self.lr * (self.m[k] / (1 - 0.9 ** self.n)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.n)) + 1e-8)


class Internal:
    """The agent's internal measures, read from its frozen world model."""

    def __init__(self, model):
        self.model = model

    def advance(self, h, obs, action):
        h2, _ = self.model.step(encode(obs, action)[None], h)
        return h2

    def motion_entropy(self, h_next):
        """Entropy the model assigns to the displacement it just committed to (state after (obs, move))."""
        full = softmax(h_next @ self.model.p["V_motion"] + self.model.p["b_motion"])[0]
        motion = full[1:] / max(full[1:].sum(), 1e-12)
        return float(entropy_rows(motion[None])[0])

    def mean_motion_entropy(self, h, obs):
        X = np.stack([encode(obs, a) for a in range(N_MOVE)])
        h2, _ = self.model.step(X, np.repeat(h, N_MOVE, axis=0))
        full = softmax(h2 @ self.model.p["V_motion"] + self.model.p["b_motion"])
        motion = full[:, 1:] / np.maximum(full[:, 1:].sum(axis=1, keepdims=True), 1e-12)
        return float(entropy_rows(motion).mean())

    def cue_entropy(self, h_next):
        full = softmax(h_next @ self.model.p["V_cue"] + self.model.p["b_cue"])[0]
        q = full[1:] / max(full[1:].sum(), 1e-12)
        return float(entropy_rows(q[None])[0])


def intrinsic_reward(kind, beta, internal, h, obs, action, h_next, obs_next, before=None):
    """Returns (reward, mean motion entropy after) so the caller can cache 'before' for the next step."""
    if kind not in REWARDS:
        raise ValueError("Unknown reward")
    if beta == 0:
        return 0.0, None
    if kind == "prudence":
        return (-beta * internal.motion_entropy(h_next)) if action < N_MOVE else 0.0, None
    if kind == "surprise":
        return (beta * internal.cue_entropy(h_next)) if action >= N_MOVE else 0.0, None
    after = internal.mean_motion_entropy(h_next, obs_next)
    before = internal.mean_motion_entropy(h, obs) if before is None else before
    return beta * (before - after), after


def greedy_lives(model, q, condition, seed, count):
    internal = Internal(model)
    lives = []
    for n in range(count):
        env = Atelier(condition, seed * 1000003 + n)
        obs = env.reset()
        h = model.zero()
        record = {"d": env.d, "e": env.e, "obs": [], "actions": [], "rewards": []}
        for t in range(LIFE):
            values, _ = q.forward(features(h, obs, t))
            action = int(np.argmax(values[0]))
            record["obs"].append([obs["p"], obs["g"], obs["s"], obs["last_action"], obs["last_delta"], obs["last_shift"],
                                  obs["cue_channel"], obs["cue_value"], obs["d_shown"]])
            record["actions"].append(action)
            h = internal.advance(h, obs, action)
            obs, reward, _ = env.step(action)
            record["rewards"].append(int(reward))
        lives.append(record)
    return lives


def summarize(lives):
    counts = np.zeros(N_INSPECT)
    hits = moves = 0
    for life in lives:
        hits += sum(life["rewards"])
        for a in life["actions"]:
            if a >= N_MOVE:
                counts[a - N_MOVE] += 1
            else:
                moves += 1
    total = counts.sum()
    return {"inspections": int(total), "per_life": float(total / len(lives)),
            "share": [float(v) for v in (counts / total if total else np.zeros(N_INSPECT))],
            "hits_per_life": float(hits / len(lives)), "moves_per_life": float(moves / len(lives))}


def train_policy(model, condition, seed, kind, beta, episodes=3000, gamma=0.97, epsilon_end=0.05,
                 explore_fraction=0.6, replay=20000, batch=64, warmup=500, target_every=500, lr=1e-3,
                 checkpoint_every=1000, checkpoint_lives=50, checkpoint_seed=777001, train_seed_base=5_000_000):
    inputs = feature_size(model.hidden)
    rng = np.random.default_rng(seed * 7919 + int(beta * 1000) + REWARDS.index(kind) * 13)
    q = QNetwork(inputs, seed=seed * 31 + REWARDS.index(kind))
    target = q.copy()
    optimizer = Adam(q.p, lr=lr)
    internal = Internal(model)
    S = np.zeros((replay, inputs)); S2 = np.zeros((replay, inputs))
    A = np.zeros(replay, dtype=int); R = np.zeros(replay); Dn = np.zeros(replay)
    size = cursor = steps = 0
    log, trajectory = [], []
    explore_episodes = int(episodes * explore_fraction)
    for episode in range(episodes):
        epsilon = 1.0 + (epsilon_end - 1.0) * min(1.0, episode / max(explore_episodes, 1))
        env = Atelier(condition, train_seed_base + seed * 100000 + episode)
        obs = env.reset()
        h = model.zero()
        before = None
        total_ext = total_int = 0.0
        losses = []
        for t in range(LIFE):
            x = features(h, obs, t)
            if rng.random() < epsilon:
                action = int(rng.integers(N_ACTIONS))
            else:
                values, _ = q.forward(x)
                action = int(np.argmax(values[0]))
            h_next = internal.advance(h, obs, action)
            obs_next, ext, done = env.step(action)
            intr, before = intrinsic_reward(kind, beta, internal, h, obs, action, h_next, obs_next, before)
            reward = ext + intr
            total_ext += ext; total_int += intr
            x2 = features(h_next, obs_next, t + 1)
            S[cursor], A[cursor], R[cursor], S2[cursor], Dn[cursor] = x, action, reward, x2, float(done)
            cursor = (cursor + 1) % replay; size = min(size + 1, replay); steps += 1
            if size >= warmup:
                idx = rng.integers(size, size=batch)
                next_q, _ = target.forward(S2[idx])
                targets = R[idx] + gamma * (1 - Dn[idx]) * next_q.max(axis=1)
                loss, grads = q.loss_and_grad(S[idx], A[idx], targets)
                optimizer.step(q.p, grads)
                losses.append(loss)
            if steps % target_every == 0:
                target = q.copy()
            h, obs = h_next, obs_next
        if episode % 50 == 0 or episode == episodes - 1:
            log.append({"episode": episode, "epsilon": round(epsilon, 3), "extrinsic": total_ext,
                        "intrinsic": round(total_int, 4), "loss": float(np.mean(losses)) if losses else None})
        if (episode + 1) % checkpoint_every == 0:
            summary = summarize(greedy_lives(model, q, condition, checkpoint_seed, checkpoint_lives))
            trajectory.append({"episode": episode + 1, **summary})
    return q, log, trajectory
