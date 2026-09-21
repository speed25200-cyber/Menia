"""Atelier: a partially observed world whose agent has a hidden constitution.

D (the "self cause") sets how the agent's motor actions move it. E (the
"world cause") sets the rhythm of the sky. Neither is ever observed directly
except in the C1 control. Inspection k=0 reveals D noisily (unless C3),
k=1 reveals E noisily, k=2 and k=3 are pure noise. The agent is never told
this table. No survival reward, self-report or subjective state is involved.
"""
import numpy as np

RING = 8
SYMBOLS = 4
DELTAS = (-2, -1, 1, 2)
LIFE = 24
N_MOVE = 4
N_INSPECT = 4
N_ACTIONS = N_MOVE + N_INSPECT
CUE_RELIABILITY = 0.8
SKY_RELIABILITY = 0.8
CONDITIONS = ("T", "C1", "C3")

# Input encoding for the world model:
# p(8) g(8) s(4) last_action(9) last_delta(5) last_shift(5) cue_channel(5) cue_value(5) d_shown(4) action(8)
# last_action is the efference copy of what the agent just did, last_delta the displacement it felt,
# last_shift the change of sky it saw. None of them reveals D or E by itself.
INPUT = RING + RING + SYMBOLS + (N_ACTIONS + 1) + (len(DELTAS) + 1) + (SYMBOLS + 1) + (N_INSPECT + 1) + (SYMBOLS + 1) + 4 + N_ACTIONS
# Targets: how the body will move (none for an inspection), how the sky will shift, which cue will appear.
HEADS = {"motion": len(DELTAS) + 1, "shift": SYMBOLS, "cue": SYMBOLS + 1}


def motor_delta(d, action):
    if not 0 <= action < N_MOVE:
        raise ValueError("motor_delta expects a move action")
    return DELTAS[(action + d) % 4]


def ring_distance(a, b):
    diff = abs(int(a) - int(b)) % RING
    return min(diff, RING - diff)


class Atelier:
    def __init__(self, condition, seed):
        if condition not in CONDITIONS:
            raise ValueError("Unknown condition")
        self.condition = condition
        self.rng = np.random.default_rng(seed)
        self.d = self.e = None
        self.p = self.g = self.s = None
        self.t = 0
        self.cue_channel = -1
        self.cue_value = -1
        self.last_action = -1
        self.last_delta = -1
        self.last_shift = -1

    def _new_target(self):
        candidates = [x for x in range(RING) if x != self.p]
        return int(self.rng.choice(candidates))

    def reset(self):
        self.d = int(self.rng.integers(4))
        self.e = int(self.rng.integers(4))
        self.p = int(self.rng.integers(RING))
        self.g = self._new_target()
        self.s = int(self.rng.integers(SYMBOLS))
        self.t = 0
        self.cue_channel = -1
        self.cue_value = -1
        self.last_action = -1
        self.last_delta = -1
        self.last_shift = -1
        return self.observation()

    def observation(self):
        return {"p": self.p, "g": self.g, "s": self.s, "last_action": self.last_action,
                "last_delta": self.last_delta, "last_shift": self.last_shift,
                "cue_channel": self.cue_channel, "cue_value": self.cue_value,
                "d_shown": self.d if self.condition == "C1" else -1}

    def _cue(self, k):
        if k == 0 and self.condition != "C3":
            truth = self.d
        elif k == 1:
            truth = self.e
        else:
            return int(self.rng.integers(SYMBOLS))
        if self.rng.random() < CUE_RELIABILITY:
            return truth
        return int(self.rng.integers(SYMBOLS))

    def step(self, action):
        if self.d is None:
            raise RuntimeError("reset first")
        if not 0 <= action < N_ACTIONS:
            raise ValueError("Invalid action")
        reward = 0
        self.last_action = action
        if action < N_MOVE:
            delta = motor_delta(self.d, action)
            self.p = (self.p + delta) % RING
            self.last_delta = DELTAS.index(delta)
            self.cue_channel = -1
            self.cue_value = -1
            if self.p == self.g:
                reward = 1
                self.g = self._new_target()
        else:
            k = action - N_MOVE
            self.last_delta = -1
            self.cue_channel = k
            self.cue_value = self._cue(k)
        previous_sky = self.s
        if self.rng.random() < SKY_RELIABILITY:
            self.s = (self.s + self.e) % SYMBOLS
        else:
            self.s = int(self.rng.integers(SYMBOLS))
        self.last_shift = (self.s - previous_sky) % SYMBOLS
        self.t += 1
        return self.observation(), reward, self.t >= LIFE


def encode(obs, action):
    """One-hot input vector for the world model."""
    x = np.zeros(INPUT)
    i = 0
    x[i + obs["p"]] = 1; i += RING
    x[i + obs["g"]] = 1; i += RING
    x[i + obs["s"]] = 1; i += SYMBOLS
    x[i + obs["last_action"] + 1] = 1; i += N_ACTIONS + 1
    x[i + obs["last_delta"] + 1] = 1; i += len(DELTAS) + 1
    x[i + obs["last_shift"] + 1] = 1; i += SYMBOLS + 1
    x[i + obs["cue_channel"] + 1] = 1; i += N_INSPECT + 1
    x[i + obs["cue_value"] + 1] = 1; i += SYMBOLS + 1
    if obs["d_shown"] >= 0:
        x[i + obs["d_shown"]] = 1
    i += 4
    x[i + action] = 1
    return x


def targets(next_obs):
    return {"motion": next_obs["last_delta"] + 1, "shift": next_obs["last_shift"], "cue": next_obs["cue_value"] + 1}


def random_lives(condition, seed, count, rng_actions=None):
    """Childhood data: uniform random actions. Returns X [T,count,INPUT], Y dict [T,count], hidden (d,e)."""
    rng_actions = rng_actions or np.random.default_rng(seed + 1)
    X = np.zeros((LIFE, count, INPUT))
    Y = {k: np.zeros((LIFE, count), dtype=int) for k in HEADS}
    hidden = np.zeros((count, 2), dtype=int)
    for n in range(count):
        env = Atelier(condition, seed * 1000003 + n)
        obs = env.reset()
        hidden[n] = (env.d, env.e)
        for t in range(LIFE):
            a = int(rng_actions.integers(N_ACTIONS))
            X[t, n] = encode(obs, a)
            obs, _, _ = env.step(a)
            for k, v in targets(obs).items():
                Y[k][t, n] = v
    return X, Y, hidden
