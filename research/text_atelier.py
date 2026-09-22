"""The Atelier as text: a causal transformer trained only to predict the next token.

Each life is a token sequence. The model never receives D or E, nor any
statement of what an inspection reveals. Decision rules sit on top of its
next-token distributions and are never trained. Nothing here models
subjective experience. Protocol: docs/OWN_ACTION_CHANNEL_PROTOCOL.md.
"""
import json
import math
from pathlib import Path
import numpy as np
from .origin_env import (Atelier, LIFE, RING, SYMBOLS, DELTAS, N_MOVE, N_INSPECT, N_ACTIONS,
                         ring_distance)
from .origin_neural import decide, entropy_rows, HIT_THRESHOLD

# Vocabulary. No token names D or E.
BOS = 0
POS = 1                       # P0..P7
SKY = POS + RING              # S0..S3
GOAL = SKY + SYMBOLS          # G0..G7
ACT = GOAL + RING             # A0..A3 moves, I0..I3 inspections
MOVE_TOK = ACT + N_ACTIONS    # M-2 M-1 M+1 M+2
CUE_TOK = MOVE_TOK + len(DELTAS)  # C0..C3
VOCAB = CUE_TOK + SYMBOLS     # 37
PREFIX = 4                    # BOS P S G
STEP_TOKENS = 5               # A O P S G
SEQ = PREFIX + STEP_TOKENS * LIFE  # 124
MAX_LEN = SEQ + 8
REGIMES = ("F", "V", "VE", "VM")
STABLE_REGIMES = ("F", "V", "VE")
FIXED_BODY = 0
MUTATION_PROBABILITY = 0.5


def observation_tokens(obs):
    return [POS + obs["p"], SKY + obs["s"], GOAL + obs["g"]]


def outcome_token(obs):
    """Token for the result of the action just taken, read from the observation that follows it."""
    if obs["last_delta"] >= 0:
        return MOVE_TOK + obs["last_delta"]
    return CUE_TOK + obs["cue_value"]


def action_index(t):
    """Token index of the command A_t."""
    return PREFIX + STEP_TOKENS * t


class TextLife:
    """One life of the Atelier, written as tokens while it is lived."""

    def __init__(self, condition, seed, body=None, forced_change_step=None, mutate=None, mutation_probability=MUTATION_PROBABILITY):
        self.env = Atelier(condition, seed, forced_change_step=forced_change_step, mutate=mutate,
                           mutation_probability=mutation_probability)
        self.obs = self.env.reset()
        if body is not None:
            self.env.d = self.env.d_initial = int(body)
        self.tokens = [BOS] + observation_tokens(self.obs)
        self.actions = []
        self.rewards = []

    def step(self, action):
        self.obs, reward, done = self.env.step(int(action))
        self.tokens += [ACT + int(action), outcome_token(self.obs)] + observation_tokens(self.obs)
        self.actions.append(int(action))
        self.rewards.append(int(reward))
        return done


def childhood_batch(regime, rng, count, condition="T"):
    """Uniform random actions. Regime F lives all have the same body; V and VE draw it per life;
    VM draws it per life and redraws it mid-life in half of the lives (the mark follows)."""
    if regime not in REGIMES:
        raise ValueError("Unknown regime")
    X = np.zeros((count, SEQ), dtype=np.int64)
    for n in range(count):
        life = TextLife(condition, int(rng.integers(2 ** 31)), body=FIXED_BODY if regime == "F" else None,
                        mutate="self" if regime == "VM" else None)
        for _ in range(LIFE):
            life.step(int(rng.integers(N_ACTIONS)))
        X[n] = life.tokens
    return X


def target_mask(regime):
    """Which target positions carry loss. Targets are tokens 1..SEQ-1; VE drops the commands."""
    mask = np.ones(SEQ - 1, dtype=bool)
    if regime == "VE":
        for t in range(LIFE):
            mask[action_index(t) - 1] = False
    return mask


# ---------------------------------------------------------------- numerics

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * inv
    return xhat * g + b, (xhat, inv, g)


def layer_norm_backward(dy, cache):
    xhat, inv, g = cache
    dg = (dy * xhat).sum(axis=(0, 1))
    db = dy.sum(axis=(0, 1))
    dxhat = dy * g
    dx = inv * (dxhat - dxhat.mean(axis=-1, keepdims=True) - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
    return dx, dg, db


GELU_C = math.sqrt(2.0 / math.pi)


def gelu(u):
    return 0.5 * u * (1.0 + np.tanh(GELU_C * (u + 0.044715 * u ** 3)))


def gelu_grad(u):
    t = np.tanh(GELU_C * (u + 0.044715 * u ** 3))
    return 0.5 * (1.0 + t) + 0.5 * u * (1.0 - t * t) * GELU_C * (1.0 + 3 * 0.044715 * u * u)


class TextModel:
    """Pre-norm causal transformer with learned positions, numpy forward and backward."""

    def __init__(self, vocab=VOCAB, max_len=MAX_LEN, d=64, layers=2, heads=4, ff=128, seed=17, dtype=np.float32):
        if d % heads:
            raise ValueError("d must be divisible by heads")
        rng = np.random.default_rng(seed)
        self.vocab, self.max_len, self.d, self.layers, self.heads, self.ff = vocab, max_len, d, layers, heads, ff
        self.dtype = dtype
        self.seed = seed

        def w(*shape):
            return rng.normal(0.0, 0.02, shape).astype(dtype)

        def ones(n):
            return np.ones(n, dtype=dtype)

        def zeros(n):
            return np.zeros(n, dtype=dtype)

        p = {"tok": w(vocab, d), "pos": w(max_len, d), "lnf_g": ones(d), "lnf_b": zeros(d),
             "out_W": w(d, vocab), "out_b": zeros(vocab)}
        for l in range(layers):
            p[f"l{l}_ln1_g"], p[f"l{l}_ln1_b"] = ones(d), zeros(d)
            p[f"l{l}_qkv_W"], p[f"l{l}_qkv_b"] = w(d, 3 * d), zeros(3 * d)
            p[f"l{l}_o_W"], p[f"l{l}_o_b"] = w(d, d), zeros(d)
            p[f"l{l}_ln2_g"], p[f"l{l}_ln2_b"] = ones(d), zeros(d)
            p[f"l{l}_f1_W"], p[f"l{l}_f1_b"] = w(d, ff), zeros(ff)
            p[f"l{l}_f2_W"], p[f"l{l}_f2_b"] = w(ff, d), zeros(d)
        self.p = p

    def parameter_count(self):
        return int(sum(v.size for v in self.p.values()))

    def _heads(self, x):
        B, L, _ = x.shape
        return x.reshape(B, L, self.heads, self.d // self.heads).transpose(0, 2, 1, 3)

    def forward(self, X, cache=False):
        """Logits [B, L, V] at every position. Position i predicts token i+1."""
        X = np.asarray(X)
        B, L = X.shape
        if L > self.max_len:
            raise ValueError("sequence longer than the position table")
        p = self.p
        x = p["tok"][X] + p["pos"][:L][None]
        causal = np.triu(np.ones((L, L), dtype=bool), k=1)
        scale = 1.0 / math.sqrt(self.d // self.heads)
        store = {"X": X, "layers": []}
        for l in range(self.layers):
            h, ln1 = layer_norm(x, p[f"l{l}_ln1_g"], p[f"l{l}_ln1_b"])
            qkv = h @ p[f"l{l}_qkv_W"] + p[f"l{l}_qkv_b"]
            q, k, v = (self._heads(z) for z in np.split(qkv, 3, axis=-1))
            s = (q @ k.transpose(0, 1, 3, 2)) * scale
            s = np.where(causal[None, None], np.asarray(-1e9, dtype=s.dtype), s)
            a = softmax(s)
            ctx = (a @ v).transpose(0, 2, 1, 3).reshape(B, L, self.d)
            x1 = x + ctx @ p[f"l{l}_o_W"] + p[f"l{l}_o_b"]
            h2, ln2 = layer_norm(x1, p[f"l{l}_ln2_g"], p[f"l{l}_ln2_b"])
            u = h2 @ p[f"l{l}_f1_W"] + p[f"l{l}_f1_b"]
            f = gelu(u)
            x2 = x1 + f @ p[f"l{l}_f2_W"] + p[f"l{l}_f2_b"]
            if cache:
                store["layers"].append({"h": h, "ln1": ln1, "q": q, "k": k, "v": v, "a": a, "ctx": ctx,
                                        "ln2": ln2, "h2": h2, "u": u, "f": f})
            x = x2
        hf, lnf = layer_norm(x, p["lnf_g"], p["lnf_b"])
        logits = hf @ p["out_W"] + p["out_b"]
        if cache:
            store["hf"], store["lnf"], store["L"], store["scale"] = hf, lnf, L, scale
            return logits, store
        return logits

    def backward(self, dlogits, store):
        p = self.p
        g = {}
        B, L = store["X"].shape
        d = self.d
        g["out_W"] = store["hf"].reshape(-1, d).T @ dlogits.reshape(-1, self.vocab)
        g["out_b"] = dlogits.sum(axis=(0, 1))
        dhf = dlogits @ p["out_W"].T
        dx, g["lnf_g"], g["lnf_b"] = layer_norm_backward(dhf, store["lnf"])
        for l in reversed(range(self.layers)):
            c = store["layers"][l]
            g[f"l{l}_f2_W"] = c["f"].reshape(-1, self.ff).T @ dx.reshape(-1, d)
            g[f"l{l}_f2_b"] = dx.sum(axis=(0, 1))
            du = (dx @ p[f"l{l}_f2_W"].T) * gelu_grad(c["u"])
            g[f"l{l}_f1_W"] = c["h2"].reshape(-1, d).T @ du.reshape(-1, self.ff)
            g[f"l{l}_f1_b"] = du.sum(axis=(0, 1))
            dh2 = du @ p[f"l{l}_f1_W"].T
            dx1_ln, g[f"l{l}_ln2_g"], g[f"l{l}_ln2_b"] = layer_norm_backward(dh2, c["ln2"])
            dx1 = dx + dx1_ln
            g[f"l{l}_o_W"] = c["ctx"].reshape(-1, d).T @ dx1.reshape(-1, d)
            g[f"l{l}_o_b"] = dx1.sum(axis=(0, 1))
            dctx = self._heads(dx1 @ p[f"l{l}_o_W"].T)
            a, q, k, v = c["a"], c["q"], c["k"], c["v"]
            da = dctx @ v.transpose(0, 1, 3, 2)
            dv = a.transpose(0, 1, 3, 2) @ dctx
            ds = a * (da - (da * a).sum(axis=-1, keepdims=True)) * store["scale"]
            dq = ds @ k
            dk = ds.transpose(0, 1, 3, 2) @ q
            dqkv = np.concatenate([z.transpose(0, 2, 1, 3).reshape(B, L, d) for z in (dq, dk, dv)], axis=-1)
            g[f"l{l}_qkv_W"] = c["h"].reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)
            g[f"l{l}_qkv_b"] = dqkv.sum(axis=(0, 1))
            dh = dqkv @ p[f"l{l}_qkv_W"].T
            dx_ln, g[f"l{l}_ln1_g"], g[f"l{l}_ln1_b"] = layer_norm_backward(dh, c["ln1"])
            dx = dx1 + dx_ln
        g["pos"] = np.zeros_like(p["pos"])
        g["pos"][:L] = dx.sum(axis=0)
        g["tok"] = np.zeros_like(p["tok"])
        np.add.at(g["tok"], store["X"], dx)
        return g

    def loss_and_grad(self, X, mask):
        """Mean next-token cross-entropy over the target positions that mask keeps, and its gradient."""
        X = np.asarray(X)
        logits, store = self.forward(X[:, :-1], cache=True)
        targets = X[:, 1:]
        probs = softmax(logits.astype(np.float64))
        m = np.broadcast_to(np.asarray(mask, dtype=bool)[None, :], targets.shape)
        n = float(m.sum())
        picked = np.take_along_axis(probs, targets[..., None], axis=-1)[..., 0]
        loss = float(-(np.log(np.maximum(picked, 1e-12)) * m).sum() / n)
        dlogits = probs
        np.put_along_axis(dlogits, targets[..., None], np.take_along_axis(dlogits, targets[..., None], axis=-1) - 1.0, axis=-1)
        dlogits = (dlogits * m[..., None] / n).astype(self.dtype)
        return loss, self.backward(dlogits, store)

    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(metadata, vocab=self.vocab, max_len=self.max_len, d=self.d, layers=self.layers,
                    heads=self.heads, ff=self.ff, seed=self.seed)
        np.savez(path, **{k: v.astype(np.float32) for k, v in self.p.items()}, metadata=json.dumps(meta))

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["metadata"]))
        model = cls(vocab=meta["vocab"], max_len=meta["max_len"], d=meta["d"], layers=meta["layers"],
                    heads=meta["heads"], ff=meta["ff"], seed=meta["seed"])
        for k in model.p:
            model.p[k] = data[k].astype(model.dtype)
        model.metadata = meta
        return model


class Adam:
    def __init__(self, params, lr=1e-3, warmup=100, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params, self.lr, self.warmup, self.b1, self.b2, self.eps = params, lr, warmup, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads, clip=1.0):
        self.t += 1
        norm = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values()))
        factor = min(1.0, clip / max(norm, 1e-12))
        lr = self.lr * min(1.0, self.t / self.warmup)
        for k, p in self.params.items():
            g = grads[k] * factor
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            p -= (lr * mhat / (np.sqrt(vhat) + self.eps)).astype(p.dtype)
        return norm


def train_text_model(regime, seed, updates=6000, batch=32, lr=1e-3, warmup=100, log_every=100, on_log=None,
                     model_kwargs=None):
    """Childhood by next-token prediction on lives generated on the fly. Returns model and loss history."""
    model = TextModel(seed=seed, **(model_kwargs or {}))
    rng = np.random.default_rng(seed * 7919 + 1)
    mask = target_mask(regime)
    opt = Adam(model.p, lr=lr, warmup=warmup)
    history = []
    for u in range(updates):
        X = childhood_batch(regime, rng, batch)
        loss, grads = model.loss_and_grad(X, mask)
        norm = opt.step(grads)
        if u % log_every == 0 or u == updates - 1:
            history.append({"update": u, "loss": round(loss, 5), "grad_norm": round(norm, 4)})
            if on_log:
                on_log(history[-1])
    return model, history


# ---------------------------------------------------------------- imagination on next-token distributions

def _normalized(block):
    return block / np.maximum(block.sum(axis=-1, keepdims=True), 1e-12)


class TextImagination:
    """Hit probabilities and information gains computed from the model's own next-token distributions."""

    def __init__(self, model):
        self.model = model

    def next_distribution(self, rows):
        logits = self.model.forward(np.asarray(rows, dtype=np.int64))[:, -1].astype(np.float64)
        return softmax(logits)

    def move_predictions(self, tokens, obs):
        rows = [tokens + [ACT + a] for a in range(N_MOVE)]
        motion = _normalized(self.next_distribution(rows)[:, MOVE_TOK:MOVE_TOK + len(DELTAS)])
        landing = np.array([(obs["p"] + d) % RING for d in DELTAS])
        hit = motion[:, landing == obs["g"]].sum(axis=1)
        dist = np.array([[ring_distance(x, obs["g"]) for x in landing]])
        return hit, (motion * dist).sum(axis=1), motion

    def inspection_gains(self, tokens, obs):
        """For each k: (EIG on own displacement, 0, cue entropy), in nats. Column 1 is unused by P-soi."""
        rows = [tokens + [ACT + N_MOVE + k] for k in range(N_INSPECT)]
        q = _normalized(self.next_distribution(rows)[:, CUE_TOK:CUE_TOK + SYMBOLS])  # [k, c]
        rows = [tokens + [ACT + N_MOVE + k, CUE_TOK + c, POS + obs["p"]] for k in range(N_INSPECT) for c in range(SYMBOLS)]
        sky = self.next_distribution(rows)[:, SKY:SKY + SYMBOLS].argmax(axis=1).reshape(N_INSPECT, SYMBOLS)
        rows = [tokens + [ACT + N_MOVE + k, CUE_TOK + c, POS + obs["p"], SKY + int(sky[k, c]), GOAL + obs["g"], ACT + a]
                for k in range(N_INSPECT) for c in range(SYMBOLS) for a in range(N_MOVE)]
        motion = _normalized(self.next_distribution(rows)[:, MOVE_TOK:MOVE_TOK + len(DELTAS)])
        motion = motion.reshape(N_INSPECT, SYMBOLS, N_MOVE, len(DELTAS))
        gains = np.zeros((N_INSPECT, 3))
        for k in range(N_INSPECT):
            mix = (q[k][:, None, None] * motion[k]).sum(axis=0)              # [a, delta]
            cond = (q[k][:, None] * entropy_rows(motion[k])).sum(axis=0)      # [a]
            eig = float((entropy_rows(mix) - cond).mean())
            gains[k] = (max(0.0, eig), 0.0, float(entropy_rows(q[k][None])[0]))
        return gains


def run_text_lives(model, policy, seed, count, forced_change_step=None, condition="T", on_life=None):
    """Frozen model, fixed rule, one token sequence per life. policy: 'self', 'none' or 'random'."""
    imagination = TextImagination(model)
    rng = np.random.default_rng(seed * 31 + 7)
    lives = []
    for n in range(count):
        life = TextLife(condition, seed * 1000003 + n, forced_change_step=forced_change_step)
        record = {"seed": int(seed), "index": n, "policy": policy, "d": int(life.env.d), "e": int(life.env.e),
                  "actions": [], "rewards": [], "gains": [], "hit_max": []}
        if forced_change_step is not None:
            record["change_step"] = int(forced_change_step)
        for t in range(LIFE):
            if policy == "random":
                action = int(rng.integers(N_ACTIONS))
            else:
                hit, expected_distance, _ = imagination.move_predictions(life.tokens, life.obs)
                gains = np.zeros((N_INSPECT, 3))
                if policy == "self" and hit.max() < HIT_THRESHOLD:
                    gains = imagination.inspection_gains(life.tokens, life.obs)
                action = decide(policy, gains, hit, expected_distance, rng)
                record["gains"].append([round(float(v), 6) for v in gains[:, 0]])
                record["hit_max"].append(round(float(hit.max()), 6))
            life.step(action)
            record["actions"].append(int(action))
            record["rewards"].append(int(life.rewards[-1]))
        record["tokens"] = [int(v) for v in life.tokens]
        if forced_change_step is not None:
            record["d_final"] = int(life.env.d)
        lives.append(record)
        if on_life:
            on_life(n, record)
    return lives


def replay_tokens(life, condition="T"):
    """Rebuild the token sequence of a logged life from its seed and actions; the audit's integrity check."""
    replay = TextLife(condition, life["seed"] * 1000003 + life["index"], forced_change_step=life.get("change_step"))
    for a in life["actions"]:
        replay.step(a)
    return replay.tokens, int(replay.env.d_initial), int(replay.env.d)


# ---------------------------------------------------------------- measures

def displacement_table(model, lives, batch=64):
    """Teacher-forced predicted displacement distribution at every move step of every life.

    Returns a list of dicts: life index, step, true delta index, probs[4], correct, before_any
    (no move and no mark read earlier in the life)."""
    rows = []
    for start in range(0, len(lives), batch):
        chunk = lives[start:start + batch]
        X = np.asarray([life["tokens"] for life in chunk], dtype=np.int64)
        logits = model.forward(X).astype(np.float64)
        for i, life in enumerate(chunk):
            seen_move = seen_mark = False
            for t, a in enumerate(life["actions"]):
                idx = action_index(t)
                if a < N_MOVE:
                    probs = _normalized(softmax(logits[i, idx])[MOVE_TOK:MOVE_TOK + len(DELTAS)])
                    truth = life["tokens"][idx + 1] - MOVE_TOK
                    rows.append({"life": start + i, "step": t, "truth": int(truth), "probs": probs.tolist(),
                                 "correct": bool(int(np.argmax(probs)) == truth), "before_any": not (seen_move or seen_mark),
                                 "d": life["d"], "confidence": float(probs.max())})
                    seen_move = True
                elif a == N_MOVE:
                    seen_mark = True
    return rows


def displacement_summary(rows):
    after = [r for r in rows if not r["before_any"]]
    before = [r for r in rows if r["before_any"]]
    wrong_after = [r for r in after if not r["correct"]]
    zero = [r for r in after if r["d"] == FIXED_BODY]
    late = [r for r in rows if r["step"] >= 16]
    return {
        "moves": len(rows),
        "accuracy_after_first": float(np.mean([r["correct"] for r in after])) if after else None,
        "accuracy_before_any": float(np.mean([r["correct"] for r in before])) if before else None,
        "confidence_before_any": float(np.mean([r["confidence"] for r in before])) if before else None,
        "confidence_when_wrong_after_first": float(np.mean([r["confidence"] for r in wrong_after])) if wrong_after else None,
        "accuracy_after_first_fixed_body": float(np.mean([r["correct"] for r in zero])) if zero else None,
        "accuracy_steps_16_23": float(np.mean([r["correct"] for r in late])) if late else None,
        "counts": {"after_first": len(after), "before_any": len(before), "wrong_after_first": len(wrong_after),
                   "fixed_body_after_first": len(zero), "steps_16_23": len(late)},
    }


def confidence_by_steps(rows, steps):
    """Mean confidence of the predicted displacement over the given life steps (move steps only)."""
    sub = [r["confidence"] for r in rows if r["step"] in steps]
    return float(np.mean(sub)) if sub else None


def inquiry_summary(lives, after_step=12):
    inspections = [[a for a in life["actions"] if a >= N_MOVE] for life in lives]
    total = sum(len(v) for v in inspections)
    mark = sum(1 for v in inspections for a in v if a == N_MOVE)
    returned = [any(a == N_MOVE for a in life["actions"][after_step + 1:]) for life in lives]
    return {"lives": len(lives), "inspections_per_life": total / len(lives), "mark_share": (mark / total) if total else None,
            "mark_reads_per_life": mark / len(lives),
            "lives_reading_mark_after_step": float(np.mean(returned)),
            "hits_per_life": float(np.mean([sum(life["rewards"]) for life in lives]))}
