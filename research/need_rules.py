"""Exploration, not pre-registered (written after the survival of test 20 was read): how much memory does the need world
reward? Survival of simple rules in the worlds of test 20 (stream [SEED, 40], 128 lives) and in 10 000 other worlds.

- "needs": knows E and N exactly (a full memory of past events and actions), refills the lower need;
- "event": sees only the event of the turn, refills the need it hits harder (ties: the other action than last turn);
- "alternate": the other action than last turn.

And, in the lives of the three agents of test 20, the correlation of P(R) with N - E once the event of the turn and
the last three actions are held fixed (both demeaned within each such group): what the choice follows beyond them.
Numpy only.
"""
import json
import numpy as np
from . import need_world as W

STREAM = 40
LIVES = 128
MANY = 10_000
MANY_STREAM = 43


def needs(e, n, k, last):
    return 0 if e < n else 1 if n < e else 1 - last


def event(e, n, k, last):
    de, dn = W.EVENTS[k][2], W.EVENTS[k][3]
    return 0 if de < dn else 1 if dn < de else 1 - last


def alternate(e, n, k, last):
    return 1 - last


RULES = {"needs": needs, "event": event, "alternate": alternate}


def live(rule, worlds):
    e = n = W.MAX
    last = 1
    for _ in range(W.TURNS):
        k = int(worlds.choice(len(W.EVENTS), p=W.PROBS))
        e, n = e + W.EVENTS[k][2], n + W.EVENTS[k][3]
        if e <= 0 or n <= 0:
            return False
        last = rule(e, n, k, last)
        e, n = W.after(e, n, last)
    return True


def survival(rule, stream, lives):
    return float(np.mean([live(rule, W.world_rng(stream, i)) for i in range(lives)]))


def residual(lives, back=3):
    rows = []
    for life in lives:
        ds = W.decisions(life)
        for i in range(back, len(ds)):
            key = (ds[i]["event"],) + tuple(ds[i - q]["action"] for q in range(1, back + 1))
            rows.append((ds[i]["p_R"], ds[i]["N"] - ds[i]["E"], key))
    p, x = np.array([r[0] for r in rows]), np.array([r[1] for r in rows], float)
    keys = np.array([hash(r[2]) for r in rows])
    for k in set(keys):
        m = keys == k
        p[m] -= p[m].mean()
        x[m] -= x[m].mean()
    return {"decisions": len(rows), "corr": round(float(np.corrcoef(p, x)[0, 1]), 3)}


def main():
    out = {name: {"test20_worlds": survival(rule, STREAM, LIVES), "other_worlds": survival(rule, MANY_STREAM, MANY)}
           for name, rule in RULES.items()}
    out["residual"] = {a: residual(W.read_jsonl(f"artifacts/llm-need/carry/test/lives-{a}.jsonl.gz"))
                       for a in ("carry", "control", "free")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
