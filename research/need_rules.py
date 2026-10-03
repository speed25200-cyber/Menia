"""Exploration, not pre-registered (written after the survival of test 20 was read): how much memory does the need world
reward? Survival of simple rules in the worlds of test 20 (stream [SEED, 40], 128 lives) and in 10 000 other worlds.

- "needs": knows E and N exactly (a full memory of past events and actions), refills the lower need;
- "event": sees only the event of the turn, refills the need it hits harder (ties: the other action than last turn);
- "alternate": the other action than last turn.
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


def main():
    out = {name: {"test20_worlds": survival(rule, STREAM, LIVES), "other_worlds": survival(rule, MANY_STREAM, MANY)}
           for name, rule in RULES.items()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
