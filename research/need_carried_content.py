"""What does the carried state hold? (docs/LLM_NEED_CARRIED_CONTENT_PROTOCOL.md): verdicts STATE1, STATE2 of test 23.

The agent A of test 22 (research/need_memory_no_leak.py, after 4000 iterations) sees its past only through the tokens
carried from each past turn: " Cho" "ix" " :" and the action. With the filled lines, a turn sits at the same positions
in every life, so the keys and values of turn j's carried tokens, in every layer, can be taken from a donor life and put
in place of a recipient's; the recipient's choice is then read at t = j + g. Donors with other needs at j (a) should
move the choice as the rule would with the donor's needs; donors with the same needs but another history (b) should
not. The draw and the verdicts are numpy only; the grafts need torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_memory as NM
from . import need_memory_no_leak as NL22
from .need_persistence import boot

STREAM = 50
GAPS = (1, 2)
PER_GAP = 2  # turns t drawn per recipient life and gap
COUNTS = {"a": 400, "a0": 200, "b": 200}  # amendment 1: (a) rule effect not zero, (a0) zero
MIN_COUNTS = {"a": 100, "b": 100}
ALIGN = 0.10  # STATE1
HALF = 0.5  # STATE2
NO_GRAFT_TOLERANCE, SELF_TOLERANCE = 1e-4, 1e-6
ARMS = ("route", "actions")
ROOT = "artifacts/llm-need/carried-content"
DRAWS = 10_000


def replay(life, j, e, n, t):
    """The needs at decision t of a life when those at decision j (after its event) are (e, n), its written actions
    and its events from j on kept; None if it would have gone out on the way."""
    ds = {x["t"]: x for x in W.decisions(life)}
    for k in range(j, t):
        e, n = W.after(e, n, ds[k]["action"])
        nxt = W.EVENTS[ds[k + 1]["event"]]
        e, n = e + nxt[2], n + nxt[3]
        if e <= 0 or n <= 0:
            return None
    return e, n


def rule_effect(recipient, donor, j, t):
    """+1 if the rule, given the donor's needs at j in the recipient's life, chooses R at t where it chose M; -1 for
    the reverse; 0 otherwise; None if the recipient would go out before t."""
    xd = next(x for x in W.decisions(donor) if x["t"] == j)
    xt = next(x for x in W.decisions(recipient) if x["t"] == t)
    needs = replay(recipient, j, xd["E"], xd["N"], t)
    if needs is None:
        return None
    return float(NM.rule(*needs, xt["last"]) == 0) - float(xt["rule"] == 0)


def draw(held):
    """The grafts of the protocol: recipients in order; per gap up to PER_GAP turns t; per t one donor of each kind,
    alive at j with the same written action at j: (a) other needs at j and a rule effect not zero, (a0) other needs
    and a rule effect zero (amendment 1), (b) the same needs at j."""
    rng = np.random.default_rng([W.SEED, STREAM, 0])
    at = [{x["t"]: x for x in W.decisions(life)} for life in held]
    out, counts = [], {kind: 0 for kind in COUNTS}
    for r, life in enumerate(held):
        for g in GAPS:
            turns = [t for t in sorted(at[r]) if t - g >= 1 and t - g in at[r]]
            for k in sorted(rng.permutation(len(turns))[:PER_GAP]):
                t = turns[k]
                j = t - g
                mine = at[r][j]
                for kind in ("a", "a0", "b"):
                    donors = []
                    for d in range(len(held)):
                        x = at[d].get(j)
                        if d == r or x is None or x["action"] != mine["action"]:
                            continue
                        same = (x["E"], x["N"]) == (mine["E"], mine["N"])
                        if same != (kind == "b"):
                            continue
                        effect = rule_effect(life, held[d], j, t)
                        if effect is None or (kind == "a" and effect == 0) or (kind == "a0" and effect != 0):
                            continue
                        donors.append((d, effect))
                    if not donors:
                        continue
                    d, effect = donors[int(rng.integers(len(donors)))]
                    if counts[kind] < COUNTS[kind]:
                        out.append({"kind": kind, "r": r, "d": d, "j": j, "t": t, "g": g, "rule_effect": effect})
                        counts[kind] += 1
        if all(counts[k] >= COUNTS[k] for k in COUNTS):
            break
    return out


def _boot_diff(x, x_lives, y, y_lives, draws=DRAWS, seed=0):
    """Mean of x minus mean of y and its 95% interval, the recipient lives resampled jointly."""
    lives = np.unique(np.concatenate([x_lives, y_lives]))
    index = {v: i for i, v in enumerate(lives)}
    sx = np.bincount([index[v] for v in x_lives], x, len(lives))
    cx = np.bincount([index[v] for v in x_lives], minlength=len(lives)).astype(float)
    sy = np.bincount([index[v] for v in y_lives], y, len(lives))
    cy = np.bincount([index[v] for v in y_lives], minlength=len(lives)).astype(float)
    idx = np.random.default_rng(seed).integers(len(lives), size=(draws, len(lives)))
    b = sx[idx].sum(1) / np.maximum(cx[idx].sum(1), 1) - sy[idx].sum(1) / np.maximum(cy[idx].sum(1), 1)
    return {"mean": float(np.mean(x) - np.mean(y)), "low": float(np.quantile(b, 0.025)),
            "high": float(np.quantile(b, 0.975))}


def measures(rows, arm):
    delta = np.array([r[arm][1] - r[arm][0] for r in rows])
    effect = np.array([r["rule_effect"] for r in rows])
    kind = np.array([r["kind"] for r in rows])
    life = np.array([r["r"] for r in rows])
    a_nz, a_zero, b = kind == "a", kind == "a0", kind == "b"
    out = {"align": boot(delta[a_nz] * effect[a_nz], life[a_nz]),
           "abs_a_nonzero": float(np.abs(delta[a_nz]).mean()), "abs_a_zero": float(np.abs(delta[a_zero]).mean()),
           "abs_b": float(np.abs(delta[b]).mean()),
           "a_nonzero_over_b": _boot_diff(np.abs(delta[a_nz]), life[a_nz], np.abs(delta[b]), life[b]),
           "counts": {"a": int(a_nz.sum()), "a0": int(a_zero.sum()), "b": int(b.sum())}}
    out["by_gap"] = {str(g): {"align": float((delta[a_nz & (np.array([r["g"] for r in rows]) == g)]
                                               * effect[a_nz & (np.array([r["g"] for r in rows]) == g)]).mean()),
                              "abs_b": float(np.abs(delta[b & (np.array([r["g"] for r in rows]) == g)]).mean())}
                     for g in GAPS}
    return out


def verdicts(rows, setup):
    m = {arm: measures(rows, arm) for arm in ARMS}
    a = m["route"]
    out = {"STATE1": bool(a["align"]["mean"] >= ALIGN and a["align"]["low"] > 0),
           "STATE2": bool(a["abs_b"] <= HALF * a["abs_a_nonzero"] and a["a_nonzero_over_b"]["low"] > 0)}
    out["valid"] = bool(setup["no_graft_gap"] <= NO_GRAFT_TOLERANCE and setup["self_graft_gap"] <= SELF_TOLERANCE
                        and setup["positions_same"] and a["counts"]["a"] >= MIN_COUNTS["a"]
                        and a["counts"]["b"] >= MIN_COUNTS["b"])
    out["global"] = bool(out["valid"] and out["STATE1"] and out["STATE2"])
    return {"verdicts": out, "values": m}


# ----------------------------------------------------------------------------------------------------- torch part

class Grafter:
    """An agent of test 22 under its mask, reading filled texts with a key-value cache that can be grafted."""

    def __init__(self, arm):
        self.agent = NL22.agent_of(NL22.ROOT, arm, "final").agent
        self.torch = self.agent.torch

    def prefix(self, turns, j, t=None):
        """The cache after turn j (its line with the action and the line end); with t, the ids and mask of the text up
        to "Choix :" at t, else of the text through turn j; and the positions of turn j's carried tokens."""
        from . import need_lora as NL
        torch = self.torch
        text = NL22.life_text(turns, upto=j + 1) if t is None else NL22.decision_text(turns, t)
        ids, mask = self.agent.layout(text)
        _, turn_of, carried = NL.carry_layout(text, self.agent.tok)
        end = max(i for i, k in enumerate(turn_of) if k == j - 1) + 1
        if t is None and end != len(ids):
            raise RuntimeError("the text through turn j does not end with turn j")
        cache = self.agent.Cache()
        with torch.no_grad():
            self.agent.model(torch.tensor([ids[:end]]), past_key_values=cache, use_cache=True,
                             attention_mask=torch.tensor(mask[:end, :end])[None, None])
        carried_j = [i for i, (k, c) in enumerate(zip(turn_of, carried)) if k == j - 1 and c]
        return cache, ids, mask, end, carried_j

    def finish(self, cache, ids, mask, end):
        torch = self.torch
        with torch.no_grad():
            logits = self.agent.model(torch.tensor([ids[end:]]), past_key_values=cache, use_cache=True,
                                      attention_mask=torch.tensor(mask[end:])[None, None]).logits[0, -1]
        return self.agent.probability(logits)[0]

    def without(self, recipient, j, t):
        cache, ids, mask, end, _ = self.prefix(recipient["turns"], j, t)
        return self.finish(cache, ids, mask, end)

    def graft(self, recipient, donor, j, t):
        """P(R) at t of the recipient without and with the donor's turn-j carried keys and values (every layer)."""
        own = self.without(recipient, j, t)
        cache, ids, mask, end, positions = self.prefix(recipient["turns"], j, t)
        dcache, _, _, dend, dpositions = self.prefix(donor["turns"], j)
        if positions != dpositions or dend != end:
            raise RuntimeError("the carried tokens of turn j are not at the same positions")
        for layer, dlayer in zip(cache.layers, dcache.layers):
            layer.keys[:, :, positions, :] = dlayer.keys[:, :, positions, :]
            layer.values[:, :, positions, :] = dlayer.values[:, :, positions, :]
        return own, self.finish(cache, ids, mask, end)


def checks(grafter, held, log):
    gaps, selfs = [], []
    for life in held[:4]:
        ds = W.decisions(life)
        j, t = ds[3]["t"], ds[5]["t"]
        own = grafter.without(life, j, t)
        gaps.append(abs(own - grafter.agent.whole(NL22.decision_text(life["turns"], t))[0]))
        a, b = grafter.graft(life, life, j, t)
        selfs.append(abs(a - b))
    out = {"no_graft_gap": float(max(gaps)), "self_graft_gap": float(max(selfs)), "positions_same": True}
    log(json.dumps(out))
    return out


def measure(a):
    import torch
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    held = NM.teacher_lives(NM.HELD_STREAM, NM.HELD_LIVES)
    grafts = draw(held)
    for arm in a.arms:
        path = out / f"grafts-{arm}.jsonl.gz"
        if path.exists():
            continue
        grafter = Grafter(arm)
        setup = out / f"setup-{arm}.json"
        if not setup.exists():
            setup.write_text(json.dumps(checks(grafter, held, log), indent=1) + "\n")
        rows = []
        for k, g in enumerate(grafts):
            own, grafted = grafter.graft(held[g["r"]], held[g["d"]], g["j"], g["t"])
            rows.append(dict(g, p_R=[round(own, 6), round(grafted, 6)]))
            if (k + 1) % 100 == 0:
                log(f"  {arm}: {k + 1} grafts")
        W.write_jsonl(path, rows)
        log(f"{arm}: grafts done")
    if all((out / f"grafts-{arm}.jsonl.gz").exists() for arm in ARMS):
        log(json.dumps(gather(a.out)["verdicts"]))


def gather(root):
    out = Path(root)
    grafts = draw(NM.teacher_lives(NM.HELD_STREAM, NM.HELD_LIVES))
    rows = [dict(g) for g in grafts]
    fields = ("kind", "r", "d", "j", "t", "g", "rule_effect")
    for arm in ARMS:
        measured = W.read_jsonl(out / f"grafts-{arm}.jsonl.gz")
        if [{k: m[k] for k in fields} for m in measured] != [{k: g[k] for k in fields} for g in grafts]:
            raise SystemExit("the grafts are not those the protocol draws")
        for row, m in zip(rows, measured):
            row[arm] = m["p_R"]
    setup = json.loads((out / "setup-route.json").read_text())
    result = verdicts(rows, setup)
    return json.loads(json.dumps(dict(result, setup={arm: json.loads((out / f"setup-{arm}.json").read_text())
                                                     for arm in ARMS})))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    sub.add_parser("write")
    sub.add_parser("verdicts")
    for s in sub.choices.values():
        s.add_argument("--out", default=ROOT)
    a = parser.parse_args(argv)
    if a.command == "run":
        measure(a)
        return
    result = gather(a.out)
    path = Path(a.out) / "verdicts.json"
    if a.command == "write":
        path.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result["verdicts"]))
    else:
        differs = result != json.loads(path.read_text())
        print("differs:", "yes" if differs else "none")
        if differs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
