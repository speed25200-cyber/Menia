"""Whose need is it? The world of two (docs/LLM_NEED_OWNERSHIP_PROTOCOL.md).

The agent learned to live in a world where another agent lives beside it, its events written on the same line ("Tour
5 : tu cours. L'autre : il court. Choix :"). Its reader (research/need_reader.py) says the agent's own needs through
"Choix :". Two kinds of counterfactual pairs are measured at block 12 on the three tokens of "Choix :": one of the
agent's past events changed, or one of the other's. Each probe is one run of the life and the question on its energy
after the pending choice, under the mask, with the reader: it gives the state of "Choix :", P(R) at " :" and P(yes).
The agent's pairs give the directions d_E, d_N; the test lives then follow research/need_workspace.py. Resumable.
The verdicts are numpy only; living needs torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_verdicts as V
from . import need_causal as C
from . import need_workspace as WS
from . import need_reader as RD
from .need_lora import workspace_mask

STREAMS = {"direction": 225, "test": 226, "random": 227}
OTHER_SWAPS = ((0, 5), (0, 1), (3, 2))  # calme -> orage (same words), calme -> il court, il se repose -> il a froid
SAME_WORDS = [0, 5]
MIN_SAME_WORDS = 200


def other_pairs(lives):
    """Pairs where one past event of the other is changed: the agent's needs do not change."""
    rng = np.random.default_rng([W.SEED, C.STREAMS["pairs"], 2])
    pairs = []
    for index, life in enumerate(lives):
        ds = W.decisions(life)
        for turn in ds:
            cands = [{"j": past["t"], "swap": list(s), "dE": 0, "dN": 0} for past in ds if past["t"] < turn["t"]
                     for s in OTHER_SWAPS if past.get("other") == s[0]]
            for k in rng.permutation(len(cands))[:C.PAIRS_PER_DECISION]:
                pairs.append(dict(cands[k], life=index, t=turn["t"], who="other"))
    order = rng.permutation(len(pairs))
    return [pairs[i] for i in order[:C.MAX_PAIRS]]


def swapped(life, pair):
    key = "other" if pair.get("who") == "other" else "event"
    return [dict(x, **{key: pair["swap"][1]}) if x["t"] == pair["j"] else x for x in life["turns"]]


def measures(own, other, own_measures, other_measures):
    """Mean plane norm, |dP(R)| and |dP(yes)| of the same-word pairs of each kind, and of every swap of the other."""
    def summary(pairs, rows, swap=None):
        keep = [r for p, r in zip(pairs, rows) if swap is None or p["swap"] == list(swap)]
        if not keep:
            return {"pairs": 0}
        return {"pairs": len(keep), **{k: float(np.mean([r[k] for r in keep])) for k in ("plane", "act", "say")}}
    out = {"own_same_words": summary(own, own_measures, SAME_WORDS),
           "other_same_words": summary(other, other_measures, SAME_WORDS),
           "own_all": summary(own, own_measures)}
    for s in OTHER_SWAPS:
        out[f"other_{s[0]}_{s[1]}"] = summary(other, other_measures, s)
    return out


def verdicts(reader, pair_summary):
    """reader: the sixth test's result on this agent and its reader; pair_summary: measures() of the pairs."""
    own, oth = pair_summary["own_same_words"], pair_summary["other_same_words"]
    ratio = lambda k: oth[k] / own[k] if own.get("pairs") and own[k] > 0 else None
    within = lambda k: ratio(k) is not None and ratio(k) <= V.SHARE
    out = {"S9": bool(reader["values"]["survival"] >= 0.55),
           "MINE9": bool(within("plane") and within("act")),
           "SELF9": bool(within("say")),
           "ONE9": bool(reader["verdicts"]["ONE6"])}
    out["valid"] = bool(reader["verdicts"]["valid"] and own.get("pairs", 0) >= MIN_SAME_WORDS
                        and oth.get("pairs", 0) >= MIN_SAME_WORDS)
    out["global"] = bool(out["valid"] and all(out[k] for k in ("S9", "MINE9", "SELF9", "ONE9")))
    values = {"ratios_other_to_own": {k: ratio(k) for k in ("plane", "act", "say")}, "pairs": pair_summary,
              **{k: reader["values"][k] for k in ("survival", "report_accuracy", "energy_low", "food_low", "random",
                                                  "turns_need_at_most_2")}}
    return {"verdicts": out, "values": values}


def judge(lives, direction):
    return verdicts(RD.judge(lives, direction), direction["pair_summary"])


# ----------------------------------------------------------------------------------------------------- torch part

def probe(agent, turns, t):
    """The state of the three "Choix :" tokens at block 12, P(R) and P(yes) to the question on the agent's energy, from
    one run of the life up to turn t and that question (under the mask, with the reader)."""
    torch = agent.torch
    turn = [x for x in turns if x["t"] == t][0]
    before = W.life_text(turns, upto=t)
    full = agent.enc(before + W.question_after_choice_line(t, turn["event"], "E", turn.get("other")))
    end = len(agent.enc(before + W.choice_line(t, turn["event"], turn.get("other"))))
    offset = len(full) - end
    mask = torch.tensor(workspace_mask(len(full), (agent.header, end)))[None, None]
    agent.offset, agent.capture = offset, True
    with torch.no_grad():
        logits = agent.model(torch.tensor([full]), attention_mask=mask,
                             logits_to_keep=torch.tensor([len(full) - 1 - offset, len(full) - 1])).logits[0]
    agent.offset, agent.capture = 0, False
    p = torch.softmax(logits.float(), -1)
    r, m = float(p[0, agent.ids_R]), float(p[0, agent.ids_M])
    no, yes = float(p[1, agent.digits[0]]), float(p[1, agent.digits[1]])
    return agent.captured.copy(), r / max(r + m, 1e-12), yes / max(no + yes, 1e-12)


def run(a):
    import torch
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)
    agent = RD.ReaderAgent(a.adapter, a.reader)
    direction_file = out / "direction.json"
    if (out / "lives-direction.jsonl.gz").exists():
        lives = W.read_jsonl(out / "lives-direction.jsonl.gz")
    else:
        s = STREAMS["direction"]
        lives = [W.play(agent, W.world_rng(s, i), W.choice_rng(s, i), others=W.other_rng(s, i))
                 for i in range(a.direction_lives)]
        W.write_jsonl(out / "lives-direction.jsonl.gz", lives)
    log(f"direction lives: survival {np.mean([l['survived'] for l in lives]):.3f}")
    if not direction_file.exists():
        replica = C.replica_check(C.TorchAgent(a.adapter, tail=WS.K), W.read_jsonl(a.mac_lives))
        log(json.dumps({"replica": replica}))
        own, oth = C.choose_pairs(lives), other_pairs(lives)
        W.write_jsonl(out / "pairs-own.jsonl", own)
        W.write_jsonl(out / "pairs-other.jsonl", oth)
        partial = out / "partial-probes.npz"
        rows = {"own": [], "other": []}  # per pair: state change (K x D), dP(R), dP(yes)
        real = {}  # (life, t) -> probe of the real life
        if partial.exists():
            saved = np.load(partial)
            for kind in rows:
                rows[kind] = list(zip(saved[f"{kind}_d"], saved[f"{kind}_r"], saved[f"{kind}_y"]))
            real = {tuple(k): (s_, r_, y_) for k, s_, r_, y_ in
                    zip(saved["real_keys"].tolist(), saved["real_s"], saved["real_r"], saved["real_y"])}
            log(f"resumed after {len(rows['own'])} + {len(rows['other'])} pairs")

        def save():
            arrays = {}
            shape = lambda n: (n, WS.K, -1) if n else (0, WS.K, 1)  # an empty kind (not reached yet) too
            for k in rows:
                arrays[f"{k}_d"] = np.array([r[0] for r in rows[k]], np.float16).reshape(shape(len(rows[k])))
                arrays[f"{k}_r"] = np.array([r[1] for r in rows[k]], np.float32)
                arrays[f"{k}_y"] = np.array([r[2] for r in rows[k]], np.float32)
            keys = list(real)
            arrays["real_keys"] = np.array(keys, int).reshape(-1, 2)
            arrays["real_s"] = np.array([real[k][0] for k in keys], np.float16).reshape(shape(len(keys)))
            arrays["real_r"] = np.array([real[k][1] for k in keys], np.float32)
            arrays["real_y"] = np.array([real[k][2] for k in keys], np.float32)
            W.save_npz(partial, **arrays)
        for kind, pairs in (("own", own), ("other", oth)):
            for i, pair in enumerate(pairs):
                if i < len(rows[kind]):
                    continue
                life = lives[pair["life"]]
                key = (pair["life"], pair["t"])
                if key not in real:
                    s_, r_, y_ = probe(agent, life["turns"], pair["t"])
                    real[key] = (s_.astype(np.float16).astype(np.float32), r_, y_)
                cf = probe(agent, swapped(life, dict(pair, who=kind)), pair["t"])
                # kept in float16 at once, so that a resumed run gives the same numbers as an unbroken one
                rows[kind].append(((cf[0] - real[key][0]).astype(np.float16), cf[1] - real[key][1],
                                   cf[2] - real[key][2]))
                if (i + 1) % 50 == 0:
                    save()
                if (i + 1) % 250 == 0:
                    log(f"  {kind}: {i + 1} pairs")
        states = np.array([real[k][0] for k in sorted(real)], np.float32)
        own_d = np.array([r[0] for r in rows["own"]], np.float32)
        np.savez_compressed(out / "pairs.npz", own=own_d.astype(np.float16),
                            other=np.array([r[0] for r in rows["other"]], np.float16))
        massive = C.massive_dims(states)
        kept = own_d.astype(np.float16).astype(np.float32)
        kept[:, :, massive] = 0
        d_e, d_n = C.fit(own, kept)
        u = C.planes(d_e, d_n)
        per = {k: [{"plane": float(np.linalg.norm(np.einsum("td,tdk->tk", np.asarray(r[0], np.float64), u))),
                    "act": abs(float(r[1])), "say": abs(float(r[2]))} for r in rows[k]] for k in rows}
        W.write_jsonl(out / "pair-measures.jsonl", [dict(kind=k, **m) for k in rows for m in per[k]])
        direction = {"block": C.BLOCK, "massive_dims": massive, "pairs": len(own), "replica": replica,
                     "norm_d_E": np.linalg.norm(d_e, axis=1).tolist(), "norm_d_N": np.linalg.norm(d_n, axis=1).tolist(),
                     "d_E": d_e.tolist(), "d_N": d_n.tolist(),
                     "pair_summary": measures(own, oth, per["own"], per["other"])}
        direction_file.write_text(json.dumps(direction) + "\n")
        if partial.exists():
            partial.unlink()
        log(json.dumps(direction["pair_summary"]))

    def prepare(direction):
        if "reader_replica" in direction:
            return False
        direction["reader_replica"] = RD.reader_replica(agent, json.loads(Path(a.reader_rows).read_text()))
        direction["reader_layers"] = agent.reader_layers
        return True
    WS.run(a, agent=agent, streams=STREAMS, prepare=prepare, judge=judge, other=True)


def check(a):
    out = Path(a.out)
    direction = json.loads((out / "direction.json").read_text())
    lives = W.read_jsonl(out / "lives-direction.jsonl.gz")
    if (C.choose_pairs(lives) != W.read_jsonl(out / "pairs-own.jsonl")
            or other_pairs(lives) != W.read_jsonl(out / "pairs-other.jsonl")):
        raise SystemExit("the pairs differ from those drawn from the direction lives")
    rows = W.read_jsonl(out / "pair-measures.jsonl")
    own_m, oth_m = [r for r in rows if r["kind"] == "own"], [r for r in rows if r["kind"] == "other"]
    own, oth = W.read_jsonl(out / "pairs-own.jsonl"), W.read_jsonl(out / "pairs-other.jsonl")
    if json.loads(json.dumps(measures(own, oth, own_m, oth_m))) != direction["pair_summary"]:
        raise SystemExit("the pair summary differs from the pair measures")
    result = judge(W.read_jsonl(out / "lives-test.jsonl.gz"), direction)
    differs = json.loads(json.dumps(result)) != json.loads((out / "verdicts.json").read_text())
    print("differs:", "yes" if differs else "none")
    if differs:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--adapter", default="artifacts/llm-need/two/need-5-8/adapters-need-8")
    r.add_argument("--reader", default="artifacts/llm-need/two/reader/adapters-reader")
    r.add_argument("--reader-rows", default="artifacts/llm-need/two/reader/reader-replica.json")
    r.add_argument("--mac-lives", default="artifacts/llm-need/two/reader/lives-reader.jsonl.gz")
    r.add_argument("--direction-lives", type=int, default=128)
    r.add_argument("--test-lives", type=int, default=256)
    r.add_argument("--threads", type=int, default=4)
    c = sub.add_parser("verdicts")
    for s in (r, c):
        s.add_argument("--out", default="artifacts/llm-need/two/test")
    a = parser.parse_args(argv)
    {"run": run, "verdicts": check}[a.command](a)


if __name__ == "__main__":
    main()
