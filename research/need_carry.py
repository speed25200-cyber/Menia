"""A state that lasts: the past seen only through the state (docs/LLM_NEED_CARRY_PROTOCOL.md): verdicts MEM1, MEM2.

Three agents, in torch on CPU: the agent that carries (the final agent and the carrying adapter learned on the Mac,
research/need_mlx.py carry), the control (the final agent alone) — both under the memory mask
(research.need_lora.carry_mask: a turn sees the header, itself, and of each earlier turn only " Cho" "ix" " :" and its
action) — and the final agent without mask (reference). Survival: 128 new lives each, the same worlds and draws.
Carried information: pairs "calme" -> "tu cours" at a past turn j of the carrying agent's lives, t - j >= 2, P(R) at t
read on the real and the changed text. Resumable. The verdicts are numpy only; living needs torch.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_verdicts as V
from . import need_causal as C
from . import need_lora as NL
from .need_persistence import boot

LIVES = 128
STREAM = 40
PAIR_STREAM = 42
PAIRS = 300
MIN_PAIRS = 150
PER_DECISION = 3
MIN_GAP = 2
SWAP = [0, 1]  # "calme" -> "tu cours": energy -2 at the swapped turn
SURVIVAL = 0.40
SURVIVAL_GAIN = 0.15
EFFECT = 0.05
BEYOND = 0.03
REPLICA_TOLERANCE = 0.02
CACHE_TOLERANCE = 1e-4
MASK_TOLERANCE = 1e-5
AGENTS = ("carry", "control", "free")
ROOT = "artifacts/llm-need/carry"
FINAL = "artifacts/llm-need/final/report/adapters-final"


def choose_pairs(lives, limit=PAIRS):
    """In the order of the lives and their decisions, up to PER_DECISION pairs per decision drawn from the stream
    [SEED, PAIR_STREAM, 0]: a past "calme" at j turned into "tu cours", t - j >= MIN_GAP, the agent alive all along
    (research.need_causal.candidate_pairs, exact replay)."""
    rng = np.random.default_rng([W.SEED, PAIR_STREAM, 0])
    out = []
    for index, life in enumerate(lives):
        for turn in W.decisions(life):
            cands = [c for c in C.candidate_pairs(life, turn) if c["swap"] == SWAP and turn["t"] - c["j"] >= MIN_GAP]
            for k in rng.permutation(len(cands))[:PER_DECISION]:
                out.append(dict(cands[k], life=index, t=turn["t"]))
                if len(out) == limit:
                    return out
    return out


def verdicts(lives, rows, setup):
    """lives: {agent: 128 lives}; rows: one per pair, {"life", "t", "j", agent: [P(R) real, P(R) changed]}."""
    surv = {a: float(np.mean([l["survived"] for l in lives[a]])) for a in AGENTS if a in lives}
    gain = V.paired_survival(lives["carry"], lives["control"])
    life = [r["life"] for r in rows]
    effect = {a: np.array([r[a][1] - r[a][0] for r in rows]) for a in AGENTS}
    carry = boot(effect["carry"], life)
    beyond = boot(effect["carry"] - effect["control"], life)
    passes = lambda d, th: d["mean"] >= th and d["low"] > 0
    mass = float(np.mean([t["mass"] for l in lives["carry"] for t in W.decisions(l)]))
    out = {"MEM1": bool(surv["carry"] >= SURVIVAL and passes(gain, SURVIVAL_GAIN)),
           "MEM2": bool(passes(carry, EFFECT) and passes(beyond, BEYOND))}
    out["valid"] = bool(setup["replica"]["mean_gap"] <= REPLICA_TOLERANCE and setup["cache_gap"] <= CACHE_TOLERANCE
                        and setup["mask_gap"] <= MASK_TOLERANCE and len(rows) >= MIN_PAIRS and mass >= 0.5)
    out["global"] = bool(out["valid"] and out["MEM1"] and out["MEM2"])
    gaps = sorted({r["t"] - r["j"] for r in rows})
    by_gap = {str(g): {"pairs": int(sum(r["t"] - r["j"] == g for r in rows)),
                       "carry": float(np.mean([e for e, r in zip(effect["carry"], rows) if r["t"] - r["j"] == g])),
                       "control": float(np.mean([e for e, r in zip(effect["control"], rows) if r["t"] - r["j"] == g]))}
              for g in gaps}
    values = {"survival": surv, "survival_gain": gain, "effect": {a: boot(effect[a], life) for a in AGENTS},
              "beyond_control": beyond, "by_gap": by_gap, "pairs": len(rows), "mass_carry": mass,
              "turns_need_at_most_2": {a: W.low_turns(lives[a]) for a in AGENTS}}
    return {"verdicts": out, "values": values}


# ----------------------------------------------------------------------------------------------------- torch part

class MaskedAgent:
    """start/decide/commit over a key-value cache under the memory mask; the final agent, with the carrying adapter
    merged on top when given (it was learned on the fused final agent)."""

    def __init__(self, adapter=None):
        import torch
        from safetensors.numpy import load_file
        from transformers import DynamicCache
        from .need_torch import load
        self.torch, self.Cache = torch, DynamicCache
        self.tok, self.model = load(FINAL)
        if adapter:
            weights = load_file(str(Path(adapter) / "adapters.safetensors"))
            scale = json.loads((Path(adapter) / "adapter_config.json").read_text())["lora_parameters"]["scale"]
            modules = dict(self.model.named_modules())
            for key in weights:
                if key.endswith(".lora_a"):
                    name = key[:-len(".lora_a")]
                    a, b = weights[key], weights[name + ".lora_b"]
                    modules[name].weight += torch.from_numpy(scale * (a @ b).T.astype(np.float32))
        enc = lambda s: self.tok.encode(s, add_special_tokens=False)
        self.R, self.M = enc(" R")[0], enc(" M")[0]
        self.calls = self.resets = 0

    def probability(self, logits):
        p = self.torch.softmax(logits.float(), -1)
        r, m = float(p[self.R]), float(p[self.M])
        return r / max(r + m, 1e-12), r + m

    def whole(self, text, hook=None):
        """P(R) and mass at the end of text, run at once under the mask; hook(layer 0 output) if given."""
        torch = self.torch
        ids, turn, carried = NL.carry_layout(text, self.tok)
        mask = torch.tensor(NL.carry_mask(turn, carried))[None, None]
        handle = self.model.model.layers[0].register_forward_hook(hook) if hook else None
        try:
            with torch.no_grad():
                logits = self.model(torch.tensor([ids]), attention_mask=mask).logits[0, -1]
        finally:
            if handle:
                handle.remove()
        return self.probability(logits)

    def start(self, header):
        self.cache, self.ids, self.text, self.pending = self.Cache(), [], header, ""

    def decide(self, line):
        torch = self.torch
        text = self.text + self.pending + line
        ids, turn, carried = NL.carry_layout(text, self.tok)
        if ids[:len(self.ids)] != self.ids:
            self.resets += 1
            self.cache, self.ids = self.Cache(), []
        past = len(self.ids)
        mask = torch.tensor(NL.carry_mask(turn, carried)[past:])[None, None]
        self.calls += 1
        with torch.no_grad():
            logits = self.model(torch.tensor([ids[past:]]), past_key_values=self.cache, attention_mask=mask,
                                use_cache=True).logits[0, -1]
        self.text, self.ids, self.pending = text, ids, ""
        return self.probability(logits)

    def commit(self, action):
        self.pending += " " + W.ACTIONS[action] + "\n"


class FreeAgent:
    """The final agent without mask (the reference), with its P(R) on a whole text."""

    def __init__(self):
        self.agent = C.TorchAgent(FINAL)

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def whole(self, text):
        return self.agent.base.p_r(text)[0], None


def checks(agents, carry_lives, rows_file, log):
    """The replica against the Mac, the cache against the whole text, and the mask (block 0 of later turns does not
    see a past event)."""
    carry = agents["carry"]
    rows = json.loads(Path(rows_file).read_text())
    gaps = [abs(carry.whole(r["text"])[0] - r["p_R"]) for r in rows]
    replica = {"decisions": len(gaps), "mean_gap": float(np.mean(gaps)), "max_gap": float(np.max(gaps))}
    life = carry_lives[0]
    cache = []
    carry.start(W.HEADER)
    for turn in W.decisions(life)[:4]:
        line = W.choice_line(turn["t"], turn["event"])
        p = carry.decide(line)[0]
        cache.append(abs(p - carry.whole(carry.text)[0]))
        carry.commit(turn["action"])
    turns = life["turns"]
    t = W.decisions(life)[5]["t"]
    first = turns[0]
    other = next(k for k in range(len(W.EVENTS)) if k != first["event"]
                 and len(carry.tok.encode(W.event_line(1, k), add_special_tokens=False))
                 == len(carry.tok.encode(W.event_line(1, first["event"]), add_special_tokens=False)))
    changed = [dict(first, event=other)] + turns[1:]
    outs = []
    for ts in (turns, changed):
        caught = {}
        carry.whole(C.decision_text(ts, t), hook=lambda m, i, o: caught.update(h=(o[0] if isinstance(o, tuple)
                                                                                  else o)[0].clone()))
        outs.append(caught["h"])
    ids, turn_of, _ = NL.carry_layout(C.decision_text(turns, t), carry.tok)
    if len(ids) != len(NL.carry_layout(C.decision_text(changed, t), carry.tok)[0]):
        raise RuntimeError("the changed event changes the number of tokens")
    later = [j for j, k in enumerate(turn_of) if k >= 1]
    mask_gap = float((outs[0][later] - outs[1][later]).abs().max())
    setup = {"replica": replica, "cache_gap": float(max(cache)), "mask_gap": mask_gap}
    log(json.dumps(setup))
    return setup


def measure(a):
    import torch
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    out = Path(a.out) / "test"
    out.mkdir(parents=True, exist_ok=True)
    agents = {"carry": MaskedAgent(Path(a.out) / "adapters-carry"), "control": MaskedAgent(), "free": FreeAgent()}
    lives = {}
    for name in AGENTS:
        final, partial = out / f"lives-{name}.jsonl.gz", out / f"partial-lives-{name}.jsonl.gz"
        if final.exists():
            lives[name] = W.read_jsonl(final)
            continue
        done = W.read_jsonl(partial) if partial.exists() else []
        for i in range(len(done), LIVES):
            done.append(W.play(agents[name], W.world_rng(STREAM, i), W.choice_rng(STREAM, i)))
            if (i + 1) % 8 == 0:
                W.write_jsonl(partial, done)
            if (i + 1) % 32 == 0:
                log(f"  {name}: {i + 1} lives")
        W.write_jsonl(final, done)
        if partial.exists():
            partial.unlink()
        lives[name] = done
        log(f"{name}: survival {np.mean([l['survived'] for l in done]):.3f}")
    setup_file = out / "setup.json"
    if not setup_file.exists():
        setup_file.write_text(json.dumps(checks(agents, lives["carry"], Path(a.out) / "carry-replica.json", log),
                                         indent=1) + "\n")
    setup = json.loads(setup_file.read_text())
    pairs_file = out / "pairs.jsonl.gz"
    if not pairs_file.exists():
        pairs = choose_pairs(lives["carry"])
        rows = []
        for k, pair in enumerate(pairs):
            life = lives["carry"][pair["life"]]
            real, changed = C.decision_text(life["turns"], pair["t"]), C.decision_text(C.swapped(life, pair), pair["t"])
            if len(agents["carry"].tok.encode(real)) != len(agents["carry"].tok.encode(changed)):
                raise RuntimeError("a swap changes the number of tokens")
            rows.append(dict(pair, **{n: [round(agents[n].whole(x)[0], 6) for x in (real, changed)] for n in AGENTS}))
            if (k + 1) % 50 == 0:
                log(f"  {k + 1} pairs")
        W.write_jsonl(pairs_file, rows)
    rows = W.read_jsonl(pairs_file)
    result = verdicts(lives, rows, setup)
    log(json.dumps(result["verdicts"]))


def gather(root):
    out = Path(root) / "test"
    lives = {n: W.read_jsonl(out / f"lives-{n}.jsonl.gz") for n in AGENTS}
    rows = W.read_jsonl(out / "pairs.jsonl.gz")
    if [{k: r[k] for k in ("life", "t", "j", "swap", "dE", "dN")} for r in rows] != \
            [{k: p[k] for k in ("life", "t", "j", "swap", "dE", "dN")} for p in choose_pairs(lives["carry"])]:
        raise SystemExit("the pairs are not those the protocol draws from the carrying agent's lives")
    setup = json.loads((out / "setup.json").read_text())
    return json.loads(json.dumps(dict(verdicts(lives, rows, setup), setup=setup)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=4)
    sub.add_parser("write")
    sub.add_parser("verdicts")
    for s in sub.choices.values():
        s.add_argument("--out", default=ROOT)
    a = parser.parse_args(argv)
    if a.command == "run":
        measure(a)
        return
    result = gather(a.out)
    path = Path(a.out) / "test" / "verdicts.json"
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
