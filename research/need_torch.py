"""Exploratory (not pre-registered, docs/LLM_NEED_RESULTS.md): the final need agent in torch on CPU, to find where
the need that drives the choice is carried. Qwen3-0.6B fp32 with the mlx LoRA adapter merged (W += scale * (A @ B).T);
its choice probabilities match the Mac's to 0.003 on test decisions. Needs torch, transformers and safetensors (not in
the CI requirements)."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from safetensors.numpy import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import need_world as W

SNAP = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
torch.set_grad_enabled(False)


def load(adapter=None, scale=20.0):
    tok = AutoTokenizer.from_pretrained(SNAP)
    model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.float32)
    model.eval()
    if adapter:
        w = load_file(adapter + "/adapters.safetensors")
        mods = dict(model.named_modules())
        for k in w:
            if k.endswith(".lora_a"):
                name = k[:-len(".lora_a")]
                a, b = w[k], w[name + ".lora_b"]
                mods[name].weight += torch.from_numpy(scale * (a @ b).T.astype(np.float32))
    return tok, model


class Agent:
    def __init__(self, adapter=None):
        self.tok, self.model = load(adapter)
        self.enc = lambda s: self.tok.encode(s, add_special_tokens=False)
        self.R, self.M = self.enc(" R")[0], self.enc(" M")[0]
        self.layers = self.model.model.layers
        self.hooks = []

    def p_r(self, text, patch=None, capture=None):
        """P(R) after text. patch: {layer: (positions, tensor[len(positions), D])} replaces the block output there.
        capture: list of layers whose outputs are returned."""
        ids = torch.tensor([self.enc(text)])
        caught, handles = {}, []
        for L in set((patch or {}).keys()) | set(capture or []):
            def hook(mod, inp, out, L=L):
                h = out[0] if isinstance(out, tuple) else out
                if capture and L in capture:
                    caught[L] = h[0].clone()
                if patch and L in patch:
                    pos, val = patch[L]
                    h = h.clone()
                    h[0, pos] = val
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            handles.append(self.layers[L].register_forward_hook(hook))
        try:
            logits = self.model(ids).logits[0, -1]
        finally:
            for hd in handles:
                hd.remove()
        p = torch.softmax(logits.float(), -1)
        pr = (p[self.R] / (p[self.R] + p[self.M])).item()
        return pr, caught, ids.shape[1]


def context_text(life, t):
    turn = [x for x in life["turns"] if x["t"] == t][0]
    return W.life_text(life["turns"], upto=t) + W.choice_line(t, turn["event"]), turn


def patching(a):
    """Changes one past \"calme\" (since the last recharge) into \"tu cours\" and, block by block, puts the
    changed run's activations back into the intact run on three groups of positions; the share of the effect restored."""
    ag = Agent(a.adapter)
    lives = W.read_jsonl(a.lives)
    CALME, COURS = 0, 1
    rng = np.random.default_rng(0)
    cands = []
    for li, life in enumerate(lives):
        ds = [x for x in life["turns"] if not x.get("dead")]
        for turn in ds:
            t = turn["t"]
            if t < 6 or turn["E"] < 5:
                continue
            last_r = max([x["t"] for x in ds if x["t"] < t and x["action"] == 0], default=0)
            js = [x["t"] for x in ds if last_r < x["t"] < t and x["event"] == CALME]
            if js:
                cands.append((li, t, int(rng.choice(js))))
    rng.shuffle(cands)
    LAYERS = list(range(0, 28, 3)) + [27]
    out = []
    t0 = time.time()
    for li, t, j in cands:
        if len(out) >= 30:
            break
        life = lives[li]
        turns = life["turns"]
        cur = [x for x in turns if x["t"] == t][0]
        clean = W.life_text(turns, upto=t) + W.choice_line(t, cur["event"])
        cf_turns = [dict(x, event=COURS) if x["t"] == j else x for x in turns]
        corr = W.life_text(cf_turns, upto=t) + W.choice_line(t, cur["event"])
        p_clean, c_clean, n = ag.p_r(clean, capture=LAYERS)
        p_corr, c_corr, n2 = ag.p_r(corr, capture=LAYERS)
        assert n == n2
        if abs(p_corr - p_clean) < 0.2:
            continue
        # token positions: event j word, history after j, current line
        pre_j = ag.enc(W.life_text(turns, upto=j) + f"Tour {j} :")
        ev = list(range(len(pre_j), len(pre_j) + 2))
        line_start = len(ag.enc(W.life_text(turns, upto=t)))
        after_j = list(range(ev[-1] + 1, line_start))
        current = list(range(line_start, n))
        groups = {"event": ev, "between": after_j, "current": current}
        rec = {"life": li, "t": t, "j": j, "p_clean": p_clean, "p_corr": p_corr, "restore": {}}
        for L in LAYERS:
            for g, pos in groups.items():
                p, _, _ = ag.p_r(clean, patch={L: (pos, c_corr[L][pos])})
                rec["restore"][f"{L}:{g}"] = (p - p_clean) / (p_corr - p_clean)
        out.append(rec)
        print(len(out), li, t, j, round(p_clean, 3), round(p_corr, 3), round(time.time() - t0), flush=True)
    Path(a.out).write_text(json.dumps(out) + "\n")
    for g in ("event", "between", "current"):
        print(g, [(L, round(float(np.mean([r["restore"][f"{L}:{g}"] for r in out])), 2)) for L in LAYERS])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="artifacts/llm-need/final/report/adapters-final")
    parser.add_argument("--lives", default="artifacts/llm-need/test-final/test/lives-test-final.jsonl.gz")
    parser.add_argument("--out", default="artifacts/llm-need/explore/patching.json")
    patching(parser.parse_args())
