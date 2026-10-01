"""The state of the lack of knowledge, in torch on CPU (docs/LLM_CURIOSITY_PROTOCOL.md): the chooser is Qwen3-0.6B
fp32 with the instinct merged, the reader its second adapter on the question tokens (as research/need_reader.py).

  locate      the gathering block b*: patching on " Cho" "ix" " :" from dashboards where the progress of base rose
  directions  d_mots, d_base, d_calcul, d_suites at b* (1 500 pairs, least squares per token), the plane of the
              need and its means, the random controls; writes interventions.npz for the lives on the Mac
  static      on the decisions of the lives C: the single push (ACT), the reader (READ, SAY), the gain (GAIN), and
              the replicas against the Mac
Resumable step by step. Needs torch; the verdicts (research/curiosity_world.py) are numpy only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import curiosity_world as CW
from .need_causal import massive_dims, random_like
from .need_lora import workspace_mask

PAIRS_LOCATE, MIN_EFFECT, SHARE = 100, 0.10, 0.5
PAIRS, PER_DASHBOARD = 1500, 3
PUSH, TAIL = 2.0, 3


def histories(life):
    """The history before each session of a life, with the session."""
    history = {}
    for s in life["sessions"]:
        yield {d: list(v) for d, v in history.items()}, s
        history.setdefault(s["choice"], []).append((s["before"], s["after"]))


def changed(history, d, units, temperature):
    """The history where the last session on d shows a progress larger by `units` (its after lowered)."""
    h = {k: list(v) for k, v in history.items()}
    before, after = h[d][-1]
    h[d][-1] = (before, max(after - units * temperature, 0.0))
    return h


class Chooser:
    def __init__(self, instinct, reader=None):
        import torch
        from .need_torch import load
        torch.set_grad_enabled(False)
        self.torch = torch
        self.tok, self.model = load(str(instinct))
        self.enc = lambda s: self.tok.encode(s, add_special_tokens=False)
        self.names = [self.enc(" " + d)[0] for d in CW.DOMAINS]
        self.digits = [self.enc("0")[0], self.enc("1")[0]]
        self.header = len(self.enc(CW.HEAD))
        self.layers = self.model.model.layers
        self.offset = 0  # the question tokens at the end of a read, where the reader acts
        self.reader_layers = 0
        if reader:
            self._reader(reader)

    def _reader(self, folder):
        from safetensors.numpy import load_file
        torch = self.torch
        weights = load_file(str(Path(folder) / "adapters.safetensors"))
        scale = json.loads((Path(folder) / "adapter_config.json").read_text())["lora_parameters"]["scale"]
        modules = dict(self.model.named_modules())
        for key in weights:
            if key.endswith(".lora_a"):
                name = key[:-len(".lora_a")]
                a = torch.from_numpy(weights[key].astype(np.float32))
                b = torch.from_numpy(weights[name + ".lora_b"].astype(np.float32))

                def hook(mod, inp, out, a=a, b=b):
                    n = self.offset
                    if not n:
                        return out
                    out = out.clone()
                    out[:, -n:] = out[:, -n:] + scale * ((inp[0][:, -n:] @ a) @ b)
                    return out
                modules[name].register_forward_hook(hook)
                self.reader_layers += 1

    def run(self, ids, block=None, edit=None, capture=None, mask=None, offset=0):
        """Logits at the last position. edit(h_tail) -> new tail on block `block`'s output at the three tokens
        before the last `offset` tokens; capture: a list of blocks whose outputs at those tokens are returned."""
        torch = self.torch
        caught, handles = {}, []
        end = -offset if offset else None
        start = -(TAIL + offset)
        for b in set([block] if edit is not None else []) | set(capture or []):
            def hook(mod, inp, out, b=b):
                h = out[0] if isinstance(out, tuple) else out
                if capture and b in capture:
                    caught[b] = h[0, start:end].clone()
                if edit is not None and b == block:
                    h = h.clone()
                    h[0, start:end] = edit(h[0, start:end])
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            handles.append(self.layers[b].register_forward_hook(hook))
        self.offset = offset
        try:
            kwargs = {} if mask is None else {"attention_mask": mask}
            logits = self.model(torch.tensor([ids]), **kwargs).logits[0, -1]
        finally:
            self.offset = 0
            for h in handles:
                h.remove()
        return torch.softmax(logits.float(), -1), caught

    def choose(self, text, block=None, edit=None, capture=None):
        p, caught = self.run(self.enc(text), block, edit, capture)
        q = np.array([float(p[i]) for i in self.names])
        return q / q.sum(), float(q.sum()), caught

    def p_yes(self, dashboard_text, domain, block=None, edit=None):
        """P(yes) of the reader to « is it <domain> you lack most? », under the mask of the workspace."""
        torch = self.torch
        text = dashboard_text + CW_QUESTION.format(domain)
        ids = self.enc(text)
        end = len(self.enc(dashboard_text))
        mask = torch.tensor(workspace_mask(len(ids), (self.header, end)))[None, None]
        p, _ = self.run(ids, block, edit, mask=mask, offset=len(ids) - end)
        no, yes = float(p[self.digits[0]]), float(p[self.digits[1]])
        return yes / max(no + yes, 1e-12), no + yes


CW_QUESTION = " ? Question : est-ce {} qui te manque le plus ? Réponds 1 pour oui, 0 pour non. Réponse :"


def contexts(pilot):
    """The dashboards of the 24 pilot lives (the direction contexts), with their histories."""
    lives = []
    for name in ("pilot-random.jsonl", "pilot-random-2.jsonl", "pilot-instinct.jsonl"):
        if (Path(pilot) / name).exists():
            lives += CW.read_lives(Path(pilot) / name)
    if (Path(pilot) / "pilot-random-2.jsonl").exists():  # the difficulty changed: the second random lives count
        lives = [l for l in lives if l["life"] >= 100]
    return [(life["life"], h) for life in lives for h, _ in histories(life)]


def locate(a, chooser, cal, log=print):
    out = Path(a.out)
    path = out / "locate.json"
    if path.exists():
        return json.loads(path.read_text())
    ctx = [c for c in contexts(a.pilot) if "base" in c[1]]
    order = np.random.default_rng([CW.SEED, 720]).permutation(len(ctx))
    records, tried = [], 0
    for i in order:
        if len(records) == PAIRS_LOCATE:
            break
        _, h = ctx[i]
        real_text, cf_text = CW.dashboard(h), CW.dashboard(changed(h, "base", 2, cal["temperature"]))
        tried += 1
        if len(chooser.enc(real_text)) != len(chooser.enc(cf_text)):
            raise RuntimeError("a pair of dashboards of different lengths")
        b_idx = CW.DOMAINS.index("base")
        p_real, _, _ = chooser.choose(real_text)
        p_cf, _, caught = chooser.choose(cf_text, capture=list(range(28)))
        if abs(p_cf[b_idx] - p_real[b_idx]) < MIN_EFFECT:
            continue
        restore = []
        for b in range(28):
            p_patch, _, _ = chooser.choose(real_text, block=b, edit=lambda t, b=b: caught[b])
            restore.append(float(p_patch[b_idx] - p_real[b_idx]))
        records.append({"p_real": float(p_real[b_idx]), "p_cf": float(p_cf[b_idx]), "restore": restore})
        if len(records) % 10 == 0:
            log(f"  locate: {len(records)} pairs of {tried} tried")
    total = sum(abs(r["p_cf"] - r["p_real"]) for r in records)
    shares = [sum(np.sign(r["p_cf"] - r["p_real"]) * r["restore"][b] for r in records) / total if total else 0.0
              for b in range(28)]
    block = next((b for b, s in enumerate(shares) if s >= SHARE), None)
    result = {"block": block, "pairs": len(records), "tried": tried, "shares": shares,
              "LOC": bool(block is not None and len(records) >= PAIRS_LOCATE)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "locate-pairs.json").write_text(json.dumps(records) + "\n")
    path.write_text(json.dumps(result, indent=1) + "\n")
    log(json.dumps({k: v for k, v in result.items() if k != "shares"}))
    return result


def directions(a, chooser, cal, block, log=print):
    out = Path(a.out)
    path = out / "interventions.npz"
    if path.exists():
        return dict(np.load(path))
    ctx = [c for c in contexts(a.pilot) if c[1]]
    g = np.random.default_rng([CW.SEED, 720])
    states, x, diffs = [], [], []
    for i in g.permutation(len(ctx)):
        _, h = ctx[i]
        _, _, real = chooser.choose(CW.dashboard(h), capture=[block])
        real = real[block].numpy()
        states.append(real)
        studied = sorted(h)
        for _ in range(min(PER_DASHBOARD, PAIRS - len(diffs))):
            d = studied[int(g.integers(len(studied)))]
            units = [-2, -1, 1, 2][int(g.integers(4))]
            _, _, cf = chooser.choose(CW.dashboard(changed(h, d, units, cal["temperature"])), capture=[block])
            diffs.append(cf[block].numpy() - real)
            row = np.zeros(len(CW.DOMAINS))
            row[CW.DOMAINS.index(d)] = units
            x.append(row)
        if len(diffs) >= PAIRS:
            break
    states, x, diffs = np.array(states), np.array(x), np.array(diffs)
    massive = massive_dims(states)
    kept = diffs.copy()
    kept[:, :, massive] = 0
    # least squares without constant, per token: diffs[:, t, :] ~ x @ D[t]  (D[t] is 4 x dim)
    dirs = np.stack([np.linalg.lstsq(x, kept[:, t, :], rcond=None)[0] for t in range(TAIL)])  # (3, 4, dim)
    plane = np.stack([np.linalg.qr(dirs[t].T)[0] for t in range(TAIL)])  # (3, dim, 4)
    means = np.einsum("ntd,tdk->tk", states, plane) / len(states)
    rng = np.random.default_rng([CW.SEED, 721, 9])
    random = rng.standard_normal(plane.shape)
    random[:, massive, :] = 0
    random_plane = np.stack([np.linalg.qr(random[t])[0] for t in range(TAIL)])
    random_means = np.einsum("ntd,tdk->tk", states, random_plane) / len(states)
    d_base = dirs[:, CW.DOMAINS.index("base"), :]
    random_direction = random_like(np.random.default_rng([CW.SEED, 722]), d_base, 1, massive)[0]
    result = {"block": np.array(block), "plane": plane, "means": means, "random_plane": random_plane,
              "random_means": random_means, "d_base": d_base, "random_direction": random_direction,
              "directions": dirs, "massive": np.array(massive), "pairs": np.array(len(diffs))}
    np.savez_compressed(path, **result)
    log(json.dumps({"pairs": len(diffs), "massive": massive, "norm_d_base": np.linalg.norm(d_base, axis=1).tolist()}))
    return result


def static(a, chooser, cal, inter, log=print):
    """On the decisions of the lives C: the single push (ACT, SAY), the gain (GAIN) and the reader (READ)."""
    out = Path(a.out)
    path = out / "static.json"
    if path.exists():
        return json.loads(path.read_text())
    block = int(inter["block"])
    torch = chooser.torch
    T = lambda v: torch.tensor(np.asarray(v, np.float32))
    push = T(PUSH * inter["d_base"])
    randoms = [T(r) for r in random_like(np.random.default_rng([CW.SEED, 721]), PUSH * inter["d_base"], 3,
                                         [int(m) for m in inter["massive"]])]
    u = inter["d_base"] / np.linalg.norm(inter["d_base"], axis=1, keepdims=True)
    random_u = inter["random_direction"] / np.linalg.norm(inter["random_direction"], axis=1, keepdims=True)
    means = {}
    b_idx = CW.DOMAINS.index("base")
    lives = CW.read_lives(Path(a.lives) / "lives-C.jsonl")
    pilot_states = [chooser.choose(CW.dashboard(h), capture=[block])[2][block].numpy()
                    for _, h in contexts(a.pilot) if h]
    for name, v in (("base", u), ("random", random_u)):
        means[name] = np.einsum("ntd,td->t", np.array(pilot_states), v) / len(pilot_states)

    def gain_edit(v, m):
        vt, mt = T(v), T(m)
        return lambda t: t + ((t * vt).sum(-1) - mt)[:, None] * vt
    rows = []
    for life in lives:
        for h, s in histories(life):
            text = CW.dashboard(h)
            truth = top_need_of(h, cal)
            row = {"life": life["life"], "k": s["k"]}
            if truth is not None:
                row["read"] = {d: chooser.p_yes(text, d)[0] for d in CW.DOMAINS}
                row["truth"] = truth
            p_real, _, _ = chooser.choose(text)
            if "base" in h and p_real[b_idx] <= 0.5:
                p_push, _, _ = chooser.choose(text, block, lambda t: t + push)
                row["act"] = {"none": float(p_real[b_idx]), "push": float(p_push[b_idx]),
                              "random": [float(chooser.choose(text, block, lambda t, r=r: t + r)[0][b_idx])
                                         for r in randoms]}
                row["say"] = {"none": chooser.p_yes(text, "base")[0],
                              "push": chooser.p_yes(text, "base", block, lambda t: t + push)[0],
                              "random": [chooser.p_yes(text, "base", block, lambda t, r=r: t + r)[0] for r in randoms]}
                cf = CW.dashboard(changed(h, "base", 1, cal["temperature"]))
                plain = chooser.choose(cf)[0][b_idx] - p_real[b_idx]
                gain = (chooser.choose(cf, block, gain_edit(u, means["base"]))[0][b_idx]
                        - chooser.choose(text, block, gain_edit(u, means["base"]))[0][b_idx])
                gain_r = (chooser.choose(cf, block, gain_edit(random_u, means["random"]))[0][b_idx]
                          - chooser.choose(text, block, gain_edit(random_u, means["random"]))[0][b_idx])
                row["gain"] = {"S": float(plain), "S_gain": float(gain), "S_random_gain": float(gain_r)}
            rows.append(row)
        log(f"  static: life {life['life']}")
    path.write_text(json.dumps(rows) + "\n")
    return rows


def top_need_of(history, cal):
    need = np.array([CW.progress(history, d) + cal["bonus"] / np.sqrt(1 + len(history.get(d, [])))
                     for d in CW.DOMAINS])
    order = np.argsort(-need)
    return None if np.isclose(need[order[0]], need[order[1]]) else CW.DOMAINS[int(order[0])]


def replicas(a, chooser, log=print):
    """The torch chooser and reader against the Mac's rows (60 dashboards; the reader's validation documents)."""
    install = Path(a.install)
    control = json.loads((install / "instinct-control.json").read_text())
    gaps = [CW.tv_distance(chooser.choose(r["text"])[0], r["probs"]) for r in control["rows"]]
    reader_rows = json.loads((install / "reader-replica.json").read_text())
    reader_gaps = []
    for r in reader_rows:
        dash = r["text"][:r["workspace"]]
        domain = next(d for d in CW.DOMAINS if f"est-ce {d} qui" in r["text"])
        reader_gaps.append(abs(chooser.p_yes(dash, domain)[0] - r["p_yes"]))
    result = {"chooser": {"dashboards": len(gaps), "mean_tv": float(np.mean(gaps))},
              "reader": {"documents": len(reader_gaps), "mean_gap": float(np.mean(reader_gaps))}}
    log(json.dumps(result))
    return result


def run(a, log=print):
    import torch
    torch.set_num_threads(a.threads)
    cal = json.loads((Path(a.pilot) / "pilot.json").read_text())
    install = Path(a.install)
    instinct = install / ("adapters-instinct-4000" if (install / "adapters-instinct-4000").exists()
                          else "adapters-instinct")
    chooser = Chooser(instinct, install / "adapters-reader")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "replicas.json").exists():
        (out / "replicas.json").write_text(json.dumps(replicas(a, chooser, log), indent=1) + "\n")
    located = locate(a, chooser, cal, log)
    if not located["LOC"]:
        log("LOC fails: the rest is not measured")
        return
    inter = directions(a, chooser, cal, located["block"], log)
    if a.step == "static":
        static(a, chooser, cal, inter, log)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--step", choices=("directions", "static"), default="directions")
    r.add_argument("--pilot", default="artifacts/llm-curiosity/pilot")
    r.add_argument("--install", default="artifacts/llm-curiosity/install")
    r.add_argument("--lives", default="artifacts/llm-curiosity/lives")
    r.add_argument("--out", default="artifacts/llm-curiosity/causal")
    r.add_argument("--threads", type=int, default=4)
    a = parser.parse_args(argv)
    run(a)


if __name__ == "__main__":
    main()
