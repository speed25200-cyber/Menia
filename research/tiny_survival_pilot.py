"""The pilot before test 31 (docs/TINY_SURVIVAL_PILOT_PLAN.md): which way of learning from survival alone lets the
control B ("actions") learn? Four settings, seeds 100 to 102, the arm "actions" only, measured on 256 pilot worlds
(stream 73) apart from the 256 worlds of measure. The model, masks, tokens and episodes are those of
research/tiny_survival.py; the reward is the same (1 per turn lived after a decision). The choice is numpy only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import tiny_survival as TS

SETTINGS = ("P1", "P2", "P3", "P4")
SEEDS = (100, 101, 102)
PILOT_WORLDS = 73
ARM = "actions"
NEEDED, TIE = 0.65, 0.005
ROOT = "artifacts/tiny-survival-pilot"
CONFIG = {
    "P1": {"algo": "reinforce", "updates": 6000, "lr": 1e-3},
    "P2": {"algo": "a2c", "updates": 4000, "lr": 3e-4, "gamma": 0.9, "epochs": 1},
    "P3": {"algo": "a2c", "updates": 4000, "lr": 3e-4, "gamma": 0.97, "epochs": 1},
    "P4": {"algo": "a2c", "updates": 4000, "lr": 3e-4, "gamma": 0.9, "epochs": 4, "clip": 0.2},
}


def choose(results):
    """results: {setting: {seed: survival}}. The setting where B survives best on average if it reaches NEEDED; at a
    tie within TIE, the simplest (the first in SETTINGS); None if none reaches NEEDED."""
    means = {s: float(np.mean(list(results[s].values()))) for s in SETTINGS if s in results}
    best = max(means.values())
    if best < NEEDED:
        return None, means
    return next(s for s in SETTINGS if s in means and means[s] >= best - TIE), means


# ----------------------------------------------------------------------------------------------------- torch part

def model(seed):
    import torch
    from torch import nn
    torch.manual_seed(int(np.random.default_rng([W.SEED, TS.INIT_STREAM, seed]).integers(2 ** 31)))

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(TS.VOCAB, TS.DIM)
            self.pos = nn.Embedding(TS.LENGTH, TS.DIM)
            layer = nn.TransformerEncoderLayer(TS.DIM, TS.HEADS, 4 * TS.DIM, dropout=0.0, batch_first=True,
                                               norm_first=True)
            self.layers = nn.TransformerEncoder(layer, TS.LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(TS.DIM, TS.VOCAB)
            self.value_head = nn.Linear(TS.DIM, 1)

        def hidden(self, ids, seen):
            x = self.tok(ids) + self.pos(torch.arange(ids.shape[1]))[None]
            return self.layers(x, mask=~seen[:ids.shape[1], :ids.shape[1]])

        def forward(self, ids, seen):
            return self.out(self.hidden(ids, seen))

        def both(self, ids, seen):
            h = self.hidden(ids, seen)
            return self.out(h), self.value_head(h)[..., 0]
    return Tiny()


def train(setting, seed, log):
    import torch
    c = CONFIG[setting]
    net = model(seed)
    seen = torch.tensor(TS.mask(ARM))
    opt = torch.optim.AdamW(net.parameters(), lr=c["lr"])
    curve, window = [], []
    for update in range(c["updates"]):
        worlds = [np.random.default_rng([W.SEED, TS.WORLD_STREAM, seed, update, i]) for i in range(TS.LIVES_PER_UPDATE)]
        rng = np.random.default_rng([W.SEED, TS.ACTION_STREAM, seed, update])
        ids, decisions, length = TS.episodes(net, ARM, worlds, lambda p, last: (rng.random(len(p)) >= p).astype(int))
        life = torch.tensor([d[0] for d in decisions])
        turn = np.array([d[1] for d in decisions])
        position = torch.tensor([d[2] for d in decisions])
        action = torch.tensor([d[3] for d in decisions])
        lived = (length[life.numpy()] - turn).astype(float)  # turns lived after the decision
        if c["algo"] == "reinforce":
            baseline = np.array([lived[turn == t].mean() for t in turn])
            advantage = torch.tensor(lived - baseline, dtype=torch.float32)
            logp = torch.log_softmax(net(ids, seen)[life, position][:, [TS.R, TS.M]], -1)
            chosen = logp.gather(1, action[:, None])[:, 0]
            loss = -(advantage * chosen).mean() - TS.ENTROPY * (-(logp.exp() * logp).sum(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            g = c["gamma"]
            ret = torch.tensor((1 - g ** lived) / (1 - g), dtype=torch.float32)  # sum of g^(k-1), k = 1..lived
            with torch.no_grad():
                logits, value = net.both(ids, seen)
                old = torch.log_softmax(logits[life, position][:, [TS.R, TS.M]], -1).gather(1, action[:, None])[:, 0]
                advantage = ret - value[life, position]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            for _ in range(c["epochs"]):
                logits, value = net.both(ids, seen)
                logp = torch.log_softmax(logits[life, position][:, [TS.R, TS.M]], -1)
                chosen = logp.gather(1, action[:, None])[:, 0]
                if "clip" in c:
                    ratio = (chosen - old).exp()
                    policy = -torch.min(ratio * advantage, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * advantage).mean()
                else:
                    policy = -(advantage * chosen).mean()
                value_loss = ((value[life, position] - ret) ** 2).mean()
                entropy = (-(logp.exp() * logp).sum(-1)).mean()
                loss = policy + 0.5 * value_loss - TS.ENTROPY * entropy
                opt.zero_grad()
                loss.backward()
                opt.step()
        window.append(float((length == W.TURNS).mean()))
        if (update + 1) % 100 == 0:
            curve.append(round(float(np.mean(window)), 4))
            window = []
    log(f"  {setting} seed {seed}: training survival {curve[-1] if curve else None}")
    return net, curve


def measure(net):
    worlds = [W.world_rng(PILOT_WORLDS, i) for i in range(256)]
    greedy = lambda p, last: np.where(p > 0.5, 0, np.where(p < 0.5, 1, 1 - last))
    _, _, length = TS.episodes(net, ARM, worlds, greedy)
    return float((length == W.TURNS).mean())


def run(a):
    import torch
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for setting in a.settings:
        for seed in a.seeds:
            path = out / f"{setting}-{seed}.json"
            if path.exists():
                continue
            net, curve = train(setting, seed, log)
            survival = measure(net)
            path.write_text(json.dumps({"setting": setting, "seed": seed, "arm": ARM, "survival": survival,
                                        "curve": curve, "config": CONFIG[setting]}) + "\n")
            log(f"{setting} seed {seed}: B survival on the pilot worlds {survival:.3f}")


def summary(root):
    results = {}
    for path in sorted(Path(root).glob("P*-*.json")):
        r = json.loads(path.read_text())
        results.setdefault(r["setting"], {})[str(r["seed"])] = r["survival"]
    chosen, means = choose(results)
    return {"chosen": chosen, "means": means, "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--settings", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    r.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    sub.add_parser("choose")
    for s in sub.choices.values():
        s.add_argument("--out", default=ROOT)
    a = parser.parse_args(argv)
    if a.command == "run":
        run(a)
        return
    result = summary(a.out)
    (Path(a.out) / "choice.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"chosen": result["chosen"], "means": result["means"]}))


if __name__ == "__main__":
    main()
