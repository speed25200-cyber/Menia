"""Train source monitors from feedback that arrives after exploratory checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from menia.source_monitor import SourceMonitor
from menia.source_environment import CONDITIONS, streams
from .evaluate_source_monitor import training_features, full_evaluation, baselines


def torch_forward(parameters, x, kind):
    import torch
    p = parameters
    state = torch.zeros((x.shape[1], 40 if kind == "window" else 16), dtype=x.dtype)
    outputs = []
    for frame in x:
        if kind == "window":
            state = torch.cat((state[:, 5:], frame), dim=1)
            logit = torch.tanh(state @ p["W1"]+p["b1"]) @ p["W2"]+p["b2"]
        else:
            if kind == "reset":
                state = torch.zeros_like(state)
            gate = torch.sigmoid(frame @ p["G"]+p["bg"])
            candidate = torch.tanh(frame @ p["W"]+state @ p["U"]+p["b"])
            state = (1-gate)*state+gate*candidate
            logit = state @ p["V"]+p["bo"]
        outputs.append(logit)
    return torch.stack(outputs)


def train(kind, seed, x, visible, mask):
    import torch
    torch.manual_seed(seed)
    model = SourceMonitor(kind, seed)
    parameters = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32)) for k, v in model.p.items()}
    x, y, mask = [torch.tensor(v, dtype=torch.float32) for v in (x, visible, mask)]
    optimizer = torch.optim.Adam(parameters.values(), lr=.003)
    losses = []
    for step in range(300):
        selected = torch.randint(x.shape[1], (32,))
        logits = torch_forward(parameters, x[:, selected], kind)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, y[:, selected], reduction="none")
                *mask[:, selected]).sum()/mask[:, selected].sum().clamp(min=1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(parameters.values()), 5.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % 100 == 0:
            print(f"{kind} seed={seed}: update {step+1}/300", flush=True)
    model.p = {k: v.detach().numpy().astype(float) for k, v in parameters.items()}
    return model, losses


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frames, truth = streams(41000, batch=512)
    mask = np.random.default_rng(41001).random(truth.shape) < .5
    x, visible = training_features(frames, truth, mask)
    report = {"status": "training", "scope": "learned synthetic source inference; subjective experience and novelty unestablished",
              "versions": {"torch": torch.__version__, "numpy": np.__version__},
              "training": {"episodes": 512, "length": 48, "seed": 41000,
                           "verified_targets": int(mask.sum()), "updates": 300, "batch": 32,
                           "learning_rate": .003, "gradient_clip": 5.},
              "models": [], "baselines": {c: baselines(c, 51000+i*1000) for i, c in enumerate(CONDITIONS)}}
    root = Path(__file__).resolve().parents[1]
    sources = ("menia/source_monitor.py", "menia/source_environment.py", "research/train_source_monitor.py",
               "research/evaluate_source_monitor.py", "docs/LEARNED_SOURCE_PROTOCOL.md")
    report["source_sha256"] = {p: hashlib.sha256((root/p).read_text(encoding="utf-8").encode()).hexdigest() for p in sources}
    def save():
        (args.out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    save()
    for kind in ("recurrent", "reset", "window"):
        for seed in (11, 23, 37):
            start = time.perf_counter()
            model, losses = train(kind, seed, x, visible, mask)
            filename = f"{kind}-{seed}.json"
            model.save(args.out/filename)
            report["models"].append({"kind": kind, "seed": seed, "checkpoint": filename,
                "checkpoint_sha256": hashlib.sha256((args.out/filename).read_bytes()).hexdigest(),
                "parameters": sum(v.size for v in model.p.values()),
                "parameters_used": sum(v.size for k, v in model.p.items() if kind != "reset" or k != "U"),
                "training_losses": losses, "evaluation": full_evaluation(model)})
            save()
            print(f"{kind} seed={seed} saved and evaluated in {time.perf_counter()-start:.1f}s", flush=True)
    report["status"] = "completed"
    save()


if __name__ == "__main__":
    main()
