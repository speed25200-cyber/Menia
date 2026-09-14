"""Train and compare real weights on a small, explicitly synthetic task."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from .recurrent import RecurrentMemory, episodes, metrics


def train(seed=17, steps=300, hidden=32):
    if steps < 1:
        raise ValueError("steps must be positive")
    model = RecurrentMemory(hidden=hidden, seed=seed)
    momentum = {k: np.zeros_like(v) for k, v in model.p.items()}
    variance = {k: np.zeros_like(v) for k, v in model.p.items()}
    log = []
    for n in range(1, steps+1):
        x, y, mask = episodes(seed*100000+n, batch=64, length=12)
        loss, grads = model.loss_and_grad(x, y, mask)
        norm = np.sqrt(sum((g*g).sum() for g in grads.values()))
        scale = min(1, 1/max(float(norm), 1e-12))
        for key, parameter in model.p.items():
            g = grads[key]*scale
            momentum[key] = .9*momentum[key] + .1*g
            variance[key] = .999*variance[key] + .001*g*g
            parameter -= .005*(momentum[key]/(1-.9**n))/(np.sqrt(variance[key]/(1-.999**n))+1e-8)
        if n == 1 or n % 50 == 0:
            log.append({"step": n, "train_cross_entropy": loss})
    return model, log


def evaluate(model, seed=910001):
    # Different seeds and longer sequences; still the SAME synthetic family.
    x, y, mask = episodes(seed, batch=256, length=32)
    full, _ = model.forward(x)
    reset, _ = model.forward(x, reset_each_step=True)
    none, _ = model.forward(np.zeros_like(x))
    rule = np.empty_like(full)
    last = np.zeros(x.shape[1], dtype=int)
    for t, obs in enumerate(x):
        last = np.where(obs[:,4] > .5, obs[:,:4].argmax(-1), last)
        rule[t] = np.eye(4)[last]
    return {"learned": metrics(full,y,mask), "reset_state": metrics(reset,y,mask),
            "no_observations": metrics(none,y,mask), "constant_class_zero": metrics(np.broadcast_to([1,0,0,0],full.shape),y,mask),
            "last_observation_rule": metrics(rule,y,mask)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--seeds", type=int, nargs="+", default=[17,29,43])
    a = p.parse_args()
    root = Path(a.out)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use an empty destination to preserve prior experiments")
    root.mkdir(parents=True, exist_ok=True)
    results = []
    start = time.perf_counter()
    for seed in a.seeds:
        model, history = train(seed=seed, steps=a.steps)
        checkpoint = root/f'memory-seed-{seed}.json'
        model.save(checkpoint, {"seed":seed,"steps":a.steps,"task":"synthetic delayed symbol recall",
            "subjective_consciousness":"not established", "train_length":12})
        # Evaluate reloaded weights, not only the in-memory model.
        result = {"seed":seed,"parameters":model.parameter_count,
            "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training":history,"evaluation":evaluate(RecurrentMemory.load(checkpoint))}
        results.append(result)
        print(json.dumps({"seed":seed,"evaluation":result["evaluation"]}), flush=True)
    report = {"status":"executed CPU experiment", "python":platform.python_version(),
              "numpy":np.__version__, "wall_seconds":time.perf_counter()-start,
              "test_seed":910001,"test_length":32,"episodes_per_seed":256,
              "limitations":["synthetic family only","not a language model","not a consciousness test",
                             "deterministic memory rule is a strong reference","not integrated with iPhone chat"],
              "runs":results}
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Cross-platform one-step reference vectors for the mobile runner.
    model = RecurrentMemory.load(root/f'memory-seed-{a.seeds[0]}.json')
    x,y,mask = episodes(2026,batch=1,length=8)
    probs,_ = model.forward(x)
    (root/'mobile-fixture.json').write_text(json.dumps({"inputs":x[:,0,:].tolist(),
        "probabilities":probs[:,0,:].tolist(),"checkpoint":f'memory-seed-{a.seeds[0]}.json'},indent=2)+'\n')

if __name__ == '__main__':
    main()
