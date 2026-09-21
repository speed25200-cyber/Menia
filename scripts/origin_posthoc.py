"""Exploratory analyses added after reading the pre-registered results (documented as such)."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.origin_neural import WorldModel, replay_states, probe_analysis, fit_probe, probe_predict
from research.origin_env import LIFE, N_MOVE
root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/origin-inquiry")
def lives(cond, seed, pol):
    return [json.loads(l) for l in (root / f"lives-{cond}-{seed}-{pol}.jsonl").read_text().splitlines()]
print("1. Sonde sur les vies P-aucune en T (corps appris en bougeant, sans enquête)")
for seed in (17, 29, 43):
    model = WorldModel.load(root / f"model-T-{seed}.json")
    L = lives("T", seed, "none"); S = np.asarray([replay_states(model, l) for l in L])
    pr = probe_analysis(S, L)
    print(f"   graine {seed}: exactitude finale {pr['final_accuracy']:.3f}, stabilité {pr['stability']:.3f}, premier décodage stable médian {pr['median_first_stable_step']}")
print("2. Vies instables de P-soi en T et première marque mensongère")
for seed in (17, 29, 43):
    model = WorldModel.load(root / f"model-T-{seed}.json")
    L = lives("T", seed, "self"); S = np.asarray([replay_states(model, l) for l in L])
    d = np.array([l["d"] for l in L]); train = np.arange(len(L)) % 2 == 0; test = ~train
    W = fit_probe(S[train].reshape(-1, S.shape[-1]), np.repeat(d[train], LIFE))
    pred = probe_predict(W, S[test].reshape(-1, S.shape[-1])).reshape(test.sum(), LIFE)
    correct = pred == d[test][:, None]
    unstable = unstable_lied = 0
    for row, life in zip(correct, [l for l, t in zip(L, test) if t]):
        lied = None
        for a, nxt in zip(life["actions"], life["obs"][1:]):
            if a == N_MOVE:
                lied = nxt[7] != life["d"]; break
        idx = np.flatnonzero(row)
        if len(idx) and not row[idx[0]:].all():
            unstable += 1; unstable_lied += bool(lied)
    print(f"   graine {seed}: {unstable} vies instables sur {int(test.sum())}, dont {unstable_lied} où la première marque mentait")
print("3. Moment des inspections, P-soi en T")
for seed in (17, 29, 43):
    L = lives("T", seed, "self")
    before = [next((t for t, a in enumerate(l["actions"]) if a < N_MOVE), LIFE) for l in L]
    after = [sum(1 for a in l["actions"][b:] if a >= N_MOVE) for l, b in zip(L, before)]
    print(f"   graine {seed}: avant le premier mouvement {np.mean(before):.2f} (max {max(before)}), après {np.mean(after):.2f}")
