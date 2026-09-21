"""Exploratory analyses added after reading the learned-inquiry results (documented as such)."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.origin_env import Atelier, LIFE, N_MOVE
from research.origin_neural import WorldModel
from research.origin_rl import QNetwork, features
root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/learned-inquiry")
world = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/origin-inquiry")
runs = []
for part in sorted(root.glob("report-rl-*.json")):
    runs.extend(json.loads(part.read_text())["runs"])
def lives(r):
    name = f"{r['condition']}-{r['seed']}-{r['reward']}-{r['beta']:g}"
    return [json.loads(l) for l in (root / f"lives-{name}.jsonl").read_text().splitlines()]
print("1. Où tombent les inspections résiduelles avec β = 0 ? (pas médian, part avant le premier mouvement)")
for r in runs:
    if r["reward"] == "prudence" and r["beta"] == 0:
        steps, before = [], 0
        for life in lives(r):
            first_move = next((t for t, a in enumerate(life["actions"]) if a < N_MOVE), LIFE)
            for t, a in enumerate(life["actions"]):
                if a >= N_MOVE:
                    steps.append(t); before += t < first_move
        print(f"   {r['condition']} {r['seed']}: {len(steps)} inspections, pas médian {np.median(steps) if steps else None}, avant le premier mouvement {before}")
print("2. Enquête apprise en T : inspections avant / après le premier mouvement, par bras")
for r in sorted(runs, key=lambda r: (r["seed"], r["reward"], r["beta"])):
    if r["condition"] != "T":
        continue
    b, a, mark_before = [], [], 0
    for life in lives(r):
        first_move = next((t for t, x in enumerate(life["actions"]) if x < N_MOVE), LIFE)
        b.append(sum(1 for x in life["actions"][:first_move] if x >= N_MOVE))
        a.append(sum(1 for x in life["actions"][first_move:] if x >= N_MOVE))
        mark_before += sum(1 for x in life["actions"][:first_move] if x == N_MOVE)
    print(f"   graine {r['seed']} {r['reward']:>11} β={r['beta']:g}: avant {np.mean(b):.2f} (dont marque {mark_before / len(b):.2f}), après {np.mean(a):.2f}")
print("3. Paralysie : valeurs Q au premier pas, meilleur mouvement contre meilleure inspection")
for r in runs:
    if r["reward"] == "prudence" and r["beta"] in (3.0, 10.0) and r["condition"] in ("T", "C3"):
        model = WorldModel.load(world / f"model-{r['condition']}-{r['seed']}.json")
        q = QNetwork.load(root / f"policy-{r['condition']}-{r['seed']}-{r['reward']}-{r['beta']:g}.json")
        gaps = []
        for n in range(50):
            env = Atelier(r["condition"], 910001 * 1000003 + n); obs = env.reset()
            values, _ = q.forward(features(model.zero(), obs, 0))
            gaps.append(values[0][:N_MOVE].max() - values[0][N_MOVE:].max())
        print(f"   {r['condition']} {r['seed']} β={r['beta']:g}: Q(meilleur mouvement) − Q(meilleure inspection) au pas 0 : moyenne {np.mean(gaps):+.2f}, min {np.min(gaps):+.2f}, max {np.max(gaps):+.2f}")
