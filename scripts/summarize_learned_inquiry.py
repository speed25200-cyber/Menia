"""Compact reading of artifacts/learned-inquiry and the dose-effect figure."""
import json
from pathlib import Path
import sys
root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/learned-inquiry")
runs = []
for part in sorted(root.glob("report-rl-*.json")):
    value = json.loads(part.read_text())
    runs.extend(value["runs"])
    print(part.name, "wall %.0fs" % value["wall_seconds"], "numpy", value["numpy"])
print("\n== condition seed reward beta | insp/life | share k0..k3 | hits/life | moves/life | mark reads/life")
for r in sorted(runs, key=lambda r: (r["condition"], r["seed"], r["reward"], r["beta"])):
    e = r["evaluation"]
    print(f"{r['condition']:>2} {r['seed']:>2} {r['reward']:>11} {r['beta']:>4g} | {e['per_life']:5.2f} | {['%.2f' % s for s in e['share']]} | {e['hits_per_life']:5.2f} | {e['moves_per_life']:5.2f} | {e['per_life'] * e['share'][0]:.2f}")
print("\n== trajectories (episode, insp/life, share0, hits) for T prudence 3")
for r in runs:
    if r["condition"] == "T" and r["reward"] == "prudence" and r["beta"] == 3.0:
        print(r["seed"], [(c["episode"], round(c["per_life"], 2), round(c["share"][0], 2), round(c["hits_per_life"], 1)) for c in r["trajectory"]])
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    betas = [0.0, 0.3, 1.0, 3.0, 10.0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for seed, color in zip((17, 29, 43), ("#1f77b4", "#2ca02c", "#9467bd")):
        sel = {r["beta"]: r for r in runs if r["condition"] == "T" and r["reward"] == "prudence" and r["seed"] == seed}
        x = [b for b in betas if b in sel]
        axes[0].plot(range(len(x)), [sel[b]["evaluation"]["per_life"] * sel[b]["evaluation"]["share"][0] for b in x], marker="o", color=color, label=f"graine {seed}")
        axes[1].plot(range(len(x)), [sel[b]["evaluation"]["hits_per_life"] for b in x], marker="o", color=color)
    for ax in axes[:2]:
        ax.set_xticks(range(len(betas))); ax.set_xticklabels([f"{b:g}" for b in betas]); ax.set_xlabel("β prudence"); ax.grid(alpha=0.3)
    axes[0].set_title("Lectures de la marque par vie, T", fontsize=10); axes[0].legend(fontsize=8)
    axes[1].set_title("Hits par vie, T", fontsize=10)
    labels, values, colors = [], [], []
    for cond, kind, beta, color in (("T", "prudence", 3.0, "#1f77b4"), ("T", "information", 3.0, "#ff7f0e"), ("T", "surprise", 3.0, "#8c564b"),
                                    ("C1", "prudence", 3.0, "#7f7f7f"), ("C3", "prudence", 3.0, "#d62728")):
        sel = [r for r in runs if r["condition"] == cond and r["reward"] == kind and r["beta"] == beta]
        for r in sel:
            labels.append(f"{cond} {kind}\n{r['seed']}"); values.append(r["evaluation"]["per_life"] * r["evaluation"]["share"][0]); colors.append(color)
    axes[2].bar(range(len(values)), values, color=colors)
    axes[2].set_xticks(range(len(values))); axes[2].set_xticklabels(labels, fontsize=6, rotation=90)
    axes[2].set_title("Lectures de la marque par vie, β = 3", fontsize=10); axes[2].grid(alpha=0.3, axis="y")
    fig.suptitle("Enquête apprise par Q-learning, politique gloutonne, 300 vies de test", fontsize=10)
    fig.tight_layout()
    fig.savefig(root / "dose-effect.png", dpi=150)
    print("\nfigure:", root / "dose-effect.png")
except ImportError:
    print("matplotlib unavailable; no figure")
