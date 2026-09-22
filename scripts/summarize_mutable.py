"""Compact reading of artifacts/mutable-body and its figure: uncertainty and self-model around the body change."""
import json
from pathlib import Path
import sys
import numpy as np
root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/mutable-body")
runs = []
for part in sorted(root.glob("report-mutable-*.json")):
    value = json.loads(part.read_text())
    runs.extend(value["runs"])
    print(part.name, "wall %.0fs" % value["wall_seconds"])
def lives(r, name):
    return [json.loads(l) for l in (root / f"lives-{r['regime']}-{r['seed']}-{name}.jsonl").read_text().splitlines()]
print("\n== regime seed | M1 new-body probe (old-body) | M2 entropy jump | M3 mark reads after change (control) | M4 hits after/before | learned policy reads")
for r in sorted(runs, key=lambda r: (r["regime"], r["seed"])):
    m1 = r["M1_update"]; m4 = r["M4_recovery"]
    learned = r["lives"].get("change-learned", {}).get("mark_reads_after_change_step")
    print(f"{r['regime']:>4} {r['seed']:>2} | {m1['new_body_accuracy']:.3f} ({m1['old_body_accuracy']:.3f}) | {r['M2_detection']['jump']:+.3f} | "
          f"{r['M3_mark_reads_after_change']:.2f} ({r['M3_control_mark_reads_after_step12']:.2f}) | {m4['hits_after_per_life']:.2f}/{m4['hits_before_per_life']:.2f} = {m4['ratio']:.2f} | {learned}")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"S": "#7f7f7f", "SN": "#7f7f7f", "MS": "#1f77b4", "MW": "#d62728", "MS10": "#17becf"}
    labels = {"S": "S, corps stable en enfance", "SN": "SN, corps stable en enfance", "MS": "MS, corps mutable", "MW": "MW, monde mutable", "MS10": "MS10, corps rarement mutable"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    seen = set()
    for r in sorted(runs, key=lambda r: (r["regime"], r["seed"])):
        c = colors[r["regime"]]; lab = labels[r["regime"]] if r["regime"] not in seen else None; seen.add(r["regime"])
        ent = np.mean([life["motion_entropy"] for life in lives(r, "change-none")], axis=0)
        axes[0].plot(range(1, 25), ent, color=c, alpha=0.8, label=lab)
        steps = list(range(17, 25))
        axes[1].plot(steps, r["M1_update"]["by_step_new"], marker="o", color=c, alpha=0.8, label=lab)
        reads = np.mean([[1 if a == 4 else 0 for a in life["actions"]] for life in lives(r, "change-self")], axis=0)
        axes[2].plot(range(1, 25), reads, color=c, alpha=0.8, label=lab)
    for ax in axes:
        ax.axvline(12.5, color="k", linestyle=":", linewidth=1); ax.grid(alpha=0.3); ax.set_xlabel("pas de la vie")
    axes[0].set_title("Incertitude du modèle sur son propre déplacement, nats", fontsize=9)
    axes[1].set_title("Décodage du nouveau corps, sonde apprise sur vies stables", fontsize=9); axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_title("Lecture de la marque par pas, politique P-soi", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle("Changement de corps imposé au pas 12 ; trois initialisations par régime d'enfance", fontsize=10)
    fig.tight_layout()
    fig.savefig(root / "body-change.png", dpi=150)
    print("\nfigure:", root / "body-change.png")
except ImportError:
    print("matplotlib unavailable; no figure")
