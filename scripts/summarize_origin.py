"""Print a compact reading of artifacts/origin-inquiry and draw the learning-trajectory figure."""
import json
from pathlib import Path
import sys
root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/origin-inquiry")
runs, refs = [], {}
for part in sorted(root.glob("report-*.json")):
    value = json.loads(part.read_text())
    runs.extend(value["runs"]); refs.update(value["references"])
    print(part.name, "wall %.0fs" % value["wall_seconds"], "numpy", value["numpy"], "python", value["python"])
print("\n== Policies (share k0,k1,k2,k3 | inspections/life | hits/life)")
for run in runs:
    print(f"{run['condition']:>2} seed {run['seed']:>2} loss {run['training'][-1]['loss']:.3f}")
    for policy, e in run["policies"].items():
        print(f"    {policy:>8}: {['%.3f' % s for s in e['share']]} | {e['per_life']:.2f} | {e['hits_per_life']:.2f}")
    p = run["policies"]["self"]["probe"]; i = run["policies"]["self"]["intervention"]
    print(f"    probe final {p['final_accuracy']:.3f} stability {p['stability']:.3f} median first stable step {p['median_first_stable_step']} | intervention {i['followed_donor_body']}/{i['tested']} = {i['rate'] if i['rate'] is None else round(i['rate'], 3)}")
    print("    trajectory:", [(c["update"], round(c["share"][0], 2), round(c["per_life"], 2)) for c in run["trajectory"]])
print("\n== References")
for condition, r in refs.items():
    for policy, e in r["oracle"].items():
        print(f"{condition:>2} oracle {policy:>8}: {['%.3f' % s for s in e['share']]} | {e['per_life']:.2f} | {e['hits_per_life']:.2f}")
    e = r["random_inspector"]
    print(f"{condition:>2} random         : {['%.3f' % s for s in e['share']]} | {e['per_life']:.2f} | {e['hits_per_life']:.2f}")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    colors = {"T": "#1f77b4", "C1": "#7f7f7f", "C3": "#d62728"}
    labels = {"T": "T orphelin, marque présente", "C1": "C1 origine donnée", "C3": "C3 sans trace"}
    seen = set()
    for run in runs:
        c = run["condition"]
        x = [t["update"] for t in run["trajectory"]]
        axes[0].plot(x, [t["share"][0] for t in run["trajectory"]], marker="o", color=colors[c], alpha=0.8, label=labels[c] if c not in seen else None)
        axes[1].plot(x, [t["per_life"] for t in run["trajectory"]], marker="o", color=colors[c], alpha=0.8, label=labels[c] if c not in seen else None)
        seen.add(c)
    for ax, title in zip(axes, ("Part des inspections vers la marque du fabricant", "Inspections par vie")):
        ax.set_xscale("log"); ax.set_xlabel("mises à jour d'apprentissage"); ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05); axes[0].legend(fontsize=8)
    fig.suptitle("Politique P-soi pendant l'enfance, 100 vies par point, trois initialisations", fontsize=10)
    fig.tight_layout()
    fig.savefig(root / "learning-trajectory.png", dpi=150)
    print("\nfigure:", root / "learning-trajectory.png")
except ImportError:
    print("matplotlib unavailable; no figure")
