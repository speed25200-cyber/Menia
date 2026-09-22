"""Compact reading of artifacts/own-action-channel and its figure: body inference by step, mark reads by step."""
import json
from pathlib import Path
import sys
import numpy as np
from research.text_atelier import TextModel, displacement_table, N_MOVE, LIFE
from research.own_action_experiment import CHANGE_STEP

root = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/own-action-channel")
out = Path(sys.argv[2] if len(sys.argv) > 2 else root / "own-action-channel.png")
extra_roots = [Path(a) for a in sys.argv[3:]]  # further roots whose models join the table and the figure
reports = [json.loads(p.read_text()) for r in [root] + extra_roots for p in sorted(r.glob("report-*.json"))]
if not reports:
    raise SystemExit("no reports")


def root_of(r):
    for candidate in [root] + extra_roots:
        if (candidate / f"report-{r['regime']}-{r['seed']}.json").exists():
            return candidate
    raise FileNotFoundError(r["regime"])


def lives(r, name):
    return [json.loads(l) for l in (root_of(r) / f"lives-{r['regime']}-{r['seed']}-{name}.jsonl").read_text().splitlines()]


print("== regime seed | R acc after first (before any) | conf wrong | M∪C inspections/life, mark share | M lives reading mark after 12 (C) | M acc 16-23 | train s")
for r in sorted(reports, key=lambda r: (r["regime"], r["seed"])):
    R, M, C = r["sets"]["R"], r["sets"]["M"], r["sets"]["C"]
    dR, dM = R["displacement"], M["displacement"]
    mc_insp = (M["inquiry"]["inspections_per_life"] + C["inquiry"]["inspections_per_life"]) / 2
    mark = (M["inquiry"]["mark_reads_per_life"] + C["inquiry"]["mark_reads_per_life"]) / 2
    share = mark / mc_insp if mc_insp else float("nan")
    fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"
    print(f"{r['regime']:>3} {r['seed']:>2} | {fmt(dR['accuracy_after_first'])} ({fmt(dR['accuracy_before_any'])}) | {fmt(dR['confidence_when_wrong_after_first'])} | "
          f"{mc_insp:.2f}, {share:.2f} | {M['inquiry']['lives_reading_mark_after_step']:.2f} ({C['inquiry']['lives_reading_mark_after_step']:.2f}) | "
          f"{fmt(dM['accuracy_steps_16_23'])} | {r['timing']['training_seconds']:.0f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"F": "#7f7f7f", "V": "#1f77b4", "VE": "#2ca02c", "VM": "#d62728"}
    labels = {"F": "F, corps fixe en enfance", "V": "V, corps variable", "VE": "VE, corps variable, efférence masquée",
              "VM": "VM, corps variable et changeant en enfance"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    curves = {"R": {}, "M": {}, "reads": {}}
    for r in reports:
        model = TextModel.load(root_of(r) / f"model-{r['regime']}-{r['seed']}.npz")
        for name in ("R", "M"):
            rows = displacement_table(model, lives(r, name))
            acc = np.full(LIFE, np.nan)
            for t in range(LIFE):
                sub = [x["correct"] for x in rows if x["step"] == t]
                if sub:
                    acc[t] = np.mean(sub)
            curves[name].setdefault(r["regime"], []).append(acc)
        reads = np.zeros(LIFE)
        played = lives(r, "M")
        for life in played:
            for t, a in enumerate(life["actions"]):
                reads[t] += a == N_MOVE
        curves["reads"].setdefault(r["regime"], []).append(reads / len(played))
    for ax, (name, title) in zip(axes, [("R", "Jeu R, actions aléatoires : déplacement prédit exact"),
                                        ("M", "Jeu M, P-soi, corps changé au pas 12 : déplacement exact"),
                                        ("reads", "Jeu M : lectures de la marque par vie et par pas")]):
        for regime in ("F", "V", "VE", "VM"):
            if regime in curves[name]:
                mean = np.nanmean(np.array(curves[name][regime]), axis=0)
                ax.plot(range(LIFE), mean, color=colors[regime], label=labels[regime], lw=2)
        if name != "R":
            ax.axvline(CHANGE_STEP, color="k", ls=":", lw=1)
        ax.set_xlabel("pas de la vie")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.02 if name != "reads" else None)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("exactitude / fréquence")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("figure", out)
except ImportError:
    print("matplotlib absent, pas de figure")
