"""Exploratory, not pre-registered: does the workspace bottleneck mask the use of memory, hue code and schema?

The published version 1 agents are lesioned with and without unlimited workspace capacity, on the same test
lives; the drop in correct choices is compared. Output: artifacts/indicator-agent/masking.json.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .indicator_agent import Params
from .indicator_experiment import evaluate, SEEDS

PAIRS = {"no_recurrence": "R", "random_code": "H", "no_schema": "R", "bag": "R"}


def run(seed, root="artifacts/indicator-agent"):
    params = Params.load(Path(root) / f"params-{seed}.json")
    variants = ["unlimited"] + [f"unlimited+{v}" for v in PAIRS]
    evaluation, _ = evaluate(params, seed, variants=variants, log=lambda m: None)
    published = json.loads((Path(root) / f"report-{seed}.json").read_text())["evaluation"]
    out = {}
    for lesion, set_name in PAIRS.items():
        key = "choice_band" if set_name == "H" else "choice"
        out[lesion] = {"set": set_name, "measure": key,
                       "limited_drop": published["agent"][set_name][key] - published[lesion][set_name][key],
                       "unlimited_drop": evaluation["unlimited"][set_name][key] - evaluation[f"unlimited+{lesion}"][set_name][key]}
    return seed, out


def main():
    with ProcessPoolExecutor(3) as pool:
        results = dict(pool.map(run, SEEDS))
    summary = {lesion: {"limited_drop": float(np.mean([results[s][lesion]["limited_drop"] for s in SEEDS])),
                        "unlimited_drop": float(np.mean([results[s][lesion]["unlimited_drop"] for s in SEEDS])),
                        "set": PAIRS[lesion]} for lesion in PAIRS}
    out = {"status": "exploratoire, non pré-enregistré", "per_seed": results, "summary": summary}
    Path("artifacts/indicator-agent/masking.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
