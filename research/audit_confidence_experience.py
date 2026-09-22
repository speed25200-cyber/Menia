"""Secondary paired analysis of public confidence/reproduction data; not a consciousness test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

EXPERIMENTS = ("concurrent", "delayed")
CONDITIONS = ("mullerlyer", "baserate", "payoff")
ROOT = Path(__file__).resolve().parents[1]
GRID = np.linspace(320., 510., 1901)


def key(row):
    return row["confidence_type"], row["participant"], row["bias_source"]


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fit_measure(length, answer, degree):
    x, y = np.asarray(length, dtype=float), np.asarray(answer, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid aligned measurements")
    z = (x-400)/30
    design = np.column_stack([z**k for k in range(degree+1)])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != degree+1:
        raise ValueError("Insufficient distinct lengths")
    # Independent normal equations on aggregated cells with trial-count weights.
    levels, inverse, count = np.unique(x, return_inverse=True, return_counts=True)
    cell_mean = np.bincount(inverse, weights=y)/count
    cell_z = (levels-400)/30
    cell_design = np.column_stack([cell_z**k for k in range(degree+1)])
    reference = np.linalg.solve(cell_design.T @ (count[:, None]*cell_design),
                                cell_design.T @ (count*cell_mean))
    max_error = float(np.max(np.abs(reference-coefficients)))
    if not np.allclose(reference, coefficients, rtol=1e-9, atol=1e-9):
        raise AssertionError("Cell-based fit disagrees with trial-based fit")
    if degree == 2:
        if coefficients[2] <= 0:
            raise ValueError("Selected cohort contains a non-convex confidence fit")
        unbounded = float(400-30*coefficients[1]/(2*coefficients[2]))
        values = coefficients[0]+coefficients[1]*(GRID-400)/30+coefficients[2]*((GRID-400)/30)**2
        threshold = float(GRID[values == values.min()].mean())
        boundary = threshold in (320., 510.)
    elif degree == 1:
        if coefficients[1] <= 0:
            raise ValueError("Selected cohort contains a non-increasing reproduction fit")
        unbounded = threshold = float(400+30*(400-coefficients[0])/coefficients[1])
        boundary = not 320 <= threshold <= 510
    else:
        raise ValueError("Unknown measure")
    return {"threshold": threshold, "unbounded": unbounded, "outside_or_at_grid_boundary": boundary,
            "independent_fit_max_error": max_error}


def paired_measure(short, long):
    return {"confidence_shift": short["confidence"]["threshold"]-long["confidence"]["threshold"],
            "reproduction_shift": short["reproduction"]["threshold"]-long["reproduction"]["threshold"]}


def summarize(rows, seed, draws=10000):
    if not rows:
        raise ValueError("No participant pairs")
    strata = []
    for experiment in EXPERIMENTS:
        selected = [r for r in rows if r["experiment"] == experiment]
        if selected:
            strata.append(np.asarray([[r["confidence_shift"], r["reproduction_shift"],
                                        r["confidence_shift"]-r["reproduction_shift"]] for r in selected]))
    data = np.concatenate(strata)
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros((draws, 3))
    for stratum in strata:
        indices = rng.integers(0, len(stratum), size=(draws, len(stratum)))
        bootstrap += stratum[indices].sum(axis=1)/len(data)
    return {"participants": len(data), "bootstrap_seed": seed, "bootstrap_draws": draws,
            "metrics": {name: {"mean": float(data[:, i].mean()), "sd": float(data[:, i].std(ddof=1)),
                               "percentile_95": np.quantile(bootstrap[:, i], [.025, .975]).tolist()}
                        for i, name in enumerate(("confidence_shift", "reproduction_shift", "difference"))}}


def audit(folder):
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    for file in manifest:
        data = (folder/file["name"]).read_bytes()
        if len(data) != file["bytes"] or hashlib.sha256(data).hexdigest() != file["sha256"]:
            raise ValueError("Changed source file: "+file["name"])
    bad_rows = read_rows(folder/"bad_fits.csv")
    bad = {key(r) for r in bad_rows}
    cohorts, groups = [], defaultdict(list)
    for experiment in EXPERIMENTS:
        raw = read_rows(folder/f"raw_{experiment}.csv")
        filtered = read_rows(folder/f"filtered_{experiment}.csv")
        raw_ids, filtered_ids = {key(r) for r in raw}, {key(r) for r in filtered}
        if not filtered_ids <= raw_ids:
            raise AssertionError("Filtered participant absent from raw data")
        # Verify rows were removed, not modified, by the supplied initial filter.
        raw_kept = [r for r in raw if key(r) in filtered_ids]
        if raw_kept != filtered:
            raise AssertionError("Filtered data differs beyond participant exclusions")
        kept = [r for r in filtered if key(r) not in bad]
        selected_ids = {key(r) for r in kept}
        cohorts.append({"experiment": experiment, "raw_participants": len(raw_ids),
                        "raw_trials": len(raw), "initially_filtered_participants": len(filtered_ids),
                        "initially_filtered_trials": len(filtered),
                        "initial_exclusions": [list(k) for k in sorted(raw_ids-filtered_ids)],
                        "curve_exclusions": len(filtered_ids-selected_ids),
                        "selected_participants": len(selected_ids), "selected_trials": len(kept),
                        "counts_by_condition": {c: sum(k[2] == c for k in selected_ids) for c in CONDITIONS}})
        for row in kept:
            if row["bias_direction"] not in ("short", "long") or row["trial_type"] not in ("decision", "reproduction"):
                raise ValueError("Unknown condition or task")
            groups[(*key(row), row["bias_direction"], row["trial_type"])].append(row)
    fitted = defaultdict(dict)
    fit_errors, boundaries = [], []
    for (experiment, participant, condition, direction, task), rows in sorted(groups.items()):
        length = np.asarray([float(r["target_length"]) for r in rows])
        if set(length) != {370., 380., 390., 400., 410., 420., 430.}:
            raise ValueError("Expected seven stimulus lengths per cell")
        if task == "decision":
            if not {r["confidence"] for r in rows} <= {"high", "low"}:
                raise ValueError("Missing or unknown decision confidence")
            measure, degree = "confidence", 2
            answer = [float(r["confidence"] == "high") for r in rows]
        else:
            measure, degree = "reproduction", 1
            answer = [abs(int(r["answer"])) for r in rows]
        result = fit_measure(length, answer, degree)
        fitted[(experiment, participant, condition, direction)][measure] = result
        fit_errors.append(result["independent_fit_max_error"])
        if result["outside_or_at_grid_boundary"]:
            boundaries.append({"experiment": experiment, "participant": participant,
                               "condition": condition, "direction": direction, "measure": measure, **result})
    pairs = []
    for experiment, participant, condition in sorted({k[:3] for k in fitted}):
        short, long = fitted[(experiment, participant, condition, "short")], fitted[(experiment, participant, condition, "long")]
        if set(short) != {"confidence", "reproduction"} or set(long) != set(short):
            raise ValueError("Incomplete paired tasks")
        pairs.append({"experiment": experiment, "participant": participant, "condition": condition,
                      **paired_measure(short, long)})
    summaries = []
    for i, experiment in enumerate((*EXPERIMENTS, "pooled")):
        for j, condition in enumerate(CONDITIONS):
            chosen = [r for r in pairs if r["condition"] == condition and (experiment == "pooled" or r["experiment"] == experiment)]
            summaries.append({"experiment": experiment, "condition": condition, **summarize(chosen, 93000+3*i+j)})
    source_paths = ("research/audit_confidence_experience.py", "docs/CONFIDENCE_EXPERIENCE_PROTOCOL.md")
    return {"format": "menia-confidence-experience-audit", "version": 1,
            "scope": "secondary human measurement constraint; no subjective consciousness demonstrated in Menia",
            "sources": manifest, "numpy_version": np.__version__, "cohorts": cohorts,
            "bad_fit_rows": len(bad_rows), "unique_bad_fit_participants": len(bad),
            "fit_count": len(fit_errors), "independent_fit_max_error": max(fit_errors),
            "grid_boundaries": boundaries, "participant_pairs": pairs, "summaries": summaries,
            "source_sha256": {p: hashlib.sha256((ROOT/p).read_text(encoding="utf-8").encode()).hexdigest() for p in source_paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = audit(args.input)
    if args.check:
        from research.learn_source_policy import compare
        compare(result, json.loads(args.output.read_text(encoding="utf-8")))
        print("Human confidence/reproduction audit reproduced.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
        print(json.dumps({"cohorts": result["cohorts"], "grid_boundaries": result["grid_boundaries"],
                          "summaries": result["summaries"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
