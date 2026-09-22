"""Replay published ownership models; audit what their response data identify."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from research.causal_binding import CausalBinding, logit, window_probability

HASHES = (
    "7331235942a311fe6c84a75c25b5de053741bd25e129c1712ecb2fc25fbf48cb",
    "4c25f3cfa8a010ae7b6c0b6b73b4482a4ece63f17be8f80c6e6a5bd8a8b725cf",
    "8818b715f32c1d711b1bacae4769424a6af5f52e019ea39a859c86967f55a9ee",
)
DELAYS = {"ownership": [-500, -300, -150, 0, 150, 300, 500],
          "synchrony": [-300, -150, -50, 0, 50, 150, 300]}
NOISE = (0, 30, 50)


def parse_counts(rows, task, allow_noninteger=False):
    headers = [f"N{noise}_{delay}" for noise in NOISE for delay in DELAYS[task]]
    if list(rows[0][1:]) != headers:
        raise ValueError("Unexpected noise/delay order")
    data = {}
    for row in rows[1:]:
        if all(v is None for v in row):
            continue
        label = row[0]
        if not isinstance(label, str) or not label.startswith("S"):
            raise ValueError("Invalid participant ID")
        key = int(label[1:])
        if key in data:
            raise ValueError("Duplicate participant")
        values = np.asarray(row[1:], dtype=float)
        if values.shape != (21,) or not np.isfinite(values).all():
            raise ValueError("Missing or misaligned counts")
        if (not allow_noninteger and np.any(values != np.floor(values))) or np.any(values < 0) or np.any(values > 12):
            raise ValueError("Counts outside twelve-trial cells")
        data[key] = values.reshape(3, 7)
    return data


def fit_rows(rows, start, end):
    data = {}
    for row in rows[start-1:end]:
        key = row[0]
        if not isinstance(key, (int, float)) or not float(key).is_integer() or key in data:
            raise ValueError("Invalid or duplicate parameter ID")
        data[int(key)] = row
    return data


def load_data(folder):
    import openpyxl
    books, manifest = [], []
    for i, expected in enumerate(HASHES, 1):
        name = f"elife-77221-fig{i}-data1-v3.xlsx"
        path = folder/name
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected:
            raise ValueError(f"Changed source: {name}")
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=False)
        try:
            sheets = {}
            for sheet in workbook:
                if any(cell.data_type == "f" for row in sheet for cell in row):
                    raise ValueError("Formula requires a separate cached-value audit")
                sheets[sheet.title] = list(sheet.values)
        finally:
            workbook.close()
        books.append(sheets)
        manifest.append({"name": name, "sha256": digest, "bytes": len(content),
                         "url": f"https://cdn.elifesciences.org/articles/77221/{name}"})
    counts = {"ownership": parse_counts(books[0]["RHIdetect"], "ownership"),
              "synchrony": parse_counts(books[2]["SynchronyDetect_Supplement3"], "synchrony", allow_noninteger=True)}
    noninteger_ids = [key for key, values in counts["synchrony"].items()
                      if np.any(values != np.floor(values))]
    if noninteger_ids != [4]:
        raise ValueError("Unexpected fractional-count audit result")
    fits = {"BCI": fit_rows(books[1]["BCI"], 3, 17),
            "FC": fit_rows(books[1]["FC"], 3, 17),
            "shared": fit_rows(books[2]["ExtensionAnalysis"], 3, 17),
            "different": fit_rows(books[2]["ExtensionAnalysis"], 22, 36)}
    for table in [*counts.values(), *fits.values()]:
        if set(table) != set(range(1, 16)):
            raise ValueError("Participant sets do not match")
    return counts, fits, manifest


def binomial_nll(yes, probability, trials=12):
    yes, probability = np.asarray(yes, dtype=float), np.asarray(probability, dtype=float)
    if yes.shape != probability.shape or not np.isfinite(yes).all() or not np.isfinite(probability).all():
        raise ValueError("Invalid likelihood arrays")
    if np.any(yes != np.floor(yes)) or np.any(yes < 0) or np.any(yes > trials):
        raise ValueError("Invalid counts")
    if np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("Invalid probabilities")
    total = 0.
    for y, p in zip(yes.flat, probability.flat):
        if (y > 0 and p == 0) or (y < trials and p == 1):
            return float("inf")
        if y > 0:
            total -= float(y)*math.log(float(p))
        if y < trials:
            total -= float(trials-y)*math.log1p(-float(p))
    return total


def predictions(task, sigmas, lapse, source_sigma, prior=None, width=None, threshold=0.):
    if (prior is None) == (width is None):
        raise ValueError("Choose BCI or fixed criterion")
    model = CausalBinding(prior, source_sigma) if prior is not None else None
    return np.array([[model.report_probability(delay, sigma, lapse, threshold) if model else
                      window_probability(delay, sigma, width, lapse)
                      for delay in DELAYS[task]] for sigma in sigmas])


def comparison(differences, seed, draws=10000):
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    sums = rng.choice(differences, size=(draws, len(differences)), replace=True).sum(axis=1)
    return {"sum": float(differences.sum()), "participant_differences": differences.tolist(),
            "negative_count": int(np.sum(differences < 0)), "n": len(differences),
            "percentile_95": np.quantile(sums, [.025, .975]).tolist(), "seed": seed, "draws": draws}


def replay(counts, fits, source_sigma):
    records = []
    for key in range(1, 16):
        values = {}
        for name in ("BCI", "FC", "shared", "different"):
            row = fits[name][key]
            if name in ("shared", "different") and np.any(counts["synchrony"][key] != np.floor(counts["synchrony"][key])):
                values[name] = {"nll": None, "published_nll": row[7], "difference": None,
                                "status": "Not replayed: synchrony counts are fractional; no rounding or inferred denominator"}
                continue
            if name == "FC":
                probability = predictions("ownership", row[1:4], row[5], source_sigma, width=row[4])
                nll, published = binomial_nll(counts["ownership"][key], probability), row[6]
            else:
                nll = 0.
                tasks = ("ownership",) if name == "BCI" else ("ownership", "synchrony")
                for task in tasks:
                    prior = row[6] if name == "different" and task == "synchrony" else row[1]
                    probability = predictions(task, row[2:5], row[5], source_sigma, prior=prior)
                    nll += binomial_nll(counts[task][key], probability)
                published = row[6] if name == "BCI" else row[7]
            values[name] = {"nll": nll, "published_nll": published, "difference": nll-published}
        records.append({"participant": key, "models": values})
    bci_fc = [2*(r["models"]["BCI"]["nll"]-r["models"]["FC"]["nll"]) for r in records]
    joint_records = [r for r in records if r["models"]["different"]["nll"] is not None]
    delta_nll = np.array([r["models"]["different"]["nll"]-r["models"]["shared"]["nll"] for r in joint_records])
    return {"source_sigma": source_sigma, "records": records,
            "joint_comparison_participant_ids": [r["participant"] for r in joint_records],
            "max_absolute_nll_error_by_model": {name: max(abs(r["models"][name]["difference"]) for r in records
                                                          if r["models"][name]["difference"] is not None)
                                                 for name in ("BCI", "FC", "shared", "different")},
            "score_scope": "Penalized NLL at published parameters, without establishing that they maximize this implementation",
            "comparisons": {"BCI_minus_FC_2deltaNLL": comparison(bci_fc, 71500),
                            "different_minus_shared_AIC_form": comparison(2+2*delta_nll, 71501),
                            "different_minus_shared_BIC_form": comparison(np.log(504)+2*delta_nll, 71501)}}


def published_comparisons(fits):
    # Arithmetic from supplied NLL columns, not a reconstruction from counts.
    result = {"scope": "Published NLL values; arithmetic only, not independently reproduced likelihoods"}
    bci_fc = [2*(fits["BCI"][key][6]-fits["FC"][key][6]) for key in range(1, 16)]
    result["BCI_minus_FC_AIC_and_BIC"] = comparison(bci_fc, 71500)
    for name, ids in (("all_15", list(range(1, 16))), ("without_S4", [key for key in range(1, 16) if key != 4])):
        delta = np.array([fits["different"][key][7]-fits["shared"][key][7] for key in ids])
        result[name] = {"ids": ids, "different_minus_shared_AIC": comparison(2+2*delta, 71501),
                        "different_minus_shared_BIC": comparison(np.log(504)+2*delta, 71501)}
    return result


def equivalence(fits):
    records = []
    for key, row in sorted(fits["different"].items()):
        for task, prior in (("ownership", row[1]), ("synchrony", row[6])):
            original = CausalBinding(prior)
            transformed = CausalBinding(.5)
            threshold = -logit(prior)
            response_error, posterior_change = 0., 0.
            for sigma in row[2:5]:
                for measurement in np.linspace(-1500, 1500, 301):
                    original_response = original.report_probability(float(measurement), sigma, row[5])
                    equivalent_response = transformed.report_probability(float(measurement), sigma, row[5], threshold)
                    response_error = max(response_error, abs(original_response-equivalent_response))
                    posterior_change = max(posterior_change, abs(original.posterior(float(measurement), sigma)-
                                                               transformed.posterior(float(measurement), sigma)))
            if response_error > 1e-12:
                raise AssertionError("Prior/threshold response invariance failed")
            records.append({"participant": key, "task": task, "published_prior": prior,
                            "shared_prior": .5, "equivalent_log_threshold": threshold,
                            "max_response_probability_error": response_error,
                            "max_posterior_change_on_grid": posterior_change})
    return {"source_sigma": 348., "grid": {"start": -1500, "end": 1500, "points": 301,
            "noise_levels": 3, "task_participant_settings": 30}, "records": records,
            "scope": "Algebraic reparameterization, not a claim about actual human decision costs"}


def run(folder):
    counts, fits, manifest = load_data(folder)
    sigma_derived = float(np.sqrt(np.mean(np.square([s for s in DELAYS["ownership"] if s]))))
    descriptive = {}
    for task in DELAYS:
        valid_ids = [key for key in range(1, 16) if not np.any(counts[task][key] != np.floor(counts[task][key]))]
        array = np.stack([counts[task][key] for key in valid_ids])
        descriptive[task] = {"yes_by_noise": array.sum(axis=(0, 2)).astype(int).tolist(),
                             "trials_by_noise": [len(valid_ids)*84]*3, "participant_ids": valid_ids,
                             "delays_ms": DELAYS[task]}
    fractional = counts["synchrony"][4]
    interventions = []
    for prior in (.5, .9):
        model = CausalBinding(prior)
        for threshold in (0., -logit(.9)):
            interventions.append({"prior": prior, "log_threshold": threshold,
                                  "measurement_and_stimulus_ms": 300., "sigma_ms": 150.,
                                  "posterior_at_measurement": model.posterior(300., 150.),
                                  "response_probability_at_stimulus": model.report_probability(300., 150., log_threshold=threshold)})
    return {"study": "Chancel, Ehrsson & Ma (2022), 10.7554/eLife.77221",
            "scope": "Model replay and identification audit; no new fit or consciousness attribution",
            "numpy": np.__version__, "source_manifest": manifest,
            "cohort": {"participant_ids": list(range(1, 16)), "participants": 15, "tasks": 2,
                       "nominal_cells": 630, "trials_per_cell_in_article": 12,
                       "usable_integer_cells": 609, "nominal_judgments_in_usable_cells": 7308,
                       "unit": "aggregated values; trial order and actual denominators unavailable",
                       "fractional_block": {"participant": 4, "task": "synchrony", "values": fractional.tolist(),
                           "fractional_cells": int(np.sum(fractional != np.floor(fractional))),
                           "omitted_cells_for_binomial_replay": 21,
                           "treatment": "Omit only this task block; retain ownership and published parameters unchanged"}},
            "descriptive": descriptive,
            "intervention_example": {"scope": "Four programmed parameter settings, not human observations",
                                      "source_sigma": 348., "lapse": 0., "records": interventions},
            "published_nll_comparisons": published_comparisons(fits),
            "replays": {"printed_source_noise": replay(counts, fits, 348.),
                        "derived_source_noise": replay(counts, fits, sigma_derived)},
            "prior_threshold_equivalence": equivalence(fits)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = (json.dumps(run(args.input), ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")
    if args.check:
        if args.output.read_bytes() != content:
            raise SystemExit("Full recomputation differs")
        print("Ownership inference audit fully reproduced")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(content)
        print(args.output)


if __name__ == "__main__":
    main()
