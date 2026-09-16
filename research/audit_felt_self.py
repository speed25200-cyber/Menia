"""Paired, aggregate-only reanalysis of six reported-self measures in public data.

See docs/FELT_SELF_PROTOCOL.md. No consciousness classifier or intervention.
The source CSV is required locally and is not redistributed in this repository.
"""

import argparse
from collections import Counter
import csv
from decimal import Decimal
from fractions import Fraction
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path

import numpy as np


DATA_HASH = "20127d6cbececaa5a5d8aeb3375b65892a168d67ebd7fa4c58761a573fd967b6"
XLSX_HASH = "9c71b185d187e3409354fcdf7a1b759df5147618183dbfc3dc10af5ad2eee549"
MEASURES = (
    ("body_boundaries", "PBBS", 1, 7),
    ("reported_self_awareness", "PCI_D15", 0, 6),
    ("reported_altered_state", "PCI_D16", 0, 6),
    ("reported_altered_experience", "PCI_D24", 0, 6),
    ("reported_volitional_control", "PCI_D19", 0, 6),
    ("reported_memory", "PCI_D20", 0, 6),
)


def exact_signed_rank(differences):
    """Conditional two-sided signed-rank distribution, with ties and zero removal.

    Integer differences preserve the precision of the source scores. Ranks are
    doubled so every subset-sum and tail probability can be counted exactly.
    """
    ordered = sorted((abs(d), 1 if d > 0 else -1) for d in differences if d != 0)
    rank = 1
    weights = []
    positive_sum = 0
    for _, items in groupby(ordered, key=lambda pair: pair[0]):
        items = list(items)
        twice_rank = 2 * rank + len(items) - 1
        weights.extend([twice_rank] * len(items))
        positive_sum += twice_rank * sum(sign > 0 for _, sign in items)
        rank += len(items)
    counts = [1]
    for weight in weights:
        updated = counts + [0] * weight
        for total, count in enumerate(counts):
            updated[total + weight] += count
        counts = updated
    assert sum(counts) == 2 ** len(weights)
    tail_at = min(positive_sum, sum(weights) - positive_sum)
    p = min(Fraction(1), Fraction(2 * sum(counts[:tail_at + 1]), 2 ** len(weights)))
    return {"nonzero_pairs": len(weights), "positive_rank_sum": positive_sum / 2,
            "minimum_rank_sum": tail_at / 2, "p_exact": str(p), "p": float(p)}


def benjamini_hochberg(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    adjusted = [0.0] * len(values)
    bound = 1.0
    for rank in range(len(values), 0, -1):
        i = order[rank - 1]
        bound = min(bound, values[i] * len(values) / rank)
        adjusted[i] = bound
    return adjusted


def load_data(path):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DATA_HASH, "Unexpected source CSV hash"
    rows = list(csv.DictReader(raw.decode("cp1252").splitlines(), delimiter=";"))
    assert len(rows) == 50
    assert len({r["VP"] for r in rows}) == 50
    scores = {}
    for name, suffix, lower, upper in MEASURES:
        pairs = []
        for row in rows:
            pair = []
            for prefix in ("F", "K"):
                d = Decimal(row[prefix + suffix].replace(",", "."))
                assert Decimal(lower) <= d <= Decimal(upper)
                scaled = d * 1000
                assert scaled == scaled.to_integral_value()
                pair.append(int(scaled))
            pairs.append(pair)
        scores[name] = np.array(pairs, dtype=np.int64)
    return rows, scores


def summarize(scores, labels):
    rng = np.random.default_rng(20260915)
    # Resample complete pairs; use the same draws for every measure.
    samples = rng.integers(0, len(labels), size=(20000, len(labels)))
    results = []
    for name, suffix, lower, upper in MEASURES:
        integer_pairs = scores[name]
        differences = integer_pairs[:, 0] - integer_pairs[:, 1]
        values = integer_pairs / 1000
        delta = differences / 1000
        rank_result = exact_signed_rank([int(d) for d in differences])
        interval = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975])
        results.append({
            "measure": name, "columns": ["F" + suffix, "K" + suffix],
            "scale": [lower, upper], "n_pairs": len(labels),
            "float_mean": float(values[:, 0].mean()), "bed_mean": float(values[:, 1].mean()),
            "float_median": float(np.median(values[:, 0])), "bed_median": float(np.median(values[:, 1])),
            "paired_mean_difference": float(delta.mean()), "paired_median_difference": float(np.median(delta)),
            "paired_mean_bootstrap_ci95": [float(v) for v in interval],
            "difference_counts": {"negative": int((differences < 0).sum()),
                                  "zero": int((differences == 0).sum()), "positive": int((differences > 0).sum())},
            "signed_rank": rank_result,
            "order_label_means": {label: float(delta[labels == label].mean()) for label in sorted(set(labels))},
        })
    for result, p in zip(results, benjamini_hochberg([r["signed_rank"]["p"] for r in results])):
        result["p_bh_six_measures"] = p
        result["reject_at_0_05_after_bh"] = p <= 0.05
    return results


def crosscheck_xlsx(path, csv_rows):
    # Optional local source check; openpyxl is not needed for the CSV audit/tests.
    from openpyxl import load_workbook
    assert hashlib.sha256(path.read_bytes()).hexdigest() == XLSX_HASH
    book = load_workbook(path, read_only=True, data_only=True)
    sheet = book.active
    rows = list(sheet.values)
    header = list(rows[0])
    records = [dict(zip(header, row)) for row in rows[1:] if row[header.index("VP")] is not None]
    assert len(records) == 50 and len({r["VP"] for r in records}) == 50
    lookup = {int(r["VP"]): r for r in records}
    assert set(lookup) == {int(r["VP"]) for r in csv_rows}
    differences = []
    scores = {}
    for name, suffix, lower, upper in MEASURES:
        pairs = []
        for original in csv_rows:
            row = lookup[int(original["VP"])]
            assert row["Reihenfolge"] == original["Reihenfolge"]
            pair = []
            for prefix in ("F", "K"):
                column = prefix + suffix
                x = Decimal(str(row[column]))
                c = Decimal(original[column].replace(",", "."))
                assert lower <= x <= upper
                # Explicit sensitivity at common 3-decimal precision, not a source edit.
                pair.append(int(x.quantize(Decimal("0.001")) * 1000))
                differences.append({"column": column, "csv": float(c), "xlsx": float(x),
                                    "absolute_difference": float(abs(x - c))})
            pairs.append(pair)
        scores[name] = np.array(pairs, dtype=np.int64)
    book.close()
    return {
        "source_url": "https://osf.io/download/y4gmp/", "source_sha256": XLSX_HASH,
        "matched_participants": 50, "compared_cells": len(differences),
        "cells_differing_beyond_0_00051": sum(d["absolute_difference"] > 0.00051 for d in differences),
        "cells_differing_beyond_0_0051": [d for d in differences if d["absolute_difference"] > 0.0051],
        "sensitivity": "XLSX scores rounded to three decimals, half-even; same paired procedure and bootstrap seed",
        "results": summarize(scores, np.array([r["Reihenfolge"] for r in csv_rows])),
    }


def run_audit(path, xlsx=None):
    rows, scores = load_data(path)
    labels = np.array([row["Reihenfolge"] for row in rows])
    report = {
        "schema": "felt-self-human-audit-v1", "protocol": "docs/FELT_SELF_PROTOCOL.md",
        "source_doi": "10.1038/s41598-024-59642-y", "source_url": "https://osf.io/download/qm7y2/",
        "source_sha256": DATA_HASH, "participants": len(rows), "sessions": 2 * len(rows),
        "data_columns": len(rows[0]), "order_labels": dict(sorted(Counter(labels).items())),
        "order_label_mapping": "Not identified; no assignment of letters to session sequences",
        "bootstrap": {"seed": 20260915, "draws": 20000, "unit": "participant pair", "interval": "percentile, unadjusted"},
        "contrast": "Floatation minus Bed", "results": summarize(scores, labels),
        "scope": ["Exploratory within-participant comparisons of published aggregate subscale scores",
                  "No item-level psychometric validation, causal mediation, or experience classifier",
                  "No reanalysis of the 2026 clinical study or neural mechanism",
                  "No implementation in Menia and no consciousness or novelty established"],
    }
    if xlsx is not None:
        report["xlsx_crosscheck"] = crosscheck_xlsx(xlsx, rows)
    return report


def compare(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            compare(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            compare(a, b)
    elif isinstance(actual, float):
        assert math.isclose(actual, expected, abs_tol=1e-10, rel_tol=0), (actual, expected)
    else:
        assert actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--xlsx", type=Path, help="Optional independent-format source check and sensitivity")
    parser.add_argument("--output", type=Path, default=Path("artifacts/felt-self-audit/report.json"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = run_audit(args.data, args.xlsx)
    if args.check:
        compare(report, json.loads(args.output.read_text(encoding="utf-8")))
        print("Felt self: source hash and aggregate report reproduced (tolerance 1e-10).")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    for row in report["results"]:
        print(row["measure"], "mean difference", round(row["paired_mean_difference"], 6),
              "BH p", round(row["p_bh_six_measures"], 6))
    print("No consciousness conclusion.")


if __name__ == "__main__":
    main()
