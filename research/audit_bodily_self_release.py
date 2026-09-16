"""Inventory the public Lyu et al. (2026) release; never execute its code.

This is an audit of data units and availability, not an estimate of consciousness.
Download the four files from https://zenodo.org/records/21536139 into --data-dir.
Only aggregate statistics and archive entry names are written to the report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile


FILES = {
    "EBS_elecloc_QCcleaned.csv": "6b18b1907de4753bd5668a1879d9d6b9",
    "Part1-EBS.zip": "dbf5fb015c5b2963246a532c896ba4fb",
    "Part2-Neuroimaging.zip": "133259cb818e89a3c6bb19682b7c08a0",
    "Part3-CCEP.zip": "aee3bd8e0c4960e592957bdb8121fdc2",
}


def counts(values):
    return dict(sorted(Counter(values).items()))


def audit(data_dir: Path) -> dict:
    manifest = {}
    payloads = {}
    archives = {}
    for name, expected_md5 in FILES.items():
        payload = (data_dir / name).read_bytes()
        observed_md5 = hashlib.md5(payload).hexdigest()
        if observed_md5 != expected_md5:
            raise ValueError(f"Source version mismatch for {name}")
        payloads[name] = payload
        manifest[name] = {
            "bytes": len(payload),
            "md5": observed_md5,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "url": f"https://zenodo.org/api/records/21536139/files/{name}/content",
        }
        if name.endswith(".zip"):
            with ZipFile(io.BytesIO(payload)) as archive:
                archives[name] = [
                    {"name": item.filename, "bytes": item.file_size}
                    for item in sorted(archive.infolist(), key=lambda i: i.filename)
                    if not item.is_dir() and not item.filename.startswith("__MACOSX/")
                ]

    reader = csv.DictReader(io.StringIO(payloads["EBS_elecloc_QCcleaned.csv"].decode("utf-8-sig")))
    columns = reader.fieldnames
    rows = list(reader)
    required = {"subject", "bipolar", "elec_1", "elec_2", "Hot_Cold", "Category"}
    if not columns or not required.issubset(columns) or not rows:
        raise ValueError("Missing required columns or empty release")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed CSV row")
    if any(not row[column] for row in rows for column in required):
        raise ValueError("Missing identity or response category")

    sites = defaultdict(list)
    contacts = defaultdict(set)
    subjects = defaultdict(set)
    for row in rows:
        key = (row["subject"], *sorted((row["elec_1"], row["elec_2"])))
        sites[key].append(row)
        subjects[row["subject"]].add(key)
        for contact in (row["elec_1"], row["elec_2"]):
            contacts[(row["subject"], contact)].add(key)

    signatures = [tuple(row[column] for column in columns) for row in rows]
    per_subject = sorted(len(keys) for keys in subjects.values())
    category_subjects = defaultdict(set)
    category_sites = defaultdict(set)
    response_cross = defaultdict(Counter)
    for key, site_rows in sites.items():
        for row in site_rows:
            category_subjects[row["Category"]].add(row["subject"])
            category_sites[row["Category"]].add(key)
            response_cross[row["Hot_Cold"]][row["Category"]] += 1

    return {
        "scope": "Public release inventory and descriptive counts; no consciousness inference",
        "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": {
            "record": "https://zenodo.org/records/21536139",
            "doi": "10.5281/zenodo.21536139",
            "license": "CC-BY-4.0",
            "authors": "Dian Lyu, Lamara Allen, Sofia Pantis, Elif Goksun Karagoz, Julian Quabs, Josef Parvizi",
            "paper_doi": "10.21203/rs.3.rs-10503199/v1",
        },
        "files": manifest,
        "archive_inventory": archives,
        "csv": {
            "columns": columns,
            "rows": len(rows),
            "subjects": len(subjects),
            "unique_subject_bipolar_labels": len({(r["subject"], r["bipolar"]) for r in rows}),
            "unique_subject_unordered_contact_pairs": len(sites),
            "duplicate_exact_rows": len(rows) - len(set(signatures)),
            "pairs_with_multiple_rows": sum(len(rs) > 1 for rs in sites.values()),
            "pairs_with_both_contact_orders": sum(
                len({(r["elec_1"], r["elec_2"]) for r in rs}) > 1
                for rs in sites.values()
            ),
            "pairs_with_conflicting_hot_cold": sum(
                len({r["Hot_Cold"] for r in rs}) > 1 for rs in sites.values()
            ),
            "pairs_with_conflicting_response_labels": sum(
                len({(r["Hot_Cold"], r["Category"]) for r in rs}) > 1
                for rs in sites.values()
            ),
            "conflicting_category_patterns": counts(
                " / ".join(sorted({r["Category"] for r in rs}))
                for rs in sites.values()
                if len({r["Category"] for r in rs}) > 1
            ),
            "rows_by_response": counts(r["Hot_Cold"] for r in rows),
            "rows_by_category": counts(r["Category"] for r in rows),
            "unique_sites_by_category": {k: len(v) for k, v in sorted(category_sites.items())},
            "subjects_by_category_nonexclusive": {k: len(v) for k, v in sorted(category_subjects.items())},
            "response_category_cross_table": {k: dict(sorted(v.items())) for k, v in sorted(response_cross.items())},
            "unique_subject_contact_keys": len(contacts),
            "contacts_shared_by_distinct_stimulation_pairs": sum(len(v) > 1 for v in contacts.values()),
            "max_distinct_pairs_sharing_one_contact": max(map(len, contacts.values())),
            "sites_per_subject": {"min": min(per_subject), "max": max(per_subject), "sum": sum(per_subject)},
            "unordered_pair_interpretation": (
                "The release dictionary says contact order is arbitrary under alternating "
                "polarity. Reversed pairs require trial-level reconciliation; no rows are "
                "deleted or relabeled here."
            ),
        },
        "not_estimated": [
            "Reproduction of CCEP connectivity effects or causal mediation",
            "Presence versus absence of subjective consciousness",
            "Transfer of bodily experience to Menia",
            "Scientific novelty of an architecture",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = audit(args.data_dir)
    if args.check:
        if report != json.loads(args.output.read_text(encoding="utf-8")):
            raise SystemExit("Report differs from the audited source files")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["csv"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
