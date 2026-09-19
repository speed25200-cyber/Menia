"""Link public bodily-self and connectivity records and explore directionality.

Record linkage is inferred and checked geometrically, not supplied by the authors.
Directional summaries do not attribute consciousness or establish causal necessity.
Sources: Zenodo 15330862 DATA/metaTable.csv and Zenodo 21536139 electrode CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile


CONNECTIVITY_SHA256 = "fa9ce3fde46c677becc89a4fa8fd380ee467dbca6d231811f92ceebe4e716e76"
BODILY_MD5 = "6b18b1907de4753bd5668a1879d9d6b9"
COHORT_MD5 = "a957990ee73239762e1177ef39872154"


def summarize(values):
    values = list(values)
    result = {"participants": len(values), "mean": statistics.mean(values) if values else None}
    if len(values) < 2:
        result["bootstrap_95_percentile"] = None
        return result
    rng = random.Random(20260914)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(5000))
    def percentile(p):
        index = (len(draws) - 1) * p
        lo = int(index)
        return draws[lo] + (index-lo) * (draws[min(lo+1, len(draws)-1)]-draws[lo])
    result["bootstrap_95_percentile"] = [percentile(0.025), percentile(0.975)]
    return result


def linked_analysis(connectivity, bodily_rows):
    body_ids = {r["subject"] for r in bodily_rows}
    suffix_ids = defaultdict(set)
    for identifier in body_ids:
        suffix_ids[identifier.rsplit("_", 1)[-1]].add(identifier)
    if any(len(ids) != 1 for ids in suffix_ids.values()):
        raise ValueError("Nonunique bodily suffixes prevent candidate linkage")
    body = defaultdict(list)
    for row in bodily_rows:
        body[(row["subject"].rsplit("_", 1)[-1], tuple(sorted((row["elec_1"], row["elec_2"]))))].append(row)

    conn_suffix_ids = defaultdict(set)
    coordinates = defaultdict(list)
    directions = defaultdict(lambda: defaultdict(list))
    categories = {}
    for key, rows in body.items():
        labels = {"Silent" if r["Hot_Cold"] == "cold" else r["Category"] for r in rows}
        categories[key] = next(iter(labels)) if len(labels) == 1 else "Ambiguous"
    matched_roles = set()
    matched_role_rows = 0
    with connectivity.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            suffix = row["aSubID"].rsplit("_", 1)[-1]
            conn_suffix_ids[suffix].add(row["aSubID"])
            if suffix not in suffix_ids:
                continue
            stim = tuple(sorted(row["stim_chan"].split("-")))
            rec = tuple(sorted(row["record_chan"].split("-")))
            if len(stim) != 2 or len(rec) != 2:
                raise ValueError("Unexpected channel naming scheme")
            try:
                f1 = float(row["peak_maxCor_clst1"])
                score_eligible = (math.isfinite(f1) and row["umapAct"] in {"0", "1"}
                                  and row["sCrossBorder"] == row["rCrossBorder"] == "0"
                                  and float(row["eudDist"]) > 5)
            except ValueError:
                score_eligible = False
            for flow, contacts, other, prefix in [
                ("outflow", stim, rec, "MNIout_coord_"),
                ("inflow", rec, stim, "MNIin_coord_"),
            ]:
                key = (suffix, contacts)
                if key not in body:
                    continue
                expected = body[key][0]
                center = [(float(expected[f"e1_mni_cords_{i}"])+float(expected[f"e2_mni_cords_{i}"]))/2 for i in (1, 2, 3)]
                actual = [float(row[prefix+str(i)]) for i in (1, 2, 3)]
                distance = math.dist(center, actual)
                coordinates[key].append(distance if math.isfinite(distance) else math.inf)
                matched_roles.add((key, flow))
                matched_role_rows += 1
                if score_eligible and categories[key] in {"Sensory-Motor", "Complex"}:
                    directions[(key, other)][flow].append(f1)
    if any(len(ids) != 1 for ids in conn_suffix_ids.values()):
        raise ValueError("Nonunique connectivity suffixes prevent candidate linkage")

    valid_sites = {key for key, ds in coordinates.items() if max(ds) <= 0.001}
    site_contrasts = defaultdict(list)
    reciprocal_pairs = Counter()
    scored_observations = Counter()
    for (key, other), flows in directions.items():
        if key not in valid_sites or set(flows) != {"inflow", "outflow"}:
            continue
        value = statistics.mean(flows["outflow"]) - statistics.mean(flows["inflow"])
        site_contrasts[key].append(value)
        reciprocal_pairs[categories[key]] += 1
        scored_observations[categories[key]] += sum(map(len, flows.values()))
    subject_values = defaultdict(lambda: defaultdict(list))
    sites_by_category = Counter()
    for key, values in site_contrasts.items():
        category = categories[key]
        subject_values[category][key[0]].append(statistics.mean(values))
        sites_by_category[category] += 1
    subject_means = {cat: {subject: statistics.mean(v) for subject, v in subjects.items()} for cat, subjects in subject_values.items()}
    summaries = {}
    for cat in ("Sensory-Motor", "Complex"):
        means = subject_means.get(cat, {})
        summaries[cat] = summarize(means[s] for s in sorted(means))
        summaries[cat].update({"sites": sites_by_category[cat], "reciprocal_seed_target_pairs": reciprocal_pairs[cat], "underlying_directional_rows": scored_observations[cat]})
    sm = subject_means.get("Sensory-Motor", {})
    clx = subject_means.get("Complex", {})
    common = sorted(set(sm) & set(clx))
    differences = [sm[s]-clx[s] for s in common]
    return {
        "method": "Candidate suffix and contact-name linkage, checked against all matched MNI midpoint coordinates",
        "author_provided_crosswalk": False,
        "shared_unique_suffixes": len(set(suffix_ids) & set(conn_suffix_ids)),
        "matched_sites_before_coordinate_check": len(coordinates),
        "matched_site_direction_keys": len(matched_roles),
        "matched_directional_row_roles": matched_role_rows,
        "coordinate_error_max": max((max(v) for v in coordinates.values()), default=None),
        "coordinate_tolerance": 0.001,
        "sites_passing_coordinate_check": len(valid_sites),
        "matched_site_categories": dict(sorted(Counter(categories[k] for k in valid_sites).items())),
        "directional_estimate": "outflow F1 minus inflow F1, paired by site and distal contacts; then equal weighting of sites within participants and participants",
        "categories": summaries,
        "paired_category_difference_SM_minus_Complex": summarize(differences),
        "participants_in_either_directional_category": len(set(sm) | set(clx)),
        "bootstrap": {"resampling_unit": "participant", "draws": 5000, "seed": 20260914, "interval": "95% percentile; exploratory and conditional on candidate linkage"},
        "interpretation": "Record linkage is inferred and geometrically corroborated; the computation is not an independent replication, a pINS/aINS contrast or an experience detector",
    }


def digest(path, algorithm):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def audit(connectivity: Path, bodily: Path, cohort: Path):
    if digest(connectivity, "sha256") != CONNECTIVITY_SHA256:
        raise ValueError("Unexpected connectivity metadata version")
    if digest(bodily, "md5") != BODILY_MD5 or digest(cohort, "md5") != COHORT_MD5:
        raise ValueError("Unexpected bodily or cohort data version")
    with bodily.open(encoding="utf-8-sig", newline="") as handle:
        bodily_rows = list(csv.DictReader(handle))
    bodily_ids = {r["subject"] for r in bodily_rows}
    connectivity_ids = set()
    regions = Counter()
    activation = Counter()
    finite_f1_by_activation = Counter()
    row_count = finite_f1 = nonfinite_f1 = missing_f1 = 0
    with connectivity.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        required = {"aSubID", "stim_chan", "record_chan", "JP_label_in", "JP_label_out", "peak_maxCor_clst1", "umapAct"}
        if not columns or not required.issubset(columns):
            raise ValueError("Missing required connectivity fields")
        for row in reader:
            if None in row or any(v is None for v in row.values()) or not row["aSubID"]:
                raise ValueError("Malformed connectivity row")
            row_count += 1
            connectivity_ids.add(row["aSubID"])
            regions.update([row["JP_label_in"], row["JP_label_out"]])
            activation[row["umapAct"]] += 1
            value = row["peak_maxCor_clst1"]
            if value in {"", "NA", "N/A"}:
                missing_f1 += 1
            elif math.isfinite(float(value)):
                finite_f1 += 1
                finite_f1_by_activation[row["umapAct"]] += 1
            else:
                nonfinite_f1 += 1

    cohort_files = []
    with ZipFile(cohort) as archive:
        for name in sorted(archive.namelist()):
            if name.startswith("__MACOSX/") or name.endswith("/"):
                continue
            entry = {"name": name, "bytes": archive.getinfo(name).file_size}
            if name.endswith(".csv"):
                reader = csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig")))
                rows = list(reader)
                entry.update({"columns": reader.fieldnames, "rows": len(rows)})
                if "subID" in (reader.fieldnames or []):
                    ids = {r["subID"] for r in rows}
                    entry["distinct_subID"] = len(ids)
                    entry["exact_id_overlap_with_bodily_release"] = len(ids & bodily_ids)
            cohort_files.append(entry)

    return {
        "scope": "Public record linkage and exploratory directional contrasts; no consciousness attribution",
        "audit_code_sha256": digest(Path(__file__), "sha256"),
        "sources": {
            "connectivity_record": "https://zenodo.org/records/15330862",
            "connectivity_member": "DATA/metaTable.csv",
            "connectivity_member_sha256": CONNECTIVITY_SHA256,
            "connectivity_member_bytes": connectivity.stat().st_size,
            "bodily_record": "https://zenodo.org/records/21536139",
            "bodily_csv_md5": BODILY_MD5,
            "cohort_archive_md5": COHORT_MD5,
        },
        "connectivity": {
            "rows": row_count,
            "participants": len(connectivity_ids),
            "columns": columns,
            "regions": sorted(regions),
            "activation_labels": dict(sorted(activation.items())),
            "finite_F1_rows": finite_f1,
            "finite_F1_by_activation_label": dict(sorted(finite_f1_by_activation.items())),
            "nonfinite_F1_rows": nonfinite_f1,
            "missing_F1_rows": missing_f1,
        },
        "bodily": {"rows": len(bodily_rows), "participants": len(bodily_ids)},
        "direct_join": {
            "exact_participant_id_overlap": len(bodily_ids & connectivity_ids),
            "overlap_after_whitespace_and_case_normalization": len(
                {s.strip().casefold() for s in bodily_ids} &
                {s.strip().casefold() for s in connectivity_ids}
            ),
            "note": "This direct string comparison precedes the separately reported candidate linkage",
            "interpretation": "Zero matching IDs does not establish that the actual people differ",
        },
        "cohort_archive": cohort_files,
        "candidate_linkage_and_exploratory_analysis": linked_analysis(connectivity, bodily_rows),
        "limitations": [
            "No supplied subject-ID crosswalk was found in the inspected cohort files",
            "An INS label does not supply an anterior/posterior insula subdivision",
            "Bodily response labels are transported only under the explicitly inferred and geometrically checked record linkage",
            "The directional contrast is exploratory and does not estimate subjective experience in Menia",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectivity", required=True, type=Path)
    parser.add_argument("--bodily", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = audit(args.connectivity, args.bodily, args.cohort)
    if args.check:
        if report != json.loads(args.output.read_text(encoding="utf-8")):
            raise SystemExit("Audit report does not match source files")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report["candidate_linkage_and_exploratory_analysis"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
