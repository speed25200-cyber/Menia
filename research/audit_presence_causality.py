"""Secondary analysis of VR presence; no test or score of AI consciousness.

The optional XLSX reader uses openpyxl; numerical tests need only NumPy.
Source workbooks are read without modifying or re-exporting them.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import numpy as np

EXCLUDED = {14, 15, 30, 27, 45, 58, 38}
MEDIATORS = ("fear", "heart_rate", "skin_conductance")
SOURCE_HASHES = {
    "PoF_Data_OSF.xlsx": "199170a82c8bd04a358fa4d76b040b40b263f0cdd886e371ac2b3b893079acb9",
    "PoF_Analyses_OSF.R": "74d87e67ee69d636f66b64810f293bd1190a55152771cbfac4fa803986a0d60b",
    "ThePresenceofFear_Supplemental_Material.docx": "acbadac4659a8dff332e5ebbe64becded886592cfd5c7d00d51f759e96a296db",
}
SOURCE_URLS = {
    "PoF_Data_OSF.xlsx": "https://osf.io/download/n4xru/",
    "PoF_Analyses_OSF.R": "https://osf.io/download/mu496/",
    "ThePresenceofFear_Supplemental_Material.docx": "https://osf.io/download/2sqvp/",
}


def index_rows(rows):
    """A blank ID is not a subject; duplicate/noninteger IDs are errors."""
    header, *body = rows
    result = {}
    for row in body:
        if row[0] is None:
            continue
        if not isinstance(row[0], (int, float)) or not float(row[0]).is_integer():
            raise ValueError("Invalid participant ID")
        key = int(row[0])
        if key in result:
            raise ValueError("Duplicate participant ID")
        result[key] = {name: value for name, value in zip(header, row) if name is not None}
    return result


def raw_numeric_cell(path, member, coordinate):
    """Read an explicit XLSX numeric value, ignoring display format only."""
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read(member))
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    cells = [cell for cell in root.findall(".//s:c", ns) if cell.get("r") == coordinate]
    if len(cells) != 1 or cells[0].get("t", "n") != "n" or cells[0].find("s:f", ns) is not None:
        raise ValueError("Expected one nonformula numeric source cell")
    value = float(cells[0].findtext("s:v", namespaces=ns))
    if not np.isfinite(value):
        raise ValueError("Nonfinite raw source value")
    return value


def load_data(folder):
    import openpyxl
    manifest = []
    for name, expected in SOURCE_HASHES.items():
        content = (folder / name).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected:
            raise ValueError(f"Changed source file: {name}")
        manifest.append({"name": name, "url": SOURCE_URLS[name], "bytes": len(content), "sha256": digest})
    workbook = openpyxl.load_workbook(folder / "PoF_Data_OSF.xlsx", read_only=True, data_only=True)
    names = ("Study 2 - Ratings", "Study 2 - HR", "Study 2 - SCL")
    try:
        sheets = [index_rows(list(workbook[name].iter_rows(values_only=True))) for name in names]
    finally:
        workbook.close()
    # Documented source-format exception, discovered before fitting any model.
    # D23 is participant 43's Curb HR: numeric XML, but Excel format mmss.0.
    imported = sheets[1][43]["Curb"]
    if not isinstance(imported, datetime):
        raise ValueError("Expected audited date-formatted numeric cell")
    raw = raw_numeric_cell(folder / "PoF_Data_OSF.xlsx", "xl/worksheets/sheet7.xml", "D23")
    sheets[1][43]["Curb"] = raw
    format_exception = {"sheet": names[1], "cell": "D23", "participant": 43,
                        "variable": "Curb", "display_format": "mmss.0",
                        "reader_value": imported.isoformat(), "raw_numeric_value": raw,
                        "treatment": "Read stored numeric XML value; source workbook unchanged"}
    if not set(sheets[0]) == set(sheets[1]) == set(sheets[2]):
        raise ValueError("Participant sets disagree across sheets")
    if len(sheets[0]) != 60 or not EXCLUDED.issubset(sheets[0]):
        raise ValueError("Unexpected source cohort")
    kept = sorted(set(sheets[0]) - EXCLUDED)
    data = []
    for key in kept:
        ratings, hr, scl = [s[key] for s in sheets]
        conditions = [r["Condition"] for r in (ratings, hr, scl)]
        if conditions[0] not in (0, 1) or len(set(conditions)) != 1:
            raise ValueError("Condition mismatch")
        presence = np.array([ratings[f"{t}_presence"] for t in ("curb", "bottom", "top", "start", "end")], dtype=float)
        fear = np.array([ratings[f"{t}_fear"] for t in ("curb", "bottom", "top", "start", "end")], dtype=float)
        heart = np.array([hr[t] for t in ("Curb", "Bottom", "Top", "Start", "End")], dtype=float)
        skin = np.array([scl[t] for t in ("Curb", "Bottom", "Top", "Start", "End")], dtype=float)
        if not all(np.isfinite(v).all() for v in (presence, fear, heart, skin)):
            raise ValueError("Missing retained measurement; no imputation")
        if np.any(presence < 1) or np.any(presence > 10) or np.any(fear < 1) or np.any(fear > 10):
            raise ValueError("Rating outside scale")
        if np.any(heart <= 0) or np.any(skin < 0):
            raise ValueError("Invalid physiological measurement")
        skin = np.log1p(skin)
        data.append([key, conditions[0], presence[:2].mean(), presence[2:].mean(),
                     fear[2:].mean()-fear[:2].mean(), heart[2:].mean()-heart[:2].mean(),
                     skin[2:].mean()-skin[:2].mean()])
    data = np.array(data, dtype=float)
    audit = {"source_records_per_sheet": [len(s) for s in sheets],
             "excluded_ids": sorted(EXCLUDED), "retained_ids": kept,
             "source_orders_equal": list(sheets[0]) == list(sheets[1]) == list(sheets[2]),
             "join": "participant_id", "retained_n": len(data),
             "control_n": int(np.sum(data[:, 1] == 0)), "height_n": int(np.sum(data[:, 1] == 1)),
             "source_manifest": manifest, "format_exceptions": [format_exception]}
    if (audit["control_n"], audit["height_n"]) != (27, 26):
        raise ValueError("Unexpected retained allocation")
    return data, audit


def fit(design, y):
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Rank deficient model")
    return coef, y-design @ coef


def sensitivity(vm, vy, covariance, a, rho):
    if vm <= 0 or vy <= 0 or abs(rho) >= 1:
        raise ValueError("Invalid sensitivity parameters")
    unexplained = vy-covariance**2/vm
    if unexplained <= 0:
        raise ValueError("Degenerate conditional covariance")
    b = covariance/vm-rho*np.sqrt(unexplained/vm)/np.sqrt(1-rho*rho)
    # Independently rearranged Theorem 4 (Imai, Keele, Yamamoto, 2010).
    partial = covariance/np.sqrt(vm*vy)
    reference = a*np.sqrt(vy/vm)*(partial-rho*np.sqrt((1-partial**2)/(1-rho**2)))
    if not np.isclose(a*b, reference, atol=1e-10, rtol=1e-10):
        raise AssertionError("Sensitivity expressions disagree")
    return float(a*b)


def mediation(x, m, y, baseline=None):
    x, m, y = [np.asarray(z, dtype=float) for z in (x, m, y)]
    if x.ndim != 1 or x.shape != m.shape or x.shape != y.shape:
        raise ValueError("Invalid aligned vectors")
    if not all(np.isfinite(z).all() for z in (x, m, y)):
        raise ValueError("Nonfinite observation")
    c = np.column_stack((np.ones(len(x)), x))
    if baseline is not None:
        baseline = np.asarray(baseline, dtype=float)
        if baseline.shape != x.shape or not np.isfinite(baseline).all():
            raise ValueError("Invalid baseline")
        c = np.column_stack((c, baseline))
    cm, rm = fit(c, m)
    cy, ry = fit(c, y)
    cm_y, error = fit(np.column_stack((c, m)), y)
    vm, vy, covariance = float(rm@rm/len(x)), float(ry@ry/len(x)), float(rm@ry/len(x))
    a, b, direct, total = float(cm[1]), float(cm_y[-1]), float(cm_y[1]), float(cy[1])
    if vm <= 0 or vy <= 0 or vy-covariance**2/vm <= 0:
        raise ValueError("Degenerate residuals")
    if not np.isclose(total, direct+a*b, atol=1e-10, rtol=1e-10):
        raise AssertionError("Effect decomposition failed")
    if not np.isclose(b, covariance/vm, atol=1e-10, rtol=1e-10):
        raise AssertionError("Residual and full regressions disagree")
    return {"a": a, "b": b, "direct": direct, "total": total, "indirect": a*b,
            "rho_zero": covariance/np.sqrt(vm*vy), "vm": vm, "vy": vy, "covariance": covariance,
            "error_variance": float(error@error/len(x)),
            "mean_coefficients_m": cm.tolist(), "mean_coefficients_y": cy.tolist()}


def equivalent_models(result):
    """Two triangular factorizations of one fitted conditional Gaussian law."""
    vm, vy, k = [result[name] for name in ("vm", "vy", "covariance")]
    forward = np.array([[1., 0.], [k/vm, 1.]])
    reverse = np.array([[1., k/vy], [0., 1.]])
    s = np.array([[vm, k], [k, vy]])
    cov_f = forward @ np.diag([vm, vy-k*k/vm]) @ forward.T
    cov_r = reverse @ np.diag([vm-k*k/vy, vy]) @ reverse.T
    means = np.column_stack([result["mean_coefficients_m"], result["mean_coefficients_y"]])
    # Structural intercept/treatment coefficients, not just marginal means.
    coef_f = means @ np.linalg.inv(forward).T
    coef_r = means @ np.linalg.inv(reverse).T
    error = max(np.abs(cov_f-s).max(), np.abs(cov_r-s).max(),
                np.abs(coef_f @ forward.T-means).max(), np.abs(coef_r @ reverse.T-means).max())
    if error > 1e-10:
        raise AssertionError("Gaussian reconstructions differ")
    return {"conditional_covariance": s.tolist(), "forward_covariance": cov_f.tolist(),
            "reverse_covariance": cov_r.tolist(), "conditional_mean_coefficients": means.tolist(),
            "forward_structural_coefficients": coef_f.tolist(), "reverse_structural_coefficients": coef_r.tolist(),
            "max_reconstruction_error": float(error),
            "effect_do_m_on_y_forward": k/vm, "effect_do_m_on_y_reverse": 0.,
            "scope": "Fitted homoskedastic conditional Gaussian law; not equality of all empirical distributions"}


def bootstrap(x, m, y, baseline, seed, draws):
    groups = [np.flatnonzero(x == value) for value in (0, 1)]
    if any(len(g) == 0 for g in groups) or draws < 2:
        raise ValueError("Two treatment groups and multiple draws required")
    rng = np.random.default_rng(seed)
    keys = ("indirect", "direct", "total", "rho_zero")
    samples = np.empty((draws, len(keys)))
    for i in range(draws):
        indices = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        result = mediation(x[indices], m[indices], y[indices], None if baseline is None else baseline[indices])
        samples[i] = [result[key] for key in keys]
    return {"draws": draws, "seed": seed, "stratum_sizes": list(map(len, groups)),
            "percentile_95": {key: np.quantile(samples[:, j], [.025, .975]).tolist() for j, key in enumerate(keys)}}


def run(folder, draws=10000):
    data, audit = load_data(folder)
    models = []
    for adjusted in (False, True):
        for j, name in enumerate(MEDIATORS):
            x, y, m = data[:, 1], data[:, 3], data[:, 4+j]
            baseline = data[:, 2] if adjusted else None
            result = mediation(x, m, y, baseline)
            points = []
            for rho in np.linspace(-.8, .8, 17):
                points.append({"rho": float(rho), "indirect": sensitivity(result["vm"], result["vy"],
                               result["covariance"], result["a"], float(rho))})
            root_value = sensitivity(result["vm"], result["vy"], result["covariance"], result["a"], result["rho_zero"])
            if abs(root_value) > 1e-10:
                raise AssertionError("Nonzero at analytical sensitivity root")
            models.append({"mediator": name, "baseline_adjusted": adjusted, "estimates": result,
                           "sensitivity": points, "indirect_at_rho_zero": root_value,
                           "bootstrap": bootstrap(x, m, y, baseline, 94100+len(models), draws),
                           "equivalent_models": equivalent_models(result) if j == 0 else None})
    group_means = {str(int(c)): {"n": int(np.sum(data[:, 1] == c)),
                   "means": dict(zip(("baseline_presence", "plank_presence", *MEDIATORS),
                                     data[data[:, 1] == c, 2:].mean(axis=0).tolist()))} for c in (0, 1)}
    return {"study": "Maymon et al. 2024, study 2; secondary exploratory analysis",
            "scope": "Presence ratings in VR; no artificial consciousness attribution",
            "numerics": {"numpy": np.__version__, "bootstrap": "stratified percentile, participant unit",
                         "covariance_divisor": "N", "sensitivity_reference": "10.1214/10-STS321"},
            "cohort": audit, "group_means": group_means, "models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = run(args.input, args.draws)
    serialized = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")
    if args.check:
        if args.output.read_bytes() != serialized:
            raise SystemExit("Report differs from full recomputation")
        print("Presence causal analysis fully reproduced")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(serialized)
        print(args.output)


if __name__ == "__main__":
    main()
