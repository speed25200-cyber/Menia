"""Reanalyse public NIMADET behavioural data; no consciousness certification.

Independent logistic comparison, not a port of the published generative fit.
Only aggregate results and source hashes are exported. Raw archives stay local.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.special import expit


COMMIT = "e18a54724dc660d814100e09b093500de85c4f5c"
SUBJECTS = (1, 2, 3, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 18, 20, 21,
            23, 26, 27, 28, 31, 32, 33, 34, 35, 36)
MODELS = ("condition", "vividness", "interaction")


def fetch_archive(subject, cache):
    name = f"S{subject:02d}-behaviour.zip"
    url = f"https://raw.githubusercontent.com/ImagineRealityLab/NIMADET/{COMMIT}/Behaviour/{name}"
    path = cache / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(8_000_001)
        if len(data) > 8_000_000:
            raise ValueError("Unexpected archive size")
        # Validate before retaining a downloaded archive; never extract members.
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if not archive.namelist():
                raise ValueError("Empty archive")
        path.write_bytes(data)
    data = path.read_bytes()
    return path, {"subject": subject, "url": url, "bytes": len(data),
                  "sha256": hashlib.sha256(data).hexdigest()}


def read_subject(path, subject):
    arrays, exclusions = [], []
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if not n.startswith("__MACOSX/")]

        def unique(basename):
            candidates = [n for n in names if Path(n).name == basename]
            if len(candidates) != 1:
                raise ValueError(f"Expected unique {basename}: {len(candidates)}")
            return candidates[0]

        mapping = loadmat(io.BytesIO(archive.read(unique(f"RMs_S{subject:02d}.mat"))),
                          variable_names=["responseMappings"], simplify_cells=True)["responseMappings"]
        assert np.asarray(mapping).shape == (4,) and set(mapping) <= {1, 2}
        for run in range(1, 5):
            name = unique(f"MT_S{subject:02d}_run{run}.mat")
            data = loadmat(io.BytesIO(archive.read(name)),
                           variable_names=["R", "C", "trials", "blocks", "miniblocks", "T"],
                           simplify_cells=True)
            response = np.asarray(data["R"], dtype=float).copy()
            trial = np.asarray(data["trials"])
            nblock, ntrial = trial.shape
            assert response.shape == (nblock, ntrial, 4)
            if mapping[run-1] == 2:
                response[:, :, 0] = 5-response[:, :, 0]
            p_ori = np.asarray(data["miniblocks"])
            i_ori = np.repeat(data["blocks"], nblock // len(data["blocks"]))
            correct = np.repeat(data["C"], nblock // len(data["C"]))
            assert p_ori.shape == i_ori.shape == correct.shape == (nblock,)
            times = np.asarray(data["T"]["presTimes"])
            assert times.ndim == 3 and times.shape[1:] == (ntrial, 3)
            assert 0 < len(times) <= nblock
            # The authors iterate only over size(T.presTimes, 1). Preserve the
            # unrecorded trailing rows as missing in our exclusion accounting.
            # No timing or response is fabricated (S33 run 1 has seven blocks).
            onset = np.full((nblock, ntrial, 3), np.nan)
            onset[:len(times)] = times-data["T"]["starttime"]
            b = np.column_stack((np.repeat(p_ori, ntrial), np.repeat(i_ori, ntrial),
                                 trial.ravel(), response[:, :, 2].ravel(),
                                 response[:, :, 0].ravel(), np.repeat(correct, ntrial),
                                 onset[:, :, 0].ravel(), onset[:, :, 1].ravel(),
                                 response[:, :, 3].ravel(), onset[:, :, 2].ravel(),
                                 response[:, :, 1].ravel()))
            # Same eleven-column missingness scope as getBehaviour.m.
            bad_check = b[:, 5] == 0
            missing = np.isnan(b).any(axis=1)
            keep = ~bad_check & ~missing
            assert np.isfinite(b[keep]).all()
            assert set(b[keep, 2]) <= {0, 1} and set(b[keep, 3]) <= {0, 1}
            assert set(b[keep, 4]) <= {1, 2, 3, 4}
            rows = np.column_stack((np.full(keep.sum(), run), b[keep, 2],
                                    (b[keep, 0] == b[keep, 1]).astype(int),
                                    b[keep, 3], b[keep, 4]))
            arrays.append(rows)
            exclusions.append({"run": run, "total": len(b), "retained": int(keep.sum()),
                               "bad_imagery_check": int(bad_check.sum()),
                               "missing_any_of_eleven": int(missing.sum()),
                               "trailing_rows_without_timing": (nblock-len(times))*ntrial,
                               "vividness_screen_first_retained": int((b[keep, 9] < b[keep, 7]).sum()),
                               "reality_screen_first_retained": int((b[keep, 7] < b[keep, 9]).sum()),
                               "excluded_union": int((~keep).sum())})
    return np.concatenate(arrays), exclusions


def design(rows, model):
    p, c, v = rows[:, 1], rows[:, 2], (rows[:, 4]-2.5)/1.5
    columns = [np.ones(len(rows)), p, c, p*c]
    if model in ("vividness", "interaction"):
        columns.append(v)
    if model == "interaction":
        columns.append(c*v)
    if model not in MODELS:
        raise ValueError(model)
    return np.column_stack(columns)


def fit_logistic(x, y):
    def objective(beta):
        logits = x @ beta
        penalty = beta.copy()
        penalty[0] = 0
        return (float(np.sum(np.logaddexp(0, logits)-y*logits) + .5*penalty@penalty),
                x.T @ (expit(logits)-y)+penalty)
    # Convex penalized likelihood. Newton plus backtracking avoids dependence on
    # the L-BFGS line-search termination code near machine precision.
    beta = np.zeros(x.shape[1])
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0
    for _ in range(100):
        value, gradient = objective(beta)
        if np.max(np.abs(gradient)) < 1e-7:
            return beta
        p = expit(x @ beta)
        hessian = x.T @ ((p*(1-p))[:, None]*x)+penalty
        direction = np.linalg.solve(hessian, gradient)
        step = 1.
        for _ in range(40):
            candidate = beta-step*direction
            if objective(candidate)[0] <= value-1e-4*step*float(gradient@direction):
                beta = candidate
                break
            step *= .5
        else:
            if np.max(np.abs(gradient)) < 1e-6:
                return beta
            raise RuntimeError("Newton line search did not converge")
    raise RuntimeError("Newton iterations did not converge")


def score(p, y):
    p = np.clip(p, 1e-12, 1-1e-12)
    return {"logloss": float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
            "brier": float(np.mean((p-y)**2))}


def evaluate(subject, rows, exclusions):
    y = rows[:, 3]
    scores, fits = {}, []
    for model in MODELS:
        prediction = np.full(len(rows), np.nan)
        x = design(rows, model)
        for held_out in range(1, 5):
            test = rows[:, 0] == held_out
            if not test.any() or test.all():
                raise ValueError("Missing independent run")
            beta = fit_logistic(x[~test], y[~test])
            prediction[test] = expit(x[test] @ beta)
            fits.append({"model": model, "held_out_run": held_out,
                         "training_trials": int((~test).sum()), "test_trials": int(test.sum()),
                         "coefficients": beta.tolist()})
        assert np.isfinite(prediction).all()
        scores[model] = score(prediction, y)
    cells = []
    for p in (0, 1):
        for c in (0, 1):
            idx = (rows[:, 1] == p) & (rows[:, 2] == c)
            if not idx.any():
                raise ValueError("Empty experimental condition")
            cells.append({"present": p, "congruent": c, "n": int(idx.sum()),
                          "real_response_rate": float(rows[idx, 3].mean()),
                          "mean_vividness": float(rows[idx, 4].mean())})
    return {"subject": subject, "trials": len(rows), "exclusions": exclusions,
            "scores": scores, "cells": cells, "fits": fits}


def summarize(results):
    rng = np.random.default_rng(20260914)
    resamples = rng.integers(0, len(results), (10_000, len(results)))
    contrasts = []
    for first, second in (("condition", "interaction"), ("condition", "vividness"),
                          ("vividness", "interaction")):
        for metric in ("logloss", "brier"):
            delta = np.array([r["scores"][first][metric]-r["scores"][second][metric] for r in results])
            lo, hi = np.quantile(delta[resamples].mean(axis=1), [.025, .975])
            contrasts.append({"first": first, "second": second, "metric": metric,
                              "mean_improvement": float(delta.mean()), "ci95": [float(lo), float(hi)],
                              "participants_improved": int((delta > 0).sum())})
    return {"participants": len(results), "trials": sum(r["trials"] for r in results),
            "mean_scores": {m: {k: float(np.mean([r["scores"][m][k] for r in results]))
                                for k in ("logloss", "brier")} for m in MODELS},
            "contrasts": contrasts}


def scale_audit(rows):
    """Demonstrate the unpenalized authors' link-function symmetry, not fit it.

    Gaussian draws here differ from MATLAB. Invariance holds for any fixed
    draws. Our separate predictive fits ARE regularized; they are not covered
    by this unpenalized invariance claim.
    """
    rng = np.random.default_rng(45001)
    n = len(rows)
    p = rng.normal(0, .1, (n, 2))
    i = rng.normal(0, .1, (n, 2))
    c = rows[:, 2].astype(bool)
    p[np.arange(n), np.where(c, 0, 1)] += rows[:, 1]
    i[:, 0] += 2.5
    alpha, beta = .8, 1.2

    def probabilities(a, b, mode, rj_slope, v_slope):
        mixed = a*p+b*i
        rj = mixed[np.arange(n), np.where(c, 0, 1)] if mode == "mixed" else a*p[np.arange(n), np.where(c, 0, 1)]
        vv = mixed[:, 0] if mode == "mixed" else b*i[:, 0]
        pr = expit(-.3+rj_slope*rj)
        cumul = expit(np.array([-1., .5, 1.8])[None, :]-v_slope*vv[:, None])
        pv = np.diff(np.column_stack((np.zeros(n), cumul, np.ones(n))), axis=1)
        return np.column_stack((pr, pv))

    errors = []
    for mode in ("separate", "mixed"):
        original = probabilities(alpha, beta, mode, .7, .4)
        scales = ((.5, .5), (2., 2.)) if mode == "mixed" else ((.5, 2.), (2., .5))
        for sa, sb in scales:
            transformed = probabilities(sa*alpha, sb*beta, mode, .7/sa, .4/sb)
            error = float(np.max(np.abs(original-transformed)))
            assert error < 1e-12
            errors.append({"mode": mode, "alpha_scale": sa, "beta_scale": sb,
                           "maximum_probability_difference": error})
    return {"scope": "Analytical symmetry instantiated on human design conditions, not MATLAB replication",
            "rows": n, "checks": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".runtime/nimadet"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/human-reality-bridge/report.json"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    results, sources, all_rows = [], [], []
    for subject in SUBJECTS:
        path, source = fetch_archive(subject, args.cache)
        rows, excluded = read_subject(path, subject)
        results.append(evaluate(subject, rows, excluded))
        sources.append(source)
        all_rows.append(rows)
        print(f"S{subject:02d}: parsed and fitted", flush=True)
    report = {"schema": 1, "upstream_commit": COMMIT,
              "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
              "analysis": "Retrospective behavioural prediction; not causal or consciousness evidence",
              "protocol": "docs/HUMAN_REALITY_BRIDGE_PROTOCOL.md",
              "sources": sources, "participants": results,
              "summary": summarize(results),
              "sensitivity_without_S07": summarize([r for r in results if r["subject"] != 7]),
              "scale_audit": scale_audit(np.concatenate(all_rows))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
