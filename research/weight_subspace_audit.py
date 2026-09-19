"""Descriptive LoRA geometry and synthetic counterexamples; no LLM inference.

Compare effective updates B @ A in a shared coordinate system, not a putative
consciousness subspace. Native-localization checkpoints use scale=1. This does
not validate their training journal, completeness, base model, or behavior.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def factors(a, b):
    arrays = []
    for value in (a, b):
        x = np.asarray(value)
        if x.dtype.kind not in "fiu" or x.ndim != 2 or min(x.shape) < 1:
            raise ValueError("Expected nonempty real numeric matrices")
        x = x.astype(np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite factor")
        arrays.append(x)
    a, b = arrays
    if a.shape[0] != b.shape[1]:
        raise ValueError("Incompatible LoRA factor shapes")
    return a, b


def spectrum(a, b):
    """Singular values of B A without allocating the dense update."""
    a, b = factors(a, b)
    _, ra = np.linalg.qr(a.T, mode="reduced")
    _, rb = np.linalg.qr(b, mode="reduced")
    return np.linalg.svd(rb @ ra.T, compute_uv=False)


def inner(a, b, c, d):
    """Frobenius inner product of B A and D C, from small Gram matrices."""
    a, b = factors(a, b)
    c, d = factors(c, d)
    if a.shape[1] != c.shape[1] or b.shape[0] != d.shape[0]:
        raise ValueError("Effective updates have different shapes")
    return float(np.trace((b.T @ d) @ (c @ a.T)))


def cosine_from_norms(product, norm_left, norm_right):
    if norm_left == 0 or norm_right == 0:
        return None
    value = float(product / (norm_left * norm_right))
    if not np.isfinite(value) or abs(value) > 1 + 1e-8:
        raise ValueError("Unstable cosine computation")
    return float(np.clip(value, -1, 1))


def raw_cosine(x, y):
    if x.shape != y.shape:
        return None
    return cosine_from_norms(float(np.sum(x * y)), float(np.linalg.norm(x)), float(np.linalg.norm(y)))


def describe(a, b):
    a, b = factors(a, b)
    s = spectrum(a, b)
    energy = s * s
    total = float(energy.sum())
    if not np.isfinite(total):
        raise ValueError("Unstable spectrum computation")
    return dict(shape=[b.shape[0], a.shape[1]], factorRank=a.shape[0],
                frobeniusNorm=float(np.sqrt(total)), singularValues=s.tolist(),
                topOneEnergyFraction=None if total == 0 else float(energy[0] / total),
                rankFor95PercentEnergy=0 if total == 0 else int(np.searchsorted(np.cumsum(energy) / total, .95) + 1))


def compare(a, b, c, d):
    a, b = factors(a, b)
    c, d = factors(c, d)
    product = inner(a, b, c, d)
    left, right = describe(a, b), describe(c, d)
    return dict(left=left, right=right, updateInnerProduct=product,
                updateCosine=cosine_from_norms(product, left["frobeniusNorm"], right["frobeniusNorm"]),
                parameterizationDependentFactorCosines=dict(a=raw_cosine(a, c), b=raw_cosine(b, d)))


def pair_modules(state):
    if not state:
        raise ValueError("Empty adapter")
    paired = {}
    for key, value in state.items():
        if not isinstance(key, str) or not key.endswith((".a", ".b")):
            raise ValueError("Only native-localization .a/.b factors are supported")
        name, part = key.rsplit(".", 1)
        if not name:
            raise ValueError("Missing module name")
        paired.setdefault(name, {})[part] = value
    result = {}
    for name, parts in paired.items():
        if set(parts) != {"a", "b"}:
            raise ValueError("Missing factor for " + name)
        result[name] = factors(parts["a"], parts["b"])
    return result


def compare_states(left, right):
    left, right = pair_modules(left), pair_modules(right)
    if left.keys() != right.keys():
        raise ValueError("Adapters have different modules")
    rows = {name: compare(*left[name], *right[name]) for name in sorted(left)}
    nleft = float(np.sqrt(sum(r["left"]["frobeniusNorm"] ** 2 for r in rows.values())))
    nright = float(np.sqrt(sum(r["right"]["frobeniusNorm"] ** 2 for r in rows.values())))
    product = sum(r["updateInnerProduct"] for r in rows.values())
    return dict(origin="supplied_adapter_geometry", scale=1.0, moduleCount=len(rows),
                coordinateSystemAssumption="same base model and corresponding modules; not checked here",
                provenanceVerified=False, behaviorMeasured=False, consciousnessMeasured=False,
                globalUpdateCosine=cosine_from_norms(product, nleft, nright), modules=rows)


def synthetic_audit():
    # Sixteen task weight vectors: a dominant common component unrelated to
    # the inputs evaluated here, plus eight independent task directions.
    weights, task_ids, task_signs = [], [], []
    for i in range(1, 9):
        for sign in (-1, 1):
            w = np.zeros(9)
            w[0], w[i] = 10, sign
            weights.append(w)
            task_ids.append(i)
            task_signs.append(sign)
    weights = np.array(weights)
    _, s, vt = np.linalg.svd(weights, full_matrices=False)
    projected = (weights @ vt[:1].T) @ vt[:1]
    centered = weights - weights.mean(axis=0)
    centered_s = np.linalg.svd(centered, compute_uv=False)
    correct, projected_correct = [], []
    for w, p, i, sign in zip(weights, projected, task_ids, task_signs):
        for x_sign in (-1, 1):
            y = sign * x_sign
            # Top direction is exactly e0 analytically; treat SVD roundoff as 0.
            wp, pp = w[i] * x_sign, p[i] * x_sign
            correct.append((1 if wp >= 0 else -1) == y)
            projected_correct.append((1 if abs(pp) < 1e-12 or pp >= 0 else -1) == y)
    eye = np.eye(2)
    q = np.array([[0., -1.], [1., 0.]])
    gauge = compare(eye, eye, q @ eye, eye @ q.T)
    zero = compare(eye, np.zeros((2, 2)), eye, np.zeros((2, 2)))
    return dict(origin="synthetic_counterexamples_not_pretrained_model_results",
                paperReplication=False, consciousnessMeasured=False,
                commonComponent=dict(models=16, tasks=16, examples=32,
                                     uncenteredTopOneEnergyFraction=float(s[0] ** 2 / np.sum(s ** 2)),
                                     centeredTopOneEnergyFraction=float(centered_s[0] ** 2 / np.sum(centered_s ** 2)),
                                     fullTaskAccuracy=float(np.mean(correct)),
                                     projectedTaskAccuracy=float(np.mean(projected_correct)),
                                     zeroTieRule="predict +1; numerical tolerance 1e-12",
                                     centering="subtract per-coordinate mean across models; not paper HOSVD"),
                gaugeChange=gauge, sharedInitialization=zero)


def read_checkpoint(path):
    from safetensors.numpy import load_file
    return load_file(str(path))


def file_digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.left) != bool(args.right):
        parser.error("Both --left and --right are required for adapter comparison")
    inputs = [p.resolve() for p in (args.left, args.right) if p]
    if args.output.resolve() in inputs:
        parser.error("Output must not overwrite an input checkpoint")
    if inputs:
        report = compare_states(read_checkpoint(args.left), read_checkpoint(args.right))
        report["inputs"] = {label: dict(file=p.name, sha256=file_digest(p))
                            for label, p in (("left", args.left), ("right", args.right))}
    else:
        report = synthetic_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), origin=report["origin"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
