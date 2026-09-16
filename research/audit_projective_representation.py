"""Mathematical controls for a projective self-model candidate, not a sentience test.

These are independent examples, NOT a reproduction of the 2025 PCM simulations.
1. Information is invariant under bijective relabeling of a joint distribution.
   Rebuilding the sensor in the new metric changes that distribution instead.
2. A non-affine projective image of a Gaussian can lack ordinary moments.
   Cutoff integrals illustrate a divergence established analytically in the note.
No downloaded code, training data, or human observations are used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


POINTS = ((-0.4, 0.2, 0.2), (0.3, -0.1, 0.5), (0.0, 0.4, 1.0),
          (0.5, 0.1, 1.5), (-0.2, -0.3, 2.0))
PRIORS = ((0.2,) * 5, (0.1, 0.15, 0.2, 0.25, 0.3),
          (0.4, 0.1, 0.1, 0.1, 0.3))
SIGMAS = (0.2, 0.6, 1.5)
GAMMAS = (0.0, 0.1, 0.5, 1.0)


def project(point, gamma):
    denominator = 1.0 + gamma * point[2]
    if denominator == 0:
        raise ValueError("Point lies on the affine-chart pole")
    return tuple(value / denominator for value in point)


def sensor(points, sigma):
    """Finite categorical likelihood from normalized distance weights, not a PDF."""
    rows = []
    for source in points:
        weights = [math.exp(-math.dist(source, target) ** 2 / (2 * sigma ** 2))
                   for target in points]
        total = math.fsum(weights)
        rows.append([weight / total for weight in weights])
    return rows


def joint(prior, channel, labels):
    return {(labels[i], labels[j]): prior[i] * channel[i][j]
            for i in range(len(labels)) for j in range(len(labels))}


def mutual_information(probabilities):
    first, second = {}, {}
    for (x, y), value in probabilities.items():
        first.setdefault(x, []).append(value)
        second.setdefault(y, []).append(value)
    first = {key: math.fsum(values) for key, values in first.items()}
    second = {key: math.fsum(values) for key, values in second.items()}
    assert math.isclose(math.fsum(probabilities.values()), 1.0, abs_tol=1e-12)
    return math.fsum(p * math.log(p / (first[x] * second[y]))
                     for (x, y), p in probabilities.items() if p > 0)


def information_audit():
    results = []
    for prior_index, prior in enumerate(PRIORS):
        for sigma in SIGMAS:
            reference_channel = sensor(POINTS, sigma)
            reference_joint = joint(prior, reference_channel, POINTS)
            reference = mutual_information(reference_joint)
            for gamma in GAMMAS:
                projected = tuple(project(point, gamma) for point in POINTS)
                assert len(set(projected)) == len(POINTS)
                inverse_error = max(math.dist(project(after, -gamma), before)
                                    for before, after in zip(POINTS, projected))
                assert inverse_error < 1e-12
                # Push forward BOTH coordinates of the SAME joint law. Sorting
                # exercises label/order independence of the marginal calculation.
                pushed = dict(sorted(((project(x, gamma), project(y, gamma)), p)
                                     for (x, y), p in reference_joint.items()))
                transported = mutual_information(pushed)
                rebuilt_channel = sensor(projected, sigma)
                rebuilt = mutual_information(joint(prior, rebuilt_channel, projected))
                channel_change = max(abs(a - b) for old, new in
                                     zip(reference_channel, rebuilt_channel)
                                     for a, b in zip(old, new))
                assert abs(transported - reference) < 1e-12
                if gamma == 0:
                    assert abs(rebuilt - reference) < 1e-12
                    assert channel_change == 0
                results.append({
                    "prior_index": prior_index, "sigma": sigma, "gamma": gamma,
                    "reference_information_nats": reference,
                    "transported_information_nats": transported,
                    "rebuilt_sensor_information_nats": rebuilt,
                    "rebuilt_minus_reference_nats": rebuilt - reference,
                    "maximum_sensor_probability_change": channel_change,
                    "inverse_coordinate_error": inverse_error,
                })
    return results


def simpson(function, left, right, intervals):
    if intervals <= 0 or intervals % 2:
        raise ValueError("Simpson integration needs a positive even interval count")
    step = (right - left) / intervals
    interior = math.fsum((4 if i % 2 else 2) * function(left + i * step)
                         for i in range(1, intervals))
    return step * (function(left) + interior + function(right)) / 3


def gaussian_density(value):
    return math.exp(-value * value / 2) / math.sqrt(2 * math.pi)


def pole_audit():
    # Z ~ N(0,1), T=Z/(1+Z), pole -1. These are cutoff, UNNORMALIZED
    # integrals over epsilon <= |Z+1| <= 1/2, not full moments or variances.
    delta = 0.5
    minimum_density = gaussian_density(-1.5)
    rows = []
    for epsilon in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        estimates, errors = [], []
        for order in (1, 2):
            def integrand(log_offset):
                offset = math.exp(log_offset)
                return math.fsum(abs(z) ** order * gaussian_density(z)
                                 for z in (-1 - offset, -1 + offset)) * offset ** (1 - order)

            coarse = simpson(integrand, math.log(epsilon), math.log(delta), 4096)
            fine = simpson(integrand, math.log(epsilon), math.log(delta), 8192)
            relative_error = abs(fine - coarse) / max(abs(fine), 1e-12)
            assert relative_error < 1e-9
            estimates.append(fine)
            errors.append(relative_error)
        first_bound = minimum_density * math.log(delta / epsilon)
        second_bound = minimum_density / 2 * (1 / epsilon - 1 / delta)
        assert estimates[0] >= first_bound and estimates[1] >= second_bound
        rows.append({
            "epsilon": epsilon, "absolute_first_cutoff_integral": estimates[0],
            "second_cutoff_integral": estimates[1],
            "absolute_first_analytic_lower_bound": first_bound,
            "second_analytic_lower_bound": second_bound,
            "maximum_quadrature_relative_change": max(errors),
        })
    assert all(b["second_cutoff_integral"] > a["second_cutoff_integral"]
               for a, b in zip(rows, rows[1:]))
    # Positive control: U~Uniform[0,1] stays away from the pole.
    mean = simpson(lambda z: z / (1 + z), 0, 1, 4096)
    second = simpson(lambda z: (z / (1 + z)) ** 2, 0, 1, 4096)
    exact_mean, exact_second = 1 - math.log(2), 1.5 - 2 * math.log(2)
    assert abs(mean - exact_mean) < 1e-12 and abs(second - exact_second) < 1e-12
    return rows, {
        "distribution": "Uniform[0,1]", "minimum_denominator": 1,
        "mean": exact_mean, "second_moment": exact_second,
        "variance": exact_second - exact_mean ** 2,
        "maximum_quadrature_absolute_error": max(abs(mean - exact_mean),
                                                   abs(second - exact_second)),
    }


def audit():
    information = information_audit()
    pole, bounded = pole_audit()
    return {
        "scope": "Independent mathematical controls; no replication, consciousness attribution or novelty claim",
        "information_design": {"points": POINTS, "priors": PRIORS,
                               "sigmas": SIGMAS, "gammas": GAMMAS},
        "information_results": information,
        "pole_design": {"distribution": "Standard normal", "map": "z/(1+z)",
                        "pole": -1, "outer_radius": 0.5,
                        "integration": "Simpson in log distance from pole, 4096 and 8192 intervals"},
        "pole_results": pole, "bounded_support_control": bounded,
        "checks": {
            "bijective_coordinate_conditions": len(information),
            "identity_sensor_controls": sum(r["gamma"] == 0 for r in information),
            "changed_sensor_conditions": sum(r["maximum_sensor_probability_change"] > 1e-12
                                             for r in information),
            "maximum_transported_information_error": max(abs(r["transported_information_nats"] -
                r["reference_information_nats"]) for r in information),
            "pole_cutoffs": len(pole),
            "bounded_support_control_passed": True,
        },
        "limits": [
            "The divergence proof is analytic; finite quadrature alone does not prove an infinite moment.",
            "Full Gaussian moments differ from local approximations, truncated laws and principal values.",
            "The discrete sensor is our constructed example, not the published PCM implementation.",
            "Sensor changes can be legitimate physical changes, not numerical errors.",
            "No inference about the presence or absence of subjective experience follows from these checks."],
        "source_sha256": hashlib.sha256(Path(__file__).read_text(encoding="utf-8")
                                         .replace("\r\n", "\n").encode()).hexdigest(),
    }


def compare(actual, expected, path="report"):
    """Portable float checks for libm differences; keys, counts and hashes exact."""
    if isinstance(actual, float):
        assert isinstance(expected, (int, float)) and math.isclose(
            actual, expected, rel_tol=1e-9, abs_tol=1e-12), path
    elif isinstance(actual, dict):
        assert isinstance(expected, dict) and actual.keys() == expected.keys(), path
        for key in actual:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)):
        assert isinstance(expected, (list, tuple)) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{path}[{index}]")
    else:
        assert actual == expected, path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = audit()
    if args.check:
        compare(report, json.loads(args.output.read_text(encoding="utf-8")))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["checks"]))
