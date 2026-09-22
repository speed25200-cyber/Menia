"""Exact finite-state audit of a minimal self-targeted synergy criterion.

Not a general PID/PIRD implementation. Nonnegative bivariate PID identities
uniquely determine synergy in the two principal examples.
"""
from collections import defaultdict
from fractions import Fraction
from itertools import permutations, product
import json
import math
from pathlib import Path


STATES = tuple(product((0, 1), repeat=2))


def joint_distribution(encoding=None, *, implementation='xor'):
    encoding = encoding or {state: state for state in STATES}
    if set(encoding) != set(STATES) or set(encoding.values()) != set(STATES):
        raise ValueError('Encoding must be a bijection over the four states')
    if implementation not in ('xor', 'table'):
        raise ValueError('Unknown implementation')
    table = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    result = defaultdict(Fraction)
    for state in STATES:
        for noise in (0, 1):
            parity = state[0] ^ state[1] if implementation == 'xor' else table[state]
            future = (parity, noise)
            result[(encoding[state], encoding[future])] += Fraction(1, 8)
    return dict(result)


def mutual_information(distribution, select_x, select_y):
    joint, px, py = defaultdict(Fraction), defaultdict(Fraction), defaultdict(Fraction)
    for event, probability in distribution.items():
        x, y = select_x(event), select_y(event)
        joint[x, y] += probability
        px[x] += probability
        py[y] += probability
    return sum(float(p)*math.log2(float(p/(px[x]*py[y]))) for (x, y), p in joint.items())


def analyze(distribution):
    before, after = defaultdict(Fraction), defaultdict(Fraction)
    for (state, future), probability in distribution.items():
        before[state] += probability
        after[future] += probability
    first = mutual_information(distribution, lambda e: e[0][0], lambda e: e[1])
    second = mutual_information(distribution, lambda e: e[0][1], lambda e: e[1])
    joint = mutual_information(distribution, lambda e: e[0], lambda e: e[1])
    # I1=R+U1, I2=R+U2, I12=R+U1+U2+Syn; assume all four atoms >=0.
    redundancy_lower = max(0.0, first+second-joint)
    redundancy_upper = min(first, second)
    return {'stationary_uniform': before == after and all(p == Fraction(1, 4) for p in before.values()),
            'i_first_future_bits': first, 'i_second_future_bits': second,
            'i_joint_future_bits': joint,
            'synergy_bounds_bits': [joint-first-second+redundancy_lower,
                                    joint-first-second+redundancy_upper]}


def evaluate():
    identity = {s: s for s in STATES}
    mixed = {(a, b): (a ^ b, b) for a, b in STATES}
    original = joint_distribution(identity)
    table = joint_distribution(identity, implementation='table')
    all_encodings = []
    for values in permutations(STATES):
        encoding = dict(zip(STATES, values))
        all_encodings.append({'encoding': [list(encoding[s]) for s in STATES],
                              **analyze(joint_distribution(encoding))})
    return {'scope': 'one-step bivariate nonnegative-PID consistency audit; not consciousness or PIRD',
            'equation': 'A_next = A XOR B; B_next = independent fair bit',
            'original': analyze(original), 'mixed_encoding': analyze(joint_distribution(mixed)),
            'same_distribution_for_xor_and_lookup': original == table,
            'exact_transition_joint': [{'current': list(s), 'future': list(t), 'probability': str(p)}
                                       for (s, t), p in sorted(original.items())],
            'all_24_state_encodings': all_encodings,
            'interpretation': 'A positive self-targeted synergy criterion is satisfied by this tiny stochastic register. '
                              'Mixing subsystem coordinates can change synergy while preserving the joint process up to isomorphism. '
                              'This does not establish absence or presence of subjective experience.'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    report = evaluate()
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('original', 'mixed_encoding', 'same_distribution_for_xor_and_lookup')}, indent=2))
