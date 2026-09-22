from fractions import Fraction
import json
from pathlib import Path
import unittest
from research.audit_self_synergy import STATES, analyze, evaluate, joint_distribution


class SelfSynergyAuditTests(unittest.TestCase):
    def test_distribution_is_normalized_and_stationary(self):
        distribution = joint_distribution()
        self.assertEqual(sum(distribution.values()), Fraction(1))
        self.assertTrue(analyze(distribution)['stationary_uniform'])

    def test_marginals_force_one_bit_of_synergy(self):
        result = analyze(joint_distribution())
        self.assertEqual(result['i_first_future_bits'], 0)
        self.assertEqual(result['i_second_future_bits'], 0)
        self.assertEqual(result['i_joint_future_bits'], 1)
        self.assertEqual(result['synergy_bounds_bits'], [1, 1])

    def test_local_relabelling_preserves_synergy(self):
        for flip_a in (0, 1):
            for flip_b in (0, 1):
                encoding = {(a, b): (a ^ flip_a, b ^ flip_b) for a, b in STATES}
                self.assertEqual(analyze(joint_distribution(encoding))['synergy_bounds_bits'], [1, 1])

    def test_joint_reencoding_is_invertible_and_removes_synergy(self):
        encoding = {(a, b): (a ^ b, b) for a, b in STATES}
        for state in STATES: self.assertEqual(encoding[encoding[state]], state)
        result = analyze(joint_distribution(encoding))
        self.assertTrue(result['stationary_uniform'])
        self.assertEqual(result['i_joint_future_bits'], 1)
        self.assertEqual(result['synergy_bounds_bits'], [0, 0])

    def test_truth_table_preserves_full_transition_distribution(self):
        self.assertEqual(joint_distribution(implementation='xor'), joint_distribution(implementation='table'))
        self.assertEqual(len(evaluate()['all_24_state_encodings']), 24)

    def test_recorded_report_matches_exact_enumeration(self):
        path = Path(__file__).resolve().parents[1]/'artifacts/self-synergy-audit/report.json'
        self.assertEqual(evaluate(), json.loads(path.read_text(encoding='utf-8')))


if __name__ == '__main__':
    unittest.main()
