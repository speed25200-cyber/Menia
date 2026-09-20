from collections import Counter
from fractions import Fraction
import unittest

from research.confidence_action_branches import compile_action_branches
from research.confidence_action_controls import compile_explicit_controls, control_grid, expected_costs, grade_decision
from tests_research.test_confidence_prompt_fork import ContextSensitiveTokenizer


class ExplicitActionControlTests(unittest.TestCase):
    def test_grid_costs_optima_and_regret_against_independent_fraction_arithmetic(self):
        grid = control_grid()
        self.assertEqual(len(grid), 18)
        self.assertEqual(len({tuple(c.values()) for c in grid}), 18)
        for case in grid+[{"pPercent": p, "verificationCost": c}
                          for p, c in ((0, 1), (100, 99), (80, 20))]:
            p, c = case['pPercent'], case['verificationCost']
            reference = {'accept': (1-Fraction(p, 100))*100, 'verify': Fraction(c)}
            for decision in ('accept', 'verify'):
                grade = grade_decision(decision, p_percent=p, verification_cost=c)
                self.assertEqual(grade['selectedExpectedCost'], reference[decision])
                self.assertEqual(grade['expectedRegret'], reference[decision]-min(reference.values()))
                self.assertEqual(grade['optimalChoice'], reference[decision] == min(reference.values()))
        tie = grade_decision('accept', p_percent=80, verification_cost=20)
        self.assertEqual(set(tie['optimalActions']), {'accept', 'verify'})
        self.assertTrue(grade_decision('verify', p_percent=80, verification_cost=20)['optimalChoice'])

    def test_invalid_decision_fails_accuracy_without_inventing_a_cost(self):
        grade = grade_decision(None, p_percent=79, verification_cost=46)
        self.assertFalse(grade['validNativeDecision'])
        self.assertFalse(grade['optimalChoice'])
        self.assertFalse(grade['regretDefined'])
        self.assertIsNone(grade['selectedExpectedCost'])
        self.assertIsNone(grade['expectedRegret'])
        for p, c in ((True, 20), (50.5, 20), (-1, 20), (101, 20), (50, 0), (50, 100), (50, False)):
            with self.assertRaises(ValueError): expected_costs(p, c)
        with self.assertRaises(ValueError): grade_decision('2', p_percent=79, verification_cost=46)

    def test_supplied_information_is_after_identical_prefix_and_mapping_is_balanced(self):
        tokenizer = ContextSensitiveTokenizer()
        native = compile_action_branches(tokenizer, 'Question technique.', '17', verification_cost=46)
        for p in (27, 79):
            fork = compile_explicit_controls(tokenizer, 'Question technique.', '17', p_percent=p, verification_cost=46)
            self.assertEqual(fork['prefixIds'], native['prefixIds'])
            self.assertEqual(len(fork['branches']), 16)
            seen = Counter()
            for branch in fork['branches'].values():
                self.assertEqual(branch['inputIds'][:len(fork['prefixIds'])], fork['prefixIds'])
                self.assertEqual(branch['messages'][:3], native['branches']['judgment']['messages'][:3])
                original = native['branches'][branch['nativeBranch']]
                query = branch['messages'][-1]['content']
                self.assertTrue(query.endswith(original['messages'][-1]['content']))
                self.assertEqual(branch['candidateTokenIds'], original['candidateTokenIds'])
                if branch['information'] == 'probability':
                    self.assertIn(f'exactement à {p} %', query)
                    self.assertNotIn('coûts moyens sont déjà calculés', query)
                else:
                    self.assertNotIn(f'{p} %', query)
                    self.assertIn(f'VALIDER = {100-p} points', query)
                    self.assertIn('DEMANDER UNE VÉRIFICATION = 46 points', query)
                    ordered = [branch['codeToMeaning'][code] for code in branch['optionOrder']]
                    labels = dict(accept='VALIDER =', verify='DEMANDER UNE VÉRIFICATION =')
                    self.assertLess(query.index(labels[ordered[0]]), query.index(labels[ordered[1]]))
                for code, meaning in branch['codeToMeaning'].items():
                    seen[branch['information'], code, meaning] += 1
            self.assertEqual(set(seen.values()), {2})
            self.assertEqual(len(seen), 16)


if __name__ == '__main__': unittest.main()
