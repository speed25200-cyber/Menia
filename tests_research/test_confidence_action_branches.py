from collections import Counter
import unittest

from research.answer_confidence_data import input_messages
from research.confidence_action_branches import action_query, compile_action_branches
from tests_research.test_confidence_prompt_fork import ContextSensitiveTokenizer


class NativeBranchPromptTests(unittest.TestCase):
    def test_every_code_mapping_and_option_order_balanced_with_same_prequery_prefix(self):
        fork = compile_action_branches(ContextSensitiveTokenizer(), 'Question technique.', '17')
        self.assertEqual(len(fork['branches']), 9)
        self.assertNotIn('Choisis', fork['prefixText'])
        self.assertNotIn('Évalue', fork['prefixText'])
        seen = Counter()
        for name, branch in fork['branches'].items():
            self.assertEqual(branch['inputIds'][:len(fork['prefixIds'])], fork['prefixIds'])
            self.assertEqual(branch['messages'][:3], input_messages('Question technique.', '17')[:3])
            self.assertEqual(len(branch['messages']), 4)
            if branch['kind'] == 'action':
                for code, meaning in branch['codeToMeaning'].items():
                    seen[(code, meaning)] += 1
                    expected = 'VALIDER' if meaning == 'accept' else 'DEMANDER UNE VÉRIFICATION'
                    self.assertIn(code+' : '+expected, branch['messages'][-1]['content'])
                neg, pos = branch['candidateTokenIds']
                self.assertEqual(branch['codeToMeaning'][chr(neg)], 'verify')
                self.assertEqual(branch['codeToMeaning'][chr(pos)], 'accept')
                self.assertNotIn('Évalue la réponse', branch['messages'][-1]['content'])
        self.assertEqual(seen, Counter({(code, meaning): 2 for code in ('2', '3', '4', '5') for meaning in ('accept', 'verify')}))

    def test_unsupported_cost_codes_or_multitoken_codes_rejected(self):
        for cost in (0, 100, .2, True):
            with self.assertRaises(ValueError): compile_action_branches(ContextSensitiveTokenizer(), 'Q', 'R', verification_cost=cost)
        with self.assertRaises(ValueError): action_query({'2': 'accept', '3': 'verify'}, ['2', '2'], 20)
        with self.assertRaises(ValueError): action_query({'A': 'accept', 'B': 'verify'}, ['A', 'B'], 20)
        class SplitCodes(ContextSensitiveTokenizer):
            def encode(self, text, *, add_special_tokens):
                return [1, 2] if text == '2' else super().encode(text, add_special_tokens=add_special_tokens)
        with self.assertRaises(ValueError): compile_action_branches(SplitCodes(), 'Q', 'R')


if __name__ == '__main__': unittest.main()
