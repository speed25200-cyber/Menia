import copy
import re
import unittest

from research.prospective_binding_permutation import binding_swap
from research.prospective_state_discovery import plan
from tests_language.test_prospective_state_discovery import TokenizerFixture


class OffsetTokenizer(TokenizerFixture):
    def __call__(self,text,**kwargs):
        spans=list(re.finditer(r'<\|im_end\|>|.',text,re.S))
        return dict(input_ids=[9 if m[0]=='<|im_end|>' else ord(m[0]) for m in spans],offset_mapping=[m.span() for m in spans])


class BindingPermutationTests(unittest.TestCase):
    def setUp(self): self.tokenizer=OffsetTokenizer(); self.case=plan()['cases'][0]
    def swap(self,case=None,prompt=None):
        case=self.case if case is None else case
        ids=self.tokenizer.encode(self.tokenizer.apply_chat_template(case['messages'])) if prompt is None else prompt
        return binding_swap(self.tokenizer,case,ids,len(ids)+3)

    def test_only_observed_bits_move_to_opposite_bit_and_restore(self):
        for case in plan()['cases']:
            result=self.swap(case); perm=result['permutation']; bits={r['position']:r['bit'] for r in result['rows']}
            self.assertEqual(len(result['movedPositions']),case['bindings'])
            self.assertTrue(all(bits[perm[pos]]==1-bit for pos,bit in bits.items()))
            self.assertTrue(all(perm[perm[i]]==i for i in range(len(perm))))
            self.assertEqual(perm[-3:],list(range(len(perm)-3,len(perm))))

    def test_mismatching_observed_prompt_or_table_is_rejected(self):
        with self.assertRaises(ValueError): self.swap(prompt=[1,2,3])
        bad=copy.deepcopy(self.case); bad['values'][0]^=1
        with self.assertRaises(ValueError): self.swap(bad)

    def test_merged_token_span_is_rejected_without_guessing(self):
        class MergedTokenizer(OffsetTokenizer):
            def __call__(self,text,**kwargs):
                value=super().__call__(text,**kwargs)
                for i,(a,b) in enumerate(value['offset_mapping']):
                    if text[a:b] in ('0','1'): value['offset_mapping'][i]=(a-1,b)
                return value
        self.tokenizer=MergedTokenizer()
        with self.assertRaises(ValueError): self.swap()


if __name__=='__main__': unittest.main()
