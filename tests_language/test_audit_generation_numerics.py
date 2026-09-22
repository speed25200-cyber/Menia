import copy
import math
import unittest

from research.audit_generation_numerics import validate_decode,branch_comparison,compare_numbers


class TokenizerFixture:
    eos_token_id=9
    def encode(self,text,**kwargs): return [int(text)]
    def decode(self,tokens,**kwargs): return ''.join(str(i) for i in tokens if i!=9)


class NumericalAuditTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer=TokenizerFixture()
        self.config=dict(codes=['2','3'],negative='verify',positive='keep')
        self.value=dict(candidateTokenIds=[2,3],candidateMeanings=['verify','keep'],tokenIds=[2,9],eosTokenIds=[9],
            validNativeResponse=True,decision='verify',firstTokenMeaning='verify',stoppedAtEOS=True,reachedTokenLimit=False,
            text='2',firstTopIsCode=True,candidateLogits=[0.,math.log(3.)],conditionalPositive=.75,candidateMass=.4)

    def test_native_decision_and_binary_probability_are_recomputed(self):
        validate_decode(self.value,self.config,self.tokenizer)
        for key,value in (('decision','keep'),('conditionalPositive',.8),('stoppedAtEOS',False),('text','3'),('candidateMass',1.1)):
            bad=dict(self.value,**{key:value})
            with self.subTest(field=key):
                with self.assertRaises(ValueError): validate_decode(bad,self.config,self.tokenizer)

    def test_code_without_eos_is_not_counted_as_a_valid_action(self):
        bad=copy.deepcopy(self.value); bad.update(tokenIds=[2,3],stoppedAtEOS=False,reachedTokenLimit=True,text='23')
        with self.assertRaises(ValueError): validate_decode(bad,self.config,self.tokenizer)
        bad.update(validNativeResponse=False,decision=None)
        validate_decode(bad,self.config,self.tokenizer)
        result=branch_comparison(bad,bad)
        self.assertTrue(result['tokenIdsEqual']); self.assertFalse(result['bothValidNative'])

    def test_recorded_comparison_arithmetic_and_boolean_types(self):
        other=dict(self.value,candidateLogits=[.5,math.log(3.)],conditionalPositive=.6,candidateMass=.3)
        result=branch_comparison(self.value,other)
        self.assertAlmostEqual(result['conditionalProbabilityDifference'],.15)
        self.assertEqual(result['maximumCandidateLogitDifference'],.5)
        with self.assertRaises(ValueError): compare_numbers(result,dict(result,maximumCandidateLogitDifference=.4))
        with self.assertRaises(ValueError): compare_numbers(True,1)


if __name__=='__main__': unittest.main()
