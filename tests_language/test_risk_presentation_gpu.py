import unittest
from research.native_choice_gpu import NativeChoiceBackend
from research.risk_presentation import make_plan,request_for
from tests_language import test_cross_model_gpu as fixture


class RiskGenerationTest(unittest.TestCase):
    def test_new_factorial_messages_reach_real_generation_without_shared_cache(self):
        fixture.CrossModelBackendTests.setUpClass()
        backend=NativeChoiceBackend.__new__(NativeChoiceBackend)
        backend.model=fixture.CrossModelBackendTests.model;backend.tokenizer=fixture.CrossModelBackendTests.tokenizer
        request=request_for(make_plan(),0)
        first=backend.generate(request);second=backend.generate(request)
        self.assertEqual(first,second)
        self.assertEqual(first[1]['effectiveGeneration']['max_new_tokens'],16)
        self.assertLessEqual(first[1]['outputTokens'],16)


if __name__=='__main__':unittest.main()
