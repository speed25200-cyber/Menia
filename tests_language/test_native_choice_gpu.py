import unittest

from research.native_choice_gpu import NativeChoiceBackend
from research.native_choice_plan import make_plan, request_for
from tests_language import test_cross_model_gpu as fixture


class NativeChoiceGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.CrossModelBackendTests.setUpClass()
        cls.backend=NativeChoiceBackend.__new__(NativeChoiceBackend)
        cls.backend.model=fixture.CrossModelBackendTests.model
        cls.backend.tokenizer=fixture.CrossModelBackendTests.tokenizer

    def test_actual_generation_uses_decision_settings_and_is_repeatable(self):
        request=request_for(make_plan(),0,[])
        first=self.backend.generate(request)
        second=self.backend.generate(request)
        self.assertEqual(first,second)
        self.assertEqual(first[1]['effectiveGeneration']['max_new_tokens'],16)
        self.assertLessEqual(first[1]['outputTokens'],16)
        self.assertEqual(first[1]['effectiveGeneration']['temperature'],.7)


if __name__=='__main__':unittest.main()
