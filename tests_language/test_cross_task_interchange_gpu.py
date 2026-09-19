import unittest
import torch

from research.cross_task_interchange_gpu import run_group
from tests_language.test_state_interchange_ops import InterchangeOperationTests


class CrossTaskGPUOperationTests(unittest.TestCase):
    def test_complete_real_model_group_uses_other_task_activations(self):
        from tests_language.test_cross_model_gpu import CrossModelBackendTests
        fixture = InterchangeOperationTests()
        fixture.setUp()
        CrossModelBackendTests.setUpClass()
        pair = {role: dict(id=role, noiseSeed=36 + i, marker=i, positivePosition=1 + i,
                           sentences=['hello hello', 'hello hello hello'])
                for i, role in enumerate(('donor', 'recipient'))}
        outputs = run_group(fixture.model, CrossModelBackendTests.tokenizer, pair, [3, 4, 5, 6])
        self.assertEqual(len(outputs), 232)
        self.assertEqual(sum('donorTask' in out['request'] for out in outputs), 96)
        self.assertTrue(all(not m._forward_hooks for m in fixture.model.modules()))
        self.assertTrue(all(not p.grad for p in fixture.model.parameters()))


if __name__ == '__main__':
    unittest.main()
