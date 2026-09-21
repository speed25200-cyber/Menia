import tempfile
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from research.workspace import (SharedWorkspace, DirectLookup, lookup_batch,
                                lookup_test, encode_examples, PAIR_IDS)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_disjoint_pairs_for_every_query_and_balanced_targets(self):
        train_pairs = PAIR_IDS[PAIR_IDS % 5 != 0]
        test_pairs = PAIR_IDS[PAIR_IDS % 5 == 0]
        self.assertEqual((len(train_pairs), len(test_pairs)), (460, 116))
        train, _, _ = encode_examples(train_pairs, torch.zeros(len(train_pairs), dtype=torch.long))
        test, query, target = lookup_test()
        train_keys = {tuple(row.tolist()) for row in train.flatten(1)}
        test_keys = {tuple(row.tolist()) for row in test.flatten(1)}
        self.assertFalse(train_keys & test_keys)
        self.assertEqual(len(test_keys), 116)
        torch.testing.assert_close(torch.bincount(target), torch.full((4,), 116))
        torch.testing.assert_close(query.reshape(116, 4, 4), torch.eye(4).expand(116, 4, 4))

    def test_private_module_access_and_causal_broadcast(self):
        model = SharedWorkspace()
        tables, query, _ = lookup_batch(8, batch=4)
        altered = tables.clone()
        altered[:, 1] = 0
        _, before = model(tables, query, return_trace=True)
        _, after = model(altered, query, return_trace=True)
        torch.testing.assert_close(before[0]['specialists'][:, 0], after[0]['specialists'][:, 0])
        self.assertFalse(torch.equal(before[1]['specialists'][:, 0], after[1]['specialists'][:, 0]))
        _, blocked_before = model(tables, query, intervention='no_broadcast', return_trace=True)
        _, blocked_after = model(altered, query, intervention='no_broadcast', return_trace=True)
        for left, right in zip(blocked_before, blocked_after):
            torch.testing.assert_close(left['specialists'][:, 0], right['specialists'][:, 0])

    def test_masked_table_has_no_influence_and_recurrence_is_prefix_causal(self):
        model = SharedWorkspace()
        tables, query, _ = lookup_batch(9, batch=3)
        changed = tables.clone()
        changed[:, 0] = torch.rand_like(changed[:, 0])
        before = model(tables, query, intervention='no_table_a')
        after = model(changed, query, intervention='no_table_a')
        torch.testing.assert_close(before['answer'], after['answer'])
        _, short = model(tables, query, rounds=2, return_trace=True)
        _, long = model(tables, query, rounds=6, return_trace=True)
        for i in range(2):
            torch.testing.assert_close(short[i]['workspace'], long[i]['workspace'])

    def test_both_readouts_use_same_workspace_and_all_paths_receive_gradients(self):
        model = SharedWorkspace()
        tables, query, labels = lookup_batch(10, batch=16)
        result, trace = model(tables, query, return_trace=True)
        torch.testing.assert_close(result['answer'], model.answer(trace[-1]['workspace']))
        torch.testing.assert_close(result['intermediate'], model.intermediate(trace[-1]['workspace']))
        F.cross_entropy(result['answer'], labels).backward()
        for name, parameter in model.named_parameters():
            if name.startswith('intermediate.'):
                continue
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_weights_only_export_and_roundtrip(self):
        tables, query, _ = lookup_batch(12, batch=4)
        for cls in (SharedWorkspace, DirectLookup):
            model = cls()
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)/'model.pt'
                torch.save(model.state_dict(), path)
                restored = cls()
                restored.load_state_dict(torch.load(path, weights_only=True))
                torch.testing.assert_close(model(tables, query)['answer'], restored(tables, query)['answer'])


if __name__ == '__main__':
    unittest.main()
