import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:  # the research CI environment installs numpy only
    HAS_TORCH = False

from research.audit_confidence_ranking import explicit_auc, verify, verify_weights
from research.confidence_ranking_analysis import analyze
from research.cross_model_prediction import digest, CELLS
from tests_research.test_confidence_ranking_study import fixture_events, write_fixture


def varied_fixture_events(path):
    """Overlapping, tied scores create nonzero bootstrap interval widths."""
    plan = None
    for event in fixture_events(path):
        if event['event'] == 'header': plan = event['plan']
        if event['event'] == 'judgment':
            call = plan['calls'][event['id']]; task = plan['tasks'][call['task']]
            truth = int(task['block'] % 3 != 0)
            noise = int(digest(['audit-fixture', call['task'], call['judge'], call['producer']])[:8], 16) % 101 / 100
            if call['judge'] == 'rank': probability = .05+.35*truth+.60*noise
            elif call['judge'] == 'ce': probability = .05+.15*truth+.80*noise
            elif call['judge'] == 'neutral': probability = .05+.90*noise
            else: probability = .1*(1+CELLS.index((task['family'], task['level'])))
            probability = round(probability, 2)
            event['scores']['conditionalCorrect'] = probability
            event['scores']['topTokenId'] = 16 if probability >= .5 else 15
        yield event


def weight_fixture(path):
    """Tiny rank-eight factors, never weights of the real Menia model."""
    import torch
    from safetensors.torch import save_file
    data = dict(starts={}, checkpoints={})
    for rep in range(3):
        initial = {}
        for projection in ('q_proj', 'v_proj'):
            name = 'model.layers.0.self_attn.'+projection
            initial[name+'.a'] = torch.full((8, 4), float(rep+1))
            initial[name+'.b'] = torch.zeros((4, 8))
        initial_file = path.with_suffix(f'.r{rep}-initial.safetensors')
        save_file(initial, str(initial_file))
        sha = hashlib.sha256(initial_file.read_bytes()).hexdigest()
        for index, arm in enumerate(('ce', 'rank', 'neutral')):
            key = f'r{rep}-{arm}'
            final = {k: v.clone()+(index+1)*.01 for k, v in initial.items()}
            checkpoint = path.with_suffix('.'+key+'.safetensors')
            save_file(final, str(checkpoint))
            data['starts'][key] = dict(initializationHash=sha, trainableParameters=sum(v.numel() for v in initial.values()))
            data['checkpoints'][key] = dict(initializationHash=sha, sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    return data


class RankingAuditPrimitiveTests(unittest.TestCase):
    def test_pair_matrix_matches_expanded_observations_and_undefined_groups(self):
        y = np.array([1, 0, 1, 0, 0, 1]); ps = np.array([.4, .4, .7, .1, .8, .3]); c = np.array([0, 0, 0, 1, 1, 1])
        weights = np.array([[1, 1, 1, 1, 1, 1], [2, 3, 1, 0, 2, 4], [1, 0, 0, 0, 0, 0]])
        actual, counts = explicit_auc(y, ps, c, weights)
        for k, w in enumerate(weights):
            ids = np.repeat(np.arange(len(y)), w); values = []
            for i in ids:
                for j in ids:
                    if y[i] == 1 and y[j] == 0 and c[i] == c[j]:
                        values.append(float(ps[i] > ps[j])+.5*float(ps[i] == ps[j]))
            self.assertEqual(counts[k], len(values))
            if values: self.assertAlmostEqual(actual[k], sum(values)/len(values))
            else: self.assertTrue(np.isnan(actual[k]))
        with self.assertRaises(AssertionError): explicit_auc(y, ps, c, -weights)

    @unittest.skipUnless(HAS_TORCH, 'torch is not installed in this environment')
    def test_actual_files_finite_matched_and_changed_with_rehashed_corruption(self):
        import torch
        from safetensors.torch import load_file, save_file
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trial.jsonl'; data = weight_fixture(path)
            checked = verify_weights(data, path)
            self.assertEqual(len(checked), 12)
            final_path = path.with_suffix('.r0-rank.safetensors')
            initial_path = path.with_suffix('.r0-initial.safetensors')
            for problem in ('nan', 'unchanged', 'shape'):
                data = weight_fixture(path)
                # Detach from the read-only mmap before deliberately replacing
                # a fixture file (Windows refuses writes to mapped files).
                state = {k: v.clone() for k, v in load_file(str(final_path)).items()}
                key = next(iter(state))
                if problem == 'nan': state[key][0, 0] = float('nan')
                elif problem == 'unchanged': state = {k: v.clone() for k, v in load_file(str(initial_path)).items()}
                else: state[key] = state[key][:-1].contiguous()
                save_file(state, str(final_path))
                data['checkpoints']['r0-rank']['sha256'] = hashlib.sha256(final_path.read_bytes()).hexdigest()
                with self.assertRaises(AssertionError): verify_weights(data, path)
            data = weight_fixture(path)
            data['starts']['r1-neutral']['initializationHash'] = 'f'*64
            with self.assertRaises(AssertionError): verify_weights(data, path)


class RankingFullAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(); cls.path = Path(cls.directory.name)/'trial.jsonl'
        write_fixture(cls.path, varied_fixture_events(cls.path)); cls.report = analyze(cls.path)

    @classmethod
    def tearDownClass(cls): cls.directory.cleanup()

    def test_full_report_separate_arithmetic_and_bootstrap(self):
        checked = verify(self.path, self.report, check_weights=False)
        type(self).audit_result = checked
        self.assertTrue(checked['verified']); self.assertFalse(checked['weightsChecked'])
        self.assertEqual(checked['origin'], 'synthetic_fixture')
        self.assertEqual(checked['comparisonsChecked'], 18)
        self.assertEqual(checked['bootstrapAucsRecomputed'], 480000)
        self.assertLess(checked['maxAbsoluteDifference'], 1e-10)
        self.assertTrue(all(c['familyInterval'][1] > c['familyInterval'][0] for c in self.report['contrasts']))
        self.assertTrue(all(0 < r['scores']['base']['rank']['withinAuc'] < 1 for r in self.report['replications'].values()))

    def test_corrupted_within_auc_interval_or_gate_rejected(self):
        for location in ('auc', 'interval', 'gate'):
            changed = copy.deepcopy(self.report)
            if location == 'auc': changed['replications']['0']['scores']['base']['rank']['withinAuc'] = .99
            elif location == 'interval': changed['contrasts'][0]['familyInterval'][0] += .01
            else: changed['gates']['0']['coveragePassed'] = False
            with self.assertRaises(AssertionError): verify(self.path, changed, check_weights=False)


if __name__ == '__main__':
    unittest.main()
