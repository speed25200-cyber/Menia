import copy
from pathlib import Path
import tempfile
import unittest

from research import prospective_binding_discovery as study
from research import prospective_state_discovery as parent_study
from tests_language.test_prospective_binding_permutation import OffsetTokenizer
from tests_language.test_prospective_state_discovery import constructed_events, write_events


class BindingDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = OffsetTokenizer()
        events = constructed_events()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'parent.jsonl'
            write_events(path, events)
            cls.parent = parent_study.read_journal(path, cls.tokenizer, require_actual=False)
        cls.events = copy.deepcopy(events)
        p = study.plan()
        cls.events[0].update(plan=p, planHash=study.digest(p), sourceHash=study.source_hash())
        for e in cls.events:
            if e['event'] != 'state':
                continue
            mask = study.binding_swap(cls.tokenizer, e['case'], e['trace']['promptTokenIds'],
                                      e['trace']['cacheTokensAfterFinalToken'])
            e['bindingMask'] = mask
            for record in e['interventions'].values():
                record.update(permutation=mask['permutation'], permutationHash=study.digest(mask['permutation']))
            if e['cacheExport']:
                e['cacheExport']['file'] = e['cacheExport']['file'].replace('prospective-state-', 'prospective-binding-')

    def read(self, events, actual=False):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'targeted.jsonl'
            write_events(path, events)
            return study.read_journal(path, self.tokenizer, require_actual=actual, fixture_parent=self.parent)

    def test_complete_reproduction_and_valid_error_contrasts(self):
        data = self.read(self.events)
        summary = study.summarize(data)
        self.assertEqual(len(data['states']), 24)
        self.assertEqual(summary['units']['actual']['correct'], 24)
        self.assertEqual(summary['pairedContrasts']['values_permuted']['forecastFollowsChangedValidTaskOutcome'], 24)
        self.assertEqual(study.plan()['cases'], parent_study.plan()['cases'])
        for state in data['states'].values():
            self.assertEqual(len(state['bindingMask']['movedPositions']), state['case']['bindings'])

    def test_changed_mask_or_unselected_position_is_rejected_after_rehash(self):
        bad = copy.deepcopy(self.events)
        bad[1]['bindingMask']['permutation'][0:2] = [1, 0]
        with self.assertRaisesRegex(ValueError, 'Targeted mask changed'):
            self.read(bad)
        bad = copy.deepcopy(self.events)
        record = bad[1]['interventions']['values_permuted']
        record['permutation'] = list(record['permutation'])
        record['permutation'][0:2] = [1, 0]
        record['permutationHash'] = study.digest(record['permutation'])
        with self.assertRaisesRegex(ValueError, 'Permutation changed'):
            self.read(bad)

    def test_changed_parent_generation_and_branch_are_rejected(self):
        bad = copy.deepcopy(self.events)
        bad[1]['trace']['snapshotCacheHash'] = 'e'*64
        with self.assertRaisesRegex(ValueError, 'Parent generation or state changed'):
            self.read(bad)
        bad = copy.deepcopy(self.events)
        e = next(e for e in bad if e['event'] == 'forecast' and e['call']['condition'] == 'actual')
        e['decoded']['candidateMass'] = .998
        with self.assertRaisesRegex(ValueError, 'Parent branch changed'):
            self.read(bad)

    def test_forecasts_must_be_frozen_before_tasks(self):
        bad = copy.deepcopy(self.events)
        next(e for e in bad if e['event'] == 'request')['phase'] = 'task'
        with self.assertRaisesRegex(ValueError, 'Request order or input'):
            self.read(bad)
        bad = [copy.deepcopy(e) for e in self.events if e['event'] != 'forecasts_complete']
        with self.assertRaises(ValueError):
            self.read(bad)

    def test_synthetic_parent_cannot_be_used_in_actual_audit(self):
        with self.assertRaisesRegex(ValueError, 'Actual GPU collection required'):
            self.read(self.events, actual=True)
        bad = copy.deepcopy(self.events)
        bad[0]['origin'] = 'transformers_gpu'
        with self.assertRaisesRegex(ValueError, 'Fixture parent forbidden'):
            self.read(bad, actual=True)


if __name__ == '__main__':
    unittest.main()
