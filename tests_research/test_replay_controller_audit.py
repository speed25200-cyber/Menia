"""Portable refits must accept roundoff without accepting corrupt exports."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research import replay_controller as pilot
from research.replay_controller_audit import audit, portable_read
from tests_research.test_replay_controller import FixtureBackend, SMALL


class PortableAuditTests(unittest.TestCase):
    def fixture(self, root):
        path = Path(root)/'journal.jsonl'
        with contextlib.redirect_stdout(io.StringIO()):
            pilot.collect(path, FixtureBackend())
        path.with_suffix('.summary.json').write_text(json.dumps(pilot.analyze(path)), encoding='utf-8')
        pilot.export_controller(path, path.with_suffix('.policy.json'))
        return path

    def write(self, path, events):
        path.write_text('\n'.join(json.dumps(e) for e in events)+'\n', encoding='utf-8')

    def test_complete_export_with_independent_score_and_svd_checks(self):
        with patch.dict(pilot.COUNTS, SMALL, clear=True), tempfile.TemporaryDirectory() as root:
            result = audit(self.fixture(root))
            self.assertEqual(result['verification']['independentlyRegradedResults'], 42)
            self.assertEqual(result['verification']['testRoutesChecked'], 96)
            self.assertEqual(result['verification']['independentCandidateScores'], 603)
            self.assertTrue(result['verification']['summaryVerified'])

    def test_roundoff_changes_fit_hash_but_is_numerically_valid(self):
        with patch.dict(pilot.COUNTS, SMALL, clear=True), tempfile.TemporaryDirectory() as root:
            path = self.fixture(root)
            events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            fit = next(e for e in events if e['event']=='fit')
            weights = fit['bundle']['models']['internal']['weights']
            weights[0] = float(np.nextafter(weights[0], np.inf))
            fit['bundleHash'] = pilot.digest(fit['bundle'])
            self.write(path, events)
            with self.assertRaises(ValueError):
                pilot.read_journal(path)
            result = audit(path)
            self.assertFalse(result['verification']['refitHashEqualsStored'])
            self.assertTrue(result['verification']['numericRefitVerified'])

    def test_bad_checksum_and_rehashed_bad_fit_are_both_rejected(self):
        with patch.dict(pilot.COUNTS, SMALL, clear=True), tempfile.TemporaryDirectory() as root:
            path = self.fixture(root)
            original = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            for rehash in (False, True):
                events = copy.deepcopy(original)
                fit = next(e for e in events if e['event']=='fit')
                fit['bundle']['models']['internal']['intercept'] += .01
                if rehash:
                    fit['bundleHash'] = pilot.digest(fit['bundle'])
                self.write(path, events)
                with self.assertRaises(ValueError):
                    portable_read(path)

    def test_incorrect_report_or_exported_policy_rejected(self):
        with patch.dict(pilot.COUNTS, SMALL, clear=True), tempfile.TemporaryDirectory() as root:
            path = self.fixture(root)
            summary_path = path.with_suffix('.summary.json')
            original = summary_path.read_text(encoding='utf-8')
            summary = json.loads(original)
            summary['byPhase']['test']['routes']['learned']['finalCorrect'] += 1
            summary_path.write_text(json.dumps(summary), encoding='utf-8')
            with self.assertRaises(ValueError):
                audit(path)
            summary_path.write_text(original, encoding='utf-8')
            export_path = path.with_suffix('.policy.json')
            exported = json.loads(export_path.read_text(encoding='utf-8'))
            exported['policies']['learned']['threshold'] = .123
            export_path.write_text(json.dumps(exported), encoding='utf-8')
            with self.assertRaises(ValueError):
                audit(path)


if __name__ == '__main__':
    unittest.main()
