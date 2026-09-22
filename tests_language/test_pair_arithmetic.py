import json
from pathlib import Path
import tempfile
import unittest

from research.audit_pair_arithmetic import verify
from research import prospective_pair_discovery as study
from tests_language import test_prospective_pair_discovery as fixtures
from tests_language.test_prospective_state_discovery import write_events


class SeparatePairArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.PairDiscoveryTests.setUpClass()
        cls.events=fixtures.PairDiscoveryTests.events
        cls.summary=study.summarize(fixtures.PairDiscoveryTests().read(cls.events))

    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        root=Path(self.directory.name); self.journal=root/'journal.jsonl'; self.summary_path=root/'summary.json'; self.tokens=root/'tokens.json'
        write_events(self.journal,self.events)
        self.summary_path.write_text(json.dumps(self.summary),encoding='utf-8')
        self.tokens.write_text(json.dumps(dict(codeTokenIds={c:[ord(c)] for c in ('0','1','A','B')})),encoding='utf-8')

    def test_separate_counts_and_probabilities_match_constructed_journal(self):
        result=verify(self.journal,self.summary_path,self.tokens)
        self.assertEqual(result['branchRecords'],2880)
        self.assertLess(result['maximumAbsoluteDifference'],1e-14)

    def test_wrong_subgroup_count_is_rejected(self):
        summary=json.loads(self.summary_path.read_text(encoding='utf-8'))
        summary['units'][0]['correct']-=1
        self.summary_path.write_text(json.dumps(summary),encoding='utf-8')
        with self.assertRaises(AssertionError): verify(self.journal,self.summary_path,self.tokens)

    def test_wrong_token_code_map_is_rejected(self):
        self.tokens.write_text(json.dumps(dict(codeTokenIds={c:[ord(c)+1] for c in ('0','1','A','B')})),encoding='utf-8')
        with self.assertRaises(AssertionError): verify(self.journal,self.summary_path,self.tokens)


if __name__=='__main__': unittest.main()
