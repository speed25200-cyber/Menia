import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import composition_diagnostic as s
from research import audit_composition_diagnostic as audit
from tests_research.test_composition_diagnostic import fixture, calibration_fixture
from tests_research.test_state_composition import write


class DiagnosticAuditTests(unittest.TestCase):
    def test_independent_arithmetic_and_corrupted_summary(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(s.CONFIG,dict(blocksPerReplication=4,replications=1,resamples=20)), patch.object(s,'calibration',side_effect=lambda:calibration_fixture(s.checkpoints())):
            p=Path(d)/'f.jsonl'; fixture(p); report=s.analyze(p)
            self.assertTrue(audit.verify(p,report)['verified'])
            broken=copy.deepcopy(report)
            broken['tables']['0/composed/hidden/trained/monitor/0']['calibratedBalancedAccuracy']=.8
            with self.assertRaisesRegex(ValueError,'calculation mismatch'):audit.verify(p,broken)

    def test_ties_and_outputs_outside_requested_code(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(s.CONFIG,dict(blocksPerReplication=2,replications=1,resamples=20)), patch.object(s,'calibration',side_effect=lambda:calibration_fixture(s.checkpoints())):
            p=Path(d)/'f.jsonl';events=fixture(p)
            for e in events:
                if e['event']!='result':continue
                if '/monitor/' in e['id']:
                    e.update(choiceLogits=[0.,0.,0.,0.],rawChoice=0,rawTokenId=10)
                else:e.update(choiceLogits=[0.,0.,3.,0.],rawChoice=2,rawTokenId=12)
            write(p,events);report=s.analyze(p)
            self.assertTrue(audit.verify(p,report)['verified'])
            self.assertEqual(report['tables']['0/base/visible/trained/marker_first/0']['nativeBalancedAccuracy'],0.)
            self.assertEqual(report['tables']['0/base/hidden/trained/monitor/1']['presenceAUROC'],.5)


if __name__=='__main__':unittest.main()
