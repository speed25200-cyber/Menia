import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from research import state_composition as s
from research.audit_state_composition import audit
from tests_research.test_state_composition import fixture,write


class CompositionAuditTests(unittest.TestCase):
    def test_oracle_and_wrong_mapping_and_summary_tamper(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(s.CONFIG,dict(trainBlocks=2,testBlocks=4,epochs=2,replications=1,resamples=40)):
            p=Path(d)/'fixture.jsonl';output=Path(d)/'audit.json'
            for wrong in (False,True):
                fixture(p,constant_mapping=wrong);a=audit(p,output)
                self.assertEqual(a['fixedReadingRuleMet'],not wrong)
                self.assertEqual((a['tables'],a['contrasts'],a['primaryContrasts'],a['controlGates']),(96,14,8,24))
                self.assertEqual(a['checkpointsVerified'],{})
                self.assertLess(a['independentMaxDifference'],1e-10)
            report=s.analyze(p);report['tables']['0/test/composed/hidden/trained/monitor/0']['correct']=.12345
            p.with_suffix('.summary.json').write_text(json.dumps(report),encoding='utf-8')
            with self.assertRaises((ValueError,AssertionError)):audit(p,output)

    def test_random_logits_ties_unused_codes_and_missing_options(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(s.CONFIG,dict(trainBlocks=4,testBlocks=4,epochs=2,replications=1,resamples=40)):
            p=Path(d)/'fixture.jsonl';events=fixture(p);rng=random.Random(19);saved={};current=None
            for event in events:
                if event['event']=='request':current=event['request']
                if event['event']!='result':continue
                key=tuple(current[k] for k in ('arm','family','block','format','task','mapping'))+(0 if current['position']==3 else current['position'],)
                if key not in saved:
                    logits=[float(rng.randrange(-3,4)) for _ in range(4)];raw=max(range(4),key=lambda i:(logits[i],-i))
                    if rng.random()<.15:raw=-1
                    saved[key]=dict(choiceLogits=logits,rawChoice=raw,rawTokenId=99 if raw==-1 else 10+raw,choiceMass=.8)
                event.update(saved[key])
            write(p,events);a=audit(p,Path(d)/'audit.json')
            self.assertLess(a['independentMaxDifference'],1e-10);self.assertFalse(a['fixedReadingRuleMet'])
            self.assertGreater(a['firstTokenOutsideOptions'],0)


if __name__=='__main__':unittest.main()
