"""Second arithmetic tested before real Qwen outcomes are reviewed."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research import audit_natural_errors as audit
from research import natural_error_analysis as primary
from research import natural_error_journal as journal
from research import natural_error_questions as questions
from tests_research.test_natural_error_journal import FixtureBackend, synthetic_output


class NaturalErrorAuditTests(unittest.TestCase):
    def test_heterogeneous_full_journal_and_altered_reports(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(questions.COUNTS,train=4,validation=2,test=4))
            stack.enter_context(patch.object(journal,'RESAMPLES',2000))
            stack.enter_context(patch.object(journal,'MIN_CLASS',2))
            path=Path(temp)/'mixed.jsonl'
            backend=FixtureBackend(path)
            def generate(task,capture):
                rng=np.random.default_rng(8000+task['id'])
                def mixed(value):
                    value['state']['input']=rng.normal(0,.2,128).tolist()
                    value['state']['middle']=(np.asarray(value['state']['middle'])+rng.normal(0,.8,128)).tolist()
                    value['state']['final']=rng.normal(0,.4,128).tolist()
                    capture(value)
                text,metrics,trace=synthetic_output(task,mixed)
                # Keep an invalid answer in a train/test stratum; it must count false.
                if task['id']%17==0: text='answer '+text
                return text,metrics,trace
            backend.generate=generate
            with contextlib.redirect_stdout(io.StringIO()):
                journal.collect(path,backend)
            report=primary.analyze(path)
            verified=audit.verify(path,report)
            self.assertTrue(verified['verified'])
            self.assertEqual((verified['overallTables'],verified['withinCellTables'],verified['contrasts'],verified['primaryContrasts']),(24,144,21,9))
            self.assertLess(verified['maxAbsoluteDifference'],1e-10)
            self.assertTrue(any(c['interval95'][0] != c['interval95'][1] for c in report['contrasts']))
            for kind in ('metric','interval','gate','origin'):
                with self.subTest(kind=kind):
                    changed=copy.deepcopy(report)
                    if kind=='metric': changed['replications']['0']['scores']['internal']['auc'] += .1
                    if kind=='interval': next(c for c in changed['contrasts'] if c['primary'])['primaryFamilyInterval'][0] -= .2
                    if kind=='gate': changed['replicatedIncrementalReadoutCriterion']=not changed['replicatedIncrementalReadoutCriterion']
                    if kind=='origin': changed['origin']='transformers_gpu'
                    with self.assertRaises(ValueError): audit.verify(path,changed)

    def test_ties_single_class_bins_and_strict_integer_grading(self):
        rows=[]
        for probability,text in [(0.,'2'),(.1,'2'),(.1,'02'),(.8,'3'),(1.,'2')]:
            rows.append(dict(task=dict(family='countA',letters='ABAC'),result=dict(status='ok',text=text),predictions={'x':probability}))
        a,b=audit.scores(rows,'x'),primary.metrics(rows,'x')
        audit.compare(a,b)
        self.assertEqual(a['correct'],3)
        self.assertEqual((a['direct'],a['wrongDirect']),(2,1))
        self.assertEqual(a['reliability'][1]['n'],2)
        self.assertEqual(a['reliability'][9]['n'],1)
        successes=[r for r in rows if audit.correct(r)]
        self.assertIsNone(audit.scores(successes,'x')['auc'])
        self.assertEqual(audit.quantiles([1.,1.,2.,4.],[0.,.5,1.]),[1.,1.5,4.])


if __name__=='__main__':unittest.main()
