"""Software and information-flow controls, not pretrained-model results."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research import replay_controller as experiment

SMALL = dict(calibration=2, round1=1, round2=1, round3=1, test=2)


class FixtureBackend:
    origin='synthetic_fixture'
    metadata={'fixture':'deterministic synthetic trace; no pretrained Qwen'}

    def generate(self,task,capture):
        rng=np.random.default_rng(task['seed'])
        state={k:rng.normal(size=experiment.DIM).tolist() for k in experiment.PROJECTIONS}
        capture(state)
        answer=int(experiment.verify_question(task['question']))
        return str(answer if state['final'][0] > 0 else answer+1), {'fixture':True}


class ReplayPilotTests(unittest.TestCase):
    def fixture(self,root):
        path=Path(root)/'journal.jsonl'
        with contextlib.redirect_stdout(io.StringIO()): experiment.collect(path,FixtureBackend())
        return path

    def test_plan_disjoint_from_earlier_pilots_and_budget_fixed(self):
        p=experiment.make_plan()
        self.assertEqual(p,experiment.make_plan())
        self.assertEqual(len(p['tasks']),720)
        self.assertEqual(len({t['question'] for t in p['tasks']}),720)
        earlier={t['question'] for plan in (experiment.cross_plan(),experiment.activation_plan(),experiment.perturbation_plan()) for t in plan['tasks']}
        self.assertFalse(earlier & {t['question'] for t in p['tasks']})
        self.assertEqual([len(v) for v in p['pools'].values()],[75,51,75])

    def test_numeric_parser_handles_format_without_using_reference(self):
        for text,value in [('15',15),('59 - 44 = 15',15),('Réponse : -3.',-3),('Answer: 8',8)]:
            self.assertEqual(experiment.integer_answer(text),value)
        for text in ('14 ou 15','15 puis 16','Je pense 15','1 = 2 = 3','', 'nan'):
            self.assertIsNone(experiment.integer_answer(text))
        task=dict(family='alternatingSum',operands=[59,44],letters='')
        score=experiment.grade_text(task,'59 - 44 = 15')
        self.assertTrue(score['numericCorrect']); self.assertFalse(score['strictCorrect'])
        self.assertFalse(experiment.grade_text(task,'59 - 44 = 14')['numericCorrect'])
        for t in experiment.make_plan()['tasks']:
            self.assertTrue(experiment.grade_text(t,experiment.verify_question(t['question']))['numericCorrect'])

    def test_complete_pipeline_three_updates_export_and_completed_noop(self):
        with patch.dict(experiment.COUNTS,SMALL,clear=True),tempfile.TemporaryDirectory() as tmp:
            p=self.fixture(tmp); report=experiment.analyze(p)
            self.assertTrue(report['complete']); self.assertEqual(report['recordedResults'],42)
            self.assertEqual(report['updates'],['round1','round2','round3'])
            self.assertEqual(len(report['contrasts']),5)
            costs=report['actualCosts']
            self.assertEqual(costs['selectedVerificationCalls']+costs['additionalAuditVerificationCalls'],42)
            out=Path(tmp)/'policy.json'; experiment.export_controller(p,out)
            exported=json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(exported['policies'],report['frozenPolicies'])
            before=p.read_bytes()
            class NoCalls(FixtureBackend):
                def generate(self,*args): raise AssertionError('Completed journal was replayed')
            experiment.collect(p,NoCalls(),resume=True)
            self.assertEqual(before,p.read_bytes())

    def test_interrupted_request_retries_same_seed_without_losing_history(self):
        with patch.dict(experiment.COUNTS,SMALL,clear=True),tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'journal.jsonl'
            with contextlib.redirect_stdout(io.StringIO()): experiment.collect(p,FixtureBackend(),limit=14)
            events=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
            # Simulate process killed after state persistence but before result.
            p.write_text('\n'.join(json.dumps(e) for e in events[:-1])+'\n',encoding='utf-8')
            self.assertIsNotNone(experiment.read_journal(p)[2])
            with contextlib.redirect_stdout(io.StringIO()): experiment.collect(p,FixtureBackend(),resume=True)
            report=experiment.analyze(p)
            self.assertTrue(report['complete']); self.assertEqual(len(report['failures']),1)
            self.assertEqual(report['failures'][0]['event'],'interrupted_request')

    def test_outcome_changes_in_final_test_cannot_change_frozen_controller(self):
        with patch.dict(experiment.COUNTS,SMALL,clear=True),tempfile.TemporaryDirectory() as tmp:
            p=self.fixture(tmp); events=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
            original=experiment.analyze(p)
            test_ids={t['id'] for t in experiment.make_plan()['tasks'] if t['phase']=='test'}
            for e in events:
                if e['event']=='result' and e['id'] in test_ids:
                    e['text']='not parseable'
                    if e['primaryAction']=='direct': e['finalText']=e['text']
            p.write_text('\n'.join(json.dumps(e) for e in events)+'\n',encoding='utf-8')
            changed=experiment.analyze(p)
            self.assertEqual(original['frozenPolicies'],changed['frozenPolicies'])
            self.assertNotEqual(original['byPhase']['test']['numericCorrect'],changed['byPhase']['test']['numericCorrect'])

    def test_tampered_forecast_policy_fit_tool_and_premature_test_rejected(self):
        with patch.dict(experiment.COUNTS,SMALL,clear=True),tempfile.TemporaryDirectory() as tmp:
            p=self.fixture(tmp); original=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
            mutations=[]
            e=copy.deepcopy(original); next(x for x in e if x['event']=='state' and x['predictions'])['predictions']['internal']=9; mutations.append(e)
            e=copy.deepcopy(original); next(x for x in e if x['event']=='update')['policies']['learned']['threshold']=.123; mutations.append(e)
            e=copy.deepcopy(original); next(x for x in e if x['event']=='fit')['bundle']['models']['internal']['intercept']+=1; mutations.append(e)
            e=copy.deepcopy(original); next(x for x in e if x['event']=='result')['verification']['text']='wrong'; mutations.append(e)
            e=copy.deepcopy(original); e=[x for x in e if not (x['event']=='update' and x['phase']=='round3')]; mutations.append(e)
            for events in mutations:
                p.write_text('\n'.join(json.dumps(e) for e in events)+'\n',encoding='utf-8')
                with self.assertRaises((ValueError,AssertionError)): experiment.analyze(p)


if __name__=='__main__': unittest.main()
