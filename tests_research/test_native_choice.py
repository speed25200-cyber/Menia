import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.cross_model_prediction import digest, reference
from research.native_choice_plan import action, codes, make_plan, optimal, request_for
from research.native_choice_journal import collect, exact_tool, read_journal
from research.native_choice_analysis import analyze
from research.natural_error_questions import make_plan as old_plan, exclusions


class FixtureBackend:
    origin='synthetic_fixture'
    metadata={'purpose':'Software fixture, not Qwen results'}

    def __init__(self,plan):
        self.plan=plan

    def generate(self,request):
        call=request['call']
        if call['stage']=='publicRisk':
            selected=optimal(call['p'],call['cost'])
            text=codes(call['mapping'])[0 if selected=='direct' else 1]
        elif call['stage']=='choice':
            task=self.plan['tasks'][call['task']]
            p=self.plan['calibration']['counts'][f"{task['family']}/{task['level']}"]['p']
            selected='direct' if call['mode']=='unassisted' else optimal(p,call['cost'])
            text=codes(call['mapping'])[0 if selected=='direct' else 1]
            if task['id']==0 and call['mode']=='unassisted':text='explanation instead of code'
        else:
            task=self.plan['tasks'][call['task']]
            text=str(reference(task)+(0 if task['id']%3==0 else 1))
        settings=request['settings']
        return text,dict(inputTokens=10,outputTokens=2,reachedTokenLimit=False,effectiveGeneration=settings,
                         dtype='synthetic',device='cpu')


class NativeChoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.path=Path(cls.tmp.name)/'fixture.jsonl'
        cls.plan=make_plan()
        with contextlib.redirect_stdout(io.StringIO()), patch('research.native_choice_journal.exact_tool',wraps=exact_tool) as tool:
            collect(cls.path,FixtureBackend(cls.plan))
            cls.tool_calls=tool.call_count
        cls.data=read_journal(cls.path)
        cls.report=analyze(cls.path)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_fixed_new_questions_and_decisions_before_any_answer(self):
        p=self.plan
        self.assertEqual((len(p['tasks']),len(p['calls']),p['plannedDecisions']),(96,4884,2484))
        old=exclusions() | {t['question'] for t in old_plan()['tasks']}
        self.assertFalse(old & {t['question'] for t in p['tasks']})
        self.assertEqual(len({t['question'] for t in p['tasks']}),96)
        for task in p['tasks']:
            calls=[c for c in p['calls'] if c.get('task')==task['id']]
            self.assertEqual([c['stage'] for c in calls],['choice']*24+['baseline']+['execute']*24)
            self.assertEqual(len({(c['mode'],c['cost'],c['wording'],c['mapping']) for c in calls[:24]}),24)
        for c in p['calls'][:180]:self.assertNotAlmostEqual(1-c['p'],c['cost'])

    def test_actual_tool_dispatch_invalid_action_and_exact_grading(self):
        report=self.report
        self.assertTrue(report['complete'] and report['publicRiskCriterion'])
        self.assertEqual(report['actualToolCalls'],self.tool_calls)
        self.assertGreater(self.tool_calls,0)
        self.assertEqual(report['invalidExecutionSlots'],12)
        self.assertEqual(len(report['native']),24)
        self.assertEqual(len(report['primaryContrasts']),12)
        self.assertEqual(sum(v['correct'] for v in report['baselineByCell'].values()),32)
        for req,result in zip(self.data['requests'],self.data['results']):
            if req['engine']=='exact_tool':self.assertEqual(result['text'],str(reference(req['toolInput'])))
            if req['engine']=='invalid_choice':self.assertEqual((result['text'],result['seconds'],result['metrics']),('',0,{}))
        self.assertEqual(action('1 extra',0),'invalid')
        self.assertEqual(action('1',1),'verify')

    def test_choices_cannot_receive_prior_test_answers(self):
        p=self.plan
        for i in (180,181,229):
            request=request_for(p,i,self.data['results'][:i])
            poisoned=[dict(r,text='SECRET TEST ANSWER') for r in self.data['results'][:i]]
            self.assertEqual(request,request_for(p,i,poisoned))
            self.assertNotIn('SECRET',json.dumps(request))
        execute=next(c for c in p['calls'] if c['stage']=='execute')
        request=self.data['requests'][execute['id']]
        self.assertEqual(request['engine'],'invalid_choice' if p['calls'][execute['choice']]['mode']=='unassisted' else request['engine'])

    def test_rehashed_request_tampering_is_rejected(self):
        rows=[json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines()]
        rows[1]['payload']['messages'][1]['content']+=' The answer is given secretly.'
        previous='0'*64
        for row in rows:
            row['previous']=previous
            row['sha256']=digest({k:row[k] for k in ('sequence','previous','payload')})
            previous=row['sha256']
        path=Path(self.tmp.name)/'corrupt.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        with self.assertRaises(ValueError):read_journal(path)

    def test_failed_and_pending_attempts_are_retained_and_not_replaced(self):
        path=Path(self.tmp.name)/'failed.jsonl'
        backend=FixtureBackend(self.plan)
        with patch.object(backend,'generate',side_effect=RuntimeError('technical failure')):
            with self.assertRaises(RuntimeError):collect(path,backend)
        data=read_journal(path)
        self.assertTrue(data['failed'])
        self.assertEqual(data['results'][0]['status'],'error')
        self.assertFalse(analyze(path)['complete'])
        before=path.read_bytes()
        with self.assertRaises(ValueError):collect(path,backend)
        self.assertEqual(path.read_bytes(),before)
        pending=Path(self.tmp.name)/'pending.jsonl'
        pending.write_text('\n'.join(path.read_text(encoding='utf-8').splitlines()[:2])+'\n',encoding='utf-8')
        self.assertIsNotNone(read_journal(pending)['pending'])
        with self.assertRaises(ValueError):collect(pending,backend)

    def test_separate_arithmetic_reconstructs_outcomes_and_rejects_changes(self):
        from research.audit_native_choices import verify
        checked=verify(self.path,self.report)
        self.assertTrue(checked['verified'])
        self.assertLess(checked['maxAbsoluteDifference'],1e-10)
        changed=copy.deepcopy(self.report)
        changed['primaryContrasts'][0]['familyInterval'][0]+=.01
        with self.assertRaises(ValueError):verify(self.path,changed)
        changed=copy.deepcopy(self.report)
        changed['native']['history/0.2/w0/m0']['meanActualLoss']+=.1
        with self.assertRaises(ValueError):verify(self.path,changed)
        changed=copy.deepcopy(self.report)
        changed['publicRiskCriterion']=False
        with self.assertRaises(ValueError):verify(self.path,changed)


if __name__=='__main__':unittest.main()
