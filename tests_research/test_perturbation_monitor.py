"""Synthetic controls validate measurement, not pretrained Qwen performance."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research import perturbation_monitor as m
from research.activation_monitor import make_plan as previous_plan


class SyntheticBackend:
    origin = 'synthetic_fixture'
    metadata = {'description':'Artificial intervention controls, not a language model'}

    def __init__(self,path,mode_only=False):
        self.path,self.mode_only = path,mode_only

    def generate(self,task,capture):
        mode = 'baseline' if task['condition'] == 'sham' else task['condition']
        if self.mode_only:
            good = mode == 'baseline'
        else:
            good = bool(int(m.digest([task['qid'],mode,'outcome'])[:8],16) % 2)
        state = {k:[0.]*m.DIM for k in ('input','middle','final')}
        state['middle'][0] = 3. if good else -3.
        capture(state)
        # Check the durable commitment occurs before the backend returns output.
        with self.path.open('rb') as f:
            f.seek(max(0,self.path.stat().st_size-20000))
            event=json.loads(f.read().splitlines()[-1])
        assert event['event'] == 'state'
        assert (event['actions'] is not None) == (task['split'] == 'test')
        expected = m.verify_question(task['question'])
        return str(int(expected)+(0 if good else 1)),{'synthetic':True}


class PerturbationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name)/'positive.jsonl'
        with contextlib.redirect_stdout(io.StringIO()):
            m.collect(cls.path,SyntheticBackend(cls.path))
        cls.header,cls.rows,_,cls.bundle,cls.gate = m.read_journal(cls.path)
        cls.report = m.analyze(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_disjoint_question_groups_hidden_conditions_and_matching_seeds(self):
        plan = m.make_plan()
        self.assertEqual(len(plan['tasks']),704)
        questions = {t['question'] for t in plan['tasks']}
        self.assertEqual(len(questions),176)
        self.assertFalse(questions & {t['question'] for t in previous_plan()['tasks']})
        for start in range(0,704,4):
            group=plan['tasks'][start:start+4]
            self.assertEqual({t['condition'] for t in group},set(m.CONDITIONS))
            for key in ('question','qid','split','seed','noiseSeed'):
                self.assertEqual(len({t[key] for t in group}),1)
            self.assertTrue(all(m.messages(t)==m.messages(group[0]) for t in group))

    def test_planted_error_signal_improves_baselines_and_executed_routes(self):
        r=self.report
        self.assertTrue(r['complete'] and r['fitChecked'] and r['gate']['passed'])
        self.assertEqual(r['independentTestQuestions'],48)
        self.assertEqual(r['availableTestTargets'],192)
        self.assertEqual(r['pairedInterventionEffects']['sham']['sameText'],48)
        self.assertLess(r['scores']['internal']['brier'],.05)
        self.assertEqual(r['scores']['internal']['finalCorrect'],192)
        for n in ('betaCell','oracleCondition','inputOnly','shuffledLabels','donorState'):
            self.assertGreater(r['scores'][n]['brier'],.1)
            self.assertGreater(r['contrasts'][n+':brier']['descriptiveQuestionClusterBootstrap95'][0],0)

    def test_condition_only_signal_does_not_beat_same_condition_donor(self):
        path=Path(self.tmp.name)/'mode-only.jsonl'
        with contextlib.redirect_stdout(io.StringIO()):
            m.collect(path,SyntheticBackend(path,mode_only=True))
        r=m.analyze(path)
        self.assertAlmostEqual(r['scores']['internal']['brier'],r['scores']['donorState']['brier'],places=12)
        self.assertLess(r['scores']['oracleCondition']['brier'],.001)

    def test_fitting_excludes_gate_and_all_test_information(self):
        poisoned=copy.deepcopy(self.rows)
        for row in poisoned:
            if row['task']['split'] in ('gate','test'):
                row['result']={'secret':'do not inspect'}
                row['state']={'secret':'do not inspect'}
        self.assertEqual(m.canonical(m.fit_bundle(poisoned)),m.canonical(self.bundle))

    def test_real_verifier_and_committed_action_are_used(self):
        task=dict(question='Calcule 83 - 94.')
        with patch.object(m,'verify_question',wraps=m.verify_question) as tool:
            out=m.execute_routes(task,'99',{'a':'direct','b':'verify'})
        self.assertEqual(out['a']['text'],'99')
        self.assertEqual(out['b']['text'],'-11')
        tool.assert_called_once_with(task['question'])
        self.assertEqual(m.verify_question('Combien de lettres A contient cette chaîne : AABCDADA ?'),'4')

    def test_tampered_forecast_action_route_and_missing_gate_rejected(self):
        events=[json.loads(s) for s in self.path.read_text(encoding='utf-8').splitlines()]
        path=Path(self.tmp.name)/'tampered.jsonl'
        for kind in ('forecast','action','route','gate'):
            changed=copy.deepcopy(events)
            state=next(e for e in changed if e['event']=='state' and e['predictions'] is not None)
            if kind=='forecast':
                state['predictions']['internal']=1-state['predictions']['internal']
            elif kind=='action':
                state['actions']['internal']='invalid'
            elif kind=='route':
                result=next(e for e in changed if e['event']=='result' and e['routes'] is not None)
                result['routes']['internal']['text']='999999'
            else:
                changed=[e for e in changed if e['event']!='gate']
            path.write_text('\n'.join(m.canonical(e) for e in changed)+'\n',encoding='utf-8')
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                m.read_journal(path)

    def test_sham_mismatch_rejected_without_selecting_accuracy(self):
        group=copy.deepcopy(self.rows[:4])
        self.assertIsNone(m.check_pair(group))
        next(r for r in group if r['task']['condition']=='sham')['result']['text']='broken'
        with self.assertRaisesRegex(ValueError,'Sham changed'):
            m.check_pair(group)

    def test_interrupted_request_is_not_replayed(self):
        path=Path(self.tmp.name)/'partial.jsonl'
        with contextlib.redirect_stdout(io.StringIO()):
            m.collect(path,SyntheticBackend(path),limit=0)
            t=m.make_plan()['tasks'][0]
            m.append(path,dict(event='request',task=t,messages=m.messages(t)))
            m.collect(path,SyntheticBackend(path),resume=True,limit=1)
        _,rows,pending,_,_=m.read_journal(path)
        self.assertEqual([r['result']['status'] for r in rows],['interrupted','ok'])
        self.assertIsNone(pending)
        self.assertFalse(m.analyze(path)['complete'])


if __name__=='__main__':
    unittest.main()
