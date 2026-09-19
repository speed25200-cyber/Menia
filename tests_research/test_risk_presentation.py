import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.risk_presentation import make_plan,request_for,decode,analyze
from research.risk_presentation_journal import collect,read_journal
from research.audit_risk_presentation import verify
from research.cross_model_prediction import digest


class Fixture:
    origin='synthetic_fixture';metadata={'purpose':'Software checks, not LLM evidence'}
    def __init__(self,first=False):self.plan=make_plan();self.first=first
    def generate(self,request):
        c=request['call'];case=self.plan['cases'][c['case']]
        direct=c['order']==0 if self.first else case['pPercent']+case['costCents']>100
        text=str(1+c['mapping'] if direct else 2-c['mapping'])
        if not self.first and c['repetition']==0:text='Code '+text+'.'
        return text,dict(inputTokens=10,outputTokens=3,reachedTokenLimit=False,effectiveGeneration=request['settings'],device='cpu',dtype='synthetic')


class RiskPresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.path=Path(cls.temp.name)/'fixture.jsonl'
        with contextlib.redirect_stdout(io.StringIO()):collect(cls.path,Fixture())
        cls.report=analyze(cls.path)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_factors_are_independent_balanced_and_no_target_is_supplied(self):
        p=make_plan();self.assertEqual(p['planned'],864)
        self.assertEqual(sum(c['pPercent']+c['costCents']>100 for c in p['cases']),9)
        self.assertTrue(all(c['pPercent']+c['costCents']!=100 for c in p['cases']))
        for c in p['calls']:
            twin=next(x for x in p['calls'] if all(x[k]==c[k] for k in ('case','information','wording','mapping','repetition')) and x['order']!=c['order'])
            a=request_for(p,c['id']);b=request_for(p,twin['id'])
            self.assertEqual(a['messages'][0],b['messages'][0]);self.assertEqual(c['seed'],twin['seed'])
            a_lines=a['messages'][1]['content'].splitlines();b_lines=b['messages'][1]['content'].splitlines()
            self.assertEqual(a_lines[:2],b_lines[:2][::-1])
            self.assertNotIn('optimal',json.dumps(a['messages']))
            if c['information']=='probability':self.assertEqual(a_lines[2],b_lines[2])
            else:
                self.assertLess(a_lines[2].index('réponse directe'),a_lines[2].index('vérification externe')) if c['order']==0 else self.assertGreater(a_lines[2].index('réponse directe'),a_lines[2].index('vérification externe'))

    def test_distinguishes_format_from_decision_and_verifies_independently(self):
        r=self.report;self.assertTrue(r['complete']);self.assertEqual(len(r['groups']),16)
        for t in r['groups'].values():self.assertEqual((t['n'],t['strictCorrect'],t['semanticCorrect'],t['strictInvalid']),(54,36,54,18))
        for t in r['orderTransitions'].values():self.assertEqual((t['semanticBothCorrect'],t['semanticChanged']),(54,0))
        self.assertTrue(verify(self.path,r)['verified'])
        for raw in ('1','Code 1','cOdE\t1.'):self.assertEqual(decode(raw,0,True),'direct')
        for raw in ('1 or 2','Code 1 because','11','Code1','1. explanation'):self.assertEqual(decode(raw,0,True),'invalid')
        changed=copy.deepcopy(r);next(iter(changed['groups'].values()))['semanticCorrect']-=1
        with self.assertRaises(ValueError):verify(self.path,changed)

    def test_first_option_heuristic_is_detected_instead_of_declared_competent(self):
        path=Path(self.temp.name)/'first.jsonl'
        with contextlib.redirect_stdout(io.StringIO()):collect(path,Fixture(first=True))
        r=analyze(path);self.assertTrue(verify(path,r)['verified'])
        for t in r['groups'].values():self.assertEqual((t['strictCorrect'],t['strictFirst']),(27,54))
        for t in r['orderTransitions'].values():self.assertEqual((t['strictChanged'],t['strictBothCorrect']),(54,0))
        self.assertFalse(any(v for g in r['capabilityGates'].values() for v in g.values()))

    def test_rehashed_prompt_tampering_and_retries_are_rejected(self):
        rows=[json.loads(l) for l in self.path.read_text(encoding='utf-8').splitlines()]
        rows[1]['payload']['messages'][1]['content']+=' Use this answer instead.'
        previous='0'*64
        for row in rows:
            row['previous']=previous;row['sha256']=digest({k:row[k] for k in ('sequence','previous','payload')});previous=row['sha256']
        path=Path(self.temp.name)/'tampered.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        with self.assertRaises(ValueError):read_journal(path)
        failed=Path(self.temp.name)/'failed.jsonl'
        with patch.object(Fixture,'generate',side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):collect(failed,Fixture())
        self.assertTrue(read_journal(failed)['failed']);self.assertFalse(analyze(failed)['complete'])
        before=failed.read_bytes()
        with self.assertRaises(ValueError):collect(failed,Fixture())
        self.assertEqual(before,failed.read_bytes())


if __name__=='__main__':unittest.main()
