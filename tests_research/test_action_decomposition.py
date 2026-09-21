import copy
import json
from pathlib import Path
import tempfile
import unittest

from research import action_decomposition as s
from research.natural_error_journal import Writer
from research.audit_action_decomposition import verify_tallies


class DecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=s.make_plan()
        pp=s.parent.make_plan()
        cls.prior=dict(header=dict(plan=pp), results=[dict(id=c['id'],text=s.parent.target(pp['testCases'][c['case']],c)) for c in pp['calls']])

    def result(self,c,text):
        return dict(event='result',id=c['id'],status='ok',engine='llm',text=text,seconds=.01,
                    metrics=dict(inputTokens=100,outputTokens=3,reachedTokenLimit=False,effectiveGeneration=self.plan['settings']),errorType=None)

    def fixture(self, double_error=False):
        results=[]
        for c in self.plan['calls']:
            text=s.expected(self.plan,c)
            if double_error and c['stage']!='replay':
                other='verify' if s.decode(self.plan,c,text)=='direct' else 'direct'
                if c['stage']=='mapping':text=s.parent.symbols(c)[other]
                elif c['stage']=='semantic':text=s.NAMES[other]
                else:text=f'{s.losses(self.plan["cases"][c["case"]])[other]/100:.2f}'
            results.append(self.result(c,text))
        return results

    def test_coverage_replay_and_both_imposed_actions(self):
        p=self.plan
        self.assertEqual(p['planned'],2448)
        self.assertTrue(all(c['stage']=='replay' for c in p['calls'][:288]))
        self.assertFalse(any(c['stage']=='replay' for c in p['calls'][288:]))
        for stage,n in [('replay',288),('number',864),('semantic',864),('mapping',432)]:
            self.assertEqual(sum(c['stage']==stage for c in p['calls']),n)
        for c in p['calls']:
            req=s.request_for(p,c['id'])
            if c['stage']=='replay':
                pc=self.prior['header']['plan']['calls'][c['parentId']]
                self.assertEqual(c['seed'],pc['seed'])
                self.assertEqual(req['messages'],s.parent.messages(p['cases'][c['case']],pc))
            if c['stage']=='mapping':
                self.assertNotIn('case',c)
                self.assertIn('Action imposée : '+s.NAMES[c['selected']],req['messages'][1]['content'])
                self.assertEqual(s.decode(p,c,s.expected(p,c)),c['selected'])

    def test_perfect_components_and_wrong_errors_must_not_be_confused(self):
        good=s.summarize(self.plan,self.fixture(),self.prior)
        self.assertTrue(good['complete']);self.assertEqual(good['replayMatched'],288)
        self.assertEqual(len(good['composed']),432)
        for t in good['composed'].values():
            self.assertEqual(t['n'],16);self.assertEqual(t['composedCorrect'],16)
            self.assertEqual(t['choiceCorrect'],16);self.assertEqual(t['compensatingErrors'],0)
        bad=s.summarize(self.plan,self.fixture(double_error=True),self.prior)
        for t in bad['composed'].values():
            self.assertEqual(t['composedCorrect'],16)
            self.assertEqual(t['choiceCorrect'],0);self.assertEqual(t['oracleEncodingCorrect'],0)
            self.assertEqual(t['encodingGivenChoiceCorrect'],0);self.assertEqual(t['compensatingErrors'],16)

    def test_invalid_choice_is_retained_and_cannot_route_to_oracle(self):
        results=self.fixture()
        c=next(c for c in self.plan['calls'] if c['stage']=='semantic')
        results[c['id']]['text']='DIRECT ou VERIFIER'
        summary=s.summarize(self.plan,results,self.prior)
        rows=[v for k,v in summary['composed'].items() if k.startswith('r0/base/semantic/w0/o0/')]
        self.assertEqual(len(rows),4)
        self.assertTrue(all(t['invalidChoice']==1 and t['composedCorrect']==15 for t in rows))

    def test_rehashed_request_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fixture.jsonl';w=Writer(p)
            header=dict(event='header',plan=self.plan,planHash=s.digest(self.plan),sourceHash=s.source_hash(),origin='synthetic_fixture',metadata={})
            w.write(header,create=True);w.write(s.request_for(self.plan,0));w.write(self.fixture()[0])
            self.assertEqual(len(s.read_journal(p)['results']),1)
            rows=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]
            rows[1]['payload']['messages'][1]['content']+='\nChoix optimal : A'
            previous='0'*64
            for row in rows:
                row['previous']=previous;row['sha256']=s.digest({k:row[k] for k in ('sequence','previous','payload')});previous=row['sha256']
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            with self.assertRaises(ValueError):s.read_journal(p)

    def test_distinct_tally_rejects_modified_summary(self):
        results=self.fixture(double_error=True)
        report=s.summarize(self.plan,results,self.prior)
        self.assertTrue(verify_tallies(self.plan,results,self.prior,report)['verified'])
        wrong=copy.deepcopy(report);next(iter(wrong['composed'].values()))['compensatingErrors']=0
        with self.assertRaises(ValueError):verify_tallies(self.plan,results,self.prior,wrong)


if __name__=='__main__':unittest.main()
