import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research import presence_specificity as s


def fixture(path, kind='semantic', visible_failure=False):
    fixed=s.plan();blocks={b['id']:b for b in fixed['blocks']};frozen=s.checkpoints()
    events=[dict(event='header',plan=fixed,planHash=s.parent.digest(fixed),sourceHash=s.source_hash(),
        parentJournalSHA256=s.PARENT_JOURNAL_SHA256,checkpoints=frozen,
        metadata=dict(origin='synthetic_fixture',choiceTokenIds=[10,11,12]))]
    for req in fixed['evaluation']:
        b=blocks[req['block']];pos=req['position'];p=int(pos in (1,2));q=req['question']
        events.append(dict(event='request',request=req,promptHash=s.parent.digest(s.prompt(b,req['family'],q,pos)[0]),
            adapterHash=frozen.get(f"r{req['replication']}-{req['arm']}"),inputTokens=110))
        y=s.target(b,q,pos)
        if req['family']=='visible': z=6*y-3
        elif req['arm'] in ('base','shuffled','visible'): z=-3.
        elif kind=='fixed': z=6*p-3.
        else: z=6*y-3.
        if visible_failure and req['family']=='visible' and req['arm']=='strong': z=-3.
        raw=int(z>0);changed=req['family']=='hidden' and p==1
        events.append(dict(event='result',id=req['id'],seconds=.01,choiceLogits=[0.,z,-20.],choiceMass=.99,
            rawChoice=raw,rawTokenId=10+raw,intervention=dict(applications=int(changed or pos==3),changed=changed,normRelativeError=.001 if changed else 0.)))
    write(path,events);return events


def write(path,events): path.write_text('\n'.join(json.dumps(e) for e in events)+'\n',encoding='utf-8')


class SpecificityTests(unittest.TestCase):
    def test_disjoint_fixed_balanced_plan(self):
        fixed=s.plan();self.assertEqual(fixed,s.plan())
        old=(s.parent.plan()['blocks']+s.parent.previous.plan()['blocks']+
             s.parent.previous.previous.plan()['blocks']+s.parent.previous.previous.previous_plan()['blocks'])
        self.assertEqual(len(fixed['evaluation']),9216);self.assertEqual(len({r['id'] for r in fixed['evaluation']}),9216)
        self.assertEqual(len(fixed['blocks']),72)
        self.assertEqual(len({x for b in fixed['blocks'] for x in b['sentences']}),144)
        self.assertFalse({x for b in fixed['blocks'] for x in b['sentences']}&{x for b in old for x in b['sentences']})
        self.assertEqual(len({b['noiseSeed'] for b in fixed['blocks']}),72)
        self.assertFalse({b['noiseSeed'] for b in fixed['blocks']}&{b['noiseSeed'] for b in old})
        for rep in range(3):
            bs=[b for b in fixed['blocks'] if b['replication']==rep]
            self.assertEqual(sum(b['marker'] for b in bs),12)
        self.assertEqual(set(fixed['checkpoints']),{f'r{r}-{a}' for r in range(3) for a in ('visible','strong','shuffled')})

    def test_hidden_condition_unobservable_in_text_and_complement_mappings(self):
        for b in s.plan()['blocks']:
            for question in s.QUESTIONS:
                texts=[s.prompt(b,'hidden',question,pos) for pos in range(4)]
                self.assertTrue(all(t==texts[0] for t in texts))
                body,spans=texts[0];self.assertEqual([body[a:z] for a,z in spans],b['sentences'])
                for pos in range(4):
                    self.assertEqual(s.target(b,'presence',pos)+s.target(b,'absence',pos),1)
                    self.assertEqual(s.target(b,'marker_first',pos)+s.target(b,'marker_second',pos),1)
            self.assertEqual(s.prompt(b,'hidden','presence',0),s.parent.prompt(b,'strong','primary',0))
            self.assertEqual(s.prompt(b,'visible','presence',1),s.parent.prompt(b,'visible','primary',1))
            self.assertEqual(s.prompt(b,'visible','presence',0),s.prompt(b,'visible','presence',3))
            self.assertNotEqual(s.prompt(b,'visible','presence',0),s.prompt(b,'visible','presence',1))

    def test_semantic_oracle_passes_and_fixed_response_detector_is_rejected(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(s.CONFIG,dict(blocksPerReplication=4,resamples=50)):
            path=Path(d)/'f.jsonl';fixture(path);r=s.analyze(path)
            self.assertTrue(r['complete']);self.assertTrue(r['fixedReadingRuleMet'])
            self.assertEqual(r['shamPairs'],384)
            fixture(path,kind='fixed');r=s.analyze(path)
            self.assertFalse(r['mappingSignalRuleMet']);self.assertTrue(r['visibleInstructionGateMet'])
            for rep in range(3):
                self.assertEqual(r['tables'][f'{rep}/strong/hidden/presence']['orientedPresenceAUROC'],1.)
                self.assertEqual(r['tables'][f'{rep}/strong/hidden/absence']['orientedPresenceAUROC'],0.)
                self.assertEqual(r['tables'][f'{rep}/strong/hidden/marker_first']['rawShift'],6.)
                c=[c for c in r['contrasts'] if c['kind']=='mappingInteraction' and c['primary'] and c['replication']==rep]
                self.assertEqual(c[0]['difference'],0.)

    def test_good_hidden_scores_do_not_hide_failed_instruction_control(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(s.CONFIG,dict(blocksPerReplication=4,resamples=20)):
            path=Path(d)/'f.jsonl';fixture(path,visible_failure=True);r=s.analyze(path)
            self.assertTrue(r['mappingSignalRuleMet']);self.assertFalse(r['visibleInstructionGateMet'])
            self.assertFalse(r['fixedReadingRuleMet'])

    def test_tampering_and_sham_difference_fail_and_partial_is_not_success(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(s.CONFIG,dict(blocksPerReplication=2,replications=1,resamples=10)):
            path=Path(d)/'f.jsonl';events=fixture(path)
            write(path,events[:2]);self.assertFalse(s.analyze(path)['complete'])
            for field in ('adapterHash','promptHash'):
                bad=copy.deepcopy(events);bad[1][field]='0'*64;write(path,bad)
                with self.assertRaises(ValueError): s.analyze(path)
            bad=copy.deepcopy(events);bad[0]['parentJournalSHA256']='0'*64;write(path,bad)
            with self.assertRaises(ValueError): s.analyze(path)
            bad=copy.deepcopy(events)
            for i,e in enumerate(bad):
                if e['event']=='request' and e['request']['position']==3:
                    bad[i+1]['choiceLogits'][0]-=.01;break
            write(path,bad)
            with self.assertRaisesRegex(ValueError,'Sham'): s.analyze(path)

    def test_auroc_orientation_matches_direct_pair_enumeration(self):
        rng=np.random.default_rng(31);a=rng.normal(size=(6,3))
        for sign in (-1,1):
            z=sign*a
            direct=np.mean([float(x>y)+.5*float(x==y) for x in z[:,1:].flat for y in z[:,0]])
            self.assertAlmostEqual(s.parent.auroc(s.parent.wins(z)),direct)
        self.assertAlmostEqual(s.parent.auroc(s.parent.wins(a))+s.parent.auroc(s.parent.wins(-a)),1.)


if __name__=='__main__': unittest.main()
