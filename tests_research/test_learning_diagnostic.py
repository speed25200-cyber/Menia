import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import learning_diagnostic as p


def trace(family,pos):
    actual=p.trace_position(family,pos)
    return dict(applications=int(actual!=0),changed=actual in (1,2),normRelativeError=.0001 if actual in (1,2) else 0.)


def fixture(path, *, constant=False):
    fixed=p.plan();blocks={b['id']:b for b in fixed['blocks']};trained={}
    events=[dict(event='header',plan=fixed,planHash=p.digest(fixed),sourceHash=p.source_hash(),
                 metadata=dict(origin='synthetic_fixture',choiceTokenIds=[10,11,12]))]
    for arm in p.ARMS:
        events.append(dict(event='training_start',arm=arm,groups=len(fixed['training']),initializationSeed=p.SEED))
        for i,g in enumerate(fixed['training'],1):
            events.append(dict(event='training_step',arm=arm,step=i,group=g,primaryLosses=[1.,1.,1.],readingLosses=[1.,1.],
                gradientNorm=1.,seconds=.01,traces=[trace(arm,i) for i in range(3)]+[trace(arm,0)]*2))
        trained[arm]=p.digest(arm)
        events.append(dict(event='training_complete',arm=arm,steps=len(fixed['training']),sha256=trained[arm]))
    for r in fixed['evaluation']:
        b=blocks[r['block']]
        events.append(dict(event='request',request=r,promptHash=p.digest(p.prompt(b,r['family'],r['task'],r['position'])[0]),
                           adapterHash=trained.get(r['arm']),inputTokens=100))
        y=p.target(b,r['task'],r['position'])
        if constant or (r['task']=='primary' and r['arm'] in ('base','shuffled')): y=2
        events.append(dict(event='result',id=r['id'],seconds=.01,choiceLogits=[5. if i==y else 0. for i in range(3)],
            choiceMass=.99,rawTokenId=10+y,rawChoice=y,intervention=trace(r['family'],r['position'])))
    path.write_text('\n'.join(json.dumps(e) for e in events)+'\n',encoding='utf-8')
    return events


class DiagnosticTests(unittest.TestCase):
    def test_fixed_plan_new_sentences_balanced_groups_and_budget(self):
        fixed=p.plan();self.assertEqual(fixed,p.plan())
        old={s for b in p.previous_plan()['blocks'] for s in b['sentences']}
        new=[s for b in fixed['blocks'] for s in b['sentences']]
        self.assertEqual(len(set(new)),64);self.assertFalse(old&set(new))
        self.assertEqual(len(fixed['training'])*5*len(p.ARMS),640)
        self.assertEqual(len(fixed['evaluation']),1792)
        self.assertEqual(len(set(r['id'] for r in fixed['evaluation'])),1792)
        self.assertTrue(all(b['split']=='train' for g in fixed['training'] for b in fixed['blocks'] if g['block']==b['id']))

    def test_hidden_targets_do_not_change_prompt_visible_control_and_sham_do(self):
        b=p.plan()['blocks'][0]
        hidden=[p.prompt(b,f,'primary',i) for f in ('weak','strong','shuffled') for i in range(4)]
        self.assertTrue(all(x==hidden[0] for x in hidden))
        self.assertEqual(p.prompt(b,'visible','primary',0),p.prompt(b,'visible','primary',3))
        self.assertNotEqual(p.prompt(b,'visible','primary',1),p.prompt(b,'visible','primary',2))
        other=dict(b,noiseSeed=999,shuffledTargets=[2,0,1])
        self.assertEqual(p.prompt(b,'strong','primary',2),p.prompt(other,'strong','primary',2))

    def test_known_signal_and_constant_response_are_distinguished(self):
        with patch.dict(p.CONFIG,dict(trainBlocks=2,testBlocks=4,epochs=1)),tempfile.TemporaryDirectory() as t:
            path=Path(t)/'x.jsonl';fixture(path)
            report=p.analyze(path)
            self.assertTrue(report['complete']);self.assertEqual(report['shamPairs'],84)
            self.assertTrue(report['diagnostic']['visibleTrainingAtLeast90Percent'])
            self.assertEqual(report['tables']['test/strong/strong/primary']['positionAccuracy'],1.)
            fixture(path,constant=True);report=p.analyze(path)
            self.assertFalse(report['diagnostic']['visibleTrainingAtLeast90Percent'])
            self.assertEqual(report['tables']['test/strong/strong/primary']['positionAccuracy'],.5)
            self.assertEqual(report['tables']['test/strong/strong/primary']['absentAccuracy'],0.)

    def test_tampering_order_private_conditions_and_sham_rejected(self):
        with patch.dict(p.CONFIG,dict(trainBlocks=2,testBlocks=4,epochs=1)),tempfile.TemporaryDirectory() as t:
            path=Path(t)/'x.jsonl';original=fixture(path)
            changes=[]
            e=copy.deepcopy(original);next(x for x in e if x['event']=='training_step')['primaryLosses'].pop();changes.append(e)
            e=copy.deepcopy(original);e=[x for x in e if not (x['event']=='training_complete' and x['arm']=='shuffled')];changes.append(e)
            e=copy.deepcopy(original);next(x for x in e if x['event']=='request')['inputTokens']=513;changes.append(e)
            e=copy.deepcopy(original);next(x for x in e if x['event']=='result')['rawChoice']=9;changes.append(e)
            e=copy.deepcopy(original);next(x for x in e if x['event']=='result' and x['id'].endswith('/3'))['choiceLogits'][0]+=.1;changes.append(e)
            for events in changes:
                path.write_text('\n'.join(json.dumps(x) for x in events)+'\n',encoding='utf-8')
                with self.assertRaises(ValueError): p.analyze(path)

    def test_partial_and_explicit_interruption_preserved(self):
        with patch.dict(p.CONFIG,dict(trainBlocks=2,testBlocks=4,epochs=1)),tempfile.TemporaryDirectory() as t:
            path=Path(t)/'x.jsonl';events=fixture(path)
            i=next(i for i,e in enumerate(events) if e['event']=='request')
            prefix=events[:i+1]
            path.write_text('\n'.join(json.dumps(x) for x in prefix)+'\n',encoding='utf-8')
            report=p.analyze(path);self.assertFalse(report['complete']);self.assertTrue(report['pendingRequest'])
            interruption=dict(event='interrupted_request',id=events[i]['request']['id'])
            all_events=prefix+[interruption]+events[i:]
            path.write_text('\n'.join(json.dumps(x) for x in all_events)+'\n',encoding='utf-8')
            report=p.analyze(path);self.assertTrue(report['complete']);self.assertEqual(report['interruptedRequests'],1)

    def test_shuffled_training_fit_is_separate_from_true_localization(self):
        with patch.dict(p.CONFIG,dict(trainBlocks=2,testBlocks=4,epochs=1)),tempfile.TemporaryDirectory() as t:
            path=Path(t)/'x.jsonl';events=fixture(path)
            blocks={b['id']:b for b in p.plan()['blocks']}
            for e in events:
                if e['event']=='result' and e['id'].startswith('shuffled/shuffled/train-') and '/primary/' in e['id']:
                    _,_,bid,_,pos=e['id'].split('/')
                    y=blocks[bid]['shuffledTargets'][int(pos)%3]
                    e.update(choiceLogits=[5. if i==y else 0. for i in range(3)],rawChoice=y,rawTokenId=10+y)
            path.write_text('\n'.join(json.dumps(x) for x in events)+'\n',encoding='utf-8')
            r=p.analyze(path);score=r['tables']['train/shuffled/shuffled/primary']
            self.assertEqual(score['trainingTargetAccuracy'],1.)
            self.assertLess(score['correct'],1.)
            self.assertTrue(r['diagnostic']['shuffled']['trainingAtLeast90Percent'])


if __name__=='__main__': unittest.main()
