import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research import presence_detection as p


def trace(family, pos):
    changed=family!='visible' and pos in (1,2)
    return dict(applications=int(changed or pos==3),changed=changed,normRelativeError=.0001 if changed else 0.)


def fixture(path, *, constant=False, blockBias=False):
    fixed=p.plan(); blocks={b['id']:b for b in fixed['blocks']}; trained={}
    events=[dict(event='header',plan=fixed,planHash=p.digest(fixed),sourceHash=p.source_hash(),
                 metadata=dict(origin='synthetic_fixture',choiceTokenIds=[10,11,12]))]
    for unit in fixed['training']:
        events.append(dict(event='training_start',unit=unit))
        for i,g in enumerate(unit['groups'],1):
            events.append(dict(event='training_step',key=unit['key'],step=i,group=g,
                primaryLosses=[1.,1.,1.],readingLosses=[1.,1.],gradientNorm=1.,seconds=.01,
                traces=[trace(unit['arm'],pos) for pos in (0,1,2,0,0)]))
        trained[unit['key']]=p.digest(unit['key'])
        events.append(dict(event='training_complete',key=unit['key'],steps=len(unit['groups']),sha256=trained[unit['key']]))
    order={b['id']:i for i,b in enumerate(fixed['blocks'])}
    for req in fixed['evaluation']:
        b=blocks[req['block']]; body,_=p.prompt(b,req['family'],req['task'],req['position'],req['layout'])
        events.append(dict(event='request',request=req,promptHash=p.digest(body),
            adapterHash=trained.get(f"r{req['replication']}-{req['arm']}"),inputTokens=100))
        y=p.target(b,req['task'],req['position'],req['layout'])
        informed=not constant and not (req['task']=='primary' and req['arm'] in ('base','shuffled'))
        if not informed: y=1
        logits=[5. if i==y else 0. for i in range(3)]
        if blockBias and req['task']=='primary':
            # A block-specific offset far larger than the effect: only paired ordering survives intact.
            logits=[0.,(1. if informed and req['position'] in (1,2) else 0.)+100.*(order[b['id']]%2),-50.]
        raw=max(range(3),key=lambda i:(logits[i],-i))
        events.append(dict(event='result',id=req['id'],seconds=.01,choiceLogits=logits,choiceMass=.99,
            rawChoice=raw,rawTokenId=10+raw,intervention=trace(req['family'],req['position'])))
    write(path,events)
    return events


def write(path,events):
    path.write_text('\n'.join(json.dumps(e) for e in events)+'\n',encoding='utf-8')


class PresenceDetectionTests(unittest.TestCase):
    def test_fixed_budget_disjoint_data_conditions_and_initializations(self):
        fixed=p.plan(); self.assertEqual(fixed,p.plan())
        old=p.previous.plan()['blocks']+p.previous.previous.plan()['blocks']+p.previous.previous.previous_plan()['blocks']
        sentences=[s for b in fixed['blocks'] for s in b['sentences']]
        self.assertEqual(len(set(sentences)),240)
        self.assertFalse(set(sentences)&{s for b in old for s in b['sentences']})
        self.assertEqual(len({b['noiseSeed'] for b in fixed['blocks']}),120)
        self.assertFalse({b['noiseSeed'] for b in fixed['blocks']}&{b['noiseSeed'] for b in old})
        self.assertEqual(len(fixed['evaluation']),12864)
        self.assertEqual(len({r['id'] for r in fixed['evaluation']}),12864)
        self.assertEqual(sum(len(u['groups']) for u in fixed['training']),576)
        self.assertFalse(set(p.INITIALIZATIONS)&set(p.previous.INITIALIZATIONS))
        self.assertEqual(list(p.CONDITIONS)[0],'trained')
        self.assertEqual(len({(c['layer'],c['strength']) for c in p.CONDITIONS.values()}),4)
        by={b['id']:b for b in fixed['blocks']}
        for r in fixed['evaluation']:
            if r['family']=='visible' or by[r['block']]['split']=='train' or r['task']=='marker':
                self.assertEqual(r['condition'],'trained')
            if by[r['block']]['split']=='train': self.assertEqual(r['layout'],'canonical')
        for rep in range(3):
            units=[u for u in fixed['training'] if u['replication']==rep]
            self.assertEqual(len({u['initializationSeed'] for u in units}),1)
            self.assertTrue(all(u['groups']==units[0]['groups'] for u in units))
            self.assertTrue(all(by[g['block']]['split']=='train' and by[g['block']]['replication']==rep for u in units for g in u['groups']))

    def test_hidden_text_never_reveals_condition_and_labels_are_balanced(self):
        block=dict(sentences=['alpha.','beta.'],marker=0,noiseSeed=21,shuffledTargets=[2,0,1])
        for layout,texts in (('canonical',['alpha.','beta.']),('both_reversed',['beta.','alpha.'])):
            body,spans=p.prompt(block,'strong','primary',1,layout)
            self.assertEqual([body[lo:hi] for lo,hi in spans],texts)
            self.assertIn('0 ou 1',body);self.assertNotIn('SIGNAL',body)
            for family in ('strong','shuffled'):
                for pos in range(4): self.assertEqual(p.prompt(block,family,'primary',pos,layout),(body,spans))
            self.assertEqual(p.prompt(dict(block,noiseSeed=999,shuffledTargets=[0,1,2]),'strong','primary',2,layout),(body,spans))
            self.assertEqual([p.target(block,'primary',i,layout) for i in range(4)],[0,1,1,0])
            visible=[p.prompt(block,'visible','primary',pos,layout)[0] for pos in range(4)]
            self.assertEqual([v.count('[SIGNAL]') for v in visible],[1,2,2,1])
            self.assertIn('1 ou 2',p.prompt(block,'strong','marker',0,layout)[0])
        self.assertEqual(p.target(block,'marker',0,'canonical'),1);self.assertEqual(p.target(block,'marker',0,'both_reversed'),1)
        self.assertEqual(p.target(dict(block,marker=1),'marker',0,'both_reversed'),2)
        self.assertEqual([p.training_label(block,'strong',i) for i in range(3)],[0,1,1])
        self.assertEqual([p.training_label(block,'shuffled',i) for i in range(3)],[1,0,1])
        for arm in ('strong','shuffled'):
            weights=[(p.training_label(block,arm,i),p.training_weight(p.training_label(block,arm,i))) for i in range(3)]
            self.assertEqual(sum(w for y,w in weights if y==0),sum(w for y,w in weights if y==1))
            self.assertEqual(sum(w for y,w in weights),p.CONFIG['primaryWeight'])

    def test_auroc_matches_direct_enumeration_and_resampling(self):
        rng=np.random.default_rng(3);scores=rng.integers(0,4,size=(7,3)).astype(float)
        direct=np.mean([(a>b)+.5*(a==b) for a in scores[:,1:].ravel() for b in scores[:,0]])
        w=p.wins(scores);self.assertAlmostEqual(p.auroc(w),direct)
        index=rng.integers(0,7,size=(5,7))
        for row,value in zip(index,p.auroc(w,index)):
            s=scores[row];self.assertAlmostEqual(value,np.mean([(a>b)+.5*(a==b) for a in s[:,1:].ravel() for b in s[:,0]]))

    def test_oracle_constant_and_block_bias_are_separated(self):
        with patch.dict(p.CONFIG,dict(trainBlocks=1,testBlocks=4,epochs=1,resamples=50)),tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.jsonl'; fixture(path)
            r=p.analyze(path)
            self.assertTrue(r['complete']);self.assertEqual(len(r['contrasts']),72)
            self.assertEqual(sum(c['primary'] for c in r['contrasts']),9)
            self.assertEqual(r['unperturbedMismatches'],0);self.assertTrue(r['fixedReadingRuleMet'])
            for key,t in r['tables'].items():
                if '/strong/strong/' in key:
                    self.assertEqual(t['correct'],1.)
                    if key.endswith('/primary'):
                        self.assertEqual((t['presenceAUROC'],t['withinBlockOrdering'],t['presenceBalancedAccuracy']),(1.,1.,1.))
                if '/base/strong/' in key and key.endswith('/primary'):
                    self.assertEqual((t['presenceAUROC'],t['presenceBalancedAccuracy']),(.5,.5))
            self.assertTrue(all(c['difference']==.5 and c['interval95']==[.5,.5] for c in r['contrasts']))
            fixture(path,constant=True);r=p.analyze(path)
            self.assertFalse(r['fixedReadingRuleMet'])
            self.assertTrue(all(c['difference']==0 for c in r['contrasts']))
            fixture(path,blockBias=True);r=p.analyze(path)
            t=r['tables']['0/test/strong/strong/later/both_reversed/primary']
            self.assertEqual(t['withinBlockOrdering'],1.);self.assertEqual(t['presenceAUROC'],.75)
            self.assertEqual(t['presenceBalancedAccuracy'],.75)

    def test_tampering_checkpoint_assignment_training_and_sham_rejected(self):
        with patch.dict(p.CONFIG,dict(trainBlocks=1,testBlocks=1,epochs=1,resamples=10)),tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.jsonl'; original=fixture(path); variations=[]
            e=copy.deepcopy(original);next(x for x in e if x['event']=='training_start')['unit']['initializationSeed']+=1;variations.append(e)
            e=copy.deepcopy(original);next(x for x in e if x['event']=='training_step')['primaryLosses'].pop();variations.append(e)
            e=copy.deepcopy(original);next(x for x in e if x['event']=='request')['adapterHash']='a'*64;variations.append(e)
            e=copy.deepcopy(original);next(x for x in e if x['event']=='request')['request']['condition']='wrong';variations.append(e)
            e=copy.deepcopy(original);next(x for x in e if x['event']=='result' and x['id'].endswith('/3'))['choiceLogits'][0]+=.1;variations.append(e)
            for events in variations:
                write(path,events)
                with self.assertRaises(ValueError): p.analyze(path)
            e=copy.deepcopy(original)
            for x in e:
                if x['event']=='result' and '/weaker/' in x['id'] and x['id'].endswith(('/0','/3')): x['choiceLogits'][2]-=1.
            write(path,e);self.assertGreater(p.analyze(path)['unperturbedMismatches'],0)

    def test_partial_and_interrupted_request_are_preserved(self):
        with patch.dict(p.CONFIG,dict(trainBlocks=1,testBlocks=1,epochs=1,resamples=10)),tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.jsonl'; events=fixture(path)
            i=next(i for i,e in enumerate(events) if e['event']=='request')
            write(path,events[:i+1]);r=p.analyze(path);self.assertTrue(r['pendingRequest']);self.assertIsNone(r['fixedReadingRuleMet'])
            write(path,events[:i+1]+[dict(event='interrupted_request',id=events[i]['request']['id'])]+events[i:])
            report=p.analyze(path);self.assertTrue(report['complete']);self.assertEqual(report['interruptedRequests'],1)


if __name__=='__main__': unittest.main()
