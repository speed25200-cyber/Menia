import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import localization_replication as p


def trace(family, pos):
    changed=family!='visible' and pos in (1,2)
    return dict(applications=int(changed or pos==3),changed=changed,normRelativeError=.0001 if changed else 0.)


def fixture(path, *, constant=False):
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
    for req in fixed['evaluation']:
        b=blocks[req['block']]; body,_=p.prompt(b,req['family'],req['task'],req['position'],req['layout'])
        events.append(dict(event='request',request=req,promptHash=p.digest(body),
            adapterHash=trained.get(f"r{req['replication']}-{req['arm']}"),inputTokens=100))
        y=p.target(b,req['task'],req['position'],req['layout'])
        if constant or (req['task']=='primary' and req['arm'] in ('base','shuffled')): y=1
        events.append(dict(event='result',id=req['id'],seconds=.01,
            choiceLogits=[5. if i==y else 0. for i in range(3)],choiceMass=.99,rawChoice=y,rawTokenId=10+y,
            intervention=trace(req['family'],req['position'])))
    write(path,events)
    return events


def write(path,events):
    path.write_text('\n'.join(json.dumps(e) for e in events)+'\n',encoding='utf-8')


class ReplicationTests(unittest.TestCase):
    def test_fixed_budget_disjoint_data_and_initializations(self):
        fixed=p.plan(); self.assertEqual(fixed,p.plan())
        old=p.previous.plan()['blocks']+p.previous.previous_plan()['blocks']
        sentences=[s for b in fixed['blocks'] for s in b['sentences']]
        self.assertEqual(len(set(sentences)),192)
        self.assertFalse(set(sentences)&{s for b in old for s in b['sentences']})
        self.assertEqual(len({b['noiseSeed'] for b in fixed['blocks']}),96)
        self.assertFalse({b['noiseSeed'] for b in fixed['blocks']}&{b['noiseSeed'] for b in old})
        self.assertEqual(len(fixed['evaluation']),12480)
        self.assertEqual(len({r['id'] for r in fixed['evaluation']}),12480)
        self.assertEqual(sum(len(u['groups']) for u in fixed['training']),288)
        self.assertEqual(len(set(p.INITIALIZATIONS)),3)
        by={b['id']:b for b in fixed['blocks']}
        for rep in range(3):
            units=[u for u in fixed['training'] if u['replication']==rep]
            self.assertEqual(len({u['initializationSeed'] for u in units}),1)
            self.assertTrue(all(u['groups']==units[0]['groups'] for u in units))
            self.assertTrue(all(by[g['block']]['split']=='train' and by[g['block']]['replication']==rep for u in units for g in u['groups']))

    def test_crossed_public_numbers_content_spans_and_hidden_assignment(self):
        block=dict(sentences=['alpha.','beta.'],marker=0,noiseSeed=21,shuffledTargets=[2,0,1])
        expected={
            'canonical':(['alpha.','beta.'],[1,2],1),
            'content_reversed':(['beta.','alpha.'],[1,2],2),
            'labels_reversed':(['alpha.','beta.'],[2,1],2),
            'both_reversed':(['beta.','alpha.'],[2,1],1),
        }
        for layout,(texts,numbers,marker) in expected.items():
            body,spans=p.prompt(block,'strong','primary',1,layout)
            self.assertEqual([body[lo:hi] for lo,hi in spans],texts)
            lines=[line for line in body.splitlines() if line.startswith('PHRASE')]
            self.assertEqual([int(line[7]) for line in lines],numbers)
            self.assertEqual(p.target(block,'marker',1,layout),marker)
            self.assertEqual([p.target(block,'primary',i,layout) for i in range(4)],[0,*numbers,0])
            for family in ('strong','shuffled'):
                for pos in range(4): self.assertEqual(p.prompt(block,family,'primary',pos,layout),(body,spans))
            modified=dict(block,noiseSeed=999,shuffledTargets=[0,1,2])
            self.assertEqual(p.prompt(modified,'strong','primary',2,layout),(body,spans))
            visible,_=p.prompt(block,'visible','primary',1,layout)
            self.assertIn(texts[0],next(line for line in visible.splitlines() if '[SIGNAL]' in line and line.startswith('PHRASE')))

    def test_oracle_and_constant_are_separated_in_every_layout(self):
        with patch.dict(p.CONFIG,dict(trainBlocks=1,testBlocks=2,epochs=1)),tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.jsonl'; fixture(path)
            r=p.analyze(path)
            self.assertTrue(r['complete']);self.assertEqual(len(r['primaryContrasts']),9)
            self.assertEqual(r['shamPairs'],270)
            for key,t in r['tables'].items():
                if '/strong/strong/' in key:
                    self.assertEqual(t['correct'],1.)
                    if key.endswith('/primary'): self.assertEqual(t['presenceBalancedAccuracy'],1.)
            self.assertTrue(all(c['difference']==.5 for c in r['primaryContrasts']))
            fixture(path,constant=True);r=p.analyze(path)
            t=r['tables']['0/test/strong/strong/labels_reversed/primary']
            self.assertEqual(t['correct'],1/3);self.assertEqual(t['positionAccuracy'],.5)
            self.assertEqual(t['presenceBalancedAccuracy'],.5)
            self.assertTrue(all(c['difference']==0 for c in r['primaryContrasts']))

    def test_tampering_checkpoint_assignment_training_and_sham_rejected(self):
        with patch.dict(p.CONFIG,dict(trainBlocks=1,testBlocks=1,epochs=1)),tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.jsonl'; original=fixture(path); variations=[]
            e=copy.deepcopy(original);next(x for x in e if x['event']=='training_start')['unit']['initializationSeed']+=1;variations.append(e)
            e=copy.deepcopy(original);next(x for x in e if x['event']=='training_step')['primaryLosses'].pop();variations.append(e)
            e=copy.deepcopy(original);next(x for x in e if x['event']=='request')['adapterHash']='a'*64;variations.append(e)
            e=copy.deepcopy(original);next(x for x in e if x['event']=='request')['request']['layout']='wrong';variations.append(e)
            e=copy.deepcopy(original);next(x for x in e if x['event']=='result' and x['id'].endswith('/3'))['choiceLogits'][0]+=.1;variations.append(e)
            for events in variations:
                write(path,events)
                with self.assertRaises(ValueError): p.analyze(path)

    def test_partial_and_interrupted_request_are_preserved(self):
        with patch.dict(p.CONFIG,dict(trainBlocks=1,testBlocks=1,epochs=1)),tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.jsonl'; events=fixture(path)
            i=next(i for i,e in enumerate(events) if e['event']=='request')
            write(path,events[:i+1]);self.assertTrue(p.analyze(path)['pendingRequest'])
            write(path,events[:i+1]+[dict(event='interrupted_request',id=events[i]['request']['id'])]+events[i:])
            report=p.analyze(path);self.assertTrue(report['complete']);self.assertEqual(report['interruptedRequests'],1)


if __name__=='__main__': unittest.main()
