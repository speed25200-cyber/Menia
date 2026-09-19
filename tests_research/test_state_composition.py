import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import state_composition as s


def trace(family,pos):
    changed=family=='hidden' and pos in (1,2)
    return dict(applications=int(changed or pos==3),changed=changed,normRelativeError=.001 if changed else 0.)


def fixture(path,constant_mapping=False):
    fixed=s.plan();blocks={b['id']:b for b in fixed['blocks']};trained={}
    events=[dict(event='header',plan=fixed,planHash=s.previous.parent.digest(fixed),sourceHash=s.source_hash(),
        metadata=dict(origin='synthetic_fixture',choiceTokenIds=[10,11,12,13]),
        parentJournalSHA256=s.previous.PARENT_JOURNAL_SHA256,parentCheckpoints=s.previous.checkpoints())]
    for unit in fixed['training']:
        events.append(dict(event='training_start',unit=unit,parentHash=fixed['parentCheckpoints'][unit['parentKey']]))
        for step,g in enumerate(unit['groups'],1):
            ex=s.training_examples(blocks[g['block']],unit['arm'],g)
            events.append(dict(event='training_step',key=unit['key'],step=step,group=g,examples=ex,
                losses=[1.]*10,traces=[trace(e['family'],e['position']) for e in ex],gradientNorm=1.,seconds=.01))
        trained[unit['key']]=s.previous.parent.digest(unit['key'])
        events.append(dict(event='training_complete',key=unit['key'],steps=len(unit['groups']),sha256=trained[unit['key']]))
    for req in fixed['evaluation']:
        b=blocks[req['block']];family=req['family'];pos=req['position'];form=req['format'];mapping=req['mapping']
        ck=fixed['parentCheckpoints'][f"r{req['replication']}-strong"] if req['arm']=='parent' else trained.get(f"r{req['replication']}-{req['arm']}")
        events.append(dict(event='request',request=req,promptHash=s.previous.parent.digest(s.prompt(b,family,req['task'],pos,form,mapping)[0]),adapterHash=ck,inputTokens=115))
        y=s.target(b,req['task'],pos,form,mapping)
        if req['task']=='monitor' and family=='hidden' and req['arm']!='composed':y=s.codes(form,mapping)[0]
        if constant_mapping and req['arm']=='composed':y=s.codes(form,0)[s.bit(b,req['task'],pos)]
        logits=[5. if i==y else -5. for i in range(4)]
        events.append(dict(event='result',id=req['id'],choiceLogits=logits,choiceMass=.99,rawChoice=y,rawTokenId=10+y,seconds=.01,intervention=trace(family,pos)))
    write(path,events);return events


def write(path,events):path.write_text('\n'.join(json.dumps(e) for e in events)+'\n',encoding='utf-8')


class CompositionTests(unittest.TestCase):
    def test_mass_uses_requested_pair_and_keeps_four_digit_mass_separate(self):
        b=s.plan()['blocks'][0];req=dict(format='new_digits',mapping=1,family='hidden',task='monitor',position=0)
        out=dict(choiceLogits=[0.,0.,0.,0.],rawTokenId=10,rawChoice=0,seconds=0.,choiceMass=.8,intervention=trace('hidden',0))
        result=s.score(out,req,b,[10,11,12,13])
        self.assertAlmostEqual(result['choiceMass'],.4);self.assertEqual(result['fourDigitMass'],.8)
        self.assertEqual(result['firstTokenIsOption'],0);self.assertEqual(result['target'],3)

    def test_fixed_budget_disjoint_data_and_matched_warm_starts(self):
        p=s.plan();self.assertEqual(p,s.plan());self.assertEqual(len(p['evaluation']),25920)
        self.assertEqual(sum(len(u['groups']) for u in p['training']),576)
        self.assertEqual(len({r['id'] for r in p['evaluation']}),25920)
        old=s.previous.plan()['blocks']+s.previous.parent.plan()['blocks']+s.previous.parent.previous.plan()['blocks']+s.previous.parent.previous.previous.plan()['blocks']+s.previous.parent.previous.previous.previous_plan()['blocks']
        self.assertEqual(len({x for b in p['blocks'] for x in b['sentences']}),240)
        self.assertFalse({x for b in p['blocks'] for x in b['sentences']}&{x for b in old for x in b['sentences']})
        self.assertEqual(len({b['noiseSeed'] for b in p['blocks']}),120)
        self.assertFalse({b['noiseSeed'] for b in p['blocks']}&{b['noiseSeed'] for b in old})
        for rep in range(3):
            units=[u for u in p['training'] if u['replication']==rep]
            self.assertEqual({u['parentKey'] for u in units},{f'r{rep}-strong'})
            self.assertTrue(all(u['groups']==units[0]['groups'] for u in units))
            self.assertEqual(sum(g['monitorFamily']=='hidden' for g in units[0]['groups']),32)

    def test_training_only_changes_hidden_supervision_and_balances_codes(self):
        b=s.plan()['blocks'][0];g=dict(epoch=1,monitorFamily='hidden',publicPosition=2)
        a=s.training_examples(b,'composed',g);c=s.training_examples(b,'shuffled',g);v=s.training_examples(b,'grammar',g)
        self.assertEqual(a[6:],c[6:]);self.assertEqual(a[6:],v[6:])
        self.assertEqual([{k:e[k] for k in ('family','task','position','mapping')} for e in a],
                         [{k:e[k] for k in ('family','task','position','mapping')} for e in c])
        self.assertTrue(all(e['family']=='visible' for e in v[:6]))
        for examples in (a,c,v):
            self.assertEqual(len(examples),10);self.assertAlmostEqual(sum(e['weight'] for e in examples),1.)
            for y in (0,1):self.assertAlmostEqual(sum(e['weight'] for e in examples if e['target']==y),.5)
        g['monitorFamily']='visible'
        self.assertEqual(s.training_examples(b,'composed',g),s.training_examples(b,'shuffled',g))
        self.assertEqual(s.training_examples(b,'composed',g),s.training_examples(b,'grammar',g))

    def test_private_intervention_never_changes_text_new_codes_do_not_renumber_lines(self):
        b=s.plan()['blocks'][0]
        for form in s.FORMATS:
            for task in s.TASKS:
                for mapping in (0,1):
                    texts=[s.prompt(b,'hidden',task,pos,form,mapping) for pos in range(4)]
                    self.assertTrue(all(t==texts[0] for t in texts))
                    body,spans=texts[0];self.assertEqual([body[a:z] for a,z in spans],b['sentences'])
                    self.assertIn('PHRASE 1:',body);self.assertIn('PHRASE 2:',body)
                    self.assertIn('2 ou 3' if form=='new_digits' else '0 ou 1',body)
                    for pos in range(3):self.assertIn(s.target(b,task,pos,form,mapping),s.codes(form,mapping))
        for mapping in (0,1):
            self.assertEqual(s.prompt(b,'hidden','monitor',0,'trained',mapping),s.previous.prompt(b,'hidden','absence' if mapping else 'presence',0))

    def test_semantic_oracle_passes_and_mapping_independent_detector_fails(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(s.CONFIG,dict(trainBlocks=2,testBlocks=4,epochs=2,resamples=20)):
            p=Path(d)/'f.jsonl';fixture(p);r=s.analyze(p)
            self.assertTrue(r['complete']);self.assertTrue(r['fixedReadingRuleMet'])
            self.assertEqual(len([c for c in r['contrasts'] if c['primary']]),24)
            fixture(p,constant_mapping=True);r=s.analyze(p)
            self.assertFalse(r['fixedReadingRuleMet']);self.assertFalse(r['signalRuleMet'])

    def test_corruption_early_evaluation_and_incomplete_run_are_rejected(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(s.CONFIG,dict(trainBlocks=2,testBlocks=2,epochs=2,replications=1,resamples=10)):
            p=Path(d)/'f.jsonl';events=fixture(p)
            write(p,events[:2]);self.assertFalse(s.analyze(p)['complete'])
            request=next(e for e in events if e['event']=='request');write(p,[events[0],request])
            with self.assertRaises(ValueError):s.read_journal(p)
            bad=copy.deepcopy(events);bad[1]['parentHash']='0'*64;write(p,bad)
            with self.assertRaises(ValueError):s.read_journal(p)
            bad=copy.deepcopy(events);bad[2]['examples'][0]['target']=1-bad[2]['examples'][0]['target'];write(p,bad)
            with self.assertRaises(ValueError):s.read_journal(p)
            bad=copy.deepcopy(events)
            for i,e in enumerate(bad):
                if e['event']=='request' and e['request']['position']==3:
                    bad[i+1]['choiceMass']-=.01;break
            write(p,bad)
            with self.assertRaisesRegex(ValueError,'Sham'):s.analyze(p)

    def test_shuffled_acquisition_is_scored_against_its_trained_targets(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(s.CONFIG,dict(trainBlocks=4,testBlocks=2,epochs=2,replications=1,resamples=10)):
            p=Path(d)/'f.jsonl';events=fixture(p);blocks={b['id']:b for b in events[0]['plan']['blocks']};changed=0
            for index,event in enumerate(events):
                if event['event']!='request':continue
                req=event['request'];b=blocks[req['block']]
                if b['split']!='train' or req['arm']!='shuffled' or req['task']!='monitor':continue
                pos=0 if req['position']==3 else req['position']
                y=s.codes(req['format'],req['mapping'])[int(b['shuffledTargets'][pos]!=0)]
                events[index+1].update(choiceLogits=[5. if i==y else -5. for i in range(4)],rawChoice=y,rawTokenId=10+y)
                changed+=1
            self.assertGreater(changed,0);write(p,events);r=s.analyze(p)
            for mapping in (0,1):
                table=r['tables'][f'0/train/shuffled/hidden/trained/monitor/{mapping}']
                self.assertEqual(table['trainingTargetAccuracy'],1.)
                self.assertLess(table['correct'],1.)
            self.assertNotIn('trainingTargetAccuracy',r['tables']['0/test/shuffled/hidden/trained/monitor/0'])


if __name__=='__main__':unittest.main()
