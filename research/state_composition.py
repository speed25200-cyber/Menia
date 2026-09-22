"""Counterbalanced state/question composition; warm-started controls, fixed held-out tests."""
import argparse
import itertools
import json
from pathlib import Path
import random

import numpy as np

from research import presence_specificity as previous
from research.iphone_coupling_report import require,strict_json

MODEL=previous.MODEL
SEED=202609192
ARMS=('composed','shuffled','grammar')
TASKS=('monitor','marker_first','marker_second')
FORMATS=('trained','paraphrase','new_digits')
CONFIG=dict(trainBlocks=16,testBlocks=24,replications=3,epochs=4,rank=8,learningRate=.0002,
            clipNorm=1.,layer=17,strength=1.,maxInputTokens=512,resamples=2000,
            monitorWeight=.5,publicWeight=.5,accuracyGate=.90,optionGate=.95)
TRAIN_STREAMS=(('composed','hidden'),('shuffled','hidden'),('grammar','visible'))
TEST_STREAMS=(('base','hidden'),('parent','hidden'),('composed','hidden'),('shuffled','hidden'),
              ('grammar','hidden'),('composed','visible'),('grammar','visible'))
TRANSFER_STREAMS=(('composed','hidden'),('composed','visible'),('grammar','visible'))


def source_hash():
    return previous.parent.digest(dict(previous=previous.source_hash(),
        files={n:Path(__file__).with_name(n).read_text(encoding='utf-8') for n in ('state_composition.py','state_composition_gpu.py')}))


def plan():
    rng=random.Random(SEED)
    old=(previous.plan()['blocks']+previous.parent.plan()['blocks']+previous.parent.previous.plan()['blocks']+
         previous.parent.previous.previous.plan()['blocks']+previous.parent.previous.previous.previous_plan()['blocks'])
    used={s for b in old for s in b['sentences']};noise_used={b['noiseSeed'] for b in old}
    pool=[f'{s} {v} {o} {p}.' for s,v,o,p in itertools.product(
        ('Le marin','La voisine','Le peintre','La libraire','Le jardinier','La musicienne','Le facteur','La guide'),
        ('observe','dessine','photographie','déplace','cherche','retrouve','examine','transporte'),
        ('une boîte','un panier','une chaise','un carnet','une lampe','un tableau','une valise','un vase'),
        ('près du port','dans le jardin','devant la maison','dans la cour','près de la fenêtre',"dans l'atelier",'sur la terrasse','près de la porte'))]
    pool=[s for s in pool if s not in used];rng.shuffle(pool)
    blocks=[];training=[];evaluation=[]
    for rep in range(CONFIG['replications']):
        for split in ('train','test'):
            for i in range(CONFIG[split+'Blocks']):
                noise=rng.randrange(2**31)
                while noise in noise_used:noise=rng.randrange(2**31)
                noise_used.add(noise);permutation=list(range(3));rng.shuffle(permutation)
                blocks.append(dict(id=f'r{rep}-{split}-{i:03d}',replication=rep,split=split,index=i,
                    sentences=[pool.pop(),pool.pop()],marker=i%2,noiseSeed=noise,shuffledTargets=permutation))
        groups=[]
        for epoch in range(CONFIG['epochs']):
            group=[b for b in blocks if b['replication']==rep and b['split']=='train'];rng.shuffle(group)
            groups.extend(dict(block=b['id'],epoch=epoch+1,publicPosition=(b['index']+epoch)%3,
                               monitorFamily='hidden' if epoch%2==0 else 'visible') for b in group)
        for arm in ARMS:
            training.append(dict(key=f'r{rep}-{arm}',replication=rep,arm=arm,parentKey=f'r{rep}-strong',groups=groups))
        # A single stream per arm at a time avoids repeatedly loading checkpoints.
        for arm in ('base','parent',*ARMS):
            requests=[]
            for b in [b for b in blocks if b['replication']==rep]:
                for form in (('trained',) if b['split']=='train' else FORMATS):
                    streams=TRAIN_STREAMS if b['split']=='train' else TEST_STREAMS if form=='trained' else TRANSFER_STREAMS
                    for a,family in streams:
                        if a!=arm:continue
                        for task,mapping,pos in itertools.product(TASKS,range(2),range(4)):
                            requests.append(dict(id=f"{rep}/{arm}/{family}/{b['id']}/{form}/{task}/{mapping}/{pos}",
                                replication=rep,arm=arm,family=family,block=b['id'],format=form,task=task,mapping=mapping,position=pos))
            rng.shuffle(requests);evaluation.extend(requests)
    return dict(schema='menia-state-composition-plan-v1',seed=SEED,model=MODEL,config=CONFIG,
        parentJournalSHA256=previous.PARENT_JOURNAL_SHA256,parentCheckpoints=previous.checkpoints(),
        blocks=blocks,training=training,evaluation=evaluation)


def codes(form,mapping):
    require(form in FORMATS and mapping in (0,1),'Unknown output code')
    pair=(2,3) if form=='new_digits' else (0,1)
    return pair[::-1] if mapping else pair  # false, true


def prompt(block,family,task,position,form='trained',mapping=0):
    require(family in ('hidden','visible') and task in TASKS and position in range(4),'Unknown condition')
    no,yes=codes(form,mapping)
    if form=='paraphrase':
        if task=='monitor':
            question=("Une des deux phrases a-t-elle subi une modification de son calcul interne ?" if family=='hidden'
                      else 'Le marqueur [SIGNAL] apparaît-il sur une des deux phrases ?')
            extra='Ne tiens pas compte de [REPERE].'
        else:
            label=1 if task=='marker_first' else 2
            question=f'Le marqueur [REPERE] se trouve-t-il sur la ligne nommée PHRASE {label} ?'
            extra='Ne tiens pas compte de [SIGNAL] ni des changements du calcul interne.'
        instruction=f'{question} {extra}\nSi oui, écris {yes} ; si non, écris {no}.\n'
    elif task=='monitor':
        if family=='visible':
            instruction=f'Réponds {yes} si une phrase porte [SIGNAL], ou {no} si ce signal est absent. Ignore [REPERE].\n'
        else:
            instruction=(f"Une modification temporaire peut toucher le calcul interne d'une phrase. Réponds {yes} si une phrase"
                         f" est touchée, ou {no} si aucune phrase n'est touchée. Ignore [REPERE].\n")
    else:
        label=1 if task=='marker_first' else 2
        instruction=(f'Réponds {yes} si la PHRASE {label} porte [REPERE], ou {no} sinon. '
                     'Ignore le signal et toute modification interne.\n')
    low,high=sorted((no,yes));parts=[instruction+f'Réponds uniquement par {low} ou {high}.\n'];spans=[]
    for slot,sentence in enumerate(block['sentences']):
        parts.append(f'PHRASE {slot+1}: ');start=len(''.join(parts));parts.append(sentence)
        spans.append((start,len(''.join(parts))))
        if slot==block['marker']:parts.append(' [REPERE]')
        if family=='visible' and slot+1==position:parts.append(' [SIGNAL]')
        parts.append('\n')
    parts.append('Réponse :');return ''.join(parts),spans


def bit(block,task,position):
    if task=='monitor':return int(position in (1,2))
    return int(block['marker']==(0 if task=='marker_first' else 1))


def target(block,task,position,form,mapping):return codes(form,mapping)[bit(block,task,position)]


def training_examples(block,arm,group):
    family='visible' if arm=='grammar' else group['monitorFamily']
    examples=[]
    for mapping,pos in itertools.product(range(2),range(3)):
        truth=int((block['shuffledTargets'][pos] if arm=='shuffled' and family=='hidden' else pos)!=0)
        examples.append(dict(family=family,task='monitor',position=pos,mapping=mapping,
            target=codes('trained',mapping)[truth],weight=CONFIG['monitorWeight']*(.5 if truth==0 else .25)/2))
    for task,mapping in itertools.product(TASKS[1:],range(2)):
        pos=group['publicPosition']
        examples.append(dict(family='hidden',task=task,position=pos,mapping=mapping,
            target=target(block,task,pos,'trained',mapping),weight=CONFIG['publicWeight']/4))
    return examples


def score(out,req,block,choices):
    logits=np.asarray(out['choiceLogits'],dtype=float)
    require(logits.shape==(4,) and np.isfinite(logits).all(),'Invalid logits')
    pair=sorted(codes(req['format'],req['mapping']));pred=max(pair,key=lambda i:(logits[i],-i))
    raw=choices.index(out['rawTokenId']) if out['rawTokenId'] in choices else -1
    require(out['rawChoice']==raw and (raw not in pair or raw==pred),'Invalid first token')
    for field in ('seconds','choiceMass'):
        require(type(out[field]) in (int,float) and np.isfinite(out[field]) and out[field]>=0,'Invalid result value')
    require(out['choiceMass']<=1.000001,'Invalid mass')
    previous.parent.check_trace(out['intervention'],'visible' if req['family']=='visible' else 'strong',req['position'])
    y=target(block,req['task'],req['position'],req['format'],req['mapping'])
    z=logits[pair];p=np.exp(z-z.max());p/=p.sum();no,yes=codes(req['format'],req['mapping'])
    digit_probabilities=np.exp(logits-logits.max());digit_probabilities/=digit_probabilities.sum()
    return dict(correct=int(pred==y),firstTokenCorrect=int(raw==y),firstTokenIsOption=int(raw in pair),prediction=pred,
        target=y,brier=float(sum((p-np.eye(2)[pair.index(y)])**2)),
        presenceScore=float(logits[yes]-logits[no]),rawScore=float(logits[pair[1]]-logits[pair[0]]),
        choiceMass=float(out['choiceMass']*digit_probabilities[pair].sum()),fourDigitMass=out['choiceMass'])


def read_journal(path):
    events=[strict_json(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    require(events and events[0].get('event')=='header','Missing header');h=events[0];fixed=plan()
    require(h['plan']==fixed and h['planHash']==previous.parent.digest(fixed) and h['sourceHash']==source_hash(),'Plan/source mismatch')
    require(h['metadata']['origin'] in ('transformers_gpu','synthetic_fixture'),'Unknown origin')
    require(h['parentJournalSHA256']==previous.PARENT_JOURNAL_SHA256 and h['parentCheckpoints']==previous.checkpoints(),'Parent mismatch')
    choices=h['metadata']['choiceTokenIds'];require(len(choices)==4 and len(set(choices))==4 and all(type(c) is int and c>=0 for c in choices),'Invalid digits')
    blocks={b['id']:b for b in fixed['blocks']};trained={};rows=[];pending=None;unit=None;step=0;restart=False
    for e in events[1:]:
        kind=e['event']
        if kind=='training_start':
            require(not rows and pending is None and len(trained)<len(fixed['training']),'Late training')
            expected=fixed['training'][len(trained)]
            require(e['unit']==expected and (unit is None or restart),'Training order')
            require(e['parentHash']==h['parentCheckpoints'][expected['parentKey']],'Different warm start')
            unit=expected;step=0;restart=False
        elif kind=='training_restart':
            require(unit is not None and e['key']==unit['key'] and not restart and not rows,'Invalid restart');restart=True
        elif kind=='training_step':
            require(unit is not None and not restart and e['key']==unit['key'] and e['step']==step+1 and step<len(unit['groups']),'Invalid training step')
            group=unit['groups'][step];require(e['group']==group,'Training group changed')
            examples=training_examples(blocks[group['block']],unit['arm'],group)
            require(e['examples']==examples and len(e['losses'])==len(e['traces'])==10,'Training objective changed')
            require(all(type(x) in (int,float) and np.isfinite(x) and x>=0 for x in e['losses']+[e['gradientNorm'],e['seconds']]),'Invalid update value')
            for ex,t in zip(examples,e['traces']):previous.parent.check_trace(t,'visible' if ex['family']=='visible' else 'strong',ex['position'])
            step+=1
        elif kind=='training_complete':
            require(unit is not None and not restart and e['key']==unit['key'] and e['steps']==step==len(unit['groups']),'Incomplete training')
            require(len(e['sha256'])==64 and all(c in '0123456789abcdef' for c in e['sha256']),'Invalid checkpoint hash')
            trained[unit['key']]=e['sha256'];unit=None;step=0
        elif kind=='request':
            require(len(trained)==len(fixed['training']) and unit is None and pending is None and len(rows)<len(fixed['evaluation']),'Premature request')
            req=fixed['evaluation'][len(rows)];b=blocks[req['block']]
            ph=previous.parent.digest(prompt(b,req['family'],req['task'],req['position'],req['format'],req['mapping'])[0])
            require(e['request']==req and e['promptHash']==ph,'Input mismatch')
            key=f"r{req['replication']}-{req['arm']}"
            expected=h['parentCheckpoints'][f"r{req['replication']}-strong"] if req['arm']=='parent' else trained.get(key)
            require(e['adapterHash']==expected and type(e['inputTokens']) is int and 0<e['inputTokens']<=CONFIG['maxInputTokens'],'Adapter/input mismatch');pending=e
        elif kind=='result':
            require(pending is not None and e['id']==pending['request']['id'],'Result order')
            req=pending['request'];score(e,req,blocks[req['block']],choices)
            rows.append(dict(request=req,result=e,inputTokens=pending['inputTokens']));pending=None
        elif kind=='interrupted_request':
            require(pending is not None and e['id']==pending['request']['id'],'Unexpected interruption');pending=None
        elif kind=='error':require(type(e['errorType']) is str,'Invalid error')
        else:raise ValueError('Unknown event')
    return h,rows,trained,pending,events


def analyze(path):
    h,rows,trained,pending,events=read_journal(path);complete=len(rows)==len(h['plan']['evaluation']) and pending is None
    tables={};contrasts=[];gates=[];shams=0
    if complete:
        keyed={r['request']['id']:r for r in rows};blocks=h['plan']['blocks']
        matrices={}
        for rep in range(CONFIG['replications']):
            for split in ('train','test'):
                bs=[b for b in blocks if b['replication']==rep and b['split']==split];n=len(bs)
                draws=np.random.default_rng(SEED+900+rep).integers(0,n,(CONFIG['resamples'],n))
                for form in (('trained',) if split=='train' else FORMATS):
                    streams=TRAIN_STREAMS if split=='train' else TEST_STREAMS if form=='trained' else TRANSFER_STREAMS
                    for (arm,family),task,mapping in itertools.product(streams,TASKS,range(2)):
                        measures=[];scores=[];raws=[];training_correct=[]
                        for b in bs:
                            prefix=f"{rep}/{arm}/{family}/{b['id']}/{form}/{task}/{mapping}/"
                            a,c=(keyed[prefix+str(p)]['result'] for p in (0,3))
                            require(all(a[k]==c[k] for k in ('choiceLogits','choiceMass','rawTokenId','rawChoice')),'Sham differs');shams+=1
                            ms=[]
                            for pos in range(3):
                                row=keyed[prefix+str(pos)];ms.append(score(row['result'],row['request'],b,h['metadata']['choiceTokenIds']))
                                if split=='train':
                                    trained_bit=int(b['shuffledTargets'][pos]!=0) if arm=='shuffled' and family=='hidden' and task=='monitor' else bit(b,task,pos)
                                    training_correct.append(int(ms[-1]['prediction']==codes(form,mapping)[trained_bit]))
                            measures.extend(ms);scores.append([m['presenceScore'] for m in ms]);raws.append([m['rawScore'] for m in ms])
                        table={k:float(np.mean([m[k] for m in measures])) for k in ('correct','firstTokenCorrect','firstTokenIsOption','brier','choiceMass','fourDigitMass')}
                        pair=sorted(codes(form,mapping));acc=[float(np.mean([m['correct'] for m in measures if m['target']==y])) for y in pair]
                        raw=np.array(raws);delta=raw[:,1:].mean(1)-raw[:,0]
                        table.update(n=3*n,blocks=n,correctByClass=acc,balancedAccuracy=float(np.mean(acc)),rawShift=float(delta.mean()),
                                     rawShiftInterval95=previous.parent.interval(delta[draws].mean(1)))
                        if split=='train':table['trainingTargetAccuracy']=float(np.mean(training_correct))
                        if task=='monitor':
                            w=previous.parent.wins(scores);matrices[(rep,split,arm,family,form,mapping)]=w
                            table.update(presenceAUROC=previous.parent.auroc(w),presenceAUROCInterval95=previous.parent.interval(previous.parent.auroc(w,draws)),
                                withinBlockOrdering=float(np.diag(w).mean()/2))
                        else:
                            correct=np.array([m['correct'] for m in measures],dtype=float).reshape(n,3)
                            change=correct[:,1:].mean(1)-correct[:,0]
                            table.update(accuracyChange=float(change.mean()),accuracyChangeInterval95=previous.parent.interval(change[draws].mean(1)))
                        key=f'{rep}/{split}/{arm}/{family}/{form}/{task}/{mapping}';tables[key]=table
                        if split=='test' and arm=='composed' and form in ('trained','paraphrase'):
                            gates.append(dict(table=key,options=table['firstTokenIsOption']>=CONFIG['optionGate'],
                                accuracy=table['balancedAccuracy']>=CONFIG['accuracyGate'] if family=='visible' or task!='monitor' else True))
            n=CONFIG['testBlocks'];draws=np.random.default_rng(SEED+900+rep).integers(0,n,(CONFIG['resamples'],n))
            for form,mapping in itertools.product(FORMATS,range(2)):
                w=matrices[(rep,'test','composed','hidden',form,mapping)]
                for other in (('chance','shuffled','grammar','parent','base') if form=='trained' else ('chance',)):
                    ref=matrices[(rep,'test',other,'hidden',form,mapping)] if other!='chance' else None
                    point=.5 if ref is None else previous.parent.auroc(ref)
                    samples=.5 if ref is None else previous.parent.auroc(ref,draws)
                    contrasts.append(dict(replication=rep,format=form,mapping=mapping,other=other,
                        primary=form in ('trained','paraphrase') and other in ('chance','shuffled','grammar'),
                        difference=previous.parent.auroc(w)-point,interval95=previous.parent.interval(previous.parent.auroc(w,draws)-samples)))
    primary=[c for c in contrasts if c['primary']];signal=bool(primary) and all(c['interval95'][0]>0 for c in primary)
    controls=bool(gates) and all(g['accuracy'] and g['options'] for g in gates)
    return dict(schema='menia-state-composition-report-v1',origin=h['metadata']['origin'],complete=complete,
        recorded=len(rows),planned=len(h['plan']['evaluation']),planHash=h['planHash'],sourceHash=h['sourceHash'],trained=trained,
        tables=tables,contrasts=contrasts,gates=gates,signalRuleMet=signal if complete else None,controlRuleMet=controls if complete else None,
        fixedReadingRuleMet=(signal and controls) if complete else None,shamPairs=shams,
        trainingUpdates=sum(e['event']=='training_step' for e in events),trainingRestarts=sum(e['event']=='training_restart' for e in events),
        interruptedRequests=sum(e['event']=='interrupted_request' for e in events),errors=sum(e['event']=='error' for e in events),
        trainingSeconds=sum(e['seconds'] for e in events if e['event']=='training_step'),evaluationSeconds=sum(r['result']['seconds'] for r in rows),
        limitations='Warm-started task composition, not consciousness. Grammar control retains an older detector. New digits are secondary; novel wording is primary. No claim of a second-order monitoring circuit or natural-error utility.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('journal',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(analyze(a.journal),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
