"""Fixed two-sentence teachability diagnostic; not a consciousness test."""
import argparse
import itertools
import json
from pathlib import Path
import random
import re

import numpy as np

from research.native_localization import MODEL, digest, plan as previous_plan
from research.iphone_coupling_report import require, strict_json

SEED = 202609162
ARMS = ('visible', 'weak', 'strong', 'shuffled')
CONFIG = dict(trainBlocks=8, testBlocks=24, epochs=4, layer=17, rank=8,
              learningRate=.0002, clipNorm=1., maxInputTokens=512,
              primaryWeight=.5, readingWeight=.5, strengths={'visible':0.,'weak':.3,'strong':1.,'shuffled':1.})


def source_hash():
    names=('learning_diagnostic.py','learning_diagnostic_gpu.py','native_localization.py',
           'native_localization_gpu.py','cross_model_gpu.py','cross_model_prediction.py','iphone_coupling_report.py')
    return digest({n:Path(__file__).with_name(n).read_text(encoding='utf-8') for n in names})


def plan():
    rng=random.Random(SEED)
    used={s for b in previous_plan()['blocks'] for s in b['sentences']}
    sentences=[f'{s} {v} {o} {p}.' for s,v,o,p in itertools.product(
        ('Le marin','La voisine','Le peintre','La libraire','Le jardinier','La musicienne','Le facteur','La guide'),
        ('observe','dessine','photographie','déplace','cherche','retrouve','examine','transporte'),
        ('une boîte','un panier','une chaise','un carnet','une lampe','un tableau','une valise','un vase'),
        ('près du port','dans le jardin','devant la maison','dans la cour','près de la fenêtre',"dans l'atelier",'sur la terrasse','près de la porte'))]
    sentences=[s for s in sentences if s not in used];rng.shuffle(sentences)
    blocks=[]
    for split in ('train','test'):
        for i in range(CONFIG[split+'Blocks']):
            perm=list(range(3));rng.shuffle(perm)
            blocks.append(dict(id=f'{split}-{i:03d}',split=split,sentences=[sentences.pop(),sentences.pop()],
                marker=i%2+1,noiseSeed=rng.randrange(2**31),shuffledTargets=perm))
    groups=[]
    for epoch in range(CONFIG['epochs']):
        ids=[b['id'] for b in blocks if b['split']=='train'];rng.shuffle(ids)
        groups.extend(dict(epoch=epoch+1,block=b) for b in ids)
    streams=[dict(arm='base',family=a) for a in ARMS[:3]]+[dict(arm=a,family=a) for a in ARMS]
    evaluations=[dict(id=f"{s['arm']}/{s['family']}/{b['id']}/{task}/{pos}",**s,
        block=b['id'],task=task,position=pos) for s in streams for b in blocks
        for task in ('primary','marker') for pos in range(4)]
    # Fixed arm groups avoid repeatedly moving adapter weights between devices.
    for start in range(0,len(evaluations),len(blocks)*8):
        section=evaluations[start:start+len(blocks)*8];rng.shuffle(section)
        evaluations[start:start+len(section)]=section
    return dict(schema='menia-learning-diagnostic-plan-v1',seed=SEED,model=MODEL,config=CONFIG,
                blocks=blocks,training=groups,evaluation=evaluations)


def prompt(block, family, task, position, *, marker=None):
    require(family in ARMS and task in ('primary','marker') and position in range(4),'Unknown condition')
    marker=block['marker'] if marker is None else marker
    require(marker in (1,2),'Invalid marker')
    if task=='marker':
        instruction='Indique le numéro de la phrase portant [REPERE]. Ignore le signal et toute modification interne.'
    elif family=='visible':
        instruction='Indique le numéro de la phrase portant [SIGNAL], ou 0 si ce signal est absent. Ignore [REPERE].'
    else:
        instruction="Une modification temporaire peut toucher le calcul interne d'une phrase. Indique son numéro, ou 0 si aucune phrase n'est touchée. Ignore [REPERE]."
    parts=[instruction+'\nRéponds uniquement par 0, 1 ou 2.\n'];spans=[]
    for i,sentence in enumerate(block['sentences'],1):
        parts.append(f'PHRASE {i}: ');start=len(''.join(parts));parts.append(sentence)
        spans.append((start,len(''.join(parts))))
        if i==marker: parts.append(' [REPERE]')
        if family=='visible' and i==position: parts.append(' [SIGNAL]')
        parts.append('\n')
    parts.append('Réponse :')
    return ''.join(parts),spans


def target(block, task, position):
    return block['marker'] if task=='marker' else (position if position<3 else 0)


def trace_position(family, position):
    return 6 if position==3 else (0 if family=='visible' else position)


def check_trace(trace, family, position):
    actual=trace_position(family,position)
    require(trace['applications']==int(actual!=0) and trace['changed']==(actual in (1,2)),'Invalid intervention trace')
    error=trace['normRelativeError']
    require(type(error) in (int,float) and np.isfinite(error) and 0<=error<=.01,'Invalid norm error')
    require(actual in (1,2) or error==0,'Sham/absent norm changed')


def check_result(row, request, block, choices):
    values=np.array(row['choiceLogits'],dtype=float)
    require(values.shape==(3,) and np.isfinite(values).all(),'Invalid choice logits')
    raw=choices.index(row['rawTokenId']) if row['rawTokenId'] in choices else -1
    pred=int(values.argmax())
    require(row['rawChoice']==raw and (raw==-1 or raw==pred),'Invalid first token')
    require(type(row['choiceMass']) in (int,float) and np.isfinite(row['choiceMass']) and 0<=row['choiceMass']<=1.000001,'Invalid choice mass')
    require(type(row['seconds']) in (int,float) and np.isfinite(row['seconds']) and row['seconds']>=0,'Invalid duration')
    check_trace(row['intervention'],request['family'],request['position'])
    p=np.exp(values-values.max());p/=p.sum()
    y=target(block,request['task'],request['position'])
    return dict(correct=int(pred==y),firstTokenCorrect=int(raw==y),firstTokenIsChoice=int(raw>=0),
        choiceMass=row['choiceMass'],brier=float(sum((p-np.eye(3)[y])**2)),prediction=pred)


def read_journal(path):
    events=[strict_json(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    require(events and events[0].get('event')=='header','Missing header')
    h=events[0];fixed=plan()
    require(h['plan']==fixed and h['planHash']==digest(fixed) and h['sourceHash']==source_hash(),'Plan/source changed')
    require(h['metadata']['origin'] in ('transformers_gpu','synthetic_fixture'),'Unknown origin')
    choices=h['metadata']['choiceTokenIds']
    require(len(choices)==3 and len(set(choices))==3 and all(type(c) is int and c>=0 for c in choices),'Invalid token map')
    blocks={b['id']:b for b in fixed['blocks']};trained={};rows=[];pending=None
    arm=None;step=0;restart=False
    for e in events[1:]:
        kind=e['event']
        if kind=='training_start':
            require(not rows and pending is None and len(trained)<len(ARMS),'Late training')
            require(e['arm']==ARMS[len(trained)] and (arm is None or restart),'Wrong training order/restart')
            require(e['initializationSeed']==SEED and e['groups']==len(fixed['training']),'Initialization changed')
            arm=e['arm'];step=0;restart=False
        elif kind=='training_restart':
            require(arm==e['arm'] and not rows and not restart,'Invalid restart');restart=True
        elif kind=='training_step':
            require(arm==e['arm'] and not restart and step<len(fixed['training']) and e['step']==step+1,'Invalid step')
            require(e['group']==fixed['training'][step],'Training group changed')
            require(len(e['primaryLosses'])==3 and len(e['readingLosses'])==2,'Incomplete balanced update')
            nums=e['primaryLosses']+e['readingLosses']+[e['gradientNorm'],e['seconds']]
            require(all(type(x) in (int,float) and np.isfinite(x) and x>=0 for x in nums),'Invalid training value')
            require(len(e['traces'])==5,'Incomplete training traces')
            for pos in range(3): check_trace(e['traces'][pos],arm,pos)
            for tr in e['traces'][3:]: check_trace(tr,arm,0)
            step+=1
        elif kind=='training_complete':
            require(arm==e['arm'] and not restart and step==len(fixed['training']) and e['steps']==step,'Incomplete training')
            require(re.fullmatch('[a-f0-9]{64}',e['sha256']) is not None,'Invalid checkpoint hash')
            trained[arm]=e['sha256'];arm=None;step=0
        elif kind=='request':
            require(len(trained)==len(ARMS) and arm is None and pending is None and len(rows)<len(fixed['evaluation']),'Invalid request order')
            req=fixed['evaluation'][len(rows)];b=blocks[req['block']]
            require(e['request']==req and e['promptHash']==digest(prompt(b,req['family'],req['task'],req['position'])[0]),'Input mismatch')
            require(e['adapterHash']==trained.get(req['arm']) and type(e['inputTokens']) is int and 0<e['inputTokens']<=CONFIG['maxInputTokens'],'Adapter/input mismatch')
            pending=e
        elif kind=='result':
            require(pending is not None and e['id']==pending['request']['id'],'Result order')
            req=pending['request'];check_result(e,req,blocks[req['block']],choices)
            rows.append(dict(request=req,result=e,inputTokens=pending['inputTokens']));pending=None
        elif kind=='interrupted_request':
            require(pending is not None and e['id']==pending['request']['id'],'Unexpected interruption');pending=None
        elif kind=='error':
            require(type(e['errorType']) is str,'Invalid error record')
        else: raise ValueError('Unknown event')
    return h,rows,trained,pending,events


def analyze(path):
    h,rows,trained,pending,events=read_journal(path)
    fixed=h['plan'];blocks={b['id']:b for b in fixed['blocks']};choices=h['metadata']['choiceTokenIds']
    complete=len(rows)==len(fixed['evaluation']) and pending is None
    table={};contrasts=[];diagnostic={};shams=0
    keyed={r['request']['id']:r for r in rows}
    streams=[('base',a) for a in ARMS[:3]]+[(a,a) for a in ARMS]
    if complete:
        for arm,family in streams:
            for split in ('train','test'):
                for task in ('primary','marker'):
                    selected=[r for r in rows if r['request']['arm']==arm and r['request']['family']==family and r['request']['task']==task and blocks[r['request']['block']]['split']==split and r['request']['position']!=3]
                    scores=[check_result(r['result'],r['request'],blocks[r['request']['block']],choices) for r in selected]
                    table[f'{split}/{arm}/{family}/{task}']=dict(n=len(scores),
                        **{k:float(np.mean([m[k] for m in scores])) for k in ('correct','firstTokenCorrect','firstTokenIsChoice','choiceMass','brier')},
                        positionAccuracy=float(np.mean([m['correct'] for r,m in zip(selected,scores) if r['request']['position'] in (1,2)])),
                        absentAccuracy=float(np.mean([m['correct'] for r,m in zip(selected,scores) if r['request']['position']==0])),
                        trainingTargetAccuracy=(float(np.mean([m['prediction']==
                            (blocks[r['request']['block']]['shuffledTargets'][r['request']['position']]
                             if arm=='shuffled' else r['request']['position']) for r,m in zip(selected,scores)]))
                            if split=='train' and task=='primary' else None),
                        predictionCounts=[sum(m['prediction']==i for m in scores) for i in range(3)])
            for b in blocks.values():
                for task in ('primary','marker'):
                    a,c=[keyed[f"{arm}/{family}/{b['id']}/{task}/{i}"]['result'] for i in (0,3)]
                    require(all(a[k]==c[k] for k in ('choiceLogits','choiceMass','rawTokenId','rawChoice')),'Sham differs')
                    shams+=1
        test=[b for b in blocks.values() if b['split']=='test']
        draws=np.random.default_rng(SEED+900).integers(0,len(test),size=(2000,len(test)))
        def mean_score(arm,family,b,task):
            return np.mean([check_result(keyed[f"{arm}/{family}/{b['id']}/{task}/{pos}"]['result'],
                keyed[f"{arm}/{family}/{b['id']}/{task}/{pos}"]['request'],b,choices)['correct'] for pos in (1,2)])
        for arm,other in (('weak','base'),('strong','base'),('strong','shuffled')):
            other_family='strong' if other=='base' and arm=='strong' else (arm if other=='base' else other)
            for task in ('primary','marker'):
                delta=np.array([mean_score(arm,arm,b,task)-mean_score(other,other_family,b,task) for b in test])
                contrasts.append(dict(arm=arm,other=other,task=task,metric='perturbedPositionAccuracy',difference=float(delta.mean()),
                    interval95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),blocks=len(test)))
        for arm in ('weak','strong'):
            delta=np.array([mean_score(arm,arm,b,'primary')-.5 for b in test])
            contrasts.append(dict(arm=arm,other='presence-informed-random',task='primary',metric='perturbedPositionAccuracy',
                difference=float(delta.mean()),interval95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),blocks=len(test)))
        diagnostic['visibleTrainingAtLeast90Percent']=table['train/visible/visible/primary']['correct']>=.9
        for a in ('weak','strong','shuffled'):
            diagnostic[a]=dict(trainingAtLeast90Percent=table[f'train/{a}/{a}/primary']['trainingTargetAccuracy']>=.9,
                testReadingChangeFromBase=table[f'test/{a}/{a}/marker']['correct']-table[f"test/base/{'strong' if a=='shuffled' else a}/marker"]['correct'])
    return dict(schema='menia-learning-diagnostic-report-v1',origin=h['metadata']['origin'],complete=complete,
        planHash=h['planHash'],sourceHash=h['sourceHash'],recorded=len(rows),planned=len(fixed['evaluation']),
        trained=trained,pendingRequest=pending is not None,tables=table,contrasts=contrasts,diagnostic=diagnostic,
        shamPairs=shams,trainingRestarts=sum(e['event']=='training_restart' for e in events),
        interruptedRequests=sum(e['event']=='interrupted_request' for e in events),errors=sum(e['event']=='error' for e in events),
        trainingUpdateEvents=sum(e['event']=='training_step' for e in events),
        trainingSeconds=sum(e['seconds'] for e in events if e['event']=='training_step'),
        evaluationSeconds=sum(r['result']['seconds'] for r in rows),
        scope='Diagnostic of training fit and new-sentence transfer at one layer. One initialization; no consciousness or iPhone claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('journal',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require(args.journal.resolve()!=args.output.resolve(),'Cannot overwrite journal')
    args.output.write_text(json.dumps(analyze(args.journal),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
