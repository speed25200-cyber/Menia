"""Frozen-adapter question-specificity diagnostic, separate from presence experiment 10."""
import argparse
import itertools
import json
from pathlib import Path
import random

import numpy as np

from research import presence_detection as parent
from research.iphone_coupling_report import require, strict_json

SEED = 202609191
CONFIG = dict(blocksPerReplication=24, replications=3, resamples=2000, maxInputTokens=512,
              layer=17, strength=1., rank=8, visibleBalancedAccuracyGate=.90)
QUESTIONS = ('presence', 'absence', 'marker_first', 'marker_second')
ARMS = ('base', 'visible', 'strong', 'shuffled')
FAMILIES = ('hidden', 'visible')
MODEL = parent.MODEL
PARENT_JOURNAL_SHA256 = 'fdbe7d9e764e509789e552591a8b13ef46ba3bf2efd3543eed08e63595acd526'
PARENT_SUMMARY = Path(__file__).resolve().parents[1]/'artifacts/presence-detection-pilot/first-audit-summary.json'


def checkpoints():
    report = strict_json(PARENT_SUMMARY.read_text(encoding='utf-8'))
    require(report['complete'] and report['origin']=='transformers_gpu', 'Parent run not complete')
    require(len(report['trained'])==9, 'Expected all nine frozen checkpoints')
    return report['trained']


def source_hash():
    return parent.digest(dict(parent=parent.source_hash(), checkpoints=checkpoints(),
        files={n: Path(__file__).with_name(n).read_text(encoding='utf-8')
               for n in ('presence_specificity.py', 'presence_specificity_gpu.py')}))


def plan():
    rng = random.Random(SEED)
    old = (parent.plan()['blocks'] + parent.previous.plan()['blocks']
           + parent.previous.previous.plan()['blocks'] + parent.previous.previous.previous_plan()['blocks'])
    used = {s for b in old for s in b['sentences']}
    noise_used = {b['noiseSeed'] for b in old}
    pool = [f'{s} {v} {o} {p}.' for s,v,o,p in itertools.product(
        ('Le marin', 'La voisine', 'Le peintre', 'La libraire', 'Le jardinier', 'La musicienne', 'Le facteur', 'La guide'),
        ('observe', 'dessine', 'photographie', 'déplace', 'cherche', 'retrouve', 'examine', 'transporte'),
        ('une boîte', 'un panier', 'une chaise', 'un carnet', 'une lampe', 'un tableau', 'une valise', 'un vase'),
        ('près du port', 'dans le jardin', 'devant la maison', 'dans la cour', 'près de la fenêtre', "dans l'atelier", 'sur la terrasse', 'près de la porte'))]
    pool = [s for s in pool if s not in used]; rng.shuffle(pool)
    blocks, evaluations = [], []
    for rep in range(CONFIG['replications']):
        group = []
        for i in range(CONFIG['blocksPerReplication']):
            noise = rng.randrange(2**31)
            while noise in noise_used: noise = rng.randrange(2**31)
            noise_used.add(noise)
            b = dict(id=f'r{rep}-specificity-{i:03d}', replication=rep,
                     sentences=[pool.pop(),pool.pop()], marker=i%2, noiseSeed=noise)
            blocks.append(b); group.append(b)
        for arm in ARMS:
            requests = []
            for b,family,question,pos in itertools.product(group,FAMILIES,QUESTIONS,range(4)):
                requests.append(dict(id=f"{rep}/{arm}/{family}/{b['id']}/{question}/{pos}",
                    replication=rep,arm=arm,family=family,block=b['id'],question=question,position=pos))
            rng.shuffle(requests); evaluations.extend(requests)
    return dict(schema='menia-presence-specificity-plan-v1', seed=SEED, config=CONFIG,
        model=MODEL, parentJournalSHA256=PARENT_JOURNAL_SHA256, checkpoints=checkpoints(),
        questions=list(QUESTIONS), families=list(FAMILIES), arms=list(ARMS),
        blocks=blocks, evaluation=evaluations)


def prompt(block, family, question, position):
    require(family in FAMILIES and question in QUESTIONS and position in range(4), 'Unknown condition')
    if question in ('presence','absence'):
        # Exactly the trained question; only its two output digits change for the inverse mapping.
        body, spans = parent.prompt(block,'visible' if family=='visible' else 'strong','primary',position)
        if question=='absence':
            before, after = ('Réponds 1 si','ou 0 si')
            body = body.replace(before,'Réponds 0 si',1).replace(after,'ou 1 si',1)
        return body, spans
    number = 1 if question=='marker_first' else 2
    instruction = (f'Réponds 1 si la PHRASE {number} porte [REPERE], ou 0 sinon. '
                   'Ignore le signal et toute modification interne.\nRéponds uniquement par 0 ou 1.\n')
    parts=[instruction]; spans=[]
    for slot,sentence in enumerate(block['sentences']):
        parts.append(f'PHRASE {slot+1}: '); start=len(''.join(parts)); parts.append(sentence)
        spans.append((start,len(''.join(parts))))
        if slot==block['marker']: parts.append(' [REPERE]')
        if family=='visible' and slot+1==position: parts.append(' [SIGNAL]')
        parts.append('\n')
    parts.append('Réponse :')
    return ''.join(parts),spans


def target(block, question, position):
    if question=='presence': return int(position in (1,2))
    if question=='absence': return int(position not in (1,2))
    return int(block['marker']==(0 if question=='marker_first' else 1))


def score(out, req, block, choices):
    # Reuse the strict binary-logit and intervention validation, then score the requested meaning.
    check = dict(req,task='primary',layout='canonical',family='visible' if req['family']=='visible' else 'strong')
    base = parent.score(out,check,block,choices)
    logits=np.asarray(out['choiceLogits'][:2],dtype=float)
    prob=np.exp(logits-logits.max()); prob/=prob.sum()
    y=target(block,req['question'],req['position'])
    return dict(base,correct=int(base['prediction']==y),firstTokenCorrect=int(out['rawChoice']==y),
        target=y, probabilityOne=float(prob[1]), brier=float(sum((prob-np.eye(2)[y])**2)))


def read_journal(path):
    events=[strict_json(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    require(events and events[0].get('event')=='header','Missing header')
    h=events[0]; fixed=plan()
    require(h['plan']==fixed and h['planHash']==parent.digest(fixed) and h['sourceHash']==source_hash(),'Plan/source mismatch')
    require(h['metadata']['origin'] in ('transformers_gpu','synthetic_fixture'),'Invalid origin')
    require(h['parentJournalSHA256']==PARENT_JOURNAL_SHA256 and h['checkpoints']==checkpoints(),'Parent lineage mismatch')
    choices=h['metadata']['choiceTokenIds']
    require(len(choices)==3 and len(set(choices))==3 and all(type(x) is int and x>=0 for x in choices),'Invalid choices')
    frozen=checkpoints(); blocks={b['id']:b for b in fixed['blocks']}; pending=None; rows=[]
    for e in events[1:]:
        if e['event']=='request':
            require(pending is None and len(rows)<len(fixed['evaluation']),'Unexpected request')
            req=fixed['evaluation'][len(rows)]; b=blocks[req['block']]
            require(e['request']==req and e['promptHash']==parent.digest(prompt(b,req['family'],req['question'],req['position'])[0]),'Input mismatch')
            require(e['adapterHash']==frozen.get(f"r{req['replication']}-{req['arm']}"),'Adapter mismatch')
            require(type(e['inputTokens']) is int and 0<e['inputTokens']<=CONFIG['maxInputTokens'],'Input length')
            pending=e
        elif e['event']=='result':
            require(pending is not None and e['id']==pending['request']['id'],'Result order')
            req=pending['request']; score(e,req,blocks[req['block']],choices)
            rows.append(dict(request=req,result=e,inputTokens=pending['inputTokens'])); pending=None
        elif e['event']=='interrupted_request':
            require(pending is not None and e['id']==pending['request']['id'],'Unexpected interruption'); pending=None
        elif e['event']=='error': require(type(e['errorType']) is str,'Invalid error')
        else: raise ValueError('Unexpected event; this experiment never trains')
    return h,rows,pending,events


def analyze(path):
    h,rows,pending,events=read_journal(path)
    complete=len(rows)==len(h['plan']['evaluation']) and pending is None
    tables={}; comparisons=[]; gates=[]; shams=0
    if complete:
        keyed={r['request']['id']:r for r in rows}; blocks=h['plan']['blocks']
        for rep in range(CONFIG['replications']):
            bs=[b for b in blocks if b['replication']==rep]; n=len(bs)
            draws=np.random.default_rng(SEED+900+rep).integers(0,n,(CONFIG['resamples'],n))
            matrices={}; shifts={}
            for arm,family,question in itertools.product(ARMS,FAMILIES,QUESTIONS):
                scores=[]; measures=[]
                for b in bs:
                    prefix=f"{rep}/{arm}/{family}/{b['id']}/{question}/"
                    a,c=(keyed[prefix+str(pos)]['result'] for pos in (0,3))
                    require(all(a[k]==c[k] for k in ('choiceLogits','choiceMass','rawTokenId','rawChoice')),'Sham differs')
                    shams+=1; group=[]
                    for pos in range(3):
                        r=keyed[prefix+str(pos)]
                        group.append(score(r['result'],r['request'],b,h['metadata']['choiceTokenIds']))
                    measures.extend(group); scores.append([x['presenceScore'] for x in group])
                s=np.asarray(scores); delta=s[:,1:].mean(1)-s[:,0]
                shifts[(arm,family,question)]=delta
                table={k:float(np.mean([m[k] for m in measures])) for k in
                       ('correct','firstTokenCorrect','firstTokenIsOption','choiceMass','brier')}
                correct_by_class=[float(np.mean([m['correct'] for m in measures if m['target']==y])) for y in (0,1)]
                table.update(n=3*n,blocks=n,correctByClass=correct_by_class,balancedAccuracy=float(np.mean(correct_by_class)),
                    rawShift=float(delta.mean()),rawShiftInterval95=parent.interval(delta[draws].mean(1)),
                    rawPresenceAUROC=parent.auroc(parent.wins(s)),
                    rawPresenceAUROCInterval95=parent.interval(parent.auroc(parent.wins(s),draws)),
                    rawScores=s.tolist())
                if question in ('presence','absence'):
                    oriented=s*(1 if question=='presence' else -1); w=parent.wins(oriented)
                    matrices[(arm,family,question)]=w
                    table.update(orientedPresenceAUROC=parent.auroc(w),
                        orientedPresenceAUROCInterval95=parent.interval(parent.auroc(w,draws)),
                        withinBlockOrdering=float(np.diag(w).mean()/2))
                else:
                    # Pair each intervention with the same block and question, separately for targets 0 and 1.
                    for y in (0,1):
                        ids=[i for i,b in enumerate(bs) if target(b,question,0)==y]
                        table[f'rawShiftTarget{y}']=float(delta[ids].mean()) if ids else None
                    prob=np.array([m['probabilityOne'] for m in measures]).reshape(n,3)
                    acc=np.array([m['correct'] for m in measures]).reshape(n,3)
                    for name,values in [('probabilityOneShift',prob[:,1:].mean(1)-prob[:,0]),
                                        ('accuracyChange',acc[:,1:].mean(1)-acc[:,0])]:
                        table[name]=float(values.mean());table[name+'Interval95']=parent.interval(values[draws].mean(1))
                tables[f'{rep}/{arm}/{family}/{question}']=table
            for question in ('presence','absence'):
                strong=matrices[('strong','hidden',question)]
                for other in ('chance','shuffled','base'):
                    w=matrices[(other,'hidden',question)] if other!='chance' else None
                    point=.5 if w is None else parent.auroc(w)
                    samples=.5 if w is None else parent.auroc(w,draws)
                    comparisons.append(dict(replication=rep,kind='orientedAUROC',question=question,other=other,
                        primary=other in ('chance','shuffled'),difference=parent.auroc(strong)-point,
                        interval95=parent.interval(parent.auroc(strong,draws)-samples)))
            for arm,family in itertools.product(ARMS,FAMILIES):
                interaction=shifts[(arm,family,'presence')]-shifts[(arm,family,'absence')]
                comparisons.append(dict(replication=rep,kind='mappingInteraction',arm=arm,family=family,
                    primary=arm=='strong' and family=='hidden',difference=float(interaction.mean()),
                    interval95=parent.interval(interaction[draws].mean(1))))
                for public in ('marker_first','marker_second'):
                    interaction=shifts[(arm,family,'presence')]-shifts[(arm,family,public)]
                    comparisons.append(dict(replication=rep,kind='questionInteraction',arm=arm,family=family,
                        question=public,primary=False,difference=float(interaction.mean()),
                        interval95=parent.interval(interaction[draws].mean(1))))
            gates.append(dict(replication=rep,
                visibleAdapterUnderstandsMappings=all(tables[f'{rep}/visible/visible/{q}']['balancedAccuracy']>=CONFIG['visibleBalancedAccuracyGate'] for q in ('presence','absence')),
                strongAdapterUnderstandsVisibleMappings=all(tables[f'{rep}/strong/visible/{q}']['balancedAccuracy']>=CONFIG['visibleBalancedAccuracyGate'] for q in ('presence','absence'))))
    primary=[c for c in comparisons if c['primary']]
    signal=bool(primary) and all(c['interval95'][0]>0 for c in primary)
    gate=bool(gates) and all(g['visibleAdapterUnderstandsMappings'] and g['strongAdapterUnderstandsVisibleMappings'] for g in gates)
    return dict(schema='menia-presence-specificity-report-v1',origin=h['metadata']['origin'],complete=complete,
        planHash=h['planHash'],sourceHash=h['sourceHash'],checkpoints=h['checkpoints'],
        recorded=len(rows),planned=len(h['plan']['evaluation']),tables=tables,contrasts=comparisons,
        mappingSignalRuleMet=signal if complete else None,visibleInstructionGateMet=gate if complete else None,
        fixedReadingRuleMet=(signal and gate) if complete else None,visibleGates=gates,shamPairs=shams,
        errors=sum(e['event']=='error' for e in events),interruptedRequests=sum(e['event']=='interrupted_request' for e in events),
        evaluationSeconds=sum(r['result']['seconds'] for r in rows),
        limitations='Frozen adapters from one selected positive detection experiment; new phrases within the same grammar. Conditional question response is not a test of consciousness, natural error prediction, or useful decisions.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('journal',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.write_text(json.dumps(analyze(args.journal),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
