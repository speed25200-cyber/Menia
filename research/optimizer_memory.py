"""Prospective, common-checkpoint diagnosis of training order and Adam memory."""
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np

from research import state_composition as previous
from research import composition_diagnostic as diagnostic
from research.iphone_coupling_report import require, strict_json

MODEL = previous.MODEL
SEED = 202609194
CONFIG = dict(replications=3, prefixSteps=48, tailSteps=16, testBlocks=24, lexicalBlocks=12,
              resamples=2000, rank=8, layer=17, strength=1., maxInputTokens=512)
ARMS = ('prefix', 'carry', 'reset_m', 'zero_grad')
FORMATS = ('trained', 'paraphrase')
STREAMS = (('hidden', 'monitor'), ('visible', 'monitor'),
           ('visible', 'marker_first'), ('visible', 'marker_second'))
LEXICON = (
    ('Le boulanger','La nageuse','Le pilote','La dentiste','Le berger','La danseuse','Le menuisier','La chercheuse'),
    ('inspecte','soulève','range','nettoie','emballe','répare','mesure','installe'),
    ('un vélo','une guitare','un seau','une échelle','un fauteuil','une clé','un miroir','une tasse'),
    ('derrière la gare','sous le hangar','à côté du lac','au fond du garage','dans le grenier','devant le théâtre','sur le balcon','près du moulin'))
SAME_LEXICON = (
    ('Le marin','La voisine','Le peintre','La libraire','Le jardinier','La musicienne','Le facteur','La guide'),
    ('observe','dessine','photographie','déplace','cherche','retrouve','examine','transporte'),
    ('une boîte','un panier','une chaise','un carnet','une lampe','un tableau','une valise','un vase'),
    ('près du port','dans le jardin','devant la maison','dans la cour','près de la fenêtre',"dans l'atelier",'sur la terrasse','près de la porte'))
digest = diagnostic.digest
interval = diagnostic.interval


def source_hash():
    return digest(dict(previous=diagnostic.source_hash(), files={n:Path(__file__).with_name(n).read_text(encoding='utf-8')
        for n in ('optimizer_memory.py','optimizer_memory_gpu.py','optimizer_memory_ops.py')}))


def plan():
    rng = random.Random(SEED)
    old = diagnostic.old_blocks() + diagnostic.plan()['blocks']
    used = {s for b in old for s in b['sentences']}; noises = {b['noiseSeed'] for b in old}
    pools = {}
    for split, lexicon in (('test', SAME_LEXICON), ('lexical', LEXICON)):
        pool = [' '.join(p)+'.' for p in itertools.product(*lexicon)]
        pools[split] = [s for s in pool if s not in used]; rng.shuffle(pools[split])
    original = previous.plan(); references = diagnostic.checkpoints()
    blocks, training, evaluation = [], [], []
    for rep in range(CONFIG['replications']):
        blocks.extend(b for b in original['blocks'] if b['replication']==rep and b['split']=='train')
        unit = next(u for u in original['training'] if u['key']==f'r{rep}-composed')
        require(len(unit['groups'])==CONFIG['prefixSteps']+CONFIG['tailSteps'], 'Incomplete original schedule')
        for arm in ARMS:
            groups = unit['groups'][:CONFIG['prefixSteps']] if arm=='prefix' else unit['groups'][CONFIG['prefixSteps']:]
            training.append(dict(key=f'r{rep}-{arm}',replication=rep,arm=arm,parentKey=f'r{rep}-strong',groups=groups))
        for split in ('test','lexical'):
            n = CONFIG[split+'Blocks']; require(n>=0 and n%2==0,'Balanced blocks required')
            for i in range(n):
                noise = rng.randrange(2**31)
                while noise in noises: noise = rng.randrange(2**31)
                noises.add(noise)
                blocks.append(dict(id=f'r{rep}-memory-{split}-{i:03d}',replication=rep,split=split,
                    index=i,marker=i%2,noiseSeed=noise,sentences=[pools[split].pop(),pools[split].pop()]))
        for arm in ARMS:
            requests = []
            for b in [b for b in blocks if b['replication']==rep]:
                forms = ('trained',) if b['split']=='train' else FORMATS
                streams = (('hidden','monitor'),) if b['split']=='train' else STREAMS
                for form,(family,task),mapping,pos in itertools.product(forms,streams,range(2),range(4)):
                    requests.append(dict(id=f"{rep}/{arm}/{family}/{b['id']}/{form}/{task}/{mapping}/{pos}",
                        replication=rep,arm=arm,family=family,block=b['id'],format=form,task=task,mapping=mapping,position=pos))
            rng.shuffle(requests); evaluation.extend(requests)
    return dict(schema='menia-optimizer-memory-plan-v1',seed=SEED,config=CONFIG,model=MODEL,
        parentJournalSHA256=diagnostic.PARENT_SHA256,compositionJournalSHA256=diagnostic.COMPOSITION_SHA256,
        parentCheckpoints=previous.previous.checkpoints(),
        carryReferences={f'r{r}-carry':references[f'r{r}-composed'] for r in range(CONFIG['replications'])},
        blocks=blocks,training=training,evaluation=evaluation)


def valid_hash(x):
    require(isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x),'Invalid hash')


def numbers(xs):
    require(all(type(x) in (float,int) and math.isfinite(x) and x>=0 for x in xs),'Invalid numeric value')


def check_moments(m, step):
    require(set(m)=={'firstMomentNorm','secondMomentNorm','minStep','maxStep','states'},'Moment fields')
    numbers([m['firstMomentNorm'],m['secondMomentNorm']])
    require(m['minStep']==m['maxStep']==step and type(m['states']) is int and m['states']>0,'Moment counters')


def read_journal(path):
    events=[strict_json(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    require(events and events[0].get('event')=='header','Missing header'); h=events[0]; fixed=plan()
    require(h['plan']==fixed and h['planHash']==digest(fixed) and h['sourceHash']==source_hash(),'Plan/source mismatch')
    require(h['metadata']['origin'] in ('transformers_gpu','synthetic_fixture'),'Unknown origin')
    choices=h['metadata']['choiceTokenIds']
    require(len(choices)==len(set(choices))==4 and all(type(c) is int and c>=0 for c in choices),'Invalid choices')
    blocks={b['id']:b for b in fixed['blocks']}; done={}; rows=[]; pending=None; unit=None; step=0; restart=False
    for e in events[1:]:
        kind=e['event']
        if kind=='training_start':
            require(not rows and pending is None and len(done)<len(fixed['training']),'Late training')
            expected=fixed['training'][len(done)]
            require(e['unit']==expected and (unit is None or restart),'Training order')
            require(e['trainableParameters']>0,'No trainable parameters')
            rep=expected['replication']; arm=expected['arm']
            if arm=='prefix':
                require(e['initialHash']==fixed['parentCheckpoints'][expected['parentKey']] and e['fork'] is None,'Initial parent')
            else:
                prefix=done[f'r{rep}-prefix']; fork=e['fork']
                require(e['initialHash']==prefix['sha256'] and fork['sourceStateHash']==prefix['optimizerStateHash'],'Different fork')
                require(fork['before']==prefix['moments'],'Changed initial moments')
                check_moments(fork['after'],CONFIG['prefixSteps']); valid_hash(fork['restoredStateHash'])
                expected_after=dict(fork['before'])
                if arm=='reset_m': expected_after['firstMomentNorm']=0.
                require(fork['after']==expected_after,'Wrong intervention on optimizer')
                if arm!='reset_m':require(fork['restoredStateHash']==fork['sourceStateHash'],'Unrequested state change')
            unit=expected;step=0;restart=False
        elif kind=='training_restart':
            require(unit is not None and e['key']==unit['key'] and not restart and not rows,'Unexpected restart');restart=True
        elif kind=='training_step':
            require(unit is not None and not restart and e['key']==unit['key'] and e['step']==step+1 and step<len(unit['groups']),'Update order')
            group=unit['groups'][step]; require(e['group']==group,'Changed schedule');numbers([e['seconds']])
            if unit['arm']=='zero_grad':
                require(e['mode']=='dense_zero' and not any(k in e for k in ('losses','traces','examples','gradientNorm')),'Zero control computed a loss')
            else:
                require(e['mode']=='objective' and e['examples']==previous.training_examples(blocks[group['block']],'composed',group),'Changed objective')
                require(len(e['losses'])==len(e['traces'])==10,'Missing training terms')
                numbers(e['losses']+[e['gradientNorm']])
                for ex,t in zip(e['examples'],e['traces']):
                    previous.previous.parent.check_trace(t,'visible' if ex['family']=='visible' else 'strong',ex['position'])
            step+=1
        elif kind=='training_complete':
            require(unit is not None and not restart and e['key']==unit['key'] and e['steps']==step==len(unit['groups']),'Incomplete training')
            valid_hash(e['sha256']);valid_hash(e['optimizerStateHash'])
            check_moments(e['moments'],CONFIG['prefixSteps']+(0 if unit['arm']=='prefix' else CONFIG['tailSteps']))
            if unit['arm']=='prefix':valid_hash(e['optimizerFileHash'])
            if unit['arm']=='carry':require(e['sha256']==fixed['carryReferences'][unit['key']],'Original checkpoint not reproduced exactly')
            done[unit['key']]=e;unit=None;step=0
        elif kind=='request':
            require(len(done)==len(fixed['training']) and unit is None and pending is None and len(rows)<len(fixed['evaluation']),'Premature request')
            req=fixed['evaluation'][len(rows)];b=blocks[req['block']]
            require(e['request']==req and e['promptHash']==digest(previous.prompt(b,req['family'],req['task'],req['position'],req['format'],req['mapping'])[0]),'Input mismatch')
            require(e['adapterHash']==done[f"r{req['replication']}-{req['arm']}"]['sha256'],'Evaluation weights changed')
            require(type(e['inputTokens']) is int and 0<e['inputTokens']<=CONFIG['maxInputTokens'],'Input length');pending=e
        elif kind=='result':
            require(pending is not None and e['id']==pending['request']['id'],'Result order')
            req=pending['request'];previous.score(e,req,blocks[req['block']],choices)
            require(e['choiceMass']>0,'Cannot reconstruct full-vocabulary loss with zero mass')
            rows.append(dict(request=req,result=e,inputTokens=pending['inputTokens']));pending=None
        elif kind=='interrupted_request':
            require(pending is not None and e['id']==pending['request']['id'],'Unexpected interruption');pending=None
        elif kind=='error':require(type(e['errorType']) is str,'Invalid error')
        else:raise ValueError('Unknown event')
    return h,rows,done,pending,events


def analyze(path):
    h,rows,done,pending,events=read_journal(path)
    complete=len(rows)==len(h['plan']['evaluation']) and pending is None
    tables={};contrasts=[];shams=0
    if complete:
        keyed={r['request']['id']:r for r in rows}
        for rep,split in itertools.product(range(CONFIG['replications']),('train','test','lexical')):
            bs=[b for b in h['plan']['blocks'] if b['replication']==rep and b['split']==split];n=len(bs)
            if not bs:continue
            rng=np.random.default_rng(SEED+rep*10+('train','test','lexical').index(split))
            strata=[np.array([i for i,b in enumerate(bs) if b['marker']==y]) for y in (0,1)]
            require(all(len(s)>0 for s in strata),'Both public classes required')
            draws=np.concatenate([rng.choice(s,(CONFIG['resamples'],len(s))) for s in strata],axis=1)
            forms=('trained',) if split=='train' else FORMATS
            streams=(('hidden','monitor'),) if split=='train' else STREAMS
            native={};losses={}
            for arm,form,(family,task),mapping in itertools.product(ARMS,forms,streams,range(2)):
                scores=[];measures=[];ces=[];block_values=[];block_losses=[]
                for b in bs:
                    prefix=f"{rep}/{arm}/{family}/{b['id']}/{form}/{task}/{mapping}/"
                    a,c=(keyed[prefix+str(p)]['result'] for p in (0,3))
                    require(all(a[k]==c[k] for k in ('choiceLogits','choiceMass','rawTokenId','rawChoice')),'Sham differs');shams+=1
                    group=[];ce=[]
                    for pos in range(3):
                        row=keyed[prefix+str(pos)];out=row['result'];m=previous.score(out,row['request'],b,h['metadata']['choiceTokenIds'])
                        z=np.array(out['choiceLogits'],dtype=float);z-=z.max()
                        ce.append(float(-math.log(out['choiceMass'])+math.log(np.exp(z).sum())-z[m['target']]))
                        group.append(m)
                    weights=np.array([.5,.25,.25] if task=='monitor' else [1/3]*3)
                    block_values.append(float(weights @ [m['firstTokenCorrect'] for m in group]))
                    block_losses.append(float(weights @ ce));ces.extend(ce);measures.extend(group)
                    scores.append([m['presenceScore'] for m in group])
                acc=np.array(block_values);ce=np.array(block_losses);k=(arm,form,family,task,mapping)
                native[k]=acc;losses[k]=ce
                table={k:float(np.mean([m[k] for m in measures])) for k in ('firstTokenCorrect','firstTokenIsOption','brier','choiceMass','fourDigitMass')}
                table.update(blocks=n,n=3*n,nativeBalancedAccuracy=float(acc.mean()),nativeInterval95=interval(acc[draws].mean(1)),
                    balancedCrossEntropy=float(ce.mean()),crossEntropyInterval95=interval(ce[draws].mean(1)))
                no,yes=previous.codes(form,mapping)
                for label,y in (('false',no),('true',yes)):
                    selected=[i for i,m in enumerate(measures) if m['target']==y]
                    table[label+'Accuracy']=float(np.mean([measures[i]['firstTokenCorrect'] for i in selected]))
                    table[label+'CrossEntropy']=float(np.mean([ces[i] for i in selected]))
                if task=='monitor':
                    wins=previous.previous.parent.wins(scores)
                    table['presenceAUROC']=previous.previous.parent.auroc(wins)
                    table['presenceAUROCInterval95']=interval(previous.previous.parent.auroc(wins,draws))
                tables[f'{rep}/{split}/{arm}/{family}/{form}/{task}/{mapping}']=table
            for form,(family,task),mapping in itertools.product(forms,streams,range(2)):
                for a,b in (('carry','prefix'),('zero_grad','prefix'),('reset_m','carry')):
                    key=lambda arm:(arm,form,family,task,mapping)
                    delta=native[key(a)]-native[key(b)];dc=losses[key(a)]-losses[key(b)]
                    contrasts.append(dict(replication=rep,split=split,format=form,family=family,task=task,mapping=mapping,
                        comparison=a+'-'+b,primary=split=='test' and form=='trained' and family=='hidden' and task=='monitor',
                        nativeDifference=float(delta.mean()),nativeInterval95=interval(delta[draws].mean(1)),
                        crossEntropyDifference=float(dc.mean()),crossEntropyInterval95=interval(dc[draws].mean(1))))
    return dict(schema='menia-optimizer-memory-report-v1',origin=h['metadata']['origin'],complete=complete,
        planHash=h['planHash'],sourceHash=h['sourceHash'],trained={k:e['sha256'] for k,e in done.items()},
        recorded=len(rows),planned=len(h['plan']['evaluation']),shamPairs=shams,tables=tables,contrasts=contrasts,
        trainingUpdates=sum(e['event']=='training_step' for e in events),
        errors=sum(e['event']=='error' for e in events),trainingRestarts=sum(e['event']=='training_restart' for e in events),
        interruptedRequests=sum(e['event']=='interrupted_request' for e in events),
        evaluationSeconds=sum(r['result']['seconds'] for r in rows),
        limitations='Post-experiment-13 causal diagnostic, no global success gate. Wording already studied. Reserved content lexicon is not unseen pretraining. Reset measures total trajectory effect, not additive mediation. No consciousness inference.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('journal',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(analyze(a.journal),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
