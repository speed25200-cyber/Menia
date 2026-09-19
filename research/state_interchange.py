"""Fixed whole-vector causal interchange pilot, downstream of experiment 14."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

import numpy as np
from research import optimizer_memory as previous
from research.interchange_hypotheses import predictions
from research.iphone_coupling_report import require,strict_json

ROOT=Path(__file__).resolve().parents[1]
SEED=202609195
MODEL=previous.MODEL
SITES=(17,23,35)
ARMS=('prefix','base')
TASKS=('monitor','marker_first')
CONFIG=dict(replications=3,testPairs=16,lexicalPairs=8,resamples=2000,rank=8,
            layer=17,strength=1.,maxInputTokens=512,baselineGate=.90)
PARENT_SHA256='0bd76014c6d80f56e6c5a1cedf71fd1546f78be18977c5a79ebd8ccd12947706'
digest=previous.digest


def checkpoints():
    report=strict_json((ROOT/'artifacts/optimizer-memory-pilot/summary.json').read_text(encoding='utf-8'))
    require(report['complete'] and report['recorded']==29184,'Incomplete parent')
    return {f'r{r}-prefix':report['trained'][f'r{r}-prefix'] for r in range(3)}


def source_hash():
    return digest(dict(previous=previous.source_hash(),files={n:Path(__file__).with_name(n).read_text(encoding='utf-8')
        for n in ('state_interchange.py','state_interchange_gpu.py','state_interchange_ops.py','interchange_hypotheses.py')}))


def plan():
    rng=random.Random(SEED)
    old=previous.plan()['blocks']+previous.diagnostic.plan()['blocks']+previous.previous.plan()['blocks']
    # Earlier native experiments are also excluded from the new sentence pool.
    for study in (previous.previous.previous,previous.previous.previous.parent,
                  previous.previous.previous.parent.previous,previous.previous.previous.parent.previous.previous):
        old+=study.plan()['blocks']
    old+=previous.previous.previous.parent.previous.previous.previous_plan()['blocks']
    used={s for b in old for s in b['sentences']};noise_used={b['noiseSeed'] for b in old}
    pools={split:[f'{a} {b} {c} {d}.' for a,b,c,d in itertools.product(*lex)]
           for split,lex in (('test',previous.SAME_LEXICON),('lexical',previous.LEXICON))}
    for split in pools:
        pools[split]=[s for s in pools[split] if s not in used];rng.shuffle(pools[split])
    pairs=[]
    for rep in range(3):
        for split in ('test','lexical'):
            for i in range(CONFIG[split+'Pairs']):
                pair=dict(id=f'r{rep}-{split}-{i:03d}',replication=rep,split=split,index=i)
                for role,marker,pos in (('donor',(i//2)%2,1+i%2),('recipient',i%2,1+(i//2)%2)):
                    noise=rng.randrange(2**31)
                    while noise in noise_used:noise=rng.randrange(2**31)
                    noise_used.add(noise)
                    pair[role]=dict(id=pair['id']+'-'+role,marker=marker,positivePosition=pos,
                        sentences=[pools[split].pop(),pools[split].pop()],noiseSeed=noise)
                pairs.append(pair)
    groups=[dict(id=f'{arm}/{pair["id"]}',arm=arm,pair=pair['id']) for arm in ARMS for pair in pairs]
    return dict(schema='menia-state-interchange-plan-v1',seed=SEED,config=CONFIG,sites=list(SITES),
        model=MODEL,parentJournalSHA256=PARENT_SHA256,checkpoints=checkpoints(),pairs=pairs,groups=groups,
        plannedForwards=len(groups)*136,alignment='none; full residual vector at last prompt token')


def requests():
    calls=[]
    for task in TASKS:
        for role,state,code in itertools.product(('donor','recipient'),range(2),range(2)):
            calls.append(dict(kind='intact',task=task,role=role,state=state,code=code))
        for site,state,code in itertools.product(SITES,range(2),range(2)):
            calls.append(dict(kind='sham',task=task,site=site,recipientState=state,recipientCode=code))
        for site,ds,rs,dc,rc in itertools.product(SITES,range(2),range(2),range(2),range(2)):
            calls.append(dict(kind='transfer',task=task,site=site,donorState=ds,recipientState=rs,donorCode=dc,recipientCode=rc))
    return [dict(id=i,**r) for i,r in enumerate(calls)]


def intact_key(r):return r['task'],r['role'],r['state'],r['code']


def semantic_predictions(pair,r):
    d=r['donorState'] if r['task']=='monitor' else int(pair['donor']['marker']==0)
    q=r['recipientState'] if r['task']=='monitor' else int(pair['recipient']['marker']==0)
    return predictions(d,q,r['donorCode'],r['recipientCode'])


def same_output(a,b):return all(a[k]==b[k] for k in ('rawTokenId','rawChoice','choiceLogits','choiceMass'))


def validate_group(pair,outputs,choices):
    fixed=requests();require(len(outputs)==len(fixed),'Wrong group size');intact={};transfers={}
    for expected,out in zip(fixed,outputs):
        require(out['request']==expected,'Request order changed')
        z=out['choiceLogits'];require(len(z)==4 and all(np.isfinite(z)),'Invalid logits')
        require(0<out['choiceMass']<=1.000001 and np.isfinite(out['seconds']) and out['seconds']>=0,'Invalid output metrics')
        raw=out['rawTokenId'];require(type(raw) is int and raw>=0,'Invalid token')
        require(out['rawChoice']==(choices.index(raw) if raw in choices else -1),'Raw choice mismatch')
        kind=expected['kind']
        role=expected['role'] if kind=='intact' else 'recipient'
        state=expected['state'] if kind=='intact' else expected['recipientState']
        code=expected['code'] if kind=='intact' else expected['recipientCode']
        b=pair[role];position=b['positivePosition'] if state else 0
        body,_=previous.previous.prompt(b,'hidden',expected['task'],position,'trained',code)
        require(out['promptHash']==digest(body),'Prompt changed')
        require(type(out['inputTokens']) is int and 0<out['inputTokens']<=CONFIG['maxInputTokens'],'Invalid prefix length')
        trace=out['intervention']
        require(trace['applications']==int(state!=0) and trace['changed']==bool(state),'Wrong physical intervention')
        require(np.isfinite(trace['normRelativeError']) and 0<=trace['normRelativeError']<=.01,'Invalid rotation norm')
        if kind=='intact':
            hashes=out['activationHashes'];require(set(hashes)=={str(s) for s in SITES},'Missing activations')
            for h in hashes.values():previous.valid_hash(h)
            intact[intact_key(expected)]=out
        else:
            site=expected['site'];recipient=intact[(expected['task'],'recipient',state,expected['recipientCode'])]
            donor=recipient if kind=='sham' else intact[(expected['task'],'donor',expected['donorState'],expected['donorCode'])]
            p=out['patch'];require(p['applications']==1 and p['site']==site,'Invalid patch')
            require(p['tokenIndex']==out['inputTokens']-1,'Patch did not target the last token')
            require(p['recipientHash']==recipient['activationHashes'][str(site)],'Recipient activation changed')
            require(p['donorHash']==p['patchedHash']==donor['activationHashes'][str(site)],'Not the exact donor activation')
            require(all(np.isfinite(p[k]) and p[k]>=0 for k in ('recipientNorm','donorNorm','displacementNorm')),'Invalid patch norms')
            if kind=='sham':require(same_output(out,recipient) and p['displacementNorm']==0.,'Self-sham differs')
            if kind=='transfer' and site==35:require(same_output(out,donor),'Last-layer copy failed')
            if kind=='transfer':transfers[(expected['task'],site,expected['donorState'],state,expected['donorCode'],expected['recipientCode'])]=out
    for task,role,code in itertools.product(TASKS,('donor','recipient'),range(2)):
        require(intact[(task,role,0,code)]['activationHashes']['17']==intact[(task,role,1,code)]['activationHashes']['17'],
                'Injected earlier positions changed the same-layer last-token capture')
    for task,rs,dc,rc in itertools.product(TASKS,range(2),range(2),range(2)):
        require(same_output(transfers[(task,17,0,rs,dc,rc)],transfers[(task,17,1,rs,dc,rc)]),'Layer-17 donor-state control failed')


def read_journal(path):
    events=[strict_json(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(events and events[0]['event']=='header','Missing header');h=events[0];fixed=plan()
    require(h['plan']==fixed and h['planHash']==digest(fixed) and h['sourceHash']==source_hash(),'Plan/source mismatch')
    require(h['metadata']['origin'] in ('transformers_gpu','synthetic_fixture'),'Unknown origin')
    choices=h['metadata']['choiceTokenIds'];require(len(choices)==len(set(choices))==4,'Invalid digit IDs')
    pairs={p['id']:p for p in fixed['pairs']};done=[];pending=None
    for e in events[1:]:
        kind=e['event']
        if kind=='group_start':
            require(pending is None and len(done)<len(fixed['groups']) and e['group']==fixed['groups'][len(done)],'Wrong start')
            pending=e['group']
        elif kind=='group_restart':
            require(pending is not None and e['id']==pending['id'],'Wrong restart');pending=None
        elif kind=='group_complete':
            require(pending is not None and e['id']==pending['id'],'Wrong completion')
            validate_group(pairs[pending['pair']],e['outputs'],choices)
            done.append(dict(group=pending,outputs=e['outputs']));pending=None
        elif kind=='error':require(pending is not None,'Error without running group')
        else:raise ValueError('Unknown event')
    return h,done,pending,events


def analyze(path):
    h,done,pending,events=read_journal(path);fixed=h['plan'];pairs={p['id']:p for p in fixed['pairs']}
    complete=pending is None and len(done)==len(fixed['groups']);tables={};baseline={};contrasts=[];prerequisites={}
    if complete:
        for rep,split,arm,task in itertools.product(range(3),('test','lexical'),ARMS,TASKS):
            groups=[g for g in done if g['group']['arm']==arm and pairs[g['group']['pair']]['replication']==rep and pairs[g['group']['pair']]['split']==split]
            rng=np.random.default_rng(SEED+rep*10+('test','lexical').index(split));n=len(groups)
            strata=[[i for i,g in enumerate(groups) if (pairs[g['group']['pair']]['donor']['marker'],pairs[g['group']['pair']]['recipient']['marker'])==(dm,rm)] for dm,rm in itertools.product(range(2),repeat=2)]
            draws=np.concatenate([rng.choice(ids,(CONFIG['resamples'],len(ids))) for ids in strata],axis=1)
            for code in (0,1):
                vals=[]
                for g in groups:
                    pair=pairs[g['group']['pair']];part=[]
                    for out in g['outputs']:
                        r=out['request']
                        if r['kind']=='intact' and r['task']==task and r['code']==code:
                            truth=r['state'] if task=='monitor' else int(pair[r['role']]['marker']==0)
                            part.append(int(out['rawTokenId']==h['metadata']['choiceTokenIds'][truth^code]))
                    vals.append(np.mean(part))
                baseline[f'{rep}/{split}/{arm}/{task}/{code}']=dict(accuracy=float(np.mean(vals)),pairs=n,
                    interval95=np.quantile(np.array(vals)[draws].mean(1),[.025,.975]).tolist())
            for site in SITES:
                metrics=[]
                for g in groups:
                    pair=pairs[g['group']['pair']];intact={intact_key(o['request']):o for o in g['outputs'] if o['request']['kind']=='intact'}
                    part=[]
                    for out in g['outputs']:
                        r=out['request']
                        if r['kind']!='transfer' or r['task']!=task or r['site']!=site:continue
                        pred=semantic_predictions(pair,r);raw=out['rawTokenId'];choices=h['metadata']['choiceTokenIds']
                        donor=intact[(task,'donor',r['donorState'],r['donorCode'])];recipient=intact[(task,'recipient',r['recipientState'],r['recipientCode'])]
                        logits=out['choiceLogits'];log_z=max(logits)+math.log(sum(math.exp(z-max(logits)) for z in logits))
                        losses={k+'CrossEntropy':-math.log(out['choiceMass'])+log_z-logits[v] for k,v in pred.items()}
                        part.append(dict(**{k:int(raw==choices[v]) for k,v in pred.items()},
                            **losses,
                            observedDonorCopy=int(raw==donor['rawTokenId']),observedRecipientRetention=int(raw==recipient['rawTokenId']),
                            optionOutput=int(raw in choices[:2]),patchNorm=out['patch']['displacementNorm']))
                    require(len(part)==16,'Incomplete factorial')
                    metrics.append({k:float(np.mean([p[k] for p in part])) for k in part[0]})
                key=f'{rep}/{split}/{arm}/{task}/{site}'
                tables[key]={k:dict(mean=float(np.mean([m[k] for m in metrics])),
                    interval95=np.quantile(np.array([m[k] for m in metrics])[draws].mean(1),[.025,.975]).tolist()) for k in metrics[0]}
                tables[key].update(pairs=n,calls=n*16)
                for alternative in ('donorAnswer','recipientUnchanged'):
                    delta=np.array([m['stateTransfer']-m[alternative] for m in metrics])
                    contrasts.append(dict(replication=rep,split=split,arm=arm,task=task,site=site,comparison='stateTransfer-'+alternative,
                        primary=split=='test' and arm=='prefix' and task=='monitor' and site==23,
                        difference=float(delta.mean()),interval95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()))
        for rep in range(3):
            checks={f'{task}/{code}':baseline[f'{rep}/test/prefix/{task}/{code}']['accuracy']>=CONFIG['baselineGate']
                    for task in TASKS for code in (0,1)}
            prerequisites[str(rep)]=dict(intactChecks=checks,allPassed=all(checks.values()),
                interpretation='Necessary native-task competence for primary mechanistic interpretation; never filters trials')
    return dict(schema='menia-state-interchange-report-v1',origin=h['metadata']['origin'],planHash=h['planHash'],sourceHash=h['sourceHash'],
        complete=complete,groups=len(done),plannedGroups=len(fixed['groups']),recorded=sum(len(g['outputs']) for g in done),planned=fixed['plannedForwards'],
        selfShams=sum(o['request']['kind']=='sham' for g in done for o in g['outputs']),
        finalLayerCopies=sum(o['request']['kind']=='transfer' and o['request']['site']==35 for g in done for o in g['outputs']),
        errors=sum(e['event']=='error' for e in events),restarts=sum(e['event']=='group_restart' for e in events),
        tables=tables,baseline=baseline,contrasts=contrasts,prerequisites=prerequisites,
        limitations='Post-14 mechanistic pilot on all three pre-tail checkpoints. Canonical wording only. No learned alignment and no global consciousness or novelty inference. Prefix selected as a common training stage after diagnosis, not a held-out model-selection result.')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('journal',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(analyze(a.journal),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
