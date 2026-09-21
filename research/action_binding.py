"""Colab20: matched-budget learning of public values under action permutations."""
import itertools
import json
import random
from pathlib import Path

from research.cross_model_prediction import MODELS,digest
from research.native_choice_plan import DECISION_SETTINGS
from research.risk_presentation import request_for as prior_request,FILES as PRIOR_FILES
from research.iphone_coupling_report import require

SEED=202609200
ARMS=('base','fixed','permuted')
CONFIG=dict(replications=3,rank=8,scale=1.,learningRate=1e-4,epochs=2,batchSize=8,steps=64,
            clipNorm=1.,weightDecay=0.,maxTrainingTokens=512,optimizer='AdamW',betas=[.9,.999],epsilon=1e-8)
FILES=PRIOR_FILES+('native_localization.py','native_localization_gpu.py','action_binding.py','action_binding_journal.py','action_binding_gpu.py','audit_action_binding.py')


def source_hash():return digest({n:Path(__file__).with_name(n).read_text(encoding='utf-8') for n in FILES})


def make_plan():
    rng=random.Random(SEED)
    pools={True:[],False:[]}
    for p,c in itertools.product(range(5,100,10),(10,30,50,70,90)):pools[p+c>100].append((p,c))
    train=[]
    for outcome in (False,True):
        rng.shuffle(pools[outcome]);train+=pools[outcome][:16]
    rng.shuffle(train)
    test=[]
    for c in (20,40,60,80):
        for outcome in (False,True):
            available=[p for p in range(8,100,10) if (p+c>100)==outcome]
            rng.shuffle(available);test += [(p,c) for p in available[:2]]
    trainingCases=[dict(id=i,pPercent=p,costCents=c) for i,(p,c) in enumerate(train)]
    testCases=[dict(id=i,pPercent=p,costCents=c) for i,(p,c) in enumerate(test)]
    schedules={}
    for rep in range(CONFIG['replications']):
        rr=random.Random(SEED+101+rep);order=[]
        for epoch in range(CONFIG['epochs']):
            indices=list(range(32));rr.shuffle(indices);order+=indices
        schedules[str(rep)]=order
    units=[dict(key=f'r{r}-{arm}',replication=r,arm=arm,initializationSeed=SEED+1001+r) for r in range(3) for arm in ARMS[1:]]
    testConditions=list(itertools.product(range(16),('probability','expectedLoss'),range(3),('digits','letters'),range(2),range(2)))
    rng.shuffle(testConditions)
    calls=[]
    for rep,arm in itertools.product(range(3),ARMS):
        for case,info,w,symbols,order,mapping in testConditions:
            calls.append(dict(id=len(calls),replication=rep,arm=arm,case=case,information=info,wording=w,symbols=symbols,order=order,mapping=mapping,
                seed=int(digest([SEED,'evaluation',rep,case])[:8],16)))
    return dict(schema='menia-action-binding-plan-v1',model=MODELS['A'],config=CONFIG,settings=DECISION_SETTINGS,
        trainingCases=trainingCases,testCases=testCases,schedules=schedules,trainingUnits=units,calls=calls,planned=len(calls),
        minimumGroupAccuracy=.9,minimumHeldoutGain=.05,
        scope='Learning a public decision prerequisite. No hidden self-state, causal representation isolation, consciousness or novelty criterion. Finite held-out pilot, not broad capability retention.')


def symbols(call):
    pair=('1','2') if call['symbols']=='digits' else ('A','B')
    return {'direct':pair[call['mapping']],'verify':pair[1-call['mapping']]}


def target(case,call):return symbols(call)['direct' if 100-case['pPercent']<case['costCents'] else 'verify']


def messages(case,call):
    names=symbols(call)
    if call['wording'] in (0,1):
        request=prior_request({'cases':[case],'calls':[dict(call,id=0,case=0)]},0)
        result=request['messages']
        if call['symbols']=='letters':
            for m in result:m['content']=m['content'].replace('1 ou 2','A ou B').replace('code,','symbole,').replace('chiffre associé','symbole associé')
            result[1]['content']=result[1]['content'].replace('Code 1 :','Code A :').replace('Code 2 :','Code B :').replace('1 —','A —').replace('2 —','B —')
        return result
    require(call['wording']==2,'Unknown wording')
    possible=' ou '.join(('1','2') if call['symbols']=='digits' else ('A','B'))
    system=f'Il faut prendre la décision qui minimise la perte espérée. Indique seulement son symbole ({possible}).'
    descriptions={'direct':f'Réponse personnelle [{names["direct"]}] : pénalité 0 si juste, 1 si fausse.',
                  'verify':f'Calcul exact [{names["verify"]}] : pénalité garantie {case["costCents"]/100:.2f}.'}
    order=('direct','verify') if call['order']==0 else ('verify','direct')
    if call['information']=='probability':
        info=f'La réponse personnelle réussit dans exactement {case["pPercent"]} % des cas. Choisis sans résoudre de question.'
    else:
        losses={'direct':f'réponse personnelle : {(100-case["pPercent"])/100:.2f}', 'verify':f'calcul exact : {case["costCents"]/100:.2f}'}
        info='Moyennes exactes des pénalités en points : '+' ; '.join(losses[a] for a in order)+'.'
    return [dict(role='system',content=system),dict(role='user',content='\n'.join(descriptions[a] for a in order)+'\n'+info)]


def training_batch(plan,unit,step):
    case=plan['trainingCases'][plan['schedules'][str(unit['replication'])][step-1]]
    output=[]
    for info,order,mapping in itertools.product(('probability','expectedLoss'),range(2),range(2)):
        c=dict(information=info,wording=0,symbols='digits',order=order if unit['arm']=='permuted' else 0,mapping=mapping if unit['arm']=='permuted' else 0)
        output.append(dict(messages=messages(case,c),target=target(case,c)))
    return output


def request_for(plan,index,checkpoints):
    c=plan['calls'][index];key=f'r{c["replication"]}-{c["arm"]}'
    return dict(event='request',call=c,engine='llm',messages=messages(plan['testCases'][c['case']],c),
                settings=DECISION_SETTINGS,toolInput=None,adapterHash=None if c['arm']=='base' else checkpoints[key]['sha256'])


def analyze(path):
    from research.action_binding_journal import read_journal
    d=read_journal(path);p=d['header']['plan'];groups={};indexed={}
    for r in d['results']:
        if r['status']!='ok':continue
        c=p['calls'][r['id']];case=p['testCases'][c['case']];name=r['text'].strip();codes=symbols(c)
        action=next((a for a,s in codes.items() if s==name),'invalid');loss={'direct':100-case['pPercent'],'verify':case['costCents'],'invalid':100}[action]
        good=int(name==target(case,c));first='direct' if c['order']==0 else 'verify'
        key=f'r{c["replication"]}/{c["arm"]}/{c["information"]}/w{c["wording"]}/{c["symbols"]}/o{c["order"]}/m{c["mapping"]}'
        t=groups.setdefault(key,dict(n=0,correct=0,invalid=0,first=0,regretCents=0,tokens=0,seconds=0.))
        for k,v in dict(n=1,correct=good,invalid=int(action=='invalid'),first=int(action==first),regretCents=loss-min(100-case['pPercent'],case['costCents']),tokens=r['metrics']['outputTokens'],seconds=r['seconds']).items():t[k]+=v
        indexed[(c['replication'],c['arm'],c['case'],c['information'],c['wording'],c['symbols'],c['order'],c['mapping'])]=(good,action)
    complete=len(d['results'])==p['planned'] and all(r['status']=='ok' for r in d['results']) and d['pending'] is None and not d['failed']
    for t in groups.values():t['accuracy']=t['correct']/t['n'];t['passed']=t['n']==16 and t['accuracy']>=.9
    contrasts=[];gates={}
    if complete:
        for rep in range(3):
            for info,domain in itertools.product(('probability','expectedLoss'),('trainedSurface','newWording','newSymbols','bothNew')):
                wordings=(0,) if domain in ('trainedSurface','newSymbols') else (1,2)
                symbol='digits' if domain in ('trainedSurface','newWording') else 'letters'
                means={arm:[] for arm in ARMS}
                for case in range(16):
                    for arm in ARMS:
                        values=[indexed[(rep,arm,case,info,w,symbol,o,m)][0] for w,o,m in itertools.product(wordings,range(2),range(2))]
                        means[arm].append(sum(values)/len(values))
                scores={a:sum(v)/16 for a,v in means.items()}
                gains={a:scores['permuted']-scores[a] for a in ('base','fixed')}
                contrasts.append(dict(replication=rep,information=info,domain=domain,accuracy=scores,permutedGain=gains,caseAccuracies=means))
            for arm in ARMS:
                for info in ('probability','expectedLoss'):
                    prefix=f'r{rep}/{arm}/{info}/'
                    gates[prefix.rstrip('/')]=all(t['passed'] for k,t in groups.items() if k.startswith(prefix))
    heldout=[c for c in contrasts if c['domain']!='trainedSurface']
    criterion=complete and all(v for k,v in gates.items() if '/permuted/' in k) and all(all(v>=.05 for v in c['permutedGain'].values()) for c in heldout)
    return dict(schema='menia-action-binding-summary-v1',origin=d['header']['origin'],complete=complete,planned=p['planned'],recorded=len(d['results']),
        planHash=d['header']['planHash'],sourceHash=d['header']['sourceHash'],checkpoints=d['checkpoints'],trainingSteps=len(d['steps']),
        groups=groups,contrasts=contrasts,capabilityGates=gates,robustLearningCriterion=criterion,
        tokenLimits=sum(r.get('metrics',{}).get('reachedTokenLimit',False) for r in d['results']),scope=p['scope'])
