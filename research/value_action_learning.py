"""Colab22: matched native choice training with informative or shuffled lookup."""
import itertools
import random
from pathlib import Path

from research import action_binding as parent
from research.risk_presentation import make_plan as risk_plan
from research.cross_model_prediction import digest, MODELS
from research.iphone_coupling_report import require

SEED=202609220
ARMS=('base','choice','linked','shuffled')
CONFIG=dict(parent.CONFIG)
FILES=parent.FILES+('value_action_learning.py','value_action_learning_journal.py',
                    'value_action_learning_gpu.py','audit_value_action_learning.py')


def source_hash():
    return digest({n:Path(__file__).with_name(n).read_text(encoding='utf-8') for n in FILES})


def make_plan():
    rng=random.Random(SEED);old=parent.make_plan()
    previous=old['trainingCases']+old['testCases']+risk_plan()['cases']
    excluded={(c['pPercent'],c['costCents']) for c in previous}
    def sample(ps,cs,n):
        out=[]
        for direct in (False,True):
            pool=[(p,c) for p,c in itertools.product(ps,cs) if (p+c>100)==direct and p+c!=100 and (p,c) not in excluded]
            require(len(pool)>=n,'Candidate pool too small');rng.shuffle(pool);out+=pool[:n]
        rng.shuffle(out)
        return [dict(id=i,pPercent=p,costCents=c) for i,(p,c) in enumerate(out)]
    train=sample(range(3,100,10),(12,32,52,72,92),16)
    test=sample(range(2,100,5),(14,34,54,74,94),12)
    schedules={}
    for rep in range(3):
        rr=random.Random(SEED+101+rep);indices=[]
        for epoch in range(2):
            block=list(range(32));rr.shuffle(block);indices+=block
        schedules[str(rep)]=indices
    units=[dict(key=f'r{r}-{a}',replication=r,arm=a,initializationSeed=SEED+1001+r) for r,a in itertools.product(range(3),ARMS[1:])]
    views=[dict(task='choice',wording=w,symbols=s,order=o,mapping=m) for w,s,o,m in itertools.product((1,2,3),('digits','letters'),range(2),range(2))]
    views += [dict(task='lookup',wording=1,symbols='digits',order=o,mapping=m,selected=a) for o,m,a in itertools.product(range(2),range(2),('direct','verify'))]
    conditions=list(itertools.product(range(24),views));rng.shuffle(conditions)
    calls=[]
    for rep,arm in itertools.product(range(3),ARMS):
        for case,v in conditions:
            calls.append(dict(id=len(calls),replication=rep,arm=arm,case=case,information='expectedLoss',**v,
                              seed=int(digest([SEED,'evaluation',rep,case])[:8],16)))
    return dict(schema='menia-value-action-learning-plan-v1',model=MODELS['A'],config=CONFIG,settings=parent.DECISION_SETTINGS,
                trainingCases=train,testCases=test,schedules=schedules,trainingUnits=units,calls=calls,planned=len(calls),
                trainingUpdates=576,minimumGroupAccuracy=.9,minimumHeldoutGain=.05,
                scope='Native public choice after matched-update task-mixture training; 24 distinct held-out cases. '
                      'No learned self-estimate, broad capability retention, consciousness or novelty criterion.')


def optimal(case):return 'direct' if 100-case['pPercent']<case['costCents'] else 'verify'


def messages(case,c):
    if c['wording']<=2:
        result=parent.messages(case,c)
    else:
        require(c['wording']==3,'Wording')
        names=parent.symbols(c);possible=' ou '.join(('1','2') if c['symbols']=='digits' else ('A','B'))
        lines={'direct':f'Mode direct ({names["direct"]}) : tu réponds ; résultat juste, coût 0 ; résultat faux, coût 1.',
               'verify':f'Mode contrôlé ({names["verify"]}) : un calcul exact fournit la réponse ; coût {case["costCents"]/100:.2f}.'}
        order=('direct','verify') if c['order']==0 else ('verify','direct')
        expected={'direct':f'mode direct : {(100-case["pPercent"])/100:.2f}', 'verify':f'mode contrôlé : {case["costCents"]/100:.2f}'}
        body='\n'.join(lines[a] for a in order)+'\nCoûts moyens exacts : '+' ; '.join(expected[a] for a in order)+'.'
        result=[dict(role='system',content=f'Parmi ces modes, prends celui au coût moyen minimal. Écris seulement son code : {possible}.'),dict(role='user',content=body)]
    if c['task']=='lookup':
        require(c['wording'] in (0,1) and c['symbols']=='digits','Lookup surface')
        result[0]['content']=(
            'Retrouve l’action dont la perte moyenne est égale à la valeur demandée. Réponds uniquement par son code, 1 ou 2.',
            'Le but est de retrouver la possibilité associée à la pénalité moyenne demandée. Écris uniquement son chiffre : 1 ou 2.'
        )[c['wording']]
        v=100-case['pPercent'] if c['selected']=='direct' else case['costCents']
        result[1]['content']+=f'\nValeur moyenne recherchée : {v/100:.2f} point.'
    return result


def target(case,c):
    return parent.symbols(c)[optimal(case) if c['task']=='choice' else c['selected']]


def training_batch(plan,unit,step):
    require(1<=step<=64 and unit['arm'] in ARMS[1:],'Training step/unit')
    case=plan['trainingCases'][plan['schedules'][str(unit['replication'])][step-1]]
    epoch=(step-1)//32;batch=[];aux=[]
    for w,o,m in itertools.product((0,1),range(2),range(2)):
        is_aux=unit['arm']!='choice' and m!=(o^epoch)
        c=dict(task='lookup' if is_aux else 'choice',information='expectedLoss',wording=w,symbols='digits',order=o,mapping=m)
        if is_aux:c['selected']=('direct','verify')[(w+o+epoch)%2];aux.append(len(batch))
        batch.append(dict(messages=messages(case,c),target=target(case,c),task=c['task']))
    if unit['arm']=='shuffled':
        values=[batch[i]['target'] for i in aux]
        random.Random(SEED+100000+unit['replication']*1000+step).shuffle(values)
        for i,v in zip(aux,values):batch[i]['target']=v
    return batch


def request_for(plan,index,checkpoints):
    c=plan['calls'][index];key=f'r{c["replication"]}-{c["arm"]}'
    return dict(event='request',call=c,engine='llm',messages=messages(plan['testCases'][c['case']],c),settings=plan['settings'],
                toolInput=None,adapterHash=None if c['arm']=='base' else checkpoints[key]['sha256'])


def analyze(path):
    from research.value_action_learning_journal import read_journal
    d=read_journal(path);p=d['header']['plan'];groups={};indexed={}
    for r in d['results']:
        if r['status']!='ok':continue
        c=p['calls'][r['id']];case=p['testCases'][c['case']];raw=r['text'].strip();codes=parent.symbols(c)
        action=next((a for a,s in codes.items() if s==raw),'invalid');good=int(raw==target(case,c))
        key=f'r{c["replication"]}/{c["arm"]}/{c["task"]}/w{c["wording"]}/{c["symbols"]}/o{c["order"]}/m{c["mapping"]}'
        if c['task']=='lookup':key+='/'+c['selected']
        t=groups.setdefault(key,dict(n=0,correct=0,invalid=0,tokens=0,seconds=0.))
        for k,v in dict(n=1,correct=good,invalid=int(action=='invalid'),tokens=r['metrics']['outputTokens'],seconds=r['seconds']).items():t[k]+=v
        if c['task']=='choice':indexed[(c['replication'],c['arm'],c['case'],c['wording'],c['symbols'],c['order'],c['mapping'])]=good
    complete=len(d['results'])==p['planned'] and all(r['status']=='ok' for r in d['results']) and d['pending'] is None and not d['failed']
    for t in groups.values():t['accuracy']=t['correct']/t['n'];t['passed']=t['n']==24 and t['accuracy']>=.9
    contrasts=[];gates={}
    if complete:
        for rep in range(3):
            for domain in ('trainedSurface','newWording','newSymbols','bothNew'):
                ws=(1,) if domain in ('trainedSurface','newSymbols') else (2,3)
                sym='digits' if domain in ('trainedSurface','newWording') else 'letters'
                values={a:[sum(indexed[(rep,a,case,w,sym,o,m)] for w,o,m in itertools.product(ws,range(2),range(2)))/(4*len(ws)) for case in range(24)] for a in ARMS}
                scores={a:sum(v)/24 for a,v in values.items()}
                contrasts.append(dict(replication=rep,domain=domain,accuracy=scores,caseAccuracies=values,
                                      linkedGain={a:scores['linked']-scores[a] for a in ('base','choice','shuffled')}))
            for arm,task in itertools.product(ARMS,('choice','lookup')):
                prefix=f'r{rep}/{arm}/{task}/';gates[prefix[:-1]]=all(t['passed'] for k,t in groups.items() if k.startswith(prefix))
    criterion=complete and all(v for k,v in gates.items() if '/linked/' in k) and all(v>=.05 for c in contrasts if c['domain']!='trainedSurface' for v in c['linkedGain'].values())
    return dict(schema='menia-value-action-learning-summary-v1',origin=d['header']['origin'],complete=complete,planned=p['planned'],recorded=len(d['results']),
                planHash=d['header']['planHash'],sourceHash=d['header']['sourceHash'],checkpoints=d['checkpoints'],trainingSteps=len(d['steps']),groups=groups,
                contrasts=contrasts,capabilityGates=gates,robustLearningCriterion=criterion,
                tokenLimits=sum(r.get('metrics',{}).get('reachedTokenLimit',False) for r in d['results']),scope=p['scope'])
