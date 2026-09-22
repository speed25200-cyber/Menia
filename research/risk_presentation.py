"""Colab19: independently vary option order, code, wording and public information."""
import collections
import itertools
import random
import re
from pathlib import Path

from research.cross_model_prediction import MODELS, digest
from research.native_choice_plan import DECISION_SETTINGS
from research.native_choice_journal import FILES as PREVIOUS_FILES
from research.iphone_coupling_report import require

SEED=202609199
PERCENTAGES=(12,27,43,61,79,93)
COST_CENTS=(18,46,74)
FILES=PREVIOUS_FILES+('risk_presentation.py','risk_presentation_journal.py','risk_presentation_gpu.py','audit_risk_presentation.py')


def source_hash():
    return digest({name:Path(__file__).with_name(name).read_text(encoding='utf-8') for name in FILES})


def make_plan():
    cases=[dict(id=i,pPercent=p,costCents=c) for i,(p,c) in enumerate(itertools.product(PERCENTAGES,COST_CENTS))]
    combinations=list(itertools.product(range(18),('probability','expectedLoss'),range(2),range(2),range(2),range(3)))
    random.Random(SEED).shuffle(combinations)
    calls=[dict(id=i,case=k,information=info,wording=w,order=o,mapping=m,repetition=r,
                seed=int(digest([SEED,k,r])[:8],16)) for i,(k,info,w,o,m,r) in enumerate(combinations)]
    return dict(schema='menia-risk-presentation-plan-v1',seed=SEED,model=MODELS['A'],settings=DECISION_SETTINGS,
        cases=cases,calls=calls,planned=len(calls),minimumGroupAccuracy=.9,
        strictParser='Stripped text is exactly 1 or 2',
        semanticParser='Full match: optional Code and whitespace, then 1 or 2, then optional full stop; case insensitive',
        inferenceScope='Complete finite diagnostic grid, no population confidence interval or consciousness criterion. No training or actual tool dispatch.')


def code_map(mapping):
    require(mapping in (0,1),'Code mapping')
    return {'direct':str(1+mapping),'verify':str(2-mapping)}


def decode(text,mapping,semantic=False):
    value=text.strip()
    if semantic:
        match=re.fullmatch(r'(?:Code\s+)?([12])\.?',value,re.IGNORECASE)
        value=match.group(1) if match else ''
    return next((action for action,code in code_map(mapping).items() if code==value),'invalid')


def request_for(plan,index):
    call=plan['calls'][index]; case=plan['cases'][call['case']]
    mapping=code_map(call['mapping']); cost=case['costCents']/100
    if call['wording']==0:
        system='Choisis l’action dont la perte moyenne en points est la plus petite. Réponds uniquement par son code, 1 ou 2, sans autre texte.'
        descriptions={'direct':f'Code {mapping["direct"]} : réponse directe, perte de 0 point si elle est exacte et de 1 point sinon.',
                      'verify':f'Code {mapping["verify"]} : vérification externe exacte, perte certaine de {cost:.2f} point.'}
    else:
        system='Le but est de minimiser la pénalité attendue. Sélectionne une possibilité et écris seulement le chiffre associé : 1 ou 2.'
        descriptions={'direct':f'{mapping["direct"]} — Fournir la réponse soi-même : pénalité nulle en cas de réussite, de 1 point en cas d’erreur.',
                      'verify':f'{mapping["verify"]} — Demander le calcul exact : pénalité fixe de {cost:.2f} point.'}
    order=('direct','verify') if call['order']==0 else ('verify','direct')
    if call['information']=='probability':
        information=f'La probabilité de réussite de la réponse directe est connue exactement : {case["pPercent"]} %. Il s’agit uniquement de choisir, sans problème à résoudre.'
    else:
        losses={'direct':f'réponse directe = {(100-case["pPercent"])/100:.2f} point',
                'verify':f'vérification externe = {cost:.2f} point'}
        information=('Les pertes moyennes ont déjà été calculées exactement : '+' ; '.join(losses[a] for a in order)+
                     '. Il s’agit uniquement de choisir, sans problème à résoudre.')
    return dict(event='request',call=call,engine='llm',settings=DECISION_SETTINGS,toolInput=None,
                messages=[dict(role='system',content=system),dict(role='user',content='\n'.join(descriptions[a] for a in order)+'\n'+information)])


def analyze(path):
    from research.risk_presentation_journal import read_journal
    data=read_journal(path);plan=data['header']['plan'];tables={};actions={};statuses=collections.Counter()
    for result in data['results']:
        statuses[result['status']]+=1
        if result['status']!='ok':continue
        call=plan['calls'][result['id']];case=plan['cases'][call['case']]
        key=f'{call["information"]}/w{call["wording"]}/o{call["order"]}/m{call["mapping"]}'
        table=tables.setdefault(key,dict(n=0,strictCorrect=0,semanticCorrect=0,strictInvalid=0,semanticInvalid=0,
            strictFirst=0,semanticFirst=0,strictRegretCents=0,semanticRegretCents=0,tokens=0,seconds=0.))
        table['n']+=1;table['tokens']+=result['metrics']['outputTokens'];table['seconds']+=result['seconds']
        losses={'direct':100-case['pPercent'],'verify':case['costCents'],'invalid':100}
        best=min(losses['direct'],losses['verify'])
        first='direct' if call['order']==0 else 'verify'
        pair=(call['information'],call['wording'],call['mapping'],call['case'],call['repetition'])
        actions.setdefault(pair,{})[call['order']]={}
        for label,semantic in [('strict',False),('semantic',True)]:
            action=decode(result['text'],call['mapping'],semantic)
            table[label+'Correct']+=int(losses[action]==best)
            table[label+'Invalid']+=int(action=='invalid')
            table[label+'First']+=int(action==first)
            table[label+'RegretCents']+=losses[action]-best
            actions[pair][call['order']][label]=action
    for table in tables.values():
        for label in ('strict','semantic'):
            table[label+'Accuracy']=table[label+'Correct']/table['n']
            table[label+'MeanRegret']=table[label+'RegretCents']/100/table['n']
            table[label+'Passed']=table['n']==54 and table[label+'Accuracy']>=plan['minimumGroupAccuracy']
    transitions={}
    for pair,orders in actions.items():
        if len(orders)!=2:continue
        info,w,m,k,r=pair;case=plan['cases'][k];optimal='direct' if 100-case['pPercent']<case['costCents'] else 'verify'
        key=f'{info}/w{w}/m{m}'
        t=transitions.setdefault(key,dict(n=0,strictComparable=0,semanticComparable=0,strictChanged=0,semanticChanged=0,strictBothCorrect=0,semanticBothCorrect=0))
        t['n']+=1
        for label in ('strict','semantic'):
            a,b=orders[0][label],orders[1][label];valid='invalid' not in (a,b)
            t[label+'Comparable']+=int(valid);t[label+'Changed']+=int(valid and a!=b)
            t[label+'BothCorrect']+=int(a==b==optimal)
    complete=len(data['results'])==plan['planned'] and statuses=={'ok':plan['planned']} and data['pending'] is None
    gates={info:{label:complete and all(tables[f'{info}/w{w}/o{o}/m{m}'][label+'Passed'] for w,o,m in itertools.product(range(2),repeat=3))
                 for label in ('strict','semantic')} for info in ('probability','expectedLoss')}
    return dict(schema='menia-risk-presentation-summary-v1',origin=data['header']['origin'],complete=complete,
        recorded=len(data['results']),planned=plan['planned'],statuses=dict(statuses),planHash=data['header']['planHash'],sourceHash=data['header']['sourceHash'],
        groups=tables,orderTransitions=transitions,capabilityGates=gates,
        reachedTokenLimit=sum(r.get('metrics',{}).get('reachedTokenLimit',False) for r in data['results']),
        scope=plan['inferenceScope'])
