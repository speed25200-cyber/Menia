"""Separate string parser and integer accounting; shares only validated input/plan."""
import hashlib
import itertools
import math
from pathlib import Path

from research.risk_presentation_journal import read_journal
from research.audit_composition_diagnostic import compare


def verify(path,report):
    data=read_journal(path);p=data['header']['plan'];rows=data['results']
    if len(rows)!=864 or any(r['status']!='ok' for r in rows) or data['pending'] is not None:raise ValueError('Incomplete collection')
    parsed={};groups={};transitions={}
    for info,w,o,m in itertools.product(('probability','expectedLoss'),range(2),range(2),range(2)):
        subset=[(c,rows[c['id']]) for c in p['calls'] if (c['information'],c['wording'],c['order'],c['mapping'])==(info,w,o,m)]
        if len(subset)!=54:raise ValueError('Unbalanced group')
        t=dict(n=54,tokens=sum(r['metrics']['outputTokens'] for _,r in subset),seconds=math.fsum(r['seconds'] for _,r in subset))
        for label in ('strict','semantic'):
            n_correct=n_invalid=n_first=regret=0
            for c,r in subset:
                raw=r['text'].strip().lower();value=raw
                if label=='semantic':
                    # Independent accepted grammar, including variable whitespace after Code.
                    if value.startswith('code'):
                        tail=value[4:]
                        value=tail.lstrip() if tail and tail[0].isspace() else ''
                    if value.endswith('.'):value=value[:-1]
                a='invalid'
                if value in ('1','2'):
                    a='direct' if value==str(1+m) else 'verify'
                case=p['cases'][c['case']];direct=100-case['pPercent'];verify=case['costCents']
                loss={'direct':direct,'verify':verify,'invalid':100}[a];best=min(direct,verify)
                n_correct+=int(loss==best);n_invalid+=int(a=='invalid');n_first+=int(a==('direct' if o==0 else 'verify'));regret+=loss-best
                parsed[(info,w,m,c['case'],c['repetition'],o,label)]=a
            t.update({label+'Correct':n_correct,label+'Invalid':n_invalid,label+'First':n_first,label+'RegretCents':regret,
                      label+'Accuracy':n_correct/54,label+'MeanRegret':regret/5400,label+'Passed':n_correct>=49})
        groups[f'{info}/w{w}/o{o}/m{m}']=t
    for info,w,m in itertools.product(('probability','expectedLoss'),range(2),range(2)):
        t={'n':54}
        for label in ('strict','semantic'):
            comparable=changed=both=0
            for case,r in itertools.product(p['cases'],range(3)):
                a,b=[parsed[(info,w,m,case['id'],r,o,label)] for o in range(2)]
                valid=a in ('direct','verify') and b in ('direct','verify')
                best='direct' if case['pPercent']+case['costCents']>100 else 'verify'
                comparable+=int(valid);changed+=int(valid and a!=b);both+=int(a==b==best)
            t.update({label+'Comparable':comparable,label+'Changed':changed,label+'BothCorrect':both})
        transitions[f'{info}/w{w}/m{m}']=t
    gates={info:{label:all(groups[f'{info}/w{w}/o{o}/m{m}'][label+'Correct']>=49 for w,o,m in itertools.product(range(2),repeat=3))
                 for label in ('strict','semantic')} for info in ('probability','expectedLoss')}
    expected=dict(schema='menia-risk-presentation-summary-v1',origin=data['header']['origin'],complete=True,recorded=864,planned=864,
        statuses={'ok':864},planHash=data['header']['planHash'],sourceHash=data['header']['sourceHash'],groups=groups,
        orderTransitions=transitions,capabilityGates=gates,reachedTokenLimit=sum(int(r['metrics']['reachedTokenLimit']) for r in rows),scope=p['inferenceScope'])
    delta=compare(expected,report)
    return dict(schema='menia-risk-presentation-verification-v1',verified=True,recorded=864,groups=16,orderComparisons=8,
        journalSHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),maxAbsoluteDifference=delta,
        scope='Separate parser and integer tally; shared validated journal/plan, no external replication.')
