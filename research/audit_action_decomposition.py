"""Separate integer tallies; shared immutable plan and journal integrity reader."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from research.action_decomposition import read_journal, load_parent
from research.iphone_coupling_report import require


def verify_tallies(plan, results, prior, report):
    require(len(results)==plan['planned'] and report['complete'],'Complete collection required')
    groups={}; index={}; replay=0
    names={'direct':'DIRECT','verify':'VERIFIER'}
    def codes(c):
        v=['1','2'] if c['symbols']=='digits' else ['A','B']
        return {'direct':v[c['mapping']], 'verify':v[1-c['mapping']]}
    def choice(case): return 'direct' if case['pPercent']+case['costCents']>100 else 'verify'
    for c,r in zip(plan['calls'],results):
        require(c['id']==r['id'] and r['status']=='ok','Successful ordered results')
        stage=c['stage']; text=r['text'].strip()
        key=f'r{c["replication"]}/{c["arm"]}/{stage}/w{c["wording"]}/o{c["order"]}'
        if stage in ('mapping','replay'):key+=f'/{c["symbols"]}/m{c["mapping"]}'
        if stage=='mapping':
            key+='/'+c['selected']; valid=set(codes(c).values()); target=codes(c)[c['selected']]
            idx=(c['replication'],c['arm'],stage,c['wording'],c['order'],c['symbols'],c['mapping'],c['selected'])
            index[idx]=text
        else:
            case=plan['cases'][c['case']]; best=choice(case)
            if stage=='replay':
                valid=set(codes(c).values());target=codes(c)[best]
                replay+=r['text']==prior['results'][c['parentId']]['text']
            else:
                allowed=names if stage=='semantic' else dict(direct=f'{(100-case["pPercent"])/100:.2f}',verify=f'{case["costCents"]/100:.2f}')
                valid=set(allowed.values());target=allowed[best]
                chosen=next((a for a in names if allowed[a]==text),None)
                index[(c['replication'],c['arm'],stage,c['wording'],c['order'],c['case'])]=chosen
        t=groups.setdefault(key,dict(n=0,correct=0,invalid=0,tokens=0,seconds=0.))
        t['n']+=1;t['correct']+=int(text==target);t['invalid']+=int(text not in valid)
        t['tokens']+=r['metrics']['outputTokens'];t['seconds']+=r['seconds']
    for t in groups.values():t['accuracy']=t['correct']/t['n']
    require(replay==288,'Exact replay gate')
    composed={}
    for c,r in zip(prior['header']['plan']['calls'],prior['results']):
        if c['information']!='expectedLoss':continue
        best=choice(plan['cases'][c['case']]);mapping=codes(c)
        for stage in ('number','semantic'):
            chosen=index[(c['replication'],c['arm'],stage,c['wording'],c['order'],c['case'])]
            prefix=(c['replication'],c['arm'],'mapping',c['wording'],c['order'],c['symbols'],c['mapping'])
            encoded=index[prefix+(chosen,)] if chosen else None
            key=f'r{c["replication"]}/{c["arm"]}/{stage}/w{c["wording"]}/o{c["order"]}/{c["symbols"]}/m{c["mapping"]}'
            t=composed.setdefault(key,dict(n=0,originalCorrect=0,choiceCorrect=0,encodingGivenChoiceCorrect=0,oracleEncodingCorrect=0,composedCorrect=0,compensatingErrors=0,invalidChoice=0))
            t['n']+=1;t['originalCorrect']+=int(r['text'].strip()==mapping[best])
            t['choiceCorrect']+=int(chosen==best);t['encodingGivenChoiceCorrect']+=int(chosen is not None and encoded==mapping[chosen])
            t['oracleEncodingCorrect']+=int(index[prefix+(best,)]==mapping[best])
            t['composedCorrect']+=int(encoded==mapping[best]);t['compensatingErrors']+=int(chosen!=best and encoded==mapping[best])
            t['invalidChoice']+=int(chosen is None)
    for t in composed.values():
        require(t['n']==16,'Case coverage')
        t['composedAccuracy']=t['composedCorrect']/16
        t['deltaVsOriginal']=(t['composedCorrect']-t['originalCorrect'])/16
        t['passed']=t['composedCorrect']>=15
    maxdiff=0.
    def compare(a,b):
        nonlocal maxdiff
        if isinstance(b,dict):
            require(isinstance(a,dict) and set(a)==set(b),'Tally fields')
            for k in b:compare(a[k],b[k])
        elif type(b) is float:
            require(type(a) in (int,float) and math.isfinite(a),'Finite tally')
            maxdiff=max(maxdiff,abs(a-b));require(abs(a-b)<1e-10,'Numeric tally differs')
        else:require(type(a) is type(b) and a==b,'Exact tally differs')
    compare(report['groups'],groups);compare(report['composed'],composed)
    require(report['replayMatched']==288 and report['recorded']==2448 and report['planned']==2448,'Reported counts')
    require(report['reachedTokenLimit']==sum(r['metrics']['reachedTokenLimit'] for r in results),'Truncation count')
    return dict(verified=True,groups=len(groups),composedGroups=len(composed),replayMatched=replay,
                maxAbsoluteDifference=maxdiff,scope='Separate counters; shared fixed plan and integrity reader, not an external replication.')


def verify(path,parent_path,report):
    d=read_journal(path);p=d['header']['plan'];prior=load_parent(parent_path,p)
    require(not d['failed'] and d['pending'] is None,'Incomplete journal')
    require(report['planHash']==d['header']['planHash'] and report['sourceHash']==d['header']['sourceHash'],'Report identity')
    output=verify_tallies(p,d['results'],prior,report)
    output.update(schema='menia-action-decomposition-verification-v1',journalSHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('journal','parent','summary','output'):parser.add_argument(name,type=Path)
    args=parser.parse_args()
    result=verify(args.journal,args.parent,json.loads(args.summary.read_text(encoding='utf-8')))
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
