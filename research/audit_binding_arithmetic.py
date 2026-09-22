"""Separate decimal arithmetic on the recorded targeted-binding experiment."""
import argparse
import collections
from decimal import Decimal, localcontext
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('journal',type=Path)
parser.add_argument('--summary',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
events=[json.loads(line)['payload'] for line in args.journal.read_text(encoding='utf-8').splitlines()]
report=json.loads(args.summary.read_text(encoding='utf-8'))
cases={e['case']['id']:e['case'] for e in events if e['event']=='state'}
branches={kind:{(e['call']['case'],e['call']['condition']):e['decoded'] for e in events if e['event']==kind}
          for kind in ('forecast','task')}
result={}; maximum=0.
with localcontext() as context:
    context.prec=60
    for condition,unit in report['units'].items():
        squared=[]; rows=[]
        for case in range(24):
            f=branches['forecast'][case,condition]; t=branches['task'][case,condition]
            expected=t['candidateTokenIds'][cases[case]['expected']]
            valid=len(t['tokenIds'])==2 and t['tokenIds'][0] in t['candidateTokenIds'] and t['tokenIds'][1] in t['eosTokenIds']
            correct=valid and t['tokenIds'][0]==expected
            logits=[Decimal(str(v)) for v in f['candidateLogits']]
            probability=1/(1+(logits[0]-logits[1]).exp())
            assert abs(float(probability)-f['conditionalPositive'])<1e-14
            squared.append((Decimal(str(f['conditionalPositive']))-int(correct))**2)
            original=branches['forecast'][case,'actual']['conditionalPositive']
            rows.append(dict(case=case,expected=cases[case]['expected'],target=cases[case]['targetKey'],
                taskText=t['text'],taskValid=valid,correct=correct,forecastText=f['text'],
                forecast=f['conditionalPositive'],forecastDelta=f['conditionalPositive']-original,
                taskMass=t['candidateMass'],forecastMass=f['candidateMass']))
        brier=float(sum(squared)/len(squared)); difference=abs(brier-unit['brierConditionalAll']); maximum=max(maximum,difference)
        assert difference<1e-14 and sum(r['correct'] for r in rows)==unit['correct']
        result[condition]=dict(brier=brier,rows=rows,
            taskTexts=dict(collections.Counter(r['taskText'] for r in rows)),
            forecastTexts=dict(collections.Counter(r['forecastText'] for r in rows)))
out=dict(schema='menia-binding-posthoc-arithmetic-v1',maximumBrierDifference=maximum,conditions=result,
         scope='Separate decimal arithmetic on the same recorded logits and outputs; no independent collection or held-out evidence.')
path=args.output
with path.open('x',encoding='utf-8',newline='\n') as stream:
    stream.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(verified=True,conditions=len(result),maximumBrierDifference=maximum)))
