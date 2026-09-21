"""Post-hoc description of constant native choices; no changed success criterion."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/value-action-learning-pilot'


def main(journal):
    receipt=json.loads((ART/'receipt.json').read_text(encoding='utf-8'))
    report=json.loads((ART/'summary.json').read_text(encoding='utf-8'))
    raw=Path(journal).read_bytes()
    assert report['complete'] and hashlib.sha256(raw).hexdigest()==receipt['journalSHA256']
    events=[json.loads(line)['payload'] for line in raw.splitlines()]
    plan=events[0]['plan'];counts=defaultdict(Counter);expected=defaultdict(Counter)
    for event in events:
        if event['event']!='result':continue
        call=plan['calls'][event['id']]
        if call['task']!='choice':continue
        key=f'r{call["replication"]}/{call["arm"]}/choice/w{call["wording"]}/{call["symbols"]}/o{call["order"]}/m{call["mapping"]}'
        case=plan['testCases'][call['case']];codes=('1','2') if call['symbols']=='digits' else ('A','B')
        wanted=codes[call['mapping'] if case['pPercent']+case['costCents']>100 else 1-call['mapping']]
        counts[key][event['text'].strip()]+=1;expected[key][wanted]+=1
    assert len(counts)==288
    constant={};by_arm=Counter()
    for key,tally in sorted(counts.items()):
        assert sum(tally.values())==24 and sorted(expected[key].values())==[12,12]
        if len(tally)==1:
            assert report['groups'][key]['correct']==12
            constant[key]=dict(outputs=dict(tally),expected=dict(expected[key]),correct=12,n=24)
            by_arm[key.split('/')[1]]+=1
    out=dict(schema='menia-value-action-constant-choice-diagnostic-v1',journalSHA256=receipt['journalSHA256'],
             examinedChoiceGroups=288,constantGroupsByArm=dict(by_arm),groups=constant,
             scope='Post-hoc output description on all choice groups, after fixed analysis. Constants give 12/24 because every group is balanced. No circuit explanation, revised criterion, or consciousness inference.')
    (ART/'constant-choice-diagnostic.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(groups=len(constant),byArm=dict(by_arm))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('journal',type=Path)
    main(parser.parse_args().journal)
