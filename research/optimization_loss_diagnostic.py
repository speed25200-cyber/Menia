"""Post-outcome loss trace from Colab 12; motivates, but cannot establish, forgetting."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics

from research import state_composition as source
from research.composition_diagnostic import COMPOSITION_SHA256
from research.iphone_coupling_report import require


def final_cross_entropy(result, target):
    """Recover target probability from the four logits and their recorded full-vocabulary mass."""
    logits = result['choiceLogits']; mass = result['choiceMass']
    require(len(logits) == 4 and 0 < mass <= 1.000001, 'Invalid recorded probability mass')
    maximum = max(logits)
    return -math.log(mass)+math.log(math.fsum(math.exp(x-maximum) for x in logits))+maximum-logits[target]


def analyze(journal):
    journal = Path(journal)
    require(hashlib.sha256(journal.read_bytes()).hexdigest() == COMPOSITION_SHA256, 'Unexpected source journal')
    h, rows, trained, pending, events = source.read_journal(journal)
    require(len(trained) == 9 and len(rows) == len(h['plan']['evaluation']) and pending is None, 'Incomplete source')
    online = collections.defaultdict(list); trace = []
    for event in events:
        if event['event'] != 'training_step': continue
        for ex, loss in zip(event['examples'], event['losses']):
            role = (ex['family']+'/'+('monitor-absent' if ex['position'] == 0 else 'monitor-present')) if ex['task'] == 'monitor' else 'public'
            online[(event['key'], event['group']['epoch'], role)].append(loss)
        trace.append(dict(key=event['key'], step=event['step'], group=event['group'],
                          losses=event['losses'], gradientNorm=event['gradientNorm']))
    summaries = []
    for (key, epoch, role), values in sorted(online.items()):
        summaries.append(dict(key=key, epoch=epoch, role=role, n=len(values),
            mean=statistics.mean(values), median=statistics.median(values),
            firstEightMean=statistics.mean(values[:8]), lastEightMean=statistics.mean(values[-8:])))
    blocks = {b['id']: b for b in h['plan']['blocks']}; final = collections.defaultdict(list)
    for row in rows:
        q = row['request']; b = blocks[q['block']]
        if q['arm'] != 'composed' or q['family'] != 'hidden' or q['task'] != 'monitor' or q['position'] == 3 or b['split'] != 'train': continue
        y = source.target(b, q['task'], q['position'], q['format'], q['mapping'])
        final[(q['replication'], q['mapping'], int(q['position'] != 0))].append(final_cross_entropy(row['result'], y))
    return dict(schema='menia-optimization-loss-diagnostic-v1', origin='post-outcome-reanalysis',
        sourceJournalSHA256=COMPOSITION_SHA256, sourcePlanHash=h['planHash'], sourceHash=h['sourceHash'],
        trainingSteps=len(trace), onlineSummaries=summaries, onlineTrace=trace,
        finalTrainingCrossEntropy=[dict(replication=r, mapping=m, present=bool(p), n=len(v), mean=statistics.mean(v))
                                   for (r,m,p),v in sorted(final.items())],
        interpretation='Online losses use changing weights and different example order; final losses use one frozen checkpoint. This is a retrospective clue, not a paired before/after causal test, native-accuracy estimate, or consciousness evidence.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('journal',type=Path)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(analyze(a.journal),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
