"""Exploratory continuation at fixed capacity; all Colab24 outcomes are already seen.

Three rank-8 adapters, six further epochs, fixed snapshots at four/eight total
epochs. Optimizer moments restart once per adapter. No fresh confirmation,
hyperparameter selection, own-answer generation, or consciousness criterion.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from research import confidence_final_training_fit as fit
from research.answer_confidence_data import input_messages
from research.answer_confidence_journal import validate_judgment
from research.confidence_ranking import CONFIG, rng_for
from research.cross_model_prediction import CELLS, canonical, digest, grade
from research.iphone_coupling_report import require, strict_json

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT/'artifacts/confidence-budget-preparation'
EPOCHS = (2, 4, 8)
BASELINE_TOLERANCE = 1e-5
FILES = tuple(sorted(set(fit.FILES + ('confidence_budget_diagnostic.py', 'confidence_budget_gpu.py'))))


def source_hash():
    return digest({n:Path(__file__).with_name(n).read_text(encoding='utf-8') for n in FILES})


def epoch_batches(rows, rep, epoch):
    """Same pairing algorithm as Colab24, extended to fixed epoch indexes 2..7."""
    require(type(rep) is int and rep in range(3) and type(epoch) is int and epoch in range(8), 'Epoch or replication')
    require(digest(rows) == fit.TRAINING_HASH, 'Only the original training rows')
    pairs = []
    for family, level in CELLS:
        group = sorted((r for r in rows if (r['replication'],r['family'],r['level']) == (rep,family,level)), key=lambda r:r['sourceId'])
        rng = rng_for('pairs',rep,epoch,family,level)
        good = [r for r in group if r['target'] == '1']; bad = [r for r in group if r['target'] == '0']
        rng.shuffle(good); rng.shuffle(bad); n = min(len(good),len(bad))
        paired = list(zip(good[:n],bad[:n])); remaining = good[n:]+bad[n:]
        require(len(remaining)%2 == 0, 'Remainder')
        paired.extend(zip(remaining[::2],remaining[1::2]))
        for left,right in paired:
            if rng.randrange(2): left,right = right,left
            pairs.append(dict(left=left,right=right,eligible=left['target'] != right['target']))
    rng_for('batch-order',rep,epoch).shuffle(pairs)
    require(len(pairs) == 288, 'Epoch budget')
    return [pairs[i:i+4] for i in range(0,288,4)]


def training_batches(rep):
    rows = fit.training_rows()
    return [batch for epoch in range(2,8) for batch in epoch_batches(rows,rep,epoch)]


def prepare_rows(ranking_journal, fit_journal):
    """Extract previously audited data, preserving the already-seen evaluation status."""
    from research.confidence_ranking_journal import read_journal as read_parent
    sources = {}
    for name,path,receipt in (
        ('ranking',Path(ranking_journal),ROOT/'artifacts/confidence-ranking-pilot/receipt.json'),
        ('fit',Path(fit_journal),ROOT/'artifacts/final-training-fit-pilot/receipt.json')):
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        require(h == json.loads(receipt.read_text(encoding='utf-8'))['journalSHA256'], 'Audited parent journal required')
        sources[name] = h
    parent = read_parent(ranking_journal); known = fit.read_journal(fit_journal)
    require(parent['complete'] and known['complete'], 'Complete parents required')
    scores = {o['call']['sourceId']:o['scores'] for o in known['judgments'] if o['call']['key'].endswith('-rank')}
    rows = []
    for r in fit.training_rows():
        rows.append(dict(sourceId=f'train-{r["sourceId"]}', replication=r['replication'], phase='train',
            family=r['family'],level=r['level'],messages=r['messages'],target=r['target'],baseline=scores[r['sourceId']]))
    for t in parent['header']['plan']['tasks']:
        answer = parent['answers'][(t['id'],'base')]
        rows.append(dict(sourceId=f'seen-{t["id"]}', replication=t['replication'],phase='seen24base',
            family=t['family'],level=t['level'],messages=input_messages(t['question'],answer['text']),
            target=str(int(grade(answer,t))),baseline=parent['judgments'][(t['id'],'rank','base')]['scores']))
    validate_rows(rows)
    return rows, dict(schema='menia-confidence-budget-data-v1',rowsHash=digest(rows),rows=len(rows),
        parentJournals=sources,trainingHash=fit.TRAINING_HASH,
        scope='1728 original training rows and 1152 already-reviewed Colab24 base answers. The latter never enter optimization and are no longer a reserved confirmation set.')


def validate_rows(rows):
    require(len(rows) == 2880 and len({r['sourceId'] for r in rows}) == 2880, 'Distinct complete inputs')
    require(Counter((r['replication'],r['phase'],r['family'],r['level']) for r in rows) ==
            Counter({(rep,phase,*cell):n for rep in range(3) for phase,n in (('train',96),('seen24base',64)) for cell in CELLS}), 'Input strata')
    expected = {f'train-{r["sourceId"]}':r for r in fit.training_rows()}
    train_questions = set(); seen_questions = set()
    for row in rows:
        require(row['target'] in ('0','1') and row['messages'] == input_messages(row['messages'][1]['content'],row['messages'][2]['content']), 'Messages or label')
        validate_judgment(dict(event='judgment',id=0,status='ok',seconds=0.,scores=row['baseline']),
            dict(call=dict(id=0),messages=row['messages']), dict(confidenceTokenIds=[15,16],vocabularySize=151936))
        if row['phase'] == 'train':
            original = expected[row['sourceId']]
            require(all(row[k] == original[k] for k in ('replication','family','level','messages','target')), 'Training input changed')
            train_questions.add(row['messages'][1]['content'])
        else: seen_questions.add(row['messages'][1]['content'])
    require(len(train_questions) == 1728 and len(seen_questions) == 1152 and not train_questions.intersection(seen_questions), 'Questions overlap')


def load_rows():
    rows = [strict_json(line) for line in (PREPARATION/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    receipt = json.loads((PREPARATION/'data.json').read_text(encoding='utf-8'))
    require(digest(rows) == receipt['rowsHash'], 'Prepared inputs changed')
    validate_rows(rows)
    return rows


def make_plan():
    data = json.loads((PREPARATION/'data.json').read_text(encoding='utf-8'))
    initial = [u for u in fit.make_plan()['units'] if u['arm'] == 'rank']
    units = [dict(key=f'r{rep}-e2',replication=rep,epoch=2) for rep in range(3)]
    units += [dict(key=f'r{rep}-e{epoch}',replication=rep,epoch=epoch) for rep in range(3) for epoch in (4,8)]
    return dict(schema='menia-confidence-budget-plan-v1',model=fit.make_plan()['model'],initial=initial,
        data=data, units=units, epochs=list(EPOCHS),additionalEpochs=6, updatesPerReplication=432,
        plannedUpdates=1296,plannedCalls=8640,examplesPerUnit=960,
        training=dict(CONFIG,epochs=6,updatesPerAdapter=432),
        batchHashes={str(rep):[digest(b) for b in training_batches(rep)] for rep in range(3)},
        optimizer='Fresh AdamW once per replication; moments then persist through all six additional epochs. Original moments were not saved.',
        snapshotTiming='Save at 144 and 432 additional steps; all six snapshots frozen before post-training assessments.',
        baselineTolerance=BASELINE_TOLERANCE,newGenerations=0,newLLMBaseWeightUpdates=0,
        selection='All three original rank-arm replications and all fixed snapshots; no best-checkpoint selection.',
        scope='Exploratory fixed-capacity optimization-budget diagnostic on old training and already-seen Colab24 base answers. No fresh confirmation, specific ranking advantage, privileged access, native action or consciousness criterion.')


def calls(rows,plan):
    require(digest(rows) == plan['data']['rowsHash'], 'Input identity')
    schedule = []
    for unit in plan['units']:
        subset = sorted((r for r in rows if r['replication'] == unit['replication']), key=lambda r:(r['phase'] != 'train',r['sourceId']))
        require(len(subset) == 960, 'Unit input count')
        for row in subset: schedule.append(dict(id=len(schedule),key=unit['key'],sourceId=row['sourceId']))
    require(len(schedule) == 8640, 'Assessment count')
    return schedule


def request_for(call,rows):
    return dict(event='request',call=call,messages=rows[call['sourceId']]['messages'])


def read_journal(path):
    events = []; previous = '0'*64
    for i,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        envelope = strict_json(line)
        require(set(envelope) == {'sequence','previous','payload','sha256'} and envelope['sequence'] == i and envelope['previous'] == previous, 'Journal order')
        require(envelope['sha256'] == digest({k:envelope[k] for k in ('sequence','previous','payload')}), 'Journal hash')
        events.append(envelope['payload']); previous = envelope['sha256']
    require(bool(events), 'Empty journal')
    header = events[0]; plan = make_plan(); rows = load_rows(); by_id = {r['sourceId']:r for r in rows}; schedule = calls(rows,plan)
    require(header['event'] == 'header' and header['plan'] == plan and header['planHash'] == digest(plan), 'Plan changed')
    require(header['sourceHash'] == source_hash() and header['callPlanHash'] == digest(schedule), 'Sources or calls changed')
    require(header['origin'] in ('transformers_gpu','synthetic_fixture') and header['metadata']['model'] == plan['model'], 'Model or origin')
    require(header['metadata']['newLLMBaseWeightUpdates'] == 0, 'Base weight updates')
    active = pending = training = None; completed = []; trained = []; steps = []; checkpoints = {}; judgments = []
    failed = final = baseline_passed = False; count = step = 0; max_baseline_delta = 0.
    for event in events[1:]:
        require(not failed and not final, 'Events after terminal')
        kind = event['event']
        if kind == 'failure': failed = True
        elif kind == 'baseline_complete':
            require(len(completed) == 3 and training is None and active is None and not baseline_passed and not trained, 'Baseline order')
            require(event['maxAbsoluteDifference'] == max_baseline_delta <= BASELINE_TOLERANCE, 'Baseline scores changed')
            baseline_passed = True
        elif kind == 'training_start':
            require(baseline_passed and len(completed) == 3 and training is None and active is None and len(trained) < 3, 'Training order')
            rep = len(trained); training = plan['initial'][rep]; step = 0
            require(event['replication'] == rep and event['initialHash'] == training['sha256'] and event['optimizerReset'] is True, 'Training initialization')
            require(event['trainableParameters'] == 2949120, 'Adapter capacity changed')
        elif kind == 'training_step':
            require(training is not None and event['replication'] == training['replication'] and event['step'] == step+1 <= 432, 'Training step order')
            require(event['dataHash'] == plan['batchHashes'][str(training['replication'])][step], 'Training batch changed')
            for field,n in (('losses',8),('pairLosses',4),('scoreDifferences',4)):
                require(len(event[field]) == n and all(type(x) in (int,float) and math.isfinite(x) for x in event[field]), 'Nonfinite training values')
            require(all(x >= 0 for x in event['losses']+event['pairLosses']), 'Negative loss')
            require(len(event['eligible']) == 4 and all(type(x) is bool for x in event['eligible']), 'Pair eligibility')
            require(len(event['inputTokens']) == 8 and all(type(x) is int and 1<x<=1792 for x in event['inputTokens']), 'Training tokens')
            for field in ('gradientNorm','seconds','objective'):
                require(type(event[field]) in (int,float) and math.isfinite(event[field]) and event[field]>=0, 'Training scalar')
            step += 1; steps.append(event)
        elif kind == 'checkpoint':
            require(training is not None and step in (144,432), 'Snapshot timing')
            key = f'r{training["replication"]}-e{2+step//72}'
            require(event['key'] == key and key not in checkpoints and event['step'] == step, 'Snapshot identity')
            require(event['checkpoint'] == Path(path).with_suffix('.'+key+'.safetensors').name and
                    isinstance(event['sha256'],str) and len(event['sha256']) == 64 and event['bytes']>0, 'Snapshot file')
            require(event['baseParameterVersionsUnchanged'] is True, 'Base parameters changed')
            checkpoints[key] = event
        elif kind == 'training_complete':
            require(training is not None and event['replication'] == training['replication'] and step == event['steps'] == 432, 'Training incomplete')
            require(all(f'r{training["replication"]}-e{epoch}' in checkpoints for epoch in (4,8)), 'Missing snapshot')
            trained.append(training); training = None
        elif kind == 'unit_start':
            require(active is None and pending is None and training is None and len(completed)<9, 'Assessment order')
            active = plan['units'][len(completed)]; count = 0
            if active['epoch'] == 2:
                require(not baseline_passed and not trained, 'Baseline timing')
                expected_hash = plan['initial'][active['replication']]['sha256']
            else:
                require(len(trained) == 3 and len(checkpoints) == 6, 'All weights must be frozen')
                expected_hash = checkpoints[active['key']]['sha256']
            require(event['key'] == active['key'] and event['sha256'] == expected_hash, 'Assessment checkpoint changed')
        elif kind == 'request':
            require(active is not None and pending is None and count<960 and len(judgments)<8640, 'Unexpected request')
            call = schedule[len(judgments)]
            require(call['key'] == active['key'] and event == request_for(call,by_id), 'Assessment input changed')
            pending = event
        elif kind == 'judgment':
            require(pending is not None, 'Unpaired judgment')
            validate_judgment(event,pending,header['metadata'])
            if active['epoch'] == 2:
                ref = by_id[pending['call']['sourceId']]['baseline']
                for field in ('conditionalCorrect','candidateMass'):
                    max_baseline_delta = max(max_baseline_delta,abs(event['scores'][field]-ref[field]))
                require(max_baseline_delta <= BASELINE_TOLERANCE, 'Baseline arithmetic changed')
            judgments.append(dict(call=pending['call'],scores=event['scores'],seconds=event['seconds']))
            pending = None; count += 1
        elif kind == 'unit_complete':
            require(active is not None and pending is None and count == event['calls'] == 960 and event['key'] == active['key'], 'Incomplete assessment')
            require(event['actualAdapterMatchesFile'] is True and event['baseParameterVersionsUnchanged'] is True, 'Evaluation changed weights')
            completed.append(active); active = None
        elif kind == 'complete':
            require(active is None and pending is None and training is None and baseline_passed and len(completed) == 9 and
                    len(trained) == 3 and len(judgments) == event['calls'] == 8640 and len(steps) == event['steps'] == 1296, 'Incomplete diagnostic')
            final = True
        else: raise ValueError('Unknown event '+kind)
    return dict(header=header,rows=by_id,judgments=judgments,steps=steps,checkpoints=checkpoints,
        complete=final and not failed,failed=failed,chainEnd=previous,maxBaselineDifference=max_baseline_delta)


def summarize(data):
    require(data['complete'], 'Complete diagnostic required')
    units = [dict(key=u['key']+'/'+phase) for u in data['header']['plan']['units'] for phase in ('train','seen24base')]
    judgments = [dict(o,call=dict(o['call'],key=o['call']['key']+'/'+data['rows'][o['call']['sourceId']]['phase'])) for o in data['judgments']]
    header = dict(data['header'],plan=dict(units=units,scope=data['header']['plan']['scope']))
    report = fit.summarize(dict(data,header=header,judgments=judgments))
    report.update(schema='menia-confidence-budget-summary-v1',newAdapterUpdates=len(data['steps']),
        baselineMaxAbsoluteDifference=data['maxBaselineDifference'],checkpoints=data['checkpoints'],
        trainingInputTokens=sum(sum(s['inputTokens']) for s in data['steps']),
        trainingStepSeconds=math.fsum(s['seconds'] for s in data['steps']))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true'); mode.add_argument('--journal',type=Path)
    parser.add_argument('--ranking-parent',type=Path); parser.add_argument('--fit-parent',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(args.ranking_parent is not None and args.fit_parent is not None and not PREPARATION.exists(), 'Fresh data preparation required')
        rows,receipt = prepare_rows(args.ranking_parent,args.fit_parent)
        PREPARATION.mkdir()
        (PREPARATION/'rows.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8',newline='\n')
        (PREPARATION/'data.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
        p = make_plan(); report = dict(plan=p,planHash=digest(p),sourceHash=source_hash(),callPlanHash=digest(calls(rows,p)))
        output = PREPARATION/'design.json'
    else:
        require(args.output is not None, 'Output required'); output = args.output
        report = summarize(read_journal(args.journal))
    with output.open('x',encoding='utf-8',newline='\n') as stream: stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('planHash','sourceHash')}))
