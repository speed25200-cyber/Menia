"""Training lineage followed by evaluation, with no restart or test-driven checkpoint selection."""
import math
from pathlib import Path

from research.cross_model_prediction import digest
from research.iphone_coupling_report import require,strict_json
from research.native_choice_journal import validate_result
from research.action_binding import make_plan,source_hash,training_batch,request_for


def read_journal(path):
    rows=[strict_json(l) for l in Path(path).read_text(encoding='utf-8').splitlines()];require(bool(rows),'Empty journal')
    previous='0'*64;events=[]
    for i,row in enumerate(rows):
        require(set(row)=={'sequence','previous','payload','sha256'} and row['sequence']==i and row['previous']==previous,'Journal order')
        require(row['sha256']==digest({k:row[k] for k in ('sequence','previous','payload')}),'Chain hash')
        previous=row['sha256'];events.append(row['payload'])
    h=events[0];p=make_plan();require(h['event']=='header' and h['plan']==p and h['planHash']==digest(p),'Plan changed')
    require(h['sourceHash']==source_hash() and h['origin'] in ('transformers_gpu','synthetic_fixture'),'Source/origin')
    starts={};steps=[];checkpoints={};results=[];pending=None;active=None;step=0;failed=False
    for e in events[1:]:
        require(not failed,'Events after failure')
        if e['event']=='failure':failed=True;continue
        if e['event']=='training_start':
            require(not results and pending is None and active is None and len(checkpoints)<6,'Training after evaluation or overlap')
            unit=p['trainingUnits'][len(checkpoints)];require(e['key']==unit['key'],'Training unit order')
            require(type(e['initializationHash']) is str and len(e['initializationHash'])==64 and e['trainableParameters']>0,'Initialization')
            pair=f'r{unit["replication"]}-fixed'
            if pair in starts:require(e['initializationHash']==starts[pair]['initializationHash'],'Unmatched initialization')
            starts[unit['key']]=e;active=unit;step=0
        elif e['event']=='training_step':
            require(active is not None and e['key']==active['key'] and e['step']==step+1<=64,'Training step order')
            require(e['dataHash']==digest(training_batch(p,active,e['step'])),'Training data changed')
            require(len(e['losses'])==8 and all(math.isfinite(v) and v>=0 for v in e['losses']),'Training losses')
            require(math.isfinite(e['gradientNorm']) and e['gradientNorm']>=0,'Gradient norm')
            require(len(e['inputTokens'])==8 and all(type(v) is int and 1<v<=512 for v in e['inputTokens']),'Training length')
            steps.append(e);step+=1
        elif e['event']=='training_complete':
            require(active is not None and e['key']==active['key'] and step==e['steps']==64,'Incomplete training')
            require(e['initializationHash']==starts[e['key']]['initializationHash'],'Initial weights changed')
            require(len(e['sha256'])==64 and e['checkpoint']==Path(path).with_suffix('.'+e['key']+'.safetensors').name,'Checkpoint identity')
            checkpoints[e['key']]=e;active=None
        elif e['event']=='request':
            require(len(checkpoints)==6 and active is None and pending is None and len(results)<p['planned'],'Unfrozen evaluation')
            pending=request_for(p,len(results),checkpoints);require(e==pending,'Evaluation request changed')
        elif e['event']=='result':
            require(pending is not None,'Unpaired result');validate_result(e,pending)
            results.append(e);pending=None;failed=e['status']!='ok'
        else:raise ValueError('Unknown event '+e['event'])
    return dict(header=h,starts=starts,steps=steps,checkpoints=checkpoints,results=results,pending=pending,activeTraining=active,failed=failed)
