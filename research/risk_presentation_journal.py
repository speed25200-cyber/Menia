"""Prospective no-retry collection and reconstruction of each factorial request."""
from pathlib import Path
import time

from research.cross_model_prediction import digest
from research.iphone_coupling_report import require,strict_json
from research.natural_error_journal import Writer
from research.native_choice_journal import validate_result
from research.risk_presentation import make_plan,source_hash,request_for


def collect(path,backend,limit=None):
    path=Path(path);require(not path.exists(),'Existing attempt; preserve without retry')
    require(limit is None or type(limit) is int and limit>=0,'Limit')
    plan=make_plan();writer=Writer(path)
    writer.write(dict(event='header',plan=plan,planHash=digest(plan),sourceHash=source_hash(),origin=backend.origin,metadata=backend.metadata),create=True)
    for i in range(plan['planned'] if limit is None else min(limit,plan['planned'])):
        request=request_for(plan,i);writer.write(request);started=time.perf_counter()
        result=dict(event='result',id=i,engine='llm',status='ok',text='',seconds=0.,metrics={},errorType=None)
        try:
            result['text'],result['metrics']=backend.generate(request)
            result['seconds']=time.perf_counter()-started;validate_result(result,request)
        except (Exception,KeyboardInterrupt) as error:
            result.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',text='',metrics={},seconds=time.perf_counter()-started,errorType=type(error).__name__)
            writer.write(result);raise
        writer.write(result)
        if (i+1)%24==0:print(f'{i+1}/{plan["planned"]} decisions recorded',flush=True)


def read_journal(path):
    rows=[strict_json(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    require(bool(rows),'Empty journal');previous='0'*64;events=[]
    for i,row in enumerate(rows):
        require(set(row)=={'sequence','previous','payload','sha256'} and row['sequence']==i and row['previous']==previous,'Journal order')
        require(row['sha256']==digest({k:row[k] for k in ('sequence','previous','payload')}),'Hash chain')
        previous=row['sha256'];events.append(row['payload'])
    header=events[0];plan=make_plan()
    require(header['event']=='header' and header['plan']==plan and header['planHash']==digest(plan),'Fixed plan')
    require(header['sourceHash']==source_hash() and header['origin'] in ('transformers_gpu','synthetic_fixture'),'Source/origin')
    results=[];pending=None;failed=False
    for event in events[1:]:
        require(not failed,'Events after failure')
        if event['event']=='request':
            require(pending is None and len(results)<plan['planned'],'Request order')
            pending=request_for(plan,len(results));require(event==pending,'Unexpected request')
        else:
            require(pending is not None,'Unpaired result');validate_result(event,pending)
            results.append(event);pending=None;failed=event['status']!='ok'
    return dict(header=header,results=results,pending=pending,failed=failed)
