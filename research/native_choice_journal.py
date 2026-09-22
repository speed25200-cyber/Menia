"""Append-only native choices followed by actual selected actions, with no retries."""
import hashlib
import math
from pathlib import Path
import time

from research.cross_model_prediction import digest, grade, reference
from research.iphone_coupling_report import require, strict_json
from research.natural_error_journal import Writer, SOURCE_FILES as PREVIOUS_SOURCES
from research.native_choice_plan import CALIBRATION_PATH, calibration, make_plan, request_for

FILES = PREVIOUS_SOURCES + ('native_choice_plan.py','native_choice_journal.py','native_choice_analysis.py','native_choice_gpu.py')


def source_hash():
    source = {name:Path(__file__).with_name(name).read_text(encoding='utf-8') for name in FILES}
    source['calibration.json'] = CALIBRATION_PATH.read_text(encoding='utf-8')
    return digest(source)


def verify_calibration_parent(path):
    raw = Path(path).read_bytes()
    cal = calibration()
    require(hashlib.sha256(raw).hexdigest() == cal['parentJournalSHA256'], 'Calibration parent bytes changed')
    events = [strict_json(line)['payload'] for line in raw.decode('utf-8').splitlines()]
    tasks = {e['task']['id']:e['task'] for e in events if e['event']=='request'}
    counts = {key:dict(n=0,correct=0) for key in cal['counts']}
    for e in events:
        if e['event']=='result' and tasks[e['id']]['split']=='train':
            task = tasks[e['id']]
            counts[f"{task['family']}/{task['level']}"]['n'] += 1
            counts[f"{task['family']}/{task['level']}"]['correct'] += int(grade(e,task))
    require(all(counts[k] == {f:cal['counts'][k][f] for f in ('n','correct')} for k in counts), 'Calibration counts differ')
    return cal['parentJournalSHA256']


def exact_tool(value):
    # Actual selected tool: deliberately separate arithmetic from reference().
    if value['family'] == 'countA':
        return sum(1 for letter in value['letters'] if letter == 'A')
    answer = 0
    for i,number in enumerate(value['operands']):
        answer += number if i % 2 == 0 else -number
    return answer


def validate_result(result, request):
    require(set(result) == {'event','id','status','engine','text','seconds','metrics','errorType'}, 'Result fields')
    require(result['event']=='result' and result['id']==request['call']['id'] and result['engine']==request['engine'], 'Result identity')
    require(type(result['seconds']) in (int,float) and math.isfinite(result['seconds']) and result['seconds'] >= 0, 'Elapsed time')
    require(result['status'] in ('ok','error','interrupted') and type(result['text']) is str, 'Result status/text')
    if result['status'] != 'ok':
        require(result['text']=='' and result['metrics']=={} and type(result['errorType']) is str, 'Failed result')
        return
    require(result['errorType'] is None, 'Success error field')
    metrics = result['metrics']
    if result['engine'] == 'exact_tool':
        require(result['text'] == str(reference(request['toolInput'])) and metrics == dict(tool='menia_exact_arithmetic_v1',calls=1), 'Tool receipt')
    elif result['engine'] == 'invalid_choice':
        require(result['text']=='' and result['seconds']==0 and metrics=={}, 'Invalid action must not execute')
    else:
        settings = request['settings']
        require(type(metrics['inputTokens']) is int and 1 <= metrics['inputTokens'] <= settings['max_input_tokens'], 'Input tokens')
        require(type(metrics['outputTokens']) is int and 1 <= metrics['outputTokens'] <= settings['max_new_tokens'], 'Output tokens')
        require(metrics['reachedTokenLimit'] == (metrics['outputTokens']==settings['max_new_tokens']), 'Limit flag')
        for key in ('max_new_tokens','temperature','top_p','top_k','min_p','do_sample','repetition_penalty','renormalize_logits','use_cache'):
            require(metrics['effectiveGeneration'][key] == settings[key], 'Generation differs: '+key)


def read_journal(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    require(bool(lines),'Empty journal')
    previous, events = '0'*64, []
    for i,line in enumerate(lines):
        event = strict_json(line)
        require(set(event)=={'sequence','previous','payload','sha256'} and event['sequence']==i and event['previous']==previous, 'Envelope order')
        require(event['sha256']==digest({k:event[k] for k in ('sequence','previous','payload')}),'Chain hash')
        previous = event['sha256']
        events.append(event['payload'])
    header, plan = events[0], make_plan()
    require(header['event']=='header' and header['plan']==plan and header['planHash']==digest(plan), 'Fixed plan')
    require(header['sourceHash']==source_hash() and header['origin'] in ('transformers_gpu','synthetic_fixture'), 'Source/origin')
    results, requests, pending, failed = [], [], None, False
    for event in events[1:]:
        require(not failed,'Events after failure')
        if event['event']=='request':
            require(pending is None and len(results)<len(plan['calls']), 'Request order')
            pending = request_for(plan,len(results),results)
            require(event==pending, 'Request differs from available information')
        else:
            require(pending is not None, 'Result without request')
            validate_result(event,pending)
            results.append(event)
            requests.append(pending)
            pending = None
            failed = event['status']!='ok'
    return dict(header=header,requests=requests,results=results,pending=pending,failed=failed,chainEnd=previous)


def collect(path, backend, *, limit=None):
    require(limit is None or type(limit) is int and limit>=0,'Invalid limit')
    path = Path(path)
    require(not path.exists(),'Attempt already exists; preserve and audit, no automatic retries')
    plan = make_plan()
    writer = Writer(path)
    writer.write(dict(event='header',plan=plan,planHash=digest(plan),sourceHash=source_hash(),origin=backend.origin,metadata=backend.metadata),create=True)
    results = []
    for index in range(len(plan['calls'])):
        if limit is not None and index>=limit:
            break
        request = request_for(plan,index,results)
        writer.write(request)
        started = time.perf_counter()
        result = dict(event='result',id=index,status='ok',engine=request['engine'],text='',seconds=0.,metrics={},errorType=None)
        try:
            if request['engine']=='llm':
                result['text'], result['metrics'] = backend.generate(request)
                result['seconds'] = time.perf_counter()-started
            elif request['engine']=='exact_tool':
                result['text'] = str(exact_tool(request['toolInput']))
                result['seconds'] = time.perf_counter()-started
                result['metrics'] = dict(tool='menia_exact_arithmetic_v1',calls=1)
            validate_result(result,request)
        except (Exception,KeyboardInterrupt) as error:
            result.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'error',
                          text='',seconds=time.perf_counter()-started,metrics={},errorType=type(error).__name__)
            writer.write(result)
            raise
        writer.write(result)
        results.append(result)
        if (index+1)%25==0 or index+1==len(plan['calls']):
            print(f'{index+1}/{len(plan["calls"])} slots recorded',flush=True)
    return len(results)
