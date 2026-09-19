"""Reconstruct training, current answers, crossed judgments and frozen baselines."""
import math
from pathlib import Path

from research.answer_confidence_plan import validate_training,training_batches,TRAINING
from research.answer_confidence_baselines import fit,features
from research.answer_confidence_study import make_plan,source_hash,request_for,evaluation_rows
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require,strict_json
from research.natural_error_journal import close_tree


def validate_answer(event,request):
    require(event['event']=='answer' and event['id']==request['call']['id'],'Answer identity')
    require(event['status']=='ok' and event['errorType'] is None,'Failed answer')
    c=event['trace']['preAnswer']
    require(type(c['vocabularySize']) is int and c['vocabularySize']>=2,'Vocabulary')
    require(type(c['topTokenId']) is int and 0<=c['topTokenId']<c['vocabularySize'],'Top token')
    require(math.isfinite(c['entropyNats']) and 0<=c['entropyNats']<=math.log(c['vocabularySize'])+1e-10,'Entropy')
    require(math.isclose(c['normalizedEntropy'],c['entropyNats']/math.log(c['vocabularySize']),abs_tol=1e-10),'Entropy scale')
    # Full sequence arithmetic and the other confidence fields are checked here.
    features(dict(family='countA',level=8),event)
    config=event['metrics']['effectiveGeneration']
    for key in ('temperature','top_p','top_k','min_p','do_sample','max_new_tokens','repetition_penalty','renormalize_logits','use_cache'):
        require(config[key]==request['settings'][key],'Generation settings changed')


def validate_judgment(event,request,metadata):
    require(event['event']=='judgment' and event['id']==request['call']['id'] and event['status']=='ok','Judgment identity')
    require(type(event['seconds']) in (int,float) and math.isfinite(event['seconds']) and event['seconds']>=0,'Judgment duration')
    s=event['scores']
    require(s['inputHash']==digest(request['messages']),'Judgment input changed')
    for k in ('conditionalCorrect','candidateMass'):
        require(type(s[k]) in (int,float) and math.isfinite(s[k]) and 0<=s[k]<=1,'Native score')
    require(type(s['inputTokens']) is int and 0<s['inputTokens']<=1792,'Judgment length')
    require(s['codeTokenIds']==metadata['confidenceTokenIds'] and len(set(s['codeTokenIds']))==2,'Native codes')
    require(type(s['topTokenId']) is int and 0<=s['topTokenId']<metadata['vocabularySize'],'Top token id')
    require(s['topIsCode']==(s['topTokenId'] in s['codeTokenIds']) and s['sampled'] is False,'Score provenance')
    require(s['source']=='Recomputed full text prefix, raw lm_head logits at temperature 1','Score source')


def read_journal(path,*,check_fit=True):
    envelopes=[strict_json(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    require(bool(envelopes),'Empty journal');events=[];previous='0'*64
    for i,row in enumerate(envelopes):
        require(set(row)=={'sequence','previous','payload','sha256'} and row['sequence']==i and row['previous']==previous,'Journal order')
        require(row['sha256']==digest({k:row[k] for k in ('sequence','previous','payload')}),'Journal hash')
        previous=row['sha256'];events.append(row['payload'])
    h=events[0];p=make_plan()
    require(h['event']=='header' and h['plan']==p and h['planHash']==digest(p),'Fixed plan changed')
    require(h['sourceHash']==source_hash() and h['origin'] in ('transformers_gpu','synthetic_fixture'),'Source or origin')
    import json
    report=json.loads((Path(__file__).parents[1]/'artifacts/answer-confidence-preparation/report.json').read_text(encoding='utf-8'))
    require(h['trainingReport']==report,'Training parent changed');data=h['trainingData'];validate_training(data,report)
    answers={};judgments={};starts={};checkpoints={};steps=[];bundles={};active=None;step=0;pending=None;index=0;failed=False;cached_batches=None
    for e in events[1:]:
        require(not failed,'Events after failure')
        if e['event']=='failure':failed=True;continue
        if e['event']=='training_start':
            require(index==0 and pending is None and active is None and len(checkpoints)<6,'Training phase')
            active=p['trainingUnits'][len(checkpoints)];key=active['key'];require(e['key']==key,'Training order')
            require(type(e['initializationHash']) is str and len(e['initializationHash'])==64 and e['trainableParameters']>0,'Training initialization')
            pair=f'r{active["replication"]}-measured'
            if pair in starts:require(e['initializationHash']==starts[pair]['initializationHash'],'Unmatched initial weights')
            starts[key]=e;step=0;cached_batches=training_batches(data,active['replication'],active['arm'])
        elif e['event']=='training_step':
            require(active is not None and e['key']==active['key'] and e['step']==step+1<=144,'Training step')
            require(e['dataHash']==digest(cached_batches[step]),'Training examples changed')
            require(len(e['losses'])==8 and all(math.isfinite(x) and x>=0 for x in e['losses']),'Training loss')
            require(math.isfinite(e['gradientNorm']) and e['gradientNorm']>=0,'Gradient norm')
            require(len(e['inputTokens'])==8 and all(type(n) is int and 1<n<=TRAINING['maxTrainingTokens'] for n in e['inputTokens']),'Training lengths')
            require(type(e['seconds']) in (int,float) and math.isfinite(e['seconds']) and e['seconds']>=0,'Step duration')
            steps.append(e);step+=1
        elif e['event']=='training_complete':
            require(active is not None and e['key']==active['key'] and step==e['steps']==144,'Incomplete training')
            require(e['initializationHash']==starts[e['key']]['initializationHash'],'Changed initialization')
            require(len(e['sha256'])==64 and e['checkpoint']==Path(path).with_suffix('.'+e['key']+'.safetensors').name,'Checkpoint')
            require(math.isfinite(e['trainingSeconds']) and e['trainingSeconds']>=0,'Training duration')
            checkpoints[e['key']]=e;active=None;cached_batches=None
        elif e['event']=='baseline':
            require(index==p['calibrationCalls'] and pending is None and active is None and len(checkpoints)==6,'Baseline timing')
            rep=e['replication'];require(rep==len(bundles)<3,'Baseline order')
            if check_fit:close_tree(e['bundle'],fit(evaluation_rows(p,answers,judgments,'calibration'),rep),'Baseline reconstruction')
            bundles[rep]=e['bundle']
        elif e['event']=='request':
            require(len(checkpoints)==6 and active is None and pending is None and index<len(p['calls']),'Unfrozen or extra call')
            require(index<p['calibrationCalls'] or len(bundles)==3,'Test precedes frozen calibration')
            pending=request_for(p,index,answers,checkpoints);require(e==pending,'Request or response text changed')
        elif e['event'] in ('answer','judgment'):
            require(pending is not None,'Unpaired result');c=pending['call']
            if c['kind']=='answer':
                validate_answer(e,pending);answers[(c['task'],c['producer'])]=e
            else:
                validate_judgment(e,pending,h['metadata']);judgments[(c['task'],c['judge'],c['producer'])]=e
            pending=None;index+=1
        else:raise ValueError('Unknown event '+e['event'])
    complete=index==p['plannedCalls'] and pending is None and active is None and len(bundles)==3 and not failed
    return dict(header=h,answers=answers,judgments=judgments,starts=starts,steps=steps,checkpoints=checkpoints,bundles=bundles,
                pending=pending,activeTraining=active,failed=failed,complete=complete,recorded=index,chainEnd=previous,fitChecked=check_fit)
