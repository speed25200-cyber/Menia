"""Verify the numerical-control journal and recompute its recorded comparisons.

Cache tensors, repeated-generation tensors and reverse-order branch records
are not exported. Their declared equalities cannot be independently rerun here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from research.confidence_generation_numerics_gpu import plan, source_hash
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require


def compare_numbers(a,b):
    require(type(a)==type(b) or type(a) in (int,float) and type(b) in (int,float), 'Value type')
    if isinstance(a,dict):
        require(a.keys()==b.keys(), 'Comparison fields')
        for k in a: compare_numbers(a[k],b[k])
    elif isinstance(a,list):
        require(len(a)==len(b), 'Comparison length')
        for x,y in zip(a,b): compare_numbers(x,y)
    elif type(a) in (int,float): require(math.isclose(a,b,rel_tol=0.,abs_tol=1e-12), 'Comparison arithmetic')
    else: require(a==b, 'Comparison value')


def branch_comparison(a,b):
    require(a['candidateTokenIds']==b['candidateTokenIds'] and a['candidateMeanings']==b['candidateMeanings'], 'Code mapping')
    return dict(tokenIdsEqual=a['tokenIds']==b['tokenIds'],
        bothValidNative=a['validNativeResponse'] and b['validNativeResponse'],decisionEqual=a['decision']==b['decision'],
        maximumCandidateLogitDifference=max(abs(x-y) for x,y in zip(a['candidateLogits'],b['candidateLogits'])),
        conditionalProbabilityDifference=abs(a['conditionalPositive']-b['conditionalPositive']),
        candidateMassDifference=abs(a['candidateMass']-b['candidateMass']))


def validate_decode(value,config,tokenizer):
    codes=[tokenizer.encode(c,add_special_tokens=False)[0] for c in config['codes']]
    meanings=[config['negative'],config['positive']]; ids=value['tokenIds']; eos=value['eosTokenIds']
    require(value['candidateTokenIds']==codes and value['candidateMeanings']==meanings, 'Decoded branch mapping')
    require(type(ids)==list and 1<=len(ids)<=2 and all(type(i)==int and 0<=i<151936 for i in ids), 'Native token IDs')
    require(type(eos)==list and tokenizer.eos_token_id in eos, 'EOS IDs')
    valid=len(ids)==2 and ids[0] in codes and ids[1] in eos
    require(value['validNativeResponse']==valid, 'Native response validity')
    require(value['decision']==(meanings[codes.index(ids[0])] if valid else None), 'Native decision')
    require(value['firstTokenMeaning']==(meanings[codes.index(ids[0])] if ids[0] in codes else None), 'First token meaning')
    require(value['stoppedAtEOS']==(ids[-1] in eos) and
        value['reachedTokenLimit']==(len(ids)==2 and ids[-1] not in eos), 'EOS accounting')
    require(value['text']==tokenizer.decode(ids,skip_special_tokens=True), 'Decoded text')
    require(value['firstTopIsCode']==(ids[0] in codes), 'Top-code accounting')
    require(len(value['candidateLogits'])==2 and all(math.isfinite(x) for x in value['candidateLogits']), 'Candidate logits')
    d=value['candidateLogits'][1]-value['candidateLogits'][0]
    expected=1/(1+math.exp(-d)) if d>=0 else math.exp(d)/(1+math.exp(d))
    require(abs(value['conditionalPositive']-expected)<=1e-12, 'Binary probability')
    require(math.isfinite(value['candidateMass']) and 0<=value['candidateMass']<=1+1e-12, 'Candidate mass')


def verify(path, tokenizer):
    path=Path(path); events=[]; previous='0'*64
    for i,line in enumerate(path.read_text(encoding='utf-8').splitlines()):
        e=json.loads(line)
        require(set(e)=={'sequence','previous','payload','sha256'} and e['sequence']==i and e['previous']==previous, 'Journal sequence')
        require(e['sha256']==digest({k:e[k] for k in ('sequence','previous','payload')}), 'Journal hash')
        previous=e['sha256']; events.append(e['payload'])
    require(bool(events), 'Empty journal'); header=events[0]; p=plan()
    require(header['event']=='header' and header['plan']==p and header['planHash']==digest(p) and header['sourceHash']==source_hash(), 'Frozen numerical plan')
    require(header['metadata']['model']==p['model'] and header['metadata']['weightUpdates']==header['metadata']['interventions']==0, 'Model or update count')
    records=[]; pending=None; terminal=None; summary=None; failure=None
    for event in events[1:]:
        require(terminal is None, 'Events after terminal')
        if event['event']=='case_start':
            require(pending is None and len(records)<6 and event['case']==p['cases'][len(records)], 'Case order')
            pending=event['case']
        elif event['event']=='case_result':
            require(pending is not None and event['id']==pending['id'], 'Unpaired case result')
            trace=event['trace']; prompt=tokenizer.apply_chat_template(pending['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            prompt_ids=tokenizer.encode(prompt,add_special_tokens=False); generated=trace['generatedTokenIds']; prefix=prompt_ids+generated
            require(trace['promptTokenIds']==prompt_ids and trace['promptHash']==digest(prompt_ids) and
                trace['generatedHash']==digest(generated) and event['prefixHash']==digest(prefix), 'Raw token hashes')
            require(1<=len(generated)<=64 and generated[-1]==tokenizer.eos_token_id and trace['finalTokenId']==generated[-1], 'Completed raw generation')
            require(trace['generationForwardInputLengths']==[len(prompt_ids)]+[1]*(len(generated)-1) and
                trace['cacheTokensBeforeFinalToken']==len(prefix)-1 and trace['cacheTokensAfterFinalToken']==len(prefix), 'Trajectory accounting')
            require(event['metrics']['inputTokens']==len(prompt_ids) and event['metrics']['outputTokens']==len(generated), 'Generation counts')
            require(event['text']==tokenizer.decode(generated,skip_special_tokens=True).strip(), 'Generation text')
            require(event['replaySchedules']==dict(same_schedule_replay=[len(prompt_ids)]+[1]*len(generated),full_prefill_replay=[len(prefix)]), 'Replay schedule')
            for key,config in p['branches'].items():
                suffix='\n'+tokenizer.apply_chat_template([dict(role='user',content=config['query'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=prefix+tokenizer.encode(suffix,add_special_tokens=False)
                require(event['branchInputHashes'][key]==digest(ids), 'Exact branch token history')
                original=event['decoded']['actual_generation'][key]
                for name in p['histories']:
                    other=event['decoded'][name][key]; validate_decode(other,config,tokenizer)
                    if name!='actual_generation': compare_numbers(event['branchComparisons'][name][key],branch_comparison(original,other))
                full=event['fullSequenceDecoded'][key]; validate_decode(full,config,tokenizer)
                compare_numbers(event['branchComparisons']['full_sequence'][key],branch_comparison(original,full))
            require(set(event['checks'])==set(p['strictControls']) and all(type(v)==bool for v in event['checks'].values()), 'Strict-control fields')
            require(event['allStrictControlsPassed']==all(event['checks'].values()), 'Strict-control conjunction')
            require(event['checks']['same_schedule_branch_outputs_exact']==
                (event['decoded']['actual_generation']==event['decoded']['same_schedule_replay']), 'Same-schedule recorded outputs')
            for cache in (event['repeatedCache'],*event['cacheComparisons'].values()):
                layers=cache['layers']; require(len(layers)==72, '36 K/V cache layers')
                require([(v['layer'],v['kind']) for v in layers]==[(i,k) for i in range(36) for k in ('key','value')], 'Layer identities')
                require(all(type(v['exact'])==bool and all(math.isfinite(v[k]) and v[k]>=0 for k in ('maximumAbsoluteDifference','absoluteL2','referenceL2')) for v in layers), 'Finite cache diagnostics')
                require(cache['exact']==all(v['exact'] for v in layers) and
                    cache['maximumAbsoluteDifference']==max(v['maximumAbsoluteDifference'] for v in layers), 'Cache diagnostic aggregation')
            require(event['checks']['same_schedule_cache_exact']==event['cacheComparisons']['same_schedule_replay']['exact'], 'Same-schedule cache flag')
            records.append(event); pending=None
        elif event['event']=='failure': terminal='failed'; failure=event
        elif event['event']=='complete':
            require(pending is None and len(records)==6 and all(r['allStrictControlsPassed'] for r in records), 'Incomplete strict control')
            terminal='completed'; summary=event['summary']
        else: raise ValueError('Unknown numerical event')
    require(terminal is not None, 'Attempt still incomplete')
    if summary is not None:
        expected=dict(schema='menia-generation-numerics-summary-v1',planHash=digest(p),sourceHash=source_hash(),
            cases=6,generations=12,branchDecodes=60,allStrictControlsPassed=True,
            fullPrefillMaximumCacheDifference=max(c['cacheComparisons']['full_prefill_replay']['maximumAbsoluteDifference'] for c in records),
            fullPrefillNativeTokenAgreements=sum(c['branchComparisons']['full_prefill_replay'][k]['tokenIdsEqual'] for c in records for k in p['branches']),
            fullSequenceNativeTokenAgreements=sum(c['branchComparisons']['full_sequence'][k]['tokenIdsEqual'] for c in records for k in p['branches']),
            actualValidNativeResponses=sum(c['decoded']['actual_generation'][k]['validNativeResponse'] for c in records for k in p['branches']),
            branchComparisonsPerReference=12,scope=p['scope'])
        compare_numbers(summary,expected)
        compare_numbers(json.loads(path.with_suffix('.summary.json').read_text(encoding='utf-8')),expected)
    return dict(schema='menia-generation-numerics-journal-audit-v1',verifiedJournal=True,status=terminal,
        journalSHA256=hashlib.sha256(path.read_bytes()).hexdigest(),chainEnd=previous,
        planHash=digest(p),sourceHash=source_hash(),recordedCases=len(records),failure=failure,summary=summary,
        auditorSourceHash=digest(Path(__file__).read_text(encoding='utf-8')),
        rawPromptAndBranchIdsChecked=True,nativeDecisionsAndBinaryProbabilitiesRecomputed=True,
        recordedBranchComparisonsRecomputed=True,cacheTensorComparisonsIndependentlyRecomputed=False,
        repeatedGenerationIndependentlyRecomputed=False,reverseBranchOrderIndependentlyRecomputed=False,
        scope='Journal identities, raw token histories and recorded numerical comparisons audited. Cache tensors and repeated/reverse-order records are not exported; their equality declarations cannot be independently reconstructed from this archive.')


if __name__=='__main__':
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('journal',type=Path)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    spec=plan()['model']; tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
    report=verify(args.journal,tokenizer)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],recordedCases=report['recordedCases'],summary=report['summary'],failure=report['failure'])))
