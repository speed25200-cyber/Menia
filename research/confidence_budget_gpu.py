"""Fixed-capacity continuation with no outcome-driven stopping or checkpoint choice."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from research import confidence_budget_diagnostic as study
from research.answer_confidence_gpu import assess_answer, encode_confidence_training
from research.answer_confidence_journal import validate_judgment
from research.confidence_ranking_gpu import train_paired_step
from research.confidence_ranking_study import source_hash as parent_source_hash
from research.cross_model_gpu import environment
from research.native_localization_gpu import install_adapters, adapter_state, load_adapter, file_hash
from research.natural_error_journal import Writer


def run(path,parent):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file, save_file
    path = Path(path); parent = Path(parent)
    if path.exists(): raise ValueError('Existing attempt; no automatic restart')
    plan = study.make_plan(); rows = study.load_rows(); by_id = {r['sourceId']:r for r in rows}; schedule = study.calls(rows,plan)
    prepared = json.loads((study.PREPARATION/'design.json').read_text(encoding='utf-8'))
    if (plan != prepared['plan'] or study.digest(plan) != prepared['planHash'] or
        study.source_hash() != prepared['sourceHash'] or study.digest(schedule) != prepared['callPlanHash']):
        raise ValueError('Frozen preparation changed')
    if parent_source_hash() != study.fit.PARENT_SOURCE_HASH: raise ValueError('Parent science changed')
    for unit in plan['initial']:
        file = parent/unit['checkpoint']
        if file.stat().st_size != unit['bytes'] or file_hash(file) != unit['sha256']: raise ValueError('Initial checkpoint identity')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    metadata = environment(); spec = plan['model']; writer = Writer(path)
    tokenizer = AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],torch_dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True,trust_remote_code=False).eval()
    if model.config._commit_hash != spec['revision']: raise ValueError('Model revision')
    eos = model.generation_config.eos_token_id; eos = eos[0] if isinstance(eos,list) else eos
    if eos != tokenizer.eos_token_id: raise ValueError('EOS mismatch')
    encoded_codes = [tokenizer.encode(s,add_special_tokens=False) for s in ('0','1')]
    if any(len(c) != 1 for c in encoded_codes) or encoded_codes[0] == encoded_codes[1]: raise ValueError('Code tokens')
    codes = [c[0] for c in encoded_codes]
    modules = install_adapters(model,rank=plan['training']['rank']); parameters = [p for m in modules.values() for p in (m.a,m.b)]
    if any(m.scale != plan['training']['scale'] for m in modules.values()): raise ValueError('Scale changed')
    adapter_ids = {id(p) for p in parameters}
    base_versions = {n:(id(p),p._version) for n,p in model.named_parameters() if id(p) not in adapter_ids}
    def unchanged_base():
        return base_versions == {n:(id(p),p._version) for n,p in model.named_parameters() if id(p) not in adapter_ids}
    metadata.update(model=spec,newLLMBaseWeightUpdates=0,adapterUpdatesPlanned=1296,
        confidenceTokenIds=codes,vocabularySize=model.config.vocab_size,eosTokenId=eos)
    writer.write(dict(event='header',plan=plan,planHash=study.digest(plan),sourceHash=study.source_hash(),
        callPlanHash=study.digest(schedule),origin='transformers_gpu',metadata=metadata),create=True)
    max_baseline_delta = 0.; checkpoints = {}; collection_complete = False

    def assess_unit(unit,file,expected_hash):
        nonlocal max_baseline_delta
        if file_hash(file) != expected_hash: raise ValueError('Checkpoint changed')
        state = load_file(str(file)); load_adapter(modules,state); model.requires_grad_(False)
        for m in modules.values(): m.enabled = True
        writer.write(dict(event='unit_start',key=unit['key'],sha256=expected_hash))
        for call in (c for c in schedule if c['key'] == unit['key']):
            request = study.request_for(call,by_id); writer.write(request)
            tick = time.perf_counter(); messages = request['messages']
            scores = assess_answer(model,tokenizer,messages[1]['content'],messages[2]['content'],max_input_tokens=1792)
            event = dict(event='judgment',id=call['id'],status='ok',seconds=time.perf_counter()-tick,scores=scores)
            validate_judgment(event,request,metadata)
            writer.write(event)
            if unit['epoch'] == 2:
                ref = by_id[call['sourceId']]['baseline']
                for field in ('conditionalCorrect','candidateMass'):
                    max_baseline_delta = max(max_baseline_delta,abs(scores[field]-ref[field]))
                if max_baseline_delta > study.BASELINE_TOLERANCE: raise ValueError('Baseline scores changed; do not train')
            if (call['id']+1)%192 == 0: print(f'{call["id"]+1}/8640 assessments recorded',flush=True)
        actual = adapter_state(modules)
        matches = actual.keys() == state.keys() and all(torch.equal(actual[k],v) for k,v in state.items())
        if not matches or not unchanged_base() or file_hash(file) != expected_hash: raise ValueError('Assessment changed state')
        writer.write(dict(event='unit_complete',key=unit['key'],calls=960,actualAdapterMatchesFile=True,baseParameterVersionsUnchanged=True))

    try:
        for unit in plan['units'][:3]:
            original = plan['initial'][unit['replication']]
            assess_unit(unit,parent/original['checkpoint'],original['sha256'])
        writer.write(dict(event='baseline_complete',maxAbsoluteDifference=max_baseline_delta))
        print('All baseline scores reproduced; beginning fixed continuation.',flush=True)
        for rep,original in enumerate(plan['initial']):
            file = parent/original['checkpoint']
            if file_hash(file) != original['sha256']: raise ValueError('Original weights changed')
            initial = load_file(str(file)); load_adapter(modules,initial)
            model.requires_grad_(False)
            for p in parameters: p.requires_grad_(True)
            optimizer = torch.optim.AdamW(parameters,lr=study.CONFIG['learningRate'],betas=tuple(study.CONFIG['betas']),
                eps=study.CONFIG['epsilon'],weight_decay=study.CONFIG['weightDecay'],foreach=False)
            writer.write(dict(event='training_start',replication=rep,initialHash=original['sha256'],optimizerReset=True,
                trainableParameters=sum(p.numel() for p in parameters)))
            cache = {}; batches = study.training_batches(rep)
            for step,batch in enumerate(batches,1):
                if study.digest(batch) != plan['batchHashes'][str(rep)][step-1]: raise ValueError('Schedule changed')
                pairs = []
                for pair in batch:
                    items = []
                    for side in ('left','right'):
                        example = pair[side]
                        if example['split'] != 'train' or example['replication'] != rep: raise ValueError('Training leakage')
                        source_id = example['sourceId']
                        if source_id not in cache:
                            cache[source_id] = encode_confidence_training(tokenizer,example,model.device,max_input_tokens=1792)
                        items.append(cache[source_id])
                    pairs.append(items)
                torch.cuda.synchronize(); tick = time.perf_counter()
                values = train_paired_step(model,optimizer,parameters,pairs,eos,codes,'rank')
                torch.cuda.synchronize(); values['seconds'] = time.perf_counter()-tick
                writer.write(dict(event='training_step',replication=rep,step=step,dataHash=study.digest(batch),**values))
                if step%24 == 0: print(f'r{rep}: continuation {step}/432',flush=True)
                if step in (144,432):
                    key = f'r{rep}-e{2+step//72}'; snapshot = path.with_suffix('.'+key+'.safetensors')
                    state = adapter_state(modules)
                    if not unchanged_base() or not any(not torch.equal(state[k],v) for k,v in initial.items()): raise ValueError('Unchanged adapter or changed base')
                    save_file(state,str(snapshot))
                    event = dict(event='checkpoint',key=key,step=step,checkpoint=snapshot.name,
                        sha256=file_hash(snapshot),bytes=snapshot.stat().st_size,baseParameterVersionsUnchanged=True)
                    writer.write(event); checkpoints[key] = event
            optimizer.zero_grad(set_to_none=True); del optimizer; del cache
            writer.write(dict(event='training_complete',replication=rep,steps=432))
        print('All six continuation snapshots frozen before post-training assessment.',flush=True)
        for unit in plan['units'][3:]:
            checkpoint = checkpoints[unit['key']]
            assess_unit(unit,path.parent/checkpoint['checkpoint'],checkpoint['sha256'])
        for unit in plan['initial']:
            if file_hash(parent/unit['checkpoint']) != unit['sha256']: raise ValueError('Original checkpoint altered')
        writer.write(dict(event='complete',calls=8640,steps=1296)); collection_complete = True
        report = study.summarize(study.read_journal(path))
        with path.with_suffix('.summary.json').open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    except BaseException as error:
        if not collection_complete: writer.write(dict(event='failure',errorType=type(error).__name__))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path); parser.add_argument('parent',type=Path)
    args = parser.parse_args(); run(args.journal,args.parent)
