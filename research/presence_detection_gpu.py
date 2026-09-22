"""Three fixed replications of binary presence detection; all adapters frozen before evaluation."""
import json
import os
from pathlib import Path
import time

import torch
from torch import nn

from research import presence_detection as study
from research.localization_replication import trace_position
from research.localization_replication_gpu import initialize
from research.native_localization_gpu import (
    install_adapters, adapter_state, load_adapter, forward, summarize_logits, append, file_hash,
)


def encode(tokenizer, block, family, task, position, device, layout='canonical', *, marker=None):
    body, spans = study.prompt(block, family, task, position, layout, marker=marker)
    text = tokenizer.apply_chat_template([dict(role='user',content=body)], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    if text.count(body)!=1: raise ValueError('User text not uniquely located')
    offset = text.index(body)
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False,
                       return_offsets_mapping=True, return_token_type_ids=False)
    offsets = inputs.pop('offset_mapping')[0].tolist()
    if len(offsets)>study.CONFIG['maxInputTokens']: raise ValueError('Prefix too long; no truncation')
    tokens = [[i for i,(a,b) in enumerate(offsets) if b>a and b>lo+offset and a<hi+offset] for lo,hi in spans]
    if any(not t for t in tokens) or len(set(sum(tokens,[])))!=sum(map(len,tokens)):
        raise ValueError('Invalid sentence spans')
    return inputs.to(device), tokens, study.digest(body)


def evaluate_forward(model, inputs, spans, block, family, position, condition='trained'):
    manipulation = dict(study.CONDITIONS[condition], noiseSeed=block['noiseSeed'])
    return forward(model, inputs, spans, manipulation, trace_position(family, position))


def balanced_update(model, optimizer, parameters, block, arm, choices, inputs_for):
    optimizer.zero_grad(set_to_none=True)
    primary, reading, traces = [], [], []
    for pos in range(3):
        inputs, spans, _ = inputs_for(block, arm, 'primary', pos)
        logits, stats = evaluate_forward(model, inputs, spans, block, arm, pos)
        label = study.training_label(block, arm, pos)
        loss = nn.functional.cross_entropy(logits[None], torch.tensor([choices[label]],device=logits.device))
        if not torch.isfinite(loss): raise ValueError('Nonfinite primary loss')
        (loss*study.training_weight(label)).backward()
        primary.append(float(loss.detach())); traces.append(stats)
    for identity in (0, 1):
        inputs, spans, _ = inputs_for(block, arm, 'marker', 0, marker=identity)
        logits, stats = evaluate_forward(model, inputs, spans, block, arm, 0)
        loss = nn.functional.cross_entropy(logits[None],torch.tensor([choices[identity+1]],device=logits.device))
        if not torch.isfinite(loss): raise ValueError('Nonfinite reading loss')
        (loss*(study.CONFIG['readingWeight']/2)).backward()
        reading.append(float(loss.detach())); traces.append(stats)
    norm = torch.nn.utils.clip_grad_norm_(parameters, study.CONFIG['clipNorm'], error_if_nonfinite=True)
    optimizer.step()
    return dict(primaryLosses=primary, readingLosses=reading, gradientNorm=float(norm), traces=traces)


def run(path, *, resume=False):
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from research.cross_model_gpu import environment
    path = Path(path)
    if path.exists() and not resume: raise ValueError('Existing journal requires --resume')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    metadata = environment()
    metadata.update(origin='transformers_gpu', models={'A':study.MODEL},
        settings=dict(dtype='bfloat16',sampling=False,thinking=False,useCache=False,firstTokenOnly=True,
                      attention='sdpa',optimizer='AdamW',weightDecay=0.,initializations=list(study.INITIALIZATIONS),
                      conditions=study.CONDITIONS,**study.CONFIG))
    tokenizer = AutoTokenizer.from_pretrained(study.MODEL['id'],revision=study.MODEL['revision'],trust_remote_code=False)
    encoded = [tokenizer.encode(str(i),add_special_tokens=False) for i in range(3)]
    if any(len(x)!=1 for x in encoded) or len({x[0] for x in encoded})!=3: raise ValueError('Invalid digits')
    choices = [x[0] for x in encoded]; metadata['choiceTokenIds']=choices
    fixed = study.plan()
    if path.exists():
        header, rows, trained, pending, events = study.read_journal(path)
        if metadata!=header['metadata']: raise ValueError('Resume environment changed')
        for key, sha in trained.items():
            if file_hash(path.with_suffix('.'+key+'.safetensors'))!=sha: raise ValueError('Completed adapter changed')
        if len(rows)==len(fixed['evaluation']) and pending is None:
            study.analyze(path); print('Tentative complète ; aucune nouvelle inférence.',flush=True); return
        if pending:
            append(path,'interrupted_request',id=pending['request']['id'],reason='Forward not durably recorded; deterministic retry')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        append(path,'header',plan=fixed,planHash=study.digest(fixed),sourceHash=study.source_hash(),metadata=metadata)
        rows=[]; trained={}; events=[]
    model = AutoModelForCausalLM.from_pretrained(study.MODEL['id'],revision=study.MODEL['revision'],
        torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True,trust_remote_code=False).eval()
    if model.config._commit_hash!=study.MODEL['revision'] or len(model.model.layers)!=36:
        raise ValueError('Unexpected Qwen revision/architecture')
    modules = install_adapters(model,rank=study.CONFIG['rank'])
    parameters = [p for p in model.parameters() if p.requires_grad]
    blocks = {b['id']:b for b in fixed['blocks']}; cache = {}
    def inputs_for(block,family,task,position,layout='canonical',*,marker=None):
        key = (block['id'],family,task,position,layout,marker)
        if key not in cache: cache[key]=encode(tokenizer,block,family,task,position,model.device,layout,marker=marker)
        return cache[key]
    for unit in fixed['training']:
        key = unit['key']; arm = unit['arm']
        if key in trained: continue
        # Every arm of a replication, including after a resume, restarts from the same seeded state.
        initial=initialize(modules,unit['initializationSeed'])
        if any(e['event']=='training_start' and e['unit']['key']==key for e in events):
            append(path,'training_restart',key=key,reason='Restart incomplete adapter from fixed initial state')
        append(path,'training_start',unit=unit,trainableParameters=sum(p.numel() for p in parameters))
        optimizer = torch.optim.AdamW(parameters,lr=study.CONFIG['learningRate'],weight_decay=0.)
        for step,group in enumerate(unit['groups'],1):
            torch.cuda.synchronize(); start=time.perf_counter()
            result=balanced_update(model,optimizer,parameters,blocks[group['block']],arm,choices,inputs_for)
            torch.cuda.synchronize()
            append(path,'training_step',key=key,step=step,group=group,seconds=time.perf_counter()-start,**result)
            if step%16==0: print(f'{key} : mise à jour {step}/{len(unit["groups"])}',flush=True)
        optimizer.zero_grad(set_to_none=True); del optimizer
        state=adapter_state(modules)
        if not any(not torch.equal(v,initial[k]) for k,v in state.items()): raise ValueError('No adapter weight changed')
        checkpoint=path.with_suffix('.'+key+'.safetensors'); save_file(state,str(checkpoint)); trained[key]=file_hash(checkpoint)
        append(path,'training_complete',key=key,steps=len(unit['groups']),sha256=trained[key])
    current=None
    for index,request in enumerate(fixed['evaluation'][len(rows):],len(rows)+1):
        arm=request['arm']; key=f"r{request['replication']}-{arm}"
        if key!=current:
            for module in modules.values(): module.enabled=arm!='base'
            if arm!='base':
                checkpoint=path.with_suffix('.'+key+'.safetensors')
                if file_hash(checkpoint)!=trained[key]: raise ValueError('Adapter changed before evaluation')
                load_adapter(modules,load_file(str(checkpoint)))
            current=key
        b=blocks[request['block']]
        inputs,spans,ph=inputs_for(b,request['family'],request['task'],request['position'],request['layout'])
        append(path,'request',request=request,promptHash=ph,adapterHash=trained.get(key),inputTokens=int(inputs.input_ids.shape[-1]))
        torch.cuda.synchronize(); start=time.perf_counter()
        with torch.inference_mode():
            logits,stats=evaluate_forward(model,inputs,spans,b,request['family'],request['position'],request['condition'])
            result=summarize_logits(logits,choices)
        torch.cuda.synchronize()
        append(path,'result',id=request['id'],seconds=time.perf_counter()-start,intervention=stats,**result)
        if index%256==0: print(f'Évaluation {index}/{len(fixed["evaluation"])}',flush=True)
    path.with_suffix('.summary.json').write_text(json.dumps(study.analyze(path),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('journal',type=Path); parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    try: run(args.journal,resume=args.resume)
    except BaseException as exc:
        if args.journal.exists(): append(args.journal,'error',errorType=type(exc).__name__,message=str(exc))
        raise
