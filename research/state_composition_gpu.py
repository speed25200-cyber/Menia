"""Counterbalanced warm-start training with all nine checkpoints frozen before evaluation."""
import json
import os
from pathlib import Path
import time

import torch
from torch import nn

from research import state_composition as study
from research.presence_specificity_gpu import verify_parent
from research.native_localization_gpu import install_adapters,adapter_state,load_adapter,forward,summarize_logits,append,file_hash
from research.localization_replication import trace_position


def encode(tokenizer,block,family,task,position,device,form='trained',mapping=0):
    body,spans=study.prompt(block,family,task,position,form,mapping)
    text=tokenizer.apply_chat_template([dict(role='user',content=body)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    if text.count(body)!=1:raise ValueError('Cannot uniquely locate user content')
    offset=text.index(body)
    inputs=tokenizer(text,return_tensors='pt',add_special_tokens=False,return_offsets_mapping=True,return_token_type_ids=False)
    offsets=inputs.pop('offset_mapping')[0].tolist()
    if len(offsets)>study.CONFIG['maxInputTokens']:raise ValueError('No truncation permitted')
    tokens=[[i for i,(a,b) in enumerate(offsets) if b>a and b>lo+offset and a<hi+offset] for lo,hi in spans]
    if any(not t for t in tokens) or len(set(sum(tokens,[])))!=sum(map(len,tokens)):raise ValueError('Invalid spans')
    return inputs.to(device),tokens,study.previous.parent.digest(body)


def evaluate_forward(model,inputs,spans,block,family,position):
    manipulation=dict(layer=study.CONFIG['layer'],strength=study.CONFIG['strength'],noiseSeed=block['noiseSeed'])
    return forward(model,inputs,spans,manipulation,trace_position('visible' if family=='visible' else 'strong',position))


def balanced_update(model,optimizer,parameters,block,arm,group,choices,inputs_for):
    optimizer.zero_grad(set_to_none=True);losses=[];traces=[]
    examples=study.training_examples(block,arm,group)
    for ex in examples:
        inputs,spans,_=inputs_for(block,ex['family'],ex['task'],ex['position'],'trained',ex['mapping'])
        logits,trace=evaluate_forward(model,inputs,spans,block,ex['family'],ex['position'])
        loss=nn.functional.cross_entropy(logits[None],torch.tensor([choices[ex['target']]],device=logits.device))
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        (loss*ex['weight']).backward();losses.append(float(loss.detach()));traces.append(trace)
    norm=torch.nn.utils.clip_grad_norm_(parameters,study.CONFIG['clipNorm'],error_if_nonfinite=True);optimizer.step()
    return dict(examples=examples,losses=losses,traces=traces,gradientNorm=float(norm))


def run(path,parent_path,*,resume=False):
    from safetensors.torch import load_file,save_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from research.cross_model_gpu import environment
    path=Path(path);parent_path=Path(parent_path)
    if path.exists() and not resume:raise ValueError('Existing journal requires --resume')
    frozen=verify_parent(parent_path)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    metadata=environment()
    metadata.update(origin='transformers_gpu',models={'A':study.MODEL},
        settings=dict(dtype='bfloat16',sampling=False,thinking=False,useCache=False,firstTokenOnly=True,
                      attention='sdpa',optimizer='AdamW',weightDecay=0.,warmStart='parent strong',**study.CONFIG))
    tokenizer=AutoTokenizer.from_pretrained(study.MODEL['id'],revision=study.MODEL['revision'],trust_remote_code=False)
    encoded=[tokenizer.encode(str(i),add_special_tokens=False) for i in range(4)]
    if any(len(x)!=1 for x in encoded) or len({x[0] for x in encoded})!=4:raise ValueError('Invalid digits')
    choices=[x[0] for x in encoded];metadata['choiceTokenIds']=choices;fixed=study.plan()
    if path.exists():
        h,rows,trained,pending,events=study.read_journal(path)
        if h['metadata']!=metadata:raise ValueError('Environment changed')
        for key,sha in trained.items():
            if file_hash(path.with_suffix('.'+key+'.safetensors'))!=sha:raise ValueError('Completed adapter changed')
        if len(rows)==len(fixed['evaluation']) and pending is None:
            print('Already complete; no training or inference repeated.',flush=True);return
        if pending:append(path,'interrupted_request',id=pending['request']['id'])
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        append(path,'header',plan=fixed,planHash=study.previous.parent.digest(fixed),sourceHash=study.source_hash(),
            metadata=metadata,parentJournalSHA256=study.previous.PARENT_JOURNAL_SHA256,parentCheckpoints=frozen)
        rows=[];trained={};events=[]
    model=AutoModelForCausalLM.from_pretrained(study.MODEL['id'],revision=study.MODEL['revision'],torch_dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True,trust_remote_code=False).eval()
    if model.config._commit_hash!=study.MODEL['revision'] or len(model.model.layers)!=36:raise ValueError('Unexpected architecture')
    modules=install_adapters(model,rank=study.CONFIG['rank']);parameters=[p for p in model.parameters() if p.requires_grad]
    blocks={b['id']:b for b in fixed['blocks']};cache={}
    def inputs_for(block,family,task,pos,form='trained',mapping=0):
        key=(block['id'],family,task,pos,form,mapping)
        if key not in cache:cache[key]=encode(tokenizer,block,family,task,pos,model.device,form,mapping)
        return cache[key]
    for unit in fixed['training']:
        key=unit['key'];arm=unit['arm']
        if key in trained:continue
        parent=parent_path.with_suffix('.'+unit['parentKey']+'.safetensors')
        if file_hash(parent)!=frozen[unit['parentKey']]:raise ValueError('Warm start changed')
        initial=load_file(str(parent));load_adapter(modules,initial)
        for m in modules.values():m.enabled=True
        if any(e['event']=='training_start' and e['unit']['key']==key for e in events):append(path,'training_restart',key=key)
        append(path,'training_start',unit=unit,parentHash=frozen[unit['parentKey']],trainableParameters=sum(p.numel() for p in parameters))
        optimizer=torch.optim.AdamW(parameters,lr=study.CONFIG['learningRate'],weight_decay=0.)
        for step,group in enumerate(unit['groups'],1):
            torch.cuda.synchronize();start=time.perf_counter()
            result=balanced_update(model,optimizer,parameters,blocks[group['block']],arm,group,choices,inputs_for)
            torch.cuda.synchronize();append(path,'training_step',key=key,step=step,group=group,seconds=time.perf_counter()-start,**result)
            if step%16==0:print(f'{key}: update {step}/{len(unit["groups"])}',flush=True)
        optimizer.zero_grad(set_to_none=True);del optimizer
        state=adapter_state(modules)
        if not any(not torch.equal(v,initial[k]) for k,v in state.items()):raise ValueError('No weights changed')
        checkpoint=path.with_suffix('.'+key+'.safetensors');save_file(state,str(checkpoint));trained[key]=file_hash(checkpoint)
        append(path,'training_complete',key=key,steps=len(unit['groups']),sha256=trained[key])
    model.requires_grad_(False);current=None
    for index,req in enumerate(fixed['evaluation'][len(rows):],len(rows)+1):
        key=f"r{req['replication']}-{req['arm']}"
        if key!=current:
            for m in modules.values():m.enabled=req['arm']!='base'
            if req['arm']=='parent':
                ck=parent_path.with_suffix(f".r{req['replication']}-strong.safetensors");sha=frozen[f"r{req['replication']}-strong"]
            elif req['arm']!='base':ck=path.with_suffix('.'+key+'.safetensors');sha=trained[key]
            else:ck=None;sha=None
            if ck is not None:
                if file_hash(ck)!=sha:raise ValueError('Evaluation checkpoint changed')
                load_adapter(modules,load_file(str(ck)))
            current=key
        b=blocks[req['block']];inputs,spans,ph=inputs_for(b,req['family'],req['task'],req['position'],req['format'],req['mapping'])
        append(path,'request',request=req,promptHash=ph,adapterHash=sha,inputTokens=int(inputs.input_ids.shape[-1]))
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            logits,trace=evaluate_forward(model,inputs,spans,b,req['family'],req['position'])
            result=summarize_logits(logits,choices)
        torch.cuda.synchronize();append(path,'result',id=req['id'],seconds=time.perf_counter()-start,intervention=trace,**result)
        if index%512==0:print(f'Evaluation {index}/{len(fixed["evaluation"])}',flush=True)
    verify_parent(parent_path)
    path.with_suffix('.summary.json').write_text(json.dumps(study.analyze(path),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('journal',type=Path);p.add_argument('--parent-journal',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:run(a.journal,a.parent_journal,resume=a.resume)
    except BaseException as exc:
        if a.journal.exists():append(a.journal,'error',errorType=type(exc).__name__,message=str(exc))
        raise
