"""Train a common prefix, fork optimizer states, freeze all arms, then evaluate."""
import copy
import json
import os
from pathlib import Path
import time

import torch

from research import optimizer_memory as study
from research.optimizer_memory_ops import fingerprint,fork_optimizer,moment_summary,zero_gradient_step
from research.state_composition_gpu import encode,evaluate_forward,balanced_update
from research.composition_diagnostic_gpu import verify_sources
from research.native_localization_gpu import install_adapters,adapter_state,load_adapter,summarize_logits,append,file_hash


def checkpoint(path,key):return Path(path).with_suffix('.'+key+'.safetensors')


def cpu_optimizer_state(optimizer):
    result=copy.deepcopy(optimizer.state_dict())
    for state in result['state'].values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor):state[key]=value.detach().cpu().clone()
    return result


def write_summary(path):
    Path(path).with_suffix('.summary.json').write_text(json.dumps(study.analyze(path),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run(path,parent,composition,*,resume=False):
    from safetensors.torch import load_file,save_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from research.cross_model_gpu import environment
    path,parent,composition=map(Path,(path,parent,composition))
    if path.exists() and not resume:raise ValueError('Existing journal requires --resume')
    frozen=verify_sources(parent,composition)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    metadata=environment();metadata.update(origin='transformers_gpu',models={'A':study.MODEL},
        settings=dict(dtype='bfloat16',sampling=False,thinking=False,useCache=False,firstTokenOnly=True,
            attention='sdpa',optimizer='AdamW',weightDecay=0.,learningRate=study.previous.CONFIG['learningRate'],**study.CONFIG))
    tok=AutoTokenizer.from_pretrained(study.MODEL['id'],revision=study.MODEL['revision'],trust_remote_code=False)
    ids=[tok.encode(str(i),add_special_tokens=False) for i in range(4)]
    if any(len(x)!=1 for x in ids) or len({x[0] for x in ids})!=4:raise ValueError('Invalid digits')
    choices=[x[0] for x in ids];metadata['choiceTokenIds']=choices;fixed=study.plan()
    for rep in range(study.CONFIG['replications']):
        if fixed['carryReferences'][f'r{rep}-carry']!=frozen[f'r{rep}-composed']:
            raise ValueError('Carry reference changed')
    if path.exists():
        h,rows,done,pending,events=study.read_journal(path)
        if h['metadata']!=metadata:raise ValueError('Environment changed')
        for key,e in done.items():
            if file_hash(checkpoint(path,key))!=e['sha256']:raise ValueError('Completed weights changed')
            if key.endswith('-prefix') and file_hash(path.with_suffix('.'+key+'.optimizer.pt'))!=e['optimizerFileHash']:
                raise ValueError('Saved fork state changed')
        if len(rows)==len(fixed['evaluation']) and pending is None:
            if not path.with_suffix('.summary.json').exists():write_summary(path)
            print('Already complete; no training or inference repeated.',flush=True);return
        if pending:append(path,'interrupted_request',id=pending['request']['id'])
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        append(path,'header',plan=fixed,planHash=study.digest(fixed),sourceHash=study.source_hash(),metadata=metadata)
        rows=[];done={};events=[]
    model=AutoModelForCausalLM.from_pretrained(study.MODEL['id'],revision=study.MODEL['revision'],torch_dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True,trust_remote_code=False).eval()
    if model.config._commit_hash!=study.MODEL['revision'] or len(model.model.layers)!=36:raise ValueError('Unexpected architecture')
    modules=install_adapters(model,rank=study.CONFIG['rank']);parameters=[p for p in model.parameters() if p.requires_grad]
    blocks={b['id']:b for b in fixed['blocks']};cache={}
    def inputs_for(block,family,task,pos,form='trained',mapping=0):
        key=(block['id'],family,task,pos,form,mapping)
        if key not in cache:cache[key]=encode(tok,block,family,task,pos,model.device,form,mapping)
        return cache[key]
    for unit in fixed['training']:
        key=unit['key'];arm=unit['arm'];rep=unit['replication']
        if key in done:continue
        if arm=='prefix':
            initial=parent.with_suffix('.'+unit['parentKey']+'.safetensors')
            initial_hash=fixed['parentCheckpoints'][unit['parentKey']]
        else:
            prefix=done[f'r{rep}-prefix'];initial=checkpoint(path,f'r{rep}-prefix');initial_hash=prefix['sha256']
        if file_hash(initial)!=initial_hash:raise ValueError('Initial weights changed')
        load_adapter(modules,load_file(str(initial)))
        for module in modules.values():module.enabled=True
        if arm=='prefix':
            optimizer=torch.optim.AdamW(parameters,lr=study.previous.CONFIG['learningRate'],weight_decay=0.);fork=None
        else:
            opt_path=path.with_suffix(f'.r{rep}-prefix.optimizer.pt')
            if file_hash(opt_path)!=prefix['optimizerFileHash']:raise ValueError('Fork file changed')
            state=torch.load(opt_path,map_location='cpu',weights_only=True)
            if fingerprint(state)!=prefix['optimizerStateHash']:raise ValueError('Fork state changed')
            optimizer,fork=fork_optimizer(parameters,state,reset_first=arm=='reset_m');del state
            if fork['before']!=prefix['moments']:raise ValueError('Fork moments changed')
        if any(e['event']=='training_start' and e['unit']['key']==key for e in events):append(path,'training_restart',key=key)
        append(path,'training_start',unit=unit,initialHash=initial_hash,fork=fork,trainableParameters=sum(p.numel() for p in parameters))
        for step,group in enumerate(unit['groups'],1):
            torch.cuda.synchronize();start=time.perf_counter()
            if arm=='zero_grad':
                zero_gradient_step(optimizer,parameters);result=dict(mode='dense_zero')
            else:
                result=balanced_update(model,optimizer,parameters,blocks[group['block']],'composed',group,choices,inputs_for)
                result['mode']='objective'
            torch.cuda.synchronize()
            append(path,'training_step',key=key,step=step,group=group,seconds=time.perf_counter()-start,**result)
            if step%8==0 or step==len(unit['groups']):print(f'{key}: update {step}/{len(unit["groups"])}',flush=True)
        optimizer.zero_grad(set_to_none=True)
        state=cpu_optimizer_state(optimizer);moments=moment_summary(optimizer)
        destination=checkpoint(path,key);save_file(adapter_state(modules),str(destination));sha=file_hash(destination)
        if arm=='carry' and sha!=fixed['carryReferences'][key]:
            raise ValueError(f'Exact reproduction failed for {key}: got {sha}, expected {fixed["carryReferences"][key]}. No evaluation permitted.')
        completion=dict(key=key,steps=len(unit['groups']),sha256=sha,optimizerStateHash=fingerprint(state),moments=moments)
        if arm=='prefix':
            opt_path=path.with_suffix('.'+key+'.optimizer.pt');torch.save(state,opt_path)
            completion['optimizerFileHash']=file_hash(opt_path)
        append(path,'training_complete',**completion);done[key]=dict(event='training_complete',**completion)
        del optimizer,state
    # A different GPU run cannot silently substitute an approximate carry checkpoint.
    for key,e in done.items():
        if file_hash(checkpoint(path,key))!=e['sha256']:raise ValueError('Frozen weights changed')
    model.requires_grad_(False);current=None
    for index,req in enumerate(fixed['evaluation'][len(rows):],len(rows)+1):
        key=f"r{req['replication']}-{req['arm']}";sha=done[key]['sha256']
        if current!=key:
            load_adapter(modules,load_file(str(checkpoint(path,key))));current=key
        b=blocks[req['block']];inputs,spans,ph=inputs_for(b,req['family'],req['task'],req['position'],req['format'],req['mapping'])
        append(path,'request',request=req,promptHash=ph,adapterHash=sha,inputTokens=int(inputs.input_ids.shape[-1]))
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            logits,trace=evaluate_forward(model,inputs,spans,b,req['family'],req['position']);out=summarize_logits(logits,choices)
        torch.cuda.synchronize();append(path,'result',id=req['id'],seconds=time.perf_counter()-start,intervention=trace,**out)
        if index%512==0:print(f'Evaluation {index}/{len(fixed["evaluation"])}',flush=True)
    verify_sources(parent,composition);write_summary(path)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('journal',type=Path)
    p.add_argument('--parent-journal',type=Path,required=True);p.add_argument('--composition-journal',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:run(a.journal,a.parent_journal,a.composition_journal,resume=a.resume)
    except BaseException as exc:
        if a.journal.exists():append(a.journal,'error',errorType=type(exc).__name__,message=str(exc))
        raise
