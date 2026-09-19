"""Actual Qwen interchange, frozen weights, canonical prompts, complete factorial."""
import json
import os
from pathlib import Path
import time

import torch
from research import state_interchange as study
from research.native_localization_gpu import append,file_hash,install_adapters,load_adapter,summarize_logits
from research.state_composition_gpu import encode
from research.state_interchange_ops import capture_forward,patch_forward,tensor_hash


def run_group(model,tokenizer,pair,choices):
    cache={};outputs=[]
    for request in study.requests():
        task=request['task'];kind=request['kind']
        torch.cuda.synchronize() if model.device.type=='cuda' else None
        start=time.perf_counter()
        if kind=='intact':
            role=request['role'];b=pair[role];state=request['state'];code=request['code']
            position=b['positivePosition'] if state else 0
            inputs,spans,ph=encode(tokenizer,b,'hidden',task,position,model.device,'trained',code)
            with torch.inference_mode():
                logits,trace,states=capture_forward(model,inputs,spans,b,position,study.SITES)
                result=summarize_logits(logits,choices)
            hashes={str(site):tensor_hash(states[site]) for site in study.SITES}
            cache[(task,role,state,code)]=dict(inputs=inputs,spans=spans,block=b,position=position,
                states=states,promptHash=ph,result=result)
            extra=dict(activationHashes=hashes,promptHash=ph,inputTokens=int(inputs.input_ids.shape[-1]))
        else:
            recipient=cache[(task,'recipient',request['recipientState'],request['recipientCode'])]
            donor=recipient if kind=='sham' else cache[(task,'donor',request['donorState'],request['donorCode'])]
            with torch.inference_mode():
                logits,trace,patch=patch_forward(model,recipient['inputs'],recipient['spans'],recipient['block'],
                    recipient['position'],request['site'],donor['states'][request['site']])
                result=summarize_logits(logits,choices)
            extra=dict(patch=patch,promptHash=recipient['promptHash'],inputTokens=int(recipient['inputs'].input_ids.shape[-1]))
        torch.cuda.synchronize() if model.device.type=='cuda' else None
        outputs.append(dict(request=request,seconds=time.perf_counter()-start,intervention=trace,**result,**extra))
    study.validate_group(pair,outputs,choices)
    return outputs


def run(path,parent,*,resume=False):
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from research.cross_model_gpu import environment
    path,parent=Path(path),Path(parent)
    if file_hash(parent)!=study.PARENT_SHA256:raise ValueError('Parent journal changed')
    frozen=study.checkpoints()
    for key,sha in frozen.items():
        if file_hash(parent.with_suffix('.'+key+'.safetensors'))!=sha:raise ValueError('Prefix checkpoint changed')
    if path.exists() and not resume:raise ValueError('Explicit resume required')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    metadata=environment();metadata.update(origin='transformers_gpu',models={'A':study.MODEL},
        settings=dict(dtype='bfloat16',sampling=False,thinking=False,useCache=False,firstTokenOnly=True,
                      attention='sdpa',alignmentTraining=False,weightUpdates=0,**study.CONFIG))
    tokenizer=AutoTokenizer.from_pretrained(study.MODEL['id'],revision=study.MODEL['revision'],trust_remote_code=False)
    encoded=[tokenizer.encode(str(i),add_special_tokens=False) for i in range(4)]
    if any(len(v)!=1 for v in encoded) or len({v[0] for v in encoded})!=4:raise ValueError('Digit tokenization changed')
    choices=[v[0] for v in encoded];metadata['choiceTokenIds']=choices;fixed=study.plan()
    if path.exists():
        header,done,pending,_=study.read_journal(path)
        if header['metadata']!=metadata:raise ValueError('Resume environment differs')
        if len(done)==len(fixed['groups']) and pending is None:
            print('Already complete; no inference repeated.',flush=True);return
        if pending:append(path,'group_restart',id=pending['id'],reason='Repeat whole incomplete inference group; retain restart count')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        append(path,'header',plan=fixed,planHash=study.digest(fixed),sourceHash=study.source_hash(),metadata=metadata)
        done=[]
    model=AutoModelForCausalLM.from_pretrained(study.MODEL['id'],revision=study.MODEL['revision'],torch_dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True,trust_remote_code=False).eval()
    if model.config._commit_hash!=study.MODEL['revision'] or len(model.model.layers)!=36:raise ValueError('Unexpected architecture')
    modules=install_adapters(model,rank=study.CONFIG['rank']);model.requires_grad_(False)
    pairs={p['id']:p for p in fixed['pairs']};current=None
    for index,group in enumerate(fixed['groups'][len(done):],len(done)+1):
        pair=pairs[group['pair']];key=f'r{pair["replication"]}-{group["arm"]}'
        if key!=current:
            for m in modules.values():m.enabled=group['arm']!='base'
            if group['arm']=='prefix':
                checkpoint=parent.with_suffix('.'+key+'.safetensors')
                if file_hash(checkpoint)!=frozen[key]:raise ValueError('Checkpoint changed')
                load_adapter(modules,load_file(str(checkpoint)))
            current=key
        append(path,'group_start',group=group)
        try:outputs=run_group(model,tokenizer,pair,choices)
        except BaseException as exc:
            append(path,'error',errorType=type(exc).__name__,message=str(exc));raise
        append(path,'group_complete',id=group['id'],outputs=outputs)
        print(f'Group {index}/{len(fixed["groups"])}; {index*136}/{fixed["plannedForwards"]} recorded forwards',flush=True)
    if file_hash(parent)!=study.PARENT_SHA256:raise ValueError('Parent journal changed during execution')
    for key,sha in frozen.items():
        if file_hash(parent.with_suffix('.'+key+'.safetensors'))!=sha:raise ValueError('Checkpoint changed during execution')
    summary=study.analyze(path)
    path.with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('journal',type=Path)
    p.add_argument('--parent-journal',type=Path,required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.journal,a.parent_journal,resume=a.resume)
