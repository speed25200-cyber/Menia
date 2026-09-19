"""Evaluate frozen Colab 10 adapters with new question mappings; no optimizer or training."""
import json
import os
from pathlib import Path
import time

import torch

from research import presence_specificity as study
from research.native_localization_gpu import install_adapters, load_adapter, forward, summarize_logits, append, file_hash
from research.localization_replication import trace_position


def encode(tokenizer,block,family,question,position,device):
    body,spans=study.prompt(block,family,question,position)
    text=tokenizer.apply_chat_template([dict(role='user',content=body)],tokenize=False,
                                      add_generation_prompt=True,enable_thinking=False)
    if text.count(body)!=1: raise ValueError('Cannot uniquely locate user content')
    offset=text.index(body)
    inputs=tokenizer(text,return_tensors='pt',add_special_tokens=False,return_offsets_mapping=True,return_token_type_ids=False)
    offsets=inputs.pop('offset_mapping')[0].tolist()
    if len(offsets)>study.CONFIG['maxInputTokens']: raise ValueError('No truncation permitted')
    tokens=[[i for i,(a,b) in enumerate(offsets) if b>a and b>lo+offset and a<hi+offset] for lo,hi in spans]
    if any(not t for t in tokens) or len(set(sum(tokens,[])))!=sum(map(len,tokens)): raise ValueError('Invalid spans')
    return inputs.to(device),tokens,study.parent.digest(body)


def verify_parent(parent_path):
    parent_path=Path(parent_path)
    if file_hash(parent_path)!=study.PARENT_JOURNAL_SHA256: raise ValueError('Parent journal hash mismatch')
    frozen=study.checkpoints()
    for key,sha in frozen.items():
        if file_hash(parent_path.with_suffix('.'+key+'.safetensors'))!=sha: raise ValueError('Frozen checkpoint mismatch: '+key)
    return frozen


def run(path,parent_path,*,resume=False):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from safetensors.torch import load_file
    from research.cross_model_gpu import environment
    path=Path(path);parent_path=Path(parent_path)
    if path.exists() and not resume: raise ValueError('Existing journal requires --resume')
    frozen=verify_parent(parent_path)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    metadata=environment()
    metadata.update(origin='transformers_gpu',model=study.MODEL,
        settings=dict(dtype='bfloat16',attention='sdpa',thinking=False,useCache=False,sampling=False,
                      firstTokenOnly=True,training=False,**study.CONFIG))
    tok=AutoTokenizer.from_pretrained(study.MODEL['id'],revision=study.MODEL['revision'],trust_remote_code=False)
    ids=[tok.encode(str(i),add_special_tokens=False) for i in range(3)]
    if any(len(x)!=1 for x in ids) or len({x[0] for x in ids})!=3: raise ValueError('Invalid digit tokens')
    choices=[x[0] for x in ids];metadata['choiceTokenIds']=choices;fixed=study.plan()
    if path.exists():
        h,rows,pending,_=study.read_journal(path)
        if h['metadata']!=metadata: raise ValueError('Environment changed')
        if len(rows)==len(fixed['evaluation']) and pending is None:
            print('Already complete; no inference repeated.',flush=True);return
        if pending: append(path,'interrupted_request',id=pending['request']['id'])
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        append(path,'header',plan=fixed,planHash=study.parent.digest(fixed),sourceHash=study.source_hash(),
               metadata=metadata,parentJournalSHA256=study.PARENT_JOURNAL_SHA256,checkpoints=frozen)
        rows=[]
    model=AutoModelForCausalLM.from_pretrained(study.MODEL['id'],revision=study.MODEL['revision'],
        torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True,trust_remote_code=False).eval()
    if model.config._commit_hash!=study.MODEL['revision'] or len(model.model.layers)!=36: raise ValueError('Unexpected architecture')
    modules=install_adapters(model,rank=study.CONFIG['rank']);model.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    blocks={b['id']:b for b in fixed['blocks']};current=None;cache={}
    for index,req in enumerate(fixed['evaluation'][len(rows):],len(rows)+1):
        key=f"r{req['replication']}-{req['arm']}"
        if current!=key:
            for m in modules.values(): m.enabled=req['arm']!='base'
            if req['arm']!='base':
                checkpoint=parent_path.with_suffix('.'+key+'.safetensors')
                if file_hash(checkpoint)!=frozen[key]: raise ValueError('Checkpoint changed')
                load_adapter(modules,load_file(str(checkpoint)))
            current=key
        b=blocks[req['block']];cache_key=(b['id'],req['family'],req['question'],req['position'])
        if cache_key not in cache: cache[cache_key]=encode(tok,b,req['family'],req['question'],req['position'],model.device)
        inputs,spans,ph=cache[cache_key]
        append(path,'request',request=req,promptHash=ph,adapterHash=frozen.get(key),inputTokens=int(inputs.input_ids.shape[-1]))
        torch.cuda.synchronize();start=time.perf_counter()
        manipulation=dict(layer=study.CONFIG['layer'],strength=study.CONFIG['strength'],noiseSeed=b['noiseSeed'])
        position=trace_position('visible' if req['family']=='visible' else 'strong',req['position'])
        with torch.inference_mode():
            logits,trace=forward(model,inputs,spans,manipulation,position)
            result=summarize_logits(logits,choices)
        torch.cuda.synchronize()
        append(path,'result',id=req['id'],seconds=time.perf_counter()-start,intervention=trace,**result)
        if index%256==0: print(f'Evaluation {index}/{len(fixed["evaluation"])}',flush=True)
    verify_parent(parent_path)
    path.with_suffix('.summary.json').write_text(json.dumps(study.analyze(path),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('journal',type=Path)
    parser.add_argument('--parent-journal',type=Path,required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try: run(args.journal,args.parent_journal,resume=args.resume)
    except BaseException as exc:
        if args.journal.exists(): append(args.journal,'error',errorType=type(exc).__name__,message=str(exc))
        raise
