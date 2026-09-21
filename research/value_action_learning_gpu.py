"""Matched Q/V LoRA: native choices, informative lookup or shuffled auxiliary labels."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn

from research import value_action_learning as study
from research.cross_model_gpu import environment,generate_text
from research.natural_error_journal import Writer
from research.native_localization_gpu import install_adapters,adapter_state,load_adapter,file_hash
from research.native_choice_journal import validate_result


from research.action_binding_gpu import encode_training, train_step


def run(path):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from safetensors.torch import save_file,load_file
    path=Path(path)
    if path.exists():raise ValueError('Attempt exists; no automatic retry or resume')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['TOKENIZERS_PARALLELISM']='false'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    metadata=environment();spec=study.MODELS['A'];p=study.make_plan();writer=Writer(path)
    metadata.update(models={'A':spec},settings=dict(training=study.CONFIG,evaluation=study.parent.DECISION_SETTINGS),newLLMBaseWeightUpdates=0,adapterUpdatesPlanned=576)
    tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True,trust_remote_code=False).eval()
    if model.config._commit_hash!=spec['revision']:raise ValueError('Model revision')
    eos=model.generation_config.eos_token_id
    eos=eos[0] if isinstance(eos,list) else eos
    if eos!=tokenizer.eos_token_id:raise ValueError('EOS mismatch')
    ids=[tokenizer.encode(s,add_special_tokens=False) for s in ('1','2','A','B')]
    if any(len(s)!=1 for s in ids) or len({s[0] for s in ids})!=4:raise ValueError('Distinct single-token symbols required')
    metadata.update(symbolTokenIds=[s[0] for s in ids],eosTokenId=eos)
    writer.write(dict(event='header',plan=p,planHash=study.digest(p),sourceHash=study.source_hash(),origin='transformers_gpu',metadata=metadata),create=True)
    try:
        modules=install_adapters(model,rank=8)
        if any(m.scale!=study.CONFIG['scale'] for m in modules.values()):raise ValueError('Adapter scale')
        parameters=[v for m in modules.values() for v in (m.a,m.b)]
        if {id(v) for v in model.parameters() if v.requires_grad}!={id(v) for v in parameters}:raise ValueError('Unexpected trainable weights')
        checkpoints={}
        for rep in range(3):
            torch.manual_seed(study.SEED+1001+rep)
            with torch.no_grad():
                for m in modules.values():nn.init.kaiming_uniform_(m.a,a=math.sqrt(5));m.b.zero_();m.enabled=True
            initial=adapter_state(modules);initial_file=path.with_suffix(f'.r{rep}-initial.safetensors');save_file(initial,str(initial_file));initial_hash=file_hash(initial_file)
            for arm in ('choice','linked','shuffled'):
                unit=next(u for u in p['trainingUnits'] if u['replication']==rep and u['arm']==arm);key=unit['key']
                load_adapter(modules,initial)
                optimizer=torch.optim.AdamW(parameters,lr=study.CONFIG['learningRate'],betas=tuple(study.CONFIG['betas']),eps=study.CONFIG['epsilon'],weight_decay=0.,foreach=False)
                writer.write(dict(event='training_start',key=key,initializationHash=initial_hash,trainableParameters=sum(v.numel() for v in parameters)))
                training_started=time.perf_counter();cache={}
                for step in range(1,65):
                    examples=study.training_batch(p,unit,step);encoded=[]
                    for example in examples:
                        h=study.digest(example)
                        if h not in cache:cache[h]=encode_training(tokenizer,example,model.device)
                        encoded.append(cache[h])
                    step_started=time.perf_counter()
                    metrics=train_step(model,optimizer,parameters,encoded,eos)
                    metrics['seconds']=time.perf_counter()-step_started
                    writer.write(dict(event='training_step',key=key,step=step,dataHash=study.digest(examples),**metrics))
                    if step%8==0:print(f'{key}: training {step}/64',flush=True)
                optimizer.zero_grad(set_to_none=True);del optimizer;del cache
                state=adapter_state(modules)
                if not any(not torch.equal(state[n],initial[n]) for n in state):raise ValueError('No changed adapter')
                checkpoint=path.with_suffix('.'+key+'.safetensors');save_file(state,str(checkpoint))
                event=dict(event='training_complete',key=key,steps=64,sha256=file_hash(checkpoint),checkpoint=checkpoint.name,initializationHash=initial_hash,trainingSeconds=time.perf_counter()-training_started)
                writer.write(event);checkpoints[key]=event
        print('All nine trained adapters frozen before evaluation.',flush=True)
        current=None
        for call in p['calls']:
            key=f'r{call["replication"]}-{call["arm"]}'
            if key!=current:
                for m in modules.values():m.enabled=call['arm']!='base'
                if call['arm']!='base':
                    checkpoint=path.with_suffix('.'+key+'.safetensors')
                    if file_hash(checkpoint)!=checkpoints[key]['sha256']:raise ValueError('Frozen checkpoint changed')
                    load_adapter(modules,load_file(str(checkpoint)))
                current=key
            request=study.request_for(p,call['id'],checkpoints);writer.write(request);started=time.perf_counter()
            text,metrics=generate_text(model,tokenizer,request['messages'],call['seed'],settings=study.parent.DECISION_SETTINGS)
            result=dict(event='result',id=call['id'],status='ok',engine='llm',text=text,seconds=time.perf_counter()-started,metrics=metrics,errorType=None)
            validate_result(result,request);writer.write(result)
            if (call['id']+1)%64==0:print(f'{call["id"]+1}/{p["planned"]} evaluations recorded',flush=True)
        report=study.analyze(path)
        path.with_suffix('.summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except BaseException as error:
        writer.write(dict(event='failure',errorType=type(error).__name__))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('journal',type=Path);run(parser.parse_args().journal)
