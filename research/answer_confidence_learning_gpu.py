"""Six matched confidence adapters, then frozen crossed answers and judgments."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn

from research import answer_confidence_study as study
from research.answer_confidence_plan import load_training,training_batches,TRAINING
from research.answer_confidence_gpu import assess_answer,encode_confidence_training
from research.answer_confidence_baselines import fit
from research.answer_confidence_journal import validate_answer,validate_judgment
from research.action_binding_gpu import train_step
from research.cross_model_gpu import environment
from research.output_confidence_trace import trace_generation
from research.natural_error_journal import Writer
from research.native_localization_gpu import install_adapters,adapter_state,load_adapter,file_hash


def run(path,data_directory):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from safetensors.torch import save_file,load_file
    path=Path(path)
    if path.exists():raise ValueError('Existing attempt; no automatic retry or resume')
    data,data_report=load_training(data_directory);p=study.make_plan()
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['TOKENIZERS_PARALLELISM']='false'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    metadata=environment();spec=p['model'];writer=Writer(path)
    tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],torch_dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True,trust_remote_code=False).eval()
    if model.config._commit_hash!=spec['revision']:raise ValueError('Model revision')
    eos=model.generation_config.eos_token_id;eos=eos[0] if isinstance(eos,list) else eos
    if eos!=tokenizer.eos_token_id:raise ValueError('EOS mismatch')
    codes=[tokenizer.encode(s,add_special_tokens=False) for s in ('0','1')]
    if any(len(c)!=1 for c in codes) or codes[0]==codes[1]:raise ValueError('Native codes')
    metadata.update(model=spec,newLLMBaseWeightUpdates=0,adapterUpdatesPlanned=864,confidenceTokenIds=[c[0] for c in codes],vocabularySize=model.config.vocab_size,eosTokenId=eos)
    writer.write(dict(event='header',plan=p,planHash=study.digest(p),sourceHash=study.source_hash(),trainingData=data,trainingReport=data_report,origin='transformers_gpu',metadata=metadata),create=True)
    try:
        modules=install_adapters(model,rank=TRAINING['rank']);parameters=[v for m in modules.values() for v in (m.a,m.b)]
        if any(m.scale!=TRAINING['scale'] for m in modules.values()):raise ValueError('Adapter scale')
        if {id(v) for v in model.parameters() if v.requires_grad}!={id(v) for v in parameters}:raise ValueError('Trainable weights')
        checkpoints={}
        for rep in range(3):
            torch.manual_seed(study.SEED+1001+rep)
            with torch.no_grad():
                for m in modules.values():nn.init.kaiming_uniform_(m.a,a=math.sqrt(5));m.b.zero_();m.enabled=True
            initial=adapter_state(modules);initial_file=path.with_suffix(f'.r{rep}-initial.safetensors');save_file(initial,str(initial_file));initial_hash=file_hash(initial_file)
            for arm in ('measured','shuffled'):
                key=f'r{rep}-{arm}';load_adapter(modules,initial)
                optimizer=torch.optim.AdamW(parameters,lr=TRAINING['learningRate'],betas=tuple(TRAINING['betas']),eps=TRAINING['epsilon'],weight_decay=0.,foreach=False)
                writer.write(dict(event='training_start',key=key,initializationHash=initial_hash,trainableParameters=sum(v.numel() for v in parameters)))
                started=time.perf_counter();cache={}
                for step,batch in enumerate(training_batches(data,rep,arm),1):
                    encoded=[]
                    for example in batch:
                        h=study.digest(example)
                        if h not in cache:cache[h]=encode_confidence_training(tokenizer,example,model.device,max_input_tokens=TRAINING['maxTrainingTokens'])
                        encoded.append(cache[h])
                    tick=time.perf_counter();metrics=train_step(model,optimizer,parameters,encoded,eos)
                    metrics['seconds']=time.perf_counter()-tick
                    writer.write(dict(event='training_step',key=key,step=step,dataHash=study.digest(batch),**metrics))
                    if step%16==0:print(f'{key}: training {step}/144',flush=True)
                optimizer.zero_grad(set_to_none=True);del optimizer;del cache
                state=adapter_state(modules)
                if not any(not torch.equal(state[n],initial[n]) for n in state):raise ValueError('Unchanged adapter')
                checkpoint=path.with_suffix('.'+key+'.safetensors');save_file(state,str(checkpoint))
                event=dict(event='training_complete',key=key,steps=144,checkpoint=checkpoint.name,sha256=file_hash(checkpoint),initializationHash=initial_hash,trainingSeconds=time.perf_counter()-started)
                writer.write(event);checkpoints[key]=event
        print('All six confidence adapters frozen before calibration or test.',flush=True)
        answers={};judgments={};current=None
        for call in p['calls']:
            if call['id']==p['calibrationCalls']:
                rows=study.evaluation_rows(p,answers,judgments,'calibration')
                for rep in range(3):writer.write(dict(event='baseline',replication=rep,bundle=fit(rows,rep)))
                print('All calibration baselines frozen before test.',flush=True)
            key=f'r{call["replication"]}-{call["modelArm"]}'
            if key!=current:
                for m in modules.values():m.enabled=call['modelArm']!='base'
                if call['modelArm']!='base':
                    checkpoint=path.with_suffix('.'+key+'.safetensors')
                    if file_hash(checkpoint)!=checkpoints[key]['sha256']:raise ValueError('Changed frozen weights')
                    load_adapter(modules,load_file(str(checkpoint)))
                current=key
            request=study.request_for(p,call['id'],answers,checkpoints);writer.write(request);started=time.perf_counter()
            if call['kind']=='answer':
                text,metrics,trace=trace_generation(model,tokenizer,request['messages'],call['seed'],settings=p['generation'])
                event=dict(event='answer',id=call['id'],status='ok',text=text,seconds=time.perf_counter()-started,metrics=metrics,trace=trace,errorType=None)
                validate_answer(event,request);answers[(call['task'],call['producer'])]=event
            else:
                t=p['tasks'][call['task']];answer=answers[(t['id'],call['producer'])]
                scores=assess_answer(model,tokenizer,t['question'],answer['text'],max_input_tokens=p['judgeSettings']['maxInputTokens'])
                event=dict(event='judgment',id=call['id'],status='ok',seconds=time.perf_counter()-started,scores=scores)
                validate_judgment(event,request,metadata);judgments[(t['id'],call['judge'],call['producer'])]=event
            writer.write(event)
            if (call['id']+1)%96==0:print(f'{call["id"]+1}/{p["plannedCalls"]} calls recorded',flush=True)
        from research.answer_confidence_analysis import analyze
        report=analyze(path)
        path.with_suffix('.summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except BaseException as error:
        writer.write(dict(event='failure',errorType=type(error).__name__));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('journal',type=Path);parser.add_argument('data',type=Path)
    args=parser.parse_args();run(args.journal,args.data)
