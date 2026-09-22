"""Collect matched suffix learning with reserved forecasts frozen before their tasks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from research import prospective_reader_learning as study


def run(path):
    import torch
    from safetensors.torch import save_file,load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from research import prospective_pair_discovery as parent
    from research.confidence_generation_continuity import capture_generated_prefix
    from research.confidence_generation_numerics import replay_generated_tokens
    from research.confidence_cached_action_decode import decode_cached_branch,_parameters
    from research.prospective_cache_interventions import permute_prefix_cache
    from research.cross_model_gpu import environment
    from research.natural_error_journal import Writer
    from research.retained_state_reader import RetainedStateReader

    path=Path(path);assert not path.exists()
    p=study.plan();prepared=json.loads(study.PREPARATION.read_text(encoding='utf-8'))
    assert prepared==dict(plan=p,planHash=study.digest(p),sourceHash=study.source_hash())
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    metadata=environment();metadata.update(model=p['model'],settings=p['settings'])
    w=Writer(path);w.write(dict(event='header',origin='transformers_gpu',metadata=metadata,**prepared),create=True)
    try:
        spec=p['model'];identity=spec['id']+'@'+spec['revision']+'/base-bfloat16-sdpa'
        tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
        producer=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],torch_dtype=torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True,trust_remote_code=False).eval().requires_grad_(False)
        assert producer.config._commit_hash==spec['revision'] and len(producer.model.layers)==36
        producer_versions=_parameters(producer);case_map={c['id']:c for c in p['cases']}
        held={};tasks={};weights={};reader=None;optimizer=None;active=None;reserved_forecasts=[]
        order=list(study.operations(p))
        for index,(kind,call) in enumerate(order):
            w.write(dict(event='request',kind=kind,call=call))
            torch.cuda.synchronize();started=time.monotonic()
            if kind=='state':
                case=case_map[call['case']]
                text,metrics,trace,actual=capture_generated_prefix(producer,tokenizer,case['messages'],case['seed'],
                    model_state_id=identity,settings=p['settings'],max_context_tokens=p['maxContextTokens'])
                replay=replay_generated_tokens(producer,actual,len(trace['promptTokenIds']),model_state_id=identity,schedule='generation')
                mask=parent.pair_swap(tokenizer,case,trace['promptTokenIds'],len(actual.prefix));args=dict(model_state_id=identity,layers=list(range(36)))
                altered,a=permute_prefix_cache(producer,actual,permutation=mask['permutation'],mode='values_only',**args)
                joint,j=permute_prefix_cache(producer,actual,permutation=mask['permutation'],mode='keys_and_values',**args)
                restored,r=permute_prefix_cache(producer,altered,permutation=mask['permutation'],mode='values_only',**args)
                assert actual.cache_hash==replay.cache_hash==restored.cache_hash
                states=dict(actual=actual,same_schedule=replay,values_permuted=altered,joint_permuted=joint,restored=restored)
                branches=parent.compile_branches(tokenizer,actual.prefix,case,p);exported=None
                if case['id'] in p['exportCacheCases']:
                    file=path.with_suffix(f'.case{case["id"]}.safetensors');assert not file.exists()
                    save_file({f'layer.{layer:02d}.{name}':v.detach().cpu().contiguous() for layer,pair in enumerate(actual.kv)
                        for name,v in zip(('key','value'),pair)},str(file))
                    exported=dict(file=file.name,bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                        cacheHash=actual.cache_hash,tensors=72)
                held[case['id']]=dict(states=states,branches=branches)
                result=dict(case=case,text=text,metrics=metrics,trace=trace,bindingMask=mask,
                    branchInputHashes={k:study.digest(v['inputIds']) for k,v in branches.items()},
                    cacheHashes={k:v.cache_hash for k,v in states.items()},cacheExport=exported,
                    interventions=dict(values_permuted=a,joint_permuted=j,restored=r))
            elif kind in ('task','coding','forecast'):
                h=held[call['case']];branch=h['branches'][study.branch_name(kind,call)]
                arm=call.get('arm','base');condition=study.memory_condition(arm,call['condition']);saved=h['states'][condition]
                if arm=='base':
                    decoded=decode_cached_branch(producer,tokenizer,branch,saved,model_state_id=identity,max_input_tokens=p['maxContextTokens'])
                    reader_result=None
                else:
                    if active!=arm:
                        reader.load_adapter(load_file(str(path.parent/weights[arm]['file'])));active=arm
                        assert reader.adapter_hash()==weights[arm]['adapterHash']
                    reader_result=reader.decode(tokenizer,saved,branch,max_input_tokens=p['maxContextTokens']);decoded=reader_result['decoded']
                result=dict(decoded=decoded,reader=reader_result,sourceCondition=condition,inputHash=study.digest(branch['inputIds']))
                if kind=='task':tasks[call['case'],call['target'],call['condition']]=result
            elif kind=='weights':
                arm=call['arm']
                if arm=='initial':
                    reader=RetainedStateReader(producer,producer_state_id=identity,**p['adapter']);active='initial'
                file=path.with_suffix('.'+arm+'.safetensors');assert not file.exists()
                state=reader.adapter_state();assert all(bool(torch.isfinite(t).all()) for t in state.values())
                save_file(state,str(file))
                result=dict(file=file.name,bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    adapterHash=reader.adapter_hash(),parameters=sum(t.numel() for t in state.values()),tensors=len(state))
                weights[arm]=result
            elif kind=='reset':
                reader.load_adapter(load_file(str(path.parent/weights['initial']['file'])))
                assert reader.adapter_hash()==weights['initial']['adapterHash'];active=call['arm']
                config=p['optimizer'];optimizer=torch.optim.AdamW(reader.adapter_parameters,lr=config['lr'],
                    betas=tuple(config['betas']),eps=config['eps'],weight_decay=config['weight_decay'])
                result=dict(adapterHash=reader.adapter_hash())
            elif kind=='step':
                assert active==call['arm'];batch=p['optimizer']['batch'];epoch=call['epoch'];start=call['batch']*batch
                examples=study.training_examples(epoch,p)[start:start+batch]
                assert len(examples)==batch and all(case_map[x['case']]['split']=='train' for x in examples)
                optimizer.zero_grad(set_to_none=True);rows=[]
                for example in examples:
                    h=held[example['case']];branch=h['branches'][study.branch_name(example['kind'],example)]
                    condition=study.memory_condition(call['arm'],example['condition']);saved=h['states'][condition]
                    label=study.target_label(example['kind'],example,tasks,case_map);target=branch['candidateTokenIds'][label]
                    loss=reader.loss(saved,branch,target,tokenizer.eos_token_id,max_input_tokens=p['maxContextTokens'])
                    assert bool(torch.isfinite(loss));(loss/batch).backward()
                    rows.append(dict(example=example,label=label,target=target,loss=float(loss.detach()),
                        inputHash=study.digest(branch['inputIds']),sourceCacheHash=saved.cache_hash,sourceCondition=condition))
                norm=torch.nn.utils.clip_grad_norm_(reader.adapter_parameters,p['optimizer']['clipNorm'],error_if_nonfinite=True)
                assert all(t.grad is None for t in producer.parameters());optimizer.step()
                result=dict(examples=rows,meanLoss=sum(r['loss'] for r in rows)/batch,gradientNorm=float(norm))
            elif kind=='forecasts_complete':
                assert len(reserved_forecasts)==p['counts']['reservedForecasts']
                result=dict(recordsHash=study.digest(reserved_forecasts))
            else:raise ValueError(kind)
            assert _parameters(producer)==producer_versions
            torch.cuda.synchronize();event=dict(event=kind,call=call,result=result,seconds=time.monotonic()-started)
            w.write(event)
            if kind=='forecast' and call['split']=='reserved':reserved_forecasts.append(event)
            if kind in ('weights','reset','forecasts_complete') or kind=='step' and call['batch']%14==0 or index%100==0:
                print(json.dumps(dict(operation=index+1,total=len(order),kind=kind,call=call,
                    meanLoss=result.get('meanLoss'))),flush=True)
        w.write(dict(event='complete',counts=p['counts'],producerUnchanged=True,peakGPUBytes=torch.cuda.max_memory_allocated()))
        print(json.dumps(dict(completed=True,counts=p['counts'])),flush=True)
    except BaseException as error:
        w.write(dict(event='failure',errorType=type(error).__name__,error=str(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--journal',type=Path,required=True)
    run(parser.parse_args().journal)
