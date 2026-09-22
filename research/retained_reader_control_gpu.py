"""One declared Q/V suffix update; task producer and retained memories stay frozen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from research import retained_reader_control as control


def run(path):
    import torch
    from safetensors.torch import save_file, load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from research import prospective_pair_discovery as parent
    from research.confidence_generation_continuity import capture_generated_prefix
    from research.confidence_cached_action_decode import decode_cached_branch
    from research.prospective_cache_interventions import permute_prefix_cache
    from research.cross_model_gpu import environment
    from research.natural_error_journal import Writer
    from research.retained_state_reader import RetainedStateReader

    path=Path(path); assert not path.exists()
    p=control.plan(); preparation=json.loads(control.PREPARATION.read_text())
    assert preparation==dict(plan=p,planHash=control.digest(p),sourceHash=control.source_hash())
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    metadata=environment(); metadata['model']=p['model']
    w=Writer(path); w.write(dict(event='header',origin='transformers_gpu',metadata=metadata,**preparation),create=True)
    try:
        spec=p['model']; identity=spec['id']+'@'+spec['revision']+'/base-bfloat16-sdpa'
        tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
        reference=parent.read_journal(control.PARENT,tokenizer); pp=parent.plan()
        producer=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],torch_dtype=torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True,trust_remote_code=False).eval().requires_grad_(False)
        assert producer.config._commit_hash==spec['revision']
        held={}; task_baselines={}; forecast_baselines={}; examples=[]
        for cid in p['cases']:
            old=reference['states'][cid]; case=old['case']
            text,metrics,trace,saved=capture_generated_prefix(producer,tokenizer,case['messages'],case['seed'],
                model_state_id=identity,settings=pp['settings'],max_context_tokens=2048)
            assert (text,metrics,trace)==(old['text'],old['metrics'],old['trace'])
            mask=old['bindingMask']; altered,record=permute_prefix_cache(producer,saved,model_state_id=identity,
                permutation=mask['permutation'],layers=pp['intervention']['layers'],mode='values_only')
            assert altered.cache_hash==old['cacheHashes']['values_permuted'] and record==old['interventions']['values_permuted']
            targets=[mask['selectedRows'][0],next(t for t in range(case['bindings']) if t not in mask['selectedRows'])]
            branches=parent.compile_branches(tokenizer,saved.prefix,case,pp)
            held[cid]=dict(actual=saved,values_permuted=altered,branches=branches)
            w.write(dict(event='state',case=cid,targets=targets,trace=trace,mask=mask,cacheHashes=dict(actual=saved.cache_hash,values_permuted=altered.cache_hash)))
            for target in targets:
                for condition,snapshot in (('actual',saved),('values_permuted',altered)):
                    call=dict(case=cid,target=target,condition=condition)
                    decoded=decode_cached_branch(producer,tokenizer,branches[f'task/{target}'],snapshot,model_state_id=identity,max_input_tokens=2048)
                    assert decoded==reference['records']['task'][parent.key(call)]['decoded']
                    task_baselines[cid,target,condition]=decoded
                    w.write(dict(event='producer_task_before',call=call,decoded=decoded))
                actual_task=task_baselines[cid,target,'values_permuted']
                label=int(actual_task['validNativeResponse'] and actual_task['decision']==('one' if case['values'][target] else 'zero'))
                for mapping in (0,1):
                    call=dict(case=cid,target=target,condition='values_permuted',mapping=mapping)
                    branch=branches[f'forecast/{target}/{mapping}']
                    decoded=decode_cached_branch(producer,tokenizer,branch,altered,model_state_id=identity,max_input_tokens=2048)
                    assert decoded==reference['records']['forecast'][parent.key(call)]['decoded']
                    forecast_baselines[cid,target,mapping]=decoded
                    examples.append((call,altered,branch,branch['candidateTokenIds'][label],label))
                    w.write(dict(event='producer_forecast',call=call,decoded=decoded,label=label))
        reader=RetainedStateReader(producer,producer_state_id=identity,**p['adapter'])
        w.write(dict(event='reader',config=reader.config,initialAdapterHash=reader.adapter_hash(),
            parameters=sum(t.numel() for t in reader.adapter_parameters),producerCopyExact=True))
        for call,saved,branch,target,label in examples:
            result=reader.decode(tokenizer,saved,branch)
            baseline=forecast_baselines[call['case'],call['target'],call['mapping']]
            assert result['decoded']['tokenIds']==baseline['tokenIds']
            w.write(dict(event='zero_reader',call=call,result=result))
        config=p['optimizer']; optimizer=torch.optim.AdamW(reader.adapter_parameters,lr=config['lr'],
            betas=tuple(config['betas']),eps=config['eps'],weight_decay=config['weight_decay'])
        optimizer.zero_grad(set_to_none=True); losses=[]; before_hash=reader.adapter_hash()
        torch.cuda.synchronize(); started=time.monotonic()
        for call,saved,branch,target,label in examples:
            loss=reader.loss(saved,branch,target,tokenizer.eos_token_id)
            (loss/len(examples)).backward(); losses.append(float(loss.detach()))
            w.write(dict(event='training_example',call=call,label=label,target=target,loss=losses[-1]))
        norm=torch.nn.utils.clip_grad_norm_(reader.adapter_parameters,config['clipNorm'],error_if_nonfinite=True)
        assert float(norm)>0 and all(t.grad is None for t in producer.parameters())
        optimizer.step(); torch.cuda.synchronize()
        after_hash=reader.adapter_hash(); assert before_hash!=after_hash
        weight_path=path.with_suffix('.adapter.safetensors'); assert not weight_path.exists()
        state=reader.adapter_state(); assert all(bool(torch.isfinite(t).all()) for t in state.values())
        save_file(state,str(weight_path))
        w.write(dict(event='updated',beforeAdapterHash=before_hash,afterAdapterHash=after_hash,
            gradientNorm=float(norm),seconds=time.monotonic()-started,meanLossBefore=sum(losses)/len(losses),
            weights=dict(file=weight_path.name,bytes=weight_path.stat().st_size,sha256=hashlib.sha256(weight_path.read_bytes()).hexdigest())))
        post={}; after_losses=[]
        for call,saved,branch,target,label in examples:
            result=reader.decode(tokenizer,saved,branch); post[parent.key(call)]=result
            loss=float(reader.loss(saved,branch,target,tokenizer.eos_token_id).detach()); after_losses.append(loss)
            w.write(dict(event='trained_reader',call=call,result=result,loss=loss,label=label))
        reader.load_adapter(load_file(str(weight_path)))
        assert reader.adapter_hash()==after_hash
        for call,saved,branch,target,label in examples:
            assert reader.decode(tokenizer,saved,branch)==post[parent.key(call)]
        w.write(dict(event='reload_verified',adapterHash=after_hash,branches=len(examples)))
        for (cid,target,condition),baseline in task_baselines.items():
            h=held[cid]; saved=h[condition]; reader.guard(saved)
            decoded=decode_cached_branch(producer,tokenizer,h['branches'][f'task/{target}'],saved,model_state_id=identity,max_input_tokens=2048)
            assert decoded==baseline
            w.write(dict(event='producer_task_after',call=dict(case=cid,target=target,condition=condition),decoded=decoded))
        w.write(dict(event='complete',counts=p['counts'],meanLossAfter=sum(after_losses)/len(after_losses),
            peakGPUBytes=torch.cuda.max_memory_allocated(),producerUnchanged=True))
        print(json.dumps(dict(completed=True,meanLossBefore=sum(losses)/len(losses),meanLossAfter=sum(after_losses)/len(after_losses))),flush=True)
    except BaseException as error:
        w.write(dict(event='failure',errorType=type(error).__name__,error=str(error))); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--journal',type=Path,required=True)
    run(parser.parse_args().journal)
