"""Collect all native forecasts before any scored task from hidden cache states."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from research import prospective_binding_discovery as study
from research.iphone_coupling_report import require


def run(path):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from research.cross_model_gpu import environment
    from research.natural_error_journal import Writer
    from research.confidence_generation_continuity import capture_generated_prefix
    from research.confidence_generation_numerics import replay_generated_tokens
    from research.confidence_cached_action_decode import decode_cached_branch, _parameters
    from research.prospective_cache_interventions import permute_prefix_cache, inverse_permutation

    path=Path(path); require(not path.exists(),'Preserve existing attempt')
    p=study.plan(); prepared=json.loads(study.PREPARATION.read_text(encoding='utf-8'))
    require(prepared==dict(plan=p,planHash=study.digest(p),sourceHash=study.source_hash()),'Frozen preparation changed')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Set CUDA workspace before startup')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    metadata=environment(); metadata.update(model=p['model'],settings=p['settings'],weightUpdates=0)
    writer=Writer(path); writer.write(dict(event='header',origin='transformers_gpu',metadata=metadata,**prepared),create=True)
    terminal=False
    try:
        spec=p['model']; identity=spec['id']+'@'+spec['revision']+'/base-bfloat16-sdpa'
        tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
        reference=study.parent_data(tokenizer); masks=study.prepared_masks()
        model=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],torch_dtype=torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True,trust_remote_code=False).eval().requires_grad_(False)
        require(model.config._commit_hash==spec['revision'] and len(model.model.layers)==36,'Exact Qwen3-4B architecture')
        parameters=_parameters(model); held={}
        for case in p['cases']:
            text,metrics,trace,actual=capture_generated_prefix(model,tokenizer,case['messages'],case['seed'],
                model_state_id=identity,settings=p['settings'],max_context_tokens=p['maxContextTokens'])
            parent=reference['states'][case['id']]
            require((text,metrics,trace)==(parent['text'],parent['metrics'],parent['trace']),'Parent state reproduction failed')
            replay=replay_generated_tokens(model,actual,len(trace['promptTokenIds']),model_state_id=identity,schedule='generation')
            mask=study.binding_swap(tokenizer,case,trace['promptTokenIds'],len(actual.prefix))
            require(mask==masks[case['id']],'Prepared mask mismatch')
            indices=mask['permutation']; args=dict(model_state_id=identity,layers=p['intervention']['layers'])
            altered,alter_record=permute_prefix_cache(model,actual,permutation=indices,mode='values_only',**args)
            joint,joint_record=permute_prefix_cache(model,actual,permutation=indices,mode='keys_and_values',**args)
            restored,restore_record=permute_prefix_cache(model,altered,permutation=inverse_permutation(indices),mode='values_only',**args)
            states=dict(actual=actual,same_schedule=replay,values_permuted=altered,joint_permuted=joint,restored=restored)
            checks=dict(same_schedule_cache_exact=actual.cache_hash==replay.cache_hash,
                restored_cache_exact=actual.cache_hash==restored.cache_hash,
                tokens_unchanged=all(s.prefix==actual.prefix for s in states.values()),parameters_unchanged=parameters==_parameters(model))
            branches=study.compile_branches(tokenizer,actual.prefix,case); cache_export=None
            if case['id'] in p['exportActualCacheCases']:
                cache_path=path.with_suffix(f'.case{case["id"]:02d}.safetensors')
                require(not cache_path.exists(),'Preserve existing cache export')
                tensors={f'layer.{layer:02d}.{kind}':value.detach().cpu().contiguous()
                    for layer,pair in enumerate(actual.kv) for kind,value in zip(('key','value'),pair)}
                save_file(tensors,str(cache_path))
                cache_export=dict(file=cache_path.name,bytes=cache_path.stat().st_size,
                    sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),cacheHash=actual.cache_hash,tensors=len(tensors))
            writer.write(dict(event='state',case=case,text=text,metrics=metrics,trace=trace,bindingMask=mask,
                branchInputHashes={k:study.digest(b['inputIds']) for k,b in branches.items()},
                cacheHashes={k:s.cache_hash for k,s in states.items()},checks=checks,cacheExport=cache_export,
                interventions=dict(values_permuted=alter_record,joint_permuted=joint_record,restored=restore_record)))
            require(all(checks.values()),'State control failed; preserve attempt')
            held[case['id']]=dict(states=states,branches=branches)
            print(f'{len(held)}/24 states retained; no scored task executed',flush=True)
        for phase in ('forecast','task'):
            recorded=[]; decoded_by_key={}
            for call in study.calls(p):
                state=held[call['case']]; branch=state['branches'][phase]; snapshot=state['states'][call['condition']]
                writer.write(dict(event='request',phase=phase,call=call,inputHash=study.digest(branch['inputIds'])))
                torch.cuda.synchronize(); started=time.monotonic()
                decoded=decode_cached_branch(model,tokenizer,branch,snapshot,model_state_id=identity,max_input_tokens=p['maxContextTokens'])
                torch.cuda.synchronize(); elapsed=time.monotonic()-started
                event=dict(event=phase,call=call,decoded=decoded,seconds=elapsed)
                writer.write(event); recorded.append(event); decoded_by_key[(call['case'],call['condition'])]=decoded
                if call['condition']=='actual':
                    require(decoded==reference['forecasts' if phase=='forecast' else 'tasks'][(call['case'],'actual')]['decoded'],'Parent branch reproduction failed')
                original=decoded_by_key.get((call['case'],'actual'))
                for control in ('same_schedule','restored'):
                    controlled=decoded_by_key.get((call['case'],control))
                    if original is not None and controlled is not None:
                        require(original==controlled,'Exact branch control failed; preserve attempt')
                if len(recorded)%20==0: print(f'{len(recorded)}/120 {phase} records',flush=True)
            require(parameters==_parameters(model),'Parameters changed')
            if phase=='forecast':
                writer.write(dict(event='forecasts_complete',count=120,forecastHash=study.digest(recorded)))
                print('All 120 forecasts durably recorded before the first scored task.',flush=True)
        writer.write(dict(event='complete',states=24,forecasts=120,tasks=120)); terminal=True
        summary=study.summarize(study.read_journal(path,tokenizer))
        with path.with_suffix('.summary.json').open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(complete=True,cases=24,forecasts=120,tasks=120)),flush=True)
        return summary
    except BaseException as error:
        if not terminal: writer.write(dict(event='failure',errorType=type(error).__name__,error=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--journal',type=Path,required=True)
    args=parser.parse_args(); run(args.journal)
