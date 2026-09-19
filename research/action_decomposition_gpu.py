"""One prospective diagnostic with six frozen adapters and exact replay controls."""
import argparse
import json
import os
from pathlib import Path
import time

from research import action_decomposition as study
from research.cross_model_gpu import environment, generate_text
from research.natural_error_journal import Writer
from research.native_choice_journal import validate_result


def run(path, parent_path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from safetensors.torch import load_file
    from research.native_localization_gpu import install_adapters, load_adapter, file_hash
    path=Path(path); parent_path=Path(parent_path)
    if path.exists(): raise ValueError('Attempt already exists; no retry')
    plan=study.make_plan(); prior=study.load_parent(parent_path,plan)
    for name,sha in plan['weightFiles'].items():
        if file_hash(parent_path.parent/name)!=sha: raise ValueError('Parent checkpoint changed')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'; os.environ['TOKENIZERS_PARALLELISM']='false'
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    metadata=environment(); metadata.update(models={'A':plan['model']},settings=plan['settings'],newWeightUpdates=0)
    spec=plan['model']; writer=Writer(path)
    writer.write(dict(event='header',plan=plan,planHash=study.digest(plan),sourceHash=study.source_hash(),origin='transformers_gpu',metadata=metadata),create=True)
    try:
        tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(spec['id'],revision=spec['revision'],torch_dtype=torch.bfloat16,
                device_map={'':'cuda:0'},attn_implementation='sdpa',use_safetensors=True,trust_remote_code=False).eval()
        if model.config._commit_hash!=spec['revision']: raise ValueError('Model revision')
        modules=install_adapters(model,rank=8); model.requires_grad_(False); current=None
        for c in plan['calls']:
            key=(c['replication'],c['arm'])
            if key!=current:
                for m in modules.values():m.enabled=c['arm']!='base'
                if c['arm']!='base':
                    name=f'action-binding-20260919-v1.r{c["replication"]}-{c["arm"]}.safetensors'
                    if file_hash(parent_path.parent/name)!=plan['weightFiles'][name]:raise ValueError('Adapter changed')
                    load_adapter(modules,load_file(str(parent_path.parent/name)))
                current=key
            req=study.request_for(plan,c['id']); writer.write(req); started=time.perf_counter()
            text,metrics=generate_text(model,tokenizer,req['messages'],c['seed'],settings=plan['settings'])
            result=dict(event='result',id=c['id'],engine='llm',status='ok',text=text,metrics=metrics,
                        seconds=time.perf_counter()-started,errorType=None)
            validate_result(result,req); writer.write(result)
            if c['stage']=='replay' and text!=prior['results'][c['parentId']]['text']:
                raise ValueError('Exact parent replay failed; preserve this attempt and stop before diagnostics')
            if (c['id']+1)%48==0: print(f'{c["id"]+1}/{plan["planned"]} recorded',flush=True)
        for name,sha in plan['weightFiles'].items():
            if file_hash(parent_path.parent/name)!=sha: raise ValueError('Checkpoint changed after inference')
        report=study.analyze(path,parent_path)
        path.with_suffix('.summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    except BaseException as error:
        writer.write(dict(event='failure',errorType=type(error).__name__)); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path); parser.add_argument('parent',type=Path)
    args=parser.parse_args();run(args.journal,args.parent)
