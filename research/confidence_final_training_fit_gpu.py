"""Read-only GPU evaluation of all final Colab24 adapters on their old training inputs."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from research import confidence_final_training_fit as study
from research.answer_confidence_gpu import assess_answer
from research.answer_confidence_journal import validate_judgment
from research.confidence_ranking_study import source_hash as parent_source_hash
from research.cross_model_gpu import environment
from research.native_localization_gpu import install_adapters, adapter_state, load_adapter, file_hash
from research.natural_error_journal import Writer


def run(path, checkpoints):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    path = Path(path); checkpoints = Path(checkpoints)
    if path.exists(): raise ValueError('Existing attempt; no automatic restart')
    plan = study.make_plan(); rows = study.training_rows(); by_id = {r['sourceId']:r for r in rows}
    schedule = study.calls(rows,plan)
    if parent_source_hash() != study.PARENT_SOURCE_HASH: raise ValueError('Parent assessment implementation changed')
    prepared = json.loads((study.ROOT/'artifacts/confidence-training-fit-preparation/design.json').read_text(encoding='utf-8'))
    if (prepared['plan'] != plan or prepared['planHash'] != study.digest(plan) or
            prepared['sourceHash'] != study.source_hash() or prepared['callPlanHash'] != study.digest(schedule)):
        raise ValueError('Prepared diagnostic identity changed')
    for unit in plan['units']:
        file = checkpoints/unit['checkpoint']
        if file.stat().st_size != unit['bytes'] or file_hash(file) != unit['sha256']: raise ValueError('Checkpoint identity')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    spec = plan['model']; metadata = environment(); writer = Writer(path)
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'], torch_dtype=torch.bfloat16,
        device_map={'':'cuda:0'}, attn_implementation='sdpa', use_safetensors=True, trust_remote_code=False).eval()
    if model.config._commit_hash != spec['revision']: raise ValueError('Model revision')
    codes = [tokenizer.encode(s, add_special_tokens=False) for s in ('0','1')]
    if any(len(c) != 1 for c in codes) or codes[0] == codes[1]: raise ValueError('Native codes')
    modules = install_adapters(model, rank=plan['rank']); model.requires_grad_(False)
    if any(m.scale != plan['scale'] for m in modules.values()): raise ValueError('Adapter scale')
    if any(p.requires_grad for p in model.parameters()): raise ValueError('Gradients must be disabled')
    adapter_ids = {id(v) for m in modules.values() for v in (m.a,m.b)}
    base_versions = {name:(id(p),p._version) for name,p in model.named_parameters() if id(p) not in adapter_ids}
    metadata.update(model=spec, newWeightUpdates=0, confidenceTokenIds=[c[0] for c in codes], vocabularySize=model.config.vocab_size)
    writer.write(dict(event='header', plan=plan, planHash=study.digest(plan), sourceHash=study.source_hash(),
                      callPlanHash=study.digest(schedule), origin='transformers_gpu', metadata=metadata), create=True)
    collection_complete = False
    try:
        for unit in plan['units']:
            file = checkpoints/unit['checkpoint']; state = load_file(str(file)); load_adapter(modules,state)
            for module in modules.values(): module.enabled = True
            writer.write(dict(event='unit_start', key=unit['key'], sha256=file_hash(file)))
            subset = [c for c in schedule if c['key'] == unit['key']]
            for call in subset:
                request = study.request_for(call, by_id); writer.write(request)
                messages = request['messages']; tick = time.perf_counter()
                scores = assess_answer(model,tokenizer,messages[1]['content'],messages[2]['content'],max_input_tokens=1792)
                event = dict(event='judgment',id=call['id'],status='ok',seconds=time.perf_counter()-tick,scores=scores)
                validate_judgment(event,request,metadata); writer.write(event)
                if (call['id']+1) % 192 == 0: print(f'{call["id"]+1}/5184 judgments recorded',flush=True)
            actual = adapter_state(modules)
            matches = all(torch.equal(actual[name],value) for name,value in state.items())
            unchanged = base_versions == {name:(id(p),p._version) for name,p in model.named_parameters() if id(p) not in adapter_ids}
            if not matches or not unchanged or file_hash(file) != unit['sha256']: raise ValueError('Model state changed during assessment')
            writer.write(dict(event='unit_complete',key=unit['key'],calls=len(subset),sha256=file_hash(file),
                              actualAdapterMatchesFile=matches,baseParameterVersionsUnchanged=unchanged))
        writer.write(dict(event='complete',calls=len(schedule)))
        collection_complete = True
        report = study.summarize(study.read_journal(path))
        with path.with_suffix('.summary.json').open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    except BaseException as error:
        if not collection_complete:
            writer.write(dict(event='failure',errorType=type(error).__name__))
        raise  # Post-collection analysis failures remain in the launcher log/status.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path); parser.add_argument('checkpoints',type=Path)
    args = parser.parse_args(); run(args.journal,args.checkpoints)
