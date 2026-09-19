"""Read frozen parents and composition adapters; diagnose without changing weights."""
import json
import os
from pathlib import Path
import time

import torch

from research import composition_diagnostic as study
from research.state_composition_gpu import encode, evaluate_forward
from research.native_localization_gpu import install_adapters, load_adapter, summarize_logits, append, file_hash


def checkpoint_path(parent, composition, key):
    return (Path(parent).with_suffix('.'+key.replace('-parent', '-strong')+'.safetensors')
            if key.endswith('-parent') else Path(composition).with_suffix('.'+key+'.safetensors'))


def verify_sources(parent, composition):
    if file_hash(parent) != study.PARENT_SHA256 or file_hash(composition) != study.COMPOSITION_SHA256:
        raise ValueError('Source journal hash mismatch')
    frozen = study.checkpoints()
    for key, sha in frozen.items():
        if file_hash(checkpoint_path(parent, composition, key)) != sha:
            raise ValueError('Frozen checkpoint mismatch: '+key)
    return frozen


def run(path, parent, composition, *, resume=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    from research.cross_model_gpu import environment
    path, parent, composition = map(Path, (path, parent, composition))
    if path.exists() and not resume: raise ValueError('Existing journal requires --resume')
    frozen = verify_sources(parent, composition)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    metadata = environment()
    metadata.update(origin='transformers_gpu', models={'A': study.MODEL},
        settings=dict(dtype='bfloat16', attention='sdpa', thinking=False, useCache=False, sampling=False,
                      firstTokenOnly=True, training=False, **study.CONFIG))
    tok = AutoTokenizer.from_pretrained(study.MODEL['id'], revision=study.MODEL['revision'], trust_remote_code=False)
    ids = [tok.encode(str(i), add_special_tokens=False) for i in range(4)]
    if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != 4: raise ValueError('Invalid digits')
    choices = [x[0] for x in ids]; metadata['choiceTokenIds'] = choices; fixed = study.plan()
    if path.exists():
        h, rows, pending, _ = study.read_journal(path)
        if h['metadata'] != metadata: raise ValueError('Environment changed')
        if len(rows) == len(fixed['evaluation']) and pending is None:
            # A previous stop can occur between the last result and summary writing.
            if not path.with_suffix('.summary.json').exists(): write_summary(path)
            print('Already complete; no inference repeated.', flush=True); return
        if pending: append(path, 'interrupted_request', id=pending['request']['id'])
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        append(path, 'header', plan=fixed, planHash=study.digest(fixed), sourceHash=study.source_hash(), metadata=metadata)
        rows = []
    model = AutoModelForCausalLM.from_pretrained(study.MODEL['id'], revision=study.MODEL['revision'],
        torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa',
        use_safetensors=True, trust_remote_code=False).eval()
    if model.config._commit_hash != study.MODEL['revision'] or len(model.model.layers) != 36:
        raise ValueError('Unexpected architecture')
    modules = install_adapters(model, rank=study.CONFIG['rank']); model.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()): raise ValueError('Trainable weights in frozen diagnostic')
    blocks = {b['id']: b for b in fixed['blocks']}; current = None; cache = {}
    for index, req in enumerate(fixed['evaluation'][len(rows):], len(rows)+1):
        key = f"r{req['replication']}-{req['arm']}"
        if current != key:
            for m in modules.values(): m.enabled = req['arm'] != 'base'
            if req['arm'] != 'base':
                checkpoint = checkpoint_path(parent, composition, key)
                if file_hash(checkpoint) != frozen[key]: raise ValueError('Checkpoint changed')
                load_adapter(modules, load_file(str(checkpoint)))
            current = key
        b = blocks[req['block']]
        ck = (b['id'], req['family'], req['task'], req['position'], req['format'], req['mapping'])
        if ck not in cache:
            cache[ck] = encode(tok, b, req['family'], req['task'], req['position'], model.device, req['format'], req['mapping'])
        inputs, spans, ph = cache[ck]
        append(path, 'request', request=req, promptHash=ph, adapterHash=frozen.get(key), inputTokens=int(inputs.input_ids.shape[-1]))
        torch.cuda.synchronize(); start = time.perf_counter()
        with torch.inference_mode():
            logits, trace = evaluate_forward(model, inputs, spans, b, req['family'], req['position'])
            result = summarize_logits(logits, choices)
        torch.cuda.synchronize()
        append(path, 'result', id=req['id'], seconds=time.perf_counter()-start, intervention=trace, **result)
        if index % 256 == 0: print(f'Evaluation {index}/{len(fixed["evaluation"])}', flush=True)
    verify_sources(parent, composition)
    write_summary(path)


def write_summary(path):
    path.with_suffix('.summary.json').write_text(json.dumps(study.analyze(path), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('journal', type=Path)
    p.add_argument('--parent-journal', type=Path, required=True)
    p.add_argument('--composition-journal', type=Path, required=True)
    p.add_argument('--resume', action='store_true'); a = p.parse_args()
    try: run(a.journal, a.parent_journal, a.composition_journal, resume=a.resume)
    except BaseException as exc:
        if a.journal.exists(): append(a.journal, 'error', errorType=type(exc).__name__, message=str(exc))
        raise
