"""Random-model control for a prefix patch with no downstream attention path.

An earlier token's final decoder-block OUTPUT cannot change the later token's
logits in this Qwen forward pass. The same site at the final input position can.
This is an architectural control, not an experiment on learned confidence.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
import transformers
from transformers import Qwen3Config,Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3

from research.confidence_prefix_interventions import forward_at_shared_prefix,tensor_hash


def run():
    seed=20260924
    torch.manual_seed(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                      max_position_embeddings=256,eos_token_id=2,pad_token_id=1)
    config._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(config).to(device='cpu',dtype=torch.float32).eval()
    original={k:tensor_hash(v) for k,v in model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    def inputs(ids):
        ids=torch.tensor([ids],dtype=torch.long)
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids))
    prefix=[1,2,3];donor_prefix=[1,6,7];basis=torch.eye(32)[:,:2];rows=[]
    for suffix in ([4,5],[6,7,8],[]):
        for layer in (0,1):
            inp=inputs(prefix+suffix)
            base,recipient=forward_at_shared_prefix(model,inp,prefix,layer)
            _,donor=forward_at_shared_prefix(model,inputs(donor_prefix),donor_prefix,layer)
            patched,event=forward_at_shared_prefix(model,inp,prefix,layer,donor=donor['before'],basis=basis)
            sham,_=forward_at_shared_prefix(model,inp,prefix,layer,donor=recipient['before'],basis=basis)
            assert event['displacementNorm']>0 and event['untouchedTokensEqual'] and torch.equal(base,sham)
            no_path=layer==1 and bool(suffix)
            delta=float((patched-base).abs().max())
            if no_path:assert torch.equal(base,patched), 'Final-block earlier-token patch reached future logits'
            else:assert delta>0, 'Positive propagation control did not move logits'
            rows.append(dict(prefixTokenIds=prefix,suffixTokenIds=suffix,layer=layer,position=event['position'],
                             rank=2,displacementNorm=event['displacementNorm'],logitMaxDifference=delta,
                             structurallyNoPath=no_path,shamLogitsIdentical=True,untouchedTokensEqual=True,
                             beforeHash=tensor_hash(event['before']),afterHash=tensor_hash(event['after'])))
    assert original=={k:tensor_hash(v) for k,v in model.state_dict().items()}
    assert torch.equal(rng,torch.get_rng_state())
    implementation=Path(inspect.getsourcefile(modeling_qwen3)).read_bytes()
    sources={name:Path(__file__).with_name(name).read_text(encoding='utf-8') for name in
             ('confidence_prefix_interventions.py','confidence_prefix_reachability.py')}
    return dict(schema='menia-confidence-prefix-reachability-v1',origin='random_miniature_qwen_cpu',
                seed=seed,torch=torch.__version__,transformers=transformers.__version__,
                config=dict(hiddenSize=32,layers=2,vocabularySize=64,attention='sdpa',dtype='float32',device='cpu'),
                sourceHash=hashlib.sha256(json.dumps(sources,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                qwenImplementationSHA256=hashlib.sha256(implementation).hexdigest(),
                originalWeightHashes=original,weightsUnchanged=True,rngUnchanged=True,cases=rows,
                scope='Arbitrary tokens and random weights. Full forward, no KV cache. Final block output only; not attention/MLP internals. A structural null is not evidence against learned confidence elsewhere. No trained Menia or action-performance result.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report=run();content=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        if args.output.read_text(encoding='utf-8')!=content:raise ValueError('Existing different reachability receipt')
    else:args.output.write_text(content,encoding='utf-8')
    print(json.dumps(dict(origin=report['origin'],cases=[{k:r[k] for k in ('suffixTokenIds','layer','structurallyNoPath','displacementNorm','logitMaxDifference')} for r in report['cases']],weightsUnchanged=True,rngUnchanged=True),indent=2))
