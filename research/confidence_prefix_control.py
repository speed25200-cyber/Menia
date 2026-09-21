"""Reproduce a synthetic CPU control, never a score of trained Menia."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import Qwen3Config,Qwen3ForCausalLM

from research.confidence_prefix_interventions import (
    forward_at_shared_prefix,final_head_null_control,tensor_hash)


def run():
    seed=20260920;torch.manual_seed(seed);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                      max_position_embeddings=256,eos_token_id=2,pad_token_id=1)
    config._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(config).to(device='cpu',dtype=torch.float32).eval()
    initial={k:tensor_hash(v) for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    def inputs(ids):
        t=torch.tensor([ids],dtype=torch.long)
        return dict(input_ids=t,attention_mask=torch.ones_like(t))
    prefix=[1,2,3];a=inputs(prefix+[4,5]);b=inputs(prefix+[6,7,8])
    logits,state=forward_at_shared_prefix(model,a,prefix,0)
    _,other=forward_at_shared_prefix(model,b,prefix,0)
    _,donor=forward_at_shared_prefix(model,inputs([1,6,7]),[1,6,7],0)
    basis=torch.eye(32)[:,:2]
    sham,sham_state=forward_at_shared_prefix(model,a,prefix,0,donor=state['before'],basis=basis)
    altered,altered_state=forward_at_shared_prefix(model,a,prefix,0,donor=donor['before'],basis=basis)
    rows=[]
    for ids in ([1,2,3,4],[1,5,6,7]):
        for shift in (-1.,0.,1.):
            before,after,event=final_head_null_control(model,inputs(ids),3,4,shift)
            rows.append(dict(inputTokenIds=ids,codeTokenIds=[3,4],**event,
                             conditionalBefore=float(torch.softmax(before[[3,4]].double(),0)[1]),
                             conditionalAfter=float(torch.softmax(after[[3,4]].double(),0)[1])))
    unchanged=initial=={k:tensor_hash(v) for k,v in model.state_dict().items()}
    same_rng=bool(torch.equal(rng,torch.get_rng_state()))
    assert unchanged and same_rng and torch.equal(sham,logits)
    difference=float((state['before']-other['before']).abs().max())
    assert difference<1e-6 and altered_state['untouchedTokensEqual']
    assert max(abs(r['actualLogOddsShift']-r['expectedLogOddsShift']) for r in rows)<1e-6
    sources={name:Path(__file__).with_name(name).read_text(encoding='utf-8') for name in
             ('confidence_prefix_interventions.py','confidence_prefix_control.py')}
    return dict(schema='menia-confidence-prefix-synthetic-control-v1',origin='random_miniature_qwen_cpu',
        sourceHash=hashlib.sha256(json.dumps(sources,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        seed=seed,torch=torch.__version__,transformers=transformers.__version__,device='cpu',dtype='float32',
        config=dict(hiddenSize=32,layers=2,vocabularySize=64,attention='sdpa'),
        originalWeightHashes=initial,weightsUnchanged=unchanged,rngUnchanged=same_rng,
        prefix=dict(ids=prefix,layer=0,position=state['position'],rank=2,
                    futureBranchStateMaxDifference=difference,shamLogitMaxDifference=float((sham-logits).abs().max()),
                    donorDisplacementNorm=altered_state['displacementNorm'],untouchedTokensEqual=altered_state['untouchedTokensEqual'],
                    patchedLogitMaxDifference=float((altered-logits).abs().max())),
        headControl=rows,
        scope='Arbitrary token IDs and random weights. No trained Menia, error labels, learned confidence basis, action performance, private self-access or consciousness evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report=run();text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        if args.output.read_text(encoding='utf-8')!=text:raise ValueError('Existing different control receipt')
    else:args.output.write_text(text,encoding='utf-8')
    print(json.dumps(dict(origin=report['origin'],prefix=report['prefix'],weightsUnchanged=report['weightsUnchanged'],
                         rngUnchanged=report['rngUnchanged'],headControl=report['headControl']),indent=2))
