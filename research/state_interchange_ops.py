"""Whole-vector interchange at the last prompt token; no learned alignment."""
import hashlib
import torch

from research.state_composition_gpu import evaluate_forward


def tensor_hash(value):
    data=value.detach().cpu().contiguous()
    return hashlib.sha256(str(data.dtype).encode()+str(tuple(data.shape)).encode()+
                          data.view(torch.uint8).numpy().tobytes()).hexdigest()


def capture_forward(model,inputs,spans,block,task_position,sites):
    """One actual intervention pass; detached copies never alias model output."""
    captures={};handles=[]
    def hook(site):
        def take(module,args,output):
            value=output[0] if isinstance(output,tuple) else output
            if site in captures or value.ndim!=3 or value.shape[0]!=1:
                raise ValueError('Expected one unbatched full-prefix pass')
            captures[site]=value[0,-1].detach().clone()
        return take
    try:
        for site in sites:
            handles.append(model.model.layers[site].register_forward_hook(hook(site)))
        logits,intervention=evaluate_forward(model,inputs,spans,block,'hidden',task_position)
    finally:
        for h in handles:h.remove()
    if set(captures)!=set(sites):raise ValueError('Missing capture')
    return logits,intervention,captures


def patch_forward(model,inputs,spans,block,task_position,site,donor):
    """Replace only the last token's full residual vector at a fixed layer."""
    stats={}
    def patch(module,args,output):
        value=output[0] if isinstance(output,tuple) else output
        if stats or value.ndim!=3 or value.shape[0]!=1:
            raise ValueError('Expected exactly one patch application')
        recipient=value[0,-1]
        if donor.shape!=recipient.shape or donor.dtype!=recipient.dtype or donor.device!=recipient.device:
            raise ValueError('Donor shape, dtype or device differs')
        if not torch.isfinite(donor).all():raise ValueError('Non-finite donor')
        result=value.clone();result[0,-1]=donor
        stats.update(site=site,tokenIndex=value.shape[1]-1,applications=1,
            recipientHash=tensor_hash(recipient),donorHash=tensor_hash(donor),
            patchedHash=tensor_hash(result[0,-1]),
            recipientNorm=float(recipient.float().norm()),donorNorm=float(donor.float().norm()),
            displacementNorm=float((donor.float()-recipient.float()).norm()))
        return (result,*output[1:]) if isinstance(output,tuple) else result
    handle=model.model.layers[site].register_forward_hook(patch)
    try:
        logits,intervention=evaluate_forward(model,inputs,spans,block,'hidden',task_position)
    finally:handle.remove()
    if stats.get('applications')!=1 or stats['donorHash']!=stats['patchedHash']:
        raise ValueError('Patch was not the exact supplied vector')
    return logits,intervention,stats
