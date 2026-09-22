"""Prepared Qwen prefix interventions; no fitted confidence direction or trial.

The shared prefix must be declared before the report/action branch. This module
checks token identity and intervention arithmetic, not the scientific meaning
of the declared prefix or learned basis. It is not part of frozen Colab23.
"""
import hashlib
import math

import torch


def _require(condition, message):
    if not condition:raise ValueError(message)


def _inputs(model, inputs):
    _require(not model.training, 'Evaluation mode required')
    _require(set(inputs)=={'input_ids','attention_mask'}, 'Only unpadded token inputs supported')
    ids=inputs['input_ids'];mask=inputs['attention_mask']
    _require(ids.ndim==2 and ids.shape[0]==1 and ids.shape[1]>0, 'One nonempty sequence required')
    _require(ids.dtype==torch.long and mask.shape==ids.shape and torch.all(mask==1), 'Unpadded token IDs required')
    return ids


def tensor_hash(value):
    t=value.detach().cpu().contiguous()
    return hashlib.sha256(str(t.dtype).encode()+str(tuple(t.shape)).encode()+t.view(torch.uint8).numpy().tobytes()).hexdigest()


def projected_interchange(recipient, donor, basis):
    """Replace only coordinates in a fixed orthonormal column basis."""
    _require(recipient.ndim==1 and donor.shape==recipient.shape, 'State shapes differ')
    _require(recipient.dtype==donor.dtype and recipient.device==donor.device==basis.device, 'State dtype/device differs')
    _require(torch.is_floating_point(recipient) and torch.is_floating_point(basis), 'Floating states and basis required')
    _require(basis.ndim==2 and basis.shape[0]==len(recipient) and 0<basis.shape[1]<len(recipient), 'Proper nonempty subspace required')
    _require(all(torch.isfinite(x).all() for x in (recipient,donor,basis)), 'Nonfinite intervention')
    dtype=torch.float64 if recipient.dtype==torch.float64 or basis.dtype==torch.float64 else torch.float32
    r,d,u=recipient.to(dtype),donor.to(dtype),basis.to(dtype)
    _require(torch.allclose(u.T@u,torch.eye(u.shape[1],device=u.device,dtype=dtype),atol=1e-5,rtol=1e-5), 'Basis must be orthonormal')
    result=(r+u@(u.T@(d-r))).to(recipient.dtype)
    _require(torch.isfinite(result).all(), 'Nonfinite patched state')
    return result


def forward_at_shared_prefix(model, inputs, prefix_ids, layer, *, donor=None, basis=None):
    """Capture or patch one block output at the final declared shared token.

    Full-prefix recomputation, no KV-cache intervention. A donor and basis must
    come from a separately frozen experiment; no labels or fitting occur here.
    """
    ids=_inputs(model,inputs)
    _require(type(prefix_ids) in (list,tuple) and len(prefix_ids)>0 and all(type(i) is int for i in prefix_ids), 'Explicit shared token IDs required')
    _require(len(prefix_ids)<=ids.shape[1] and ids[0,:len(prefix_ids)].tolist()==list(prefix_ids), 'Branch changed the declared prefix')
    _require(type(layer) is int and 0<=layer<len(model.model.layers), 'Layer outside model')
    _require((donor is None)==(basis is None), 'Donor and basis must be supplied together')
    position=len(prefix_ids)-1;captured={}
    def hook(module,args,output):
        value=output[0] if isinstance(output,tuple) else output
        _require(not captured and value.ndim==3 and value.shape[:2]==ids.shape, 'Expected one complete forward pass')
        before=value[0,position].detach().clone()
        after=before.clone() if donor is None else projected_interchange(before,donor,basis)
        captured.update(before=before,after=after.detach().clone(),position=position,layer=layer,
                        prefixHash=tensor_hash(ids[0,:len(prefix_ids)]),
                        rank=0 if donor is None else basis.shape[1],
                        displacementNorm=float((after.double()-before.double()).norm()))
        if donor is None:return None
        changed=value.clone();changed[0,position]=after
        # All other tokens are copied unchanged at this intervention site.
        captured['untouchedTokensEqual']=bool(torch.equal(changed[:,:position],value[:,:position]) and torch.equal(changed[:,position+1:],value[:,position+1:]))
        return (changed,*output[1:]) if isinstance(output,tuple) else changed
    handle=model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].detach().clone()
    finally:handle.remove()
    _require(bool(captured), 'Missing prefix capture')
    return logits,captured


def final_head_null_control(model, inputs, negative_code, positive_code, log_odds_shift):
    """Deliberately steer the reporting head without learning error prediction.

    At the output of final norm, delta = shift*(W1-W0)/||W1-W0||^2 changes
    the binary conditional log odds by shift, up to finite precision. This
    is a negative control for mechanistic interpretation, not a self-model.
    """
    _inputs(model,inputs)
    weights=model.lm_head.weight
    _require(type(negative_code) is int and type(positive_code) is int and
             0<=negative_code<len(weights) and 0<=positive_code<len(weights) and negative_code!=positive_code, 'Distinct vocabulary codes required')
    _require(type(log_odds_shift) in (int,float) and math.isfinite(log_odds_shift), 'Finite shift required')
    difference=(weights[positive_code].detach().double()-weights[negative_code].detach().double())
    norm2=float(difference.square().sum());_require(norm2>0 and math.isfinite(norm2), 'Degenerate output direction')
    displacement=log_odds_shift*difference/norm2;captured={}
    with torch.inference_mode():
        baseline=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].detach().clone()
    def hook(module,args,output):
        _require(not captured and output.ndim==3 and output.shape[0]==1, 'Expected one final norm pass')
        changed=output.clone();changed[0,-1]=(output[0,-1].double()+displacement).to(output.dtype)
        actual=changed[0,-1].double()-output[0,-1].double()
        captured.update(expectedLogOddsShift=float(log_odds_shift),realizedHeadProjection=float(difference@actual),
                        displacementNorm=float(actual.norm()),site='final norm output / last token',
                        scope='Output-direction control only; no error labels or learned self-state')
        return changed
    handle=model.model.norm.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            shifted=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].detach().clone()
    finally:handle.remove()
    _require(bool(captured), 'Missing final-norm intervention')
    captured['actualLogOddsShift']=float((shifted[positive_code].double()-shifted[negative_code].double())-
                                        (baseline[positive_code].double()-baseline[negative_code].double()))
    return baseline,shifted,captured
