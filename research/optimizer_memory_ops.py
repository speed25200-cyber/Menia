"""Controlled AdamW state interventions; these are optimizer diagnostics, not cognition."""
import copy
import hashlib
import json
import math

import torch


def fingerprint(state):
    h = hashlib.sha256(json.dumps(state['param_groups'],sort_keys=True,separators=(',',':')).encode())
    for key, values in sorted(state['state'].items()):
        for name, value in sorted(values.items()):
            h.update(f'{key}/{name}:'.encode())
            if isinstance(value,torch.Tensor):
                cpu=value.detach().cpu().contiguous()
                h.update(str((str(cpu.dtype),tuple(cpu.shape))).encode()); h.update(cpu.numpy().tobytes())
            else:h.update(json.dumps(value,sort_keys=True).encode())
    return h.hexdigest()


def moment_summary(optimizer):
    states=list(optimizer.state.values())
    if not states:raise ValueError('An initialized AdamW state is required')
    if any(set(s)!={'step','exp_avg','exp_avg_sq'} for s in states):
        raise ValueError('Only ordinary AdamW without AMSGrad is supported')
    norms={name:math.sqrt(math.fsum(float(s[name].double().square().sum()) for s in states))
           for name in ('exp_avg','exp_avg_sq')}
    steps=[int(s['step'].item()) for s in states]
    return dict(firstMomentNorm=norms['exp_avg'],secondMomentNorm=norms['exp_avg_sq'],
                minStep=min(steps),maxStep=max(steps),states=len(states))


def fork_optimizer(parameters,state,*,reset_first=False):
    """Restore a private copy; a reset changes only exp_avg, preserving step and exp_avg_sq."""
    if len(state['param_groups'])!=1 or state['param_groups'][0]['weight_decay']!=0:
        raise ValueError('Expected one group and no weight decay')
    optimizer=torch.optim.AdamW(parameters,lr=state['param_groups'][0]['lr'],weight_decay=0.)
    optimizer.load_state_dict(copy.deepcopy(state))
    before=moment_summary(optimizer)
    if reset_first:
        with torch.no_grad():
            for s in optimizer.state.values():s['exp_avg'].zero_()
    after=moment_summary(optimizer)
    if any(before[k]!=after[k] for k in ('secondMomentNorm','minStep','maxStep','states')):
        raise ValueError('Reset altered more than the first moment')
    return optimizer,dict(before=before,after=after,sourceStateHash=fingerprint(state),
                          restoredStateHash=fingerprint(optimizer.state_dict()))


def zero_gradient_step(optimizer,parameters):
    """Dense zero gradients still advance Adam moments; None gradients would skip updates."""
    optimizer.zero_grad(set_to_none=True)
    for parameter in parameters:parameter.grad=torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
