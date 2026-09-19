"""Descriptive weight-space audit; independent Adam zero-gradient formula.

This is an auxiliary check added after training, not a new behavioral endpoint.
Low-rank parameter distances depend on parameterization and are not effect sizes
in behavior or an additive attribution of the carried first moment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def zero_gradient_displacement(moment, second, *, step, steps, lr, betas, eps):
    """Ideal real-valued Adam trajectory, evaluated in float64, not torch.optim."""
    m, v = np.asarray(moment, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if m.shape != v.shape or not np.isfinite(m).all() or not np.isfinite(v).all() or (v < 0).any():
        raise ValueError('Invalid Adam moments')
    b1, b2 = betas
    if not (0 <= b1 < 1 and 0 <= b2 < 1 and step >= 0 and steps >= 0 and eps > 0 and lr > 0):
        raise ValueError('Unsupported Adam settings')
    delta = np.zeros_like(m)
    for k in range(1, steps+1):
        first_corrected = b1**k*m / (1-b1**(step+k))
        second_corrected = b2**k*v / (1-b2**(step+k))
        delta -= lr*first_corrected / (np.sqrt(second_corrected)+eps)
    return delta


def cosine(a, b):
    denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denominator) if denominator else None


def audit(folder, freeze):
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from research.native_localization_gpu import install_adapters
    from research.optimizer_memory_ops import fingerprint
    folder = Path(folder)
    # Parameter order is architecture order, not lexicographic safetensors order
    # (which places layer 10 before layer 2). Meta tensors allocate no weights.
    with torch.device('meta'):
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=8, hidden_size=8, intermediate_size=16,
            num_hidden_layers=36, num_attention_heads=2, num_key_value_heads=1, head_dim=4))
        install_adapters(model, rank=8)
    names = [name for name,p in model.named_parameters() if p.requires_grad]
    assert len(names) == 144
    reports = []
    for rep in range(3):
        checkpoints = {}
        for arm in ('prefix','carry','reset_m','zero_grad'):
            key = f'r{rep}-{arm}'
            path = folder/f'memory-20260919-v1.{key}.safetensors'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == freeze['checkpoints'][key]['sha256']
            tensors = load_file(str(path)); assert set(tensors) == set(names)
            checkpoints[arm] = np.concatenate([tensors[name].double().numpy().ravel() for name in names])
        path = folder/f'memory-20260919-v1.r{rep}-prefix.optimizer.pt'
        reference = freeze['optimizerStates'][f'r{rep}-prefix']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference['fileSHA256']
        state = torch.load(path,map_location='cpu',weights_only=True)
        assert fingerprint(state) == reference['stateSHA256']
        assert len(state['param_groups']) == 1
        group=state['param_groups'][0]
        assert group['weight_decay']==0 and not group['amsgrad'] and not group['maximize']
        assert group['params'] == list(range(144))
        deltas=[]
        for i,name in enumerate(names):
            s=state['state'][i]
            assert s['exp_avg'].shape == tensors[name].shape and s['step'].item()==48
            deltas.append(zero_gradient_displacement(s['exp_avg'].numpy(),s['exp_avg_sq'].numpy(),
                step=48,steps=16,lr=group['lr'],betas=group['betas'],eps=group['eps']).ravel())
        predicted=np.concatenate(deltas)
        actual={arm: checkpoints[arm]-checkpoints['prefix'] for arm in ('carry','reset_m','zero_grad')}
        residual=actual['zero_grad']-predicted
        reports.append(dict(replication=rep, parameters=len(predicted),
            prefixNorm=float(np.linalg.norm(checkpoints['prefix'])),
            displacementNorm={arm:float(np.linalg.norm(value)) for arm,value in actual.items()},
            cosines=dict(carryVsZeroGradient=cosine(actual['carry'],actual['zero_grad']),
                         carryVsReset=cosine(actual['carry'],actual['reset_m'])),
            zeroGradientFormula=dict(predictedNorm=float(np.linalg.norm(predicted)),
                residualNorm=float(np.linalg.norm(residual)),maxAbsoluteResidual=float(np.max(np.abs(residual))),
                relativeResidual=float(np.linalg.norm(residual)/np.linalg.norm(predicted)),
                cosine=cosine(predicted,actual['zero_grad']))))
    return dict(schema='menia-optimizer-displacement-audit-v1', origin=freeze['origin'],
        planHash=freeze['planHash'],sourceHash=freeze['sourceHash'],replications=reports,
        scope='All twelve saved adapters and three optimizer states hash-checked. Float64 closed Adam formula compared with actual FP32 zero-gradient checkpoints; residuals include rounding. Descriptive parameter geometry only, no behavioral or causal-mediation conclusion.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=audit(a.folder,json.loads(a.freeze.read_text(encoding='utf-8')))
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
