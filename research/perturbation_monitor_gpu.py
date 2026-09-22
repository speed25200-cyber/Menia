"""Transient intervention, independent RNG, same prompt and sampling seed per pair."""
import json
from pathlib import Path

import numpy as np

from research.activation_monitor_gpu import ActivationBackend, sample_with_states
from research.cross_model_prediction import SETTINGS, digest
from research.perturbation_monitor import CONDITIONS, analyze, collect


def rotate_vector(vector, strength, seed):
    """Rotate toward a random orthogonal direction while preserving norm in float32.

    Local NumPy RNG never consumes torch's sampling RNG. Float conversion and
    casting incur rounding; the runtime checks the recorded relative norm error.
    """
    import torch
    v = vector.float()
    norm = v.norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError('Invalid residual vector norm')
    noise = torch.as_tensor(np.random.default_rng(seed).standard_normal(v.numel()),device=v.device,dtype=v.dtype)
    noise = noise - (noise @ v)/(v @ v)*v
    n = noise.norm()
    if not torch.isfinite(n) or n <= 1e-10:
        raise ValueError('Degenerate orthogonal direction')
    changed = ((v + strength * norm * noise/n) / np.sqrt(1+strength**2)).to(vector.dtype)
    error = float(abs(changed.float().norm()/norm-1))
    if error > .01 or not torch.isfinite(changed).all():
        raise ValueError('Rotation failed finite/norm check')
    return changed, error


def sample_with_intervention(model,tokenizer,task,capture,*,settings=SETTINGS):
    condition = task['condition']
    if condition not in CONDITIONS:
        raise ValueError('Unknown intervention')
    calls, norm_error, relative_change = 0, 0., 0.
    def hook(module,inputs,output):
        nonlocal calls,norm_error,relative_change
        if calls:
            return None
        calls += 1
        original = output[0] if isinstance(output,tuple) else output
        changed = original.clone()
        if condition != 'sham':
            v = original[0,-1]
            replacement,norm_error = rotate_vector(v,CONDITIONS[condition],task['noiseSeed'])
            relative_change = float((replacement.float()-v.float()).norm()/v.float().norm())
            if relative_change < .1:
                raise ValueError('Intervention did not change the residual')
            changed[0,-1] = replacement
        return (changed,)+output[1:] if isinstance(output,tuple) else changed
    # Registered before the readout hook: middle/final states include intervention.
    handle = None
    try:
        if condition != 'baseline':
            handle = model.model.layers[len(model.model.layers)//2-1].register_forward_hook(hook)
        text,metrics = sample_with_states(model,tokenizer,task,capture,settings=settings)
        if condition != 'baseline' and calls != 1:
            raise ValueError('Intervention did not run exactly once')
        metrics['intervention'] = dict(condition=condition,applications=calls,normRelativeError=norm_error,
                                       relativeChange=relative_change,site='last prefix token, middle block output')
        return text,metrics
    finally:
        if handle is not None:
            handle.remove()


class PerturbationBackend(ActivationBackend):
    def __init__(self):
        super().__init__()
        self.metadata['perturbationSourceHash'] = digest({name:Path(__file__).with_name(name).read_text(encoding='utf-8')
            for name in ('perturbation_monitor.py','perturbation_monitor_gpu.py')})
        self.metadata['intervention'] = dict(conditions=CONDITIONS,position='last prefix token',
            firstPrefillOnly=True,normRelativeTolerance=.01,randomSource='local numpy PCG64, disjoint from sampling')

    def generate(self,task,capture):
        return sample_with_intervention(self.model,self.tokenizer,task,capture)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal',type=Path)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    backend = PerturbationBackend()
    try:
        collect(args.journal,backend,resume=args.resume)
    finally:
        if args.journal.exists():
            args.journal.with_suffix('.summary.json').write_text(json.dumps(analyze(args.journal),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
