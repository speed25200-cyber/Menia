"""Reversible cache permutations for prospective state-consequence experiments.

At a future attention query, jointly permuting all selected prefix K/V pairs
preserves attention in exact arithmetic. Permuting only V breaks the pairing.
Floating-point reduction order can still change a joint-permutation result.
Neither operator supplies an intervention label or new token to the model.
"""
import math

import torch

from research.confidence_cached_action_decode import PrefixSnapshot, _cache_hash, _model, _parameters
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require


def inverse_permutation(permutation):
    require(type(permutation) in (list, tuple) and len(permutation) > 0 and
        all(type(i) is int for i in permutation) and sorted(permutation) == list(range(len(permutation))),
        'Complete permutation required')
    inverse = [0]*len(permutation)
    for output, source in enumerate(permutation): inverse[source] = output
    return inverse


def permute_prefix_cache(model, saved, *, model_state_id, layers, permutation, mode):
    """Return a private transformed snapshot, preserving tokens and source cache.

    A caller must separately establish a task consequence and forecast it from
    independent copies. This operation alone is not a self-monitoring test.
    """
    _model(model)
    require(isinstance(saved, PrefixSnapshot) and saved.owner() is model, 'Snapshot owner')
    require(saved.model_state_id == model_state_id and saved.parameters == _parameters(model), 'Model state changed')
    require(_cache_hash(saved.kv) == saved.cache_hash, 'Source cache changed')
    require(mode in ('values_only', 'keys_and_values'), 'Unknown permutation mode')
    require(type(layers) in (list, tuple) and len(layers) > 0 and
        all(type(i) is int and 0 <= i < len(saved.kv) for i in layers) and
        list(layers) == sorted(set(layers)), 'Sorted distinct cache layers required')
    inverse_permutation(permutation)
    require(len(permutation) == len(saved.prefix), 'Permutation must cover the retained prefix')
    selected = set(layers); changed = 0; squared_displacements = []; squared_references = []
    maximum = 0.; pairs = []
    for layer, pair in enumerate(saved.kv):
        transformed = []
        for kind, before in enumerate(pair):
            require(before.ndim == 4 and before.shape[0] == 1 and before.shape[-2] == len(saved.prefix)
                and bool(torch.isfinite(before).all()), 'Invalid retained cache')
            if layer in selected and (kind == 1 or mode == 'keys_and_values'):
                indices = torch.tensor(permutation, dtype=torch.long, device=before.device)
                after = before.index_select(-2, indices).detach().clone()
            else: after = before.detach().clone()
            delta = after.double()-before.double()
            changed += int(not torch.equal(before, after))
            squared_displacements.append(float(delta.square().sum()))
            squared_references.append(float(before.double().square().sum()))
            maximum = max(maximum, float(delta.abs().max()))
            transformed.append(after)
        pairs.append(tuple(transformed))
    kv = tuple(pairs); result_hash = _cache_hash(kv)
    require(_cache_hash(saved.kv) == saved.cache_hash and saved.parameters == _parameters(model),
        'Source snapshot or model changed during permutation')
    record = dict(schema='menia-prefix-cache-permutation-v1', mode=mode, layers=list(layers),
        permutation=list(permutation), permutationHash=digest(list(permutation)),
        prefixTokens=len(saved.prefix), sourceCacheHash=saved.cache_hash, resultCacheHash=result_hash,
        changedTensors=changed, maximumAbsoluteDisplacement=maximum,
        absoluteL2=math.sqrt(math.fsum(squared_displacements)),
        referenceL2=math.sqrt(math.fsum(squared_references)), tokensUnchanged=True,
        sourceCacheUnchanged=True, parameterVersionsUnchanged=True,
        scope='Reversible experimental state intervention. No task effect, forecast or consciousness claim.')
    capture = dict(saved.capture, cacheIntervention=record)
    transformed = PrefixSnapshot(saved.prefix, kv, capture, saved.owner, saved.parameters,
        saved.model_state_id, result_hash)
    return transformed, record
