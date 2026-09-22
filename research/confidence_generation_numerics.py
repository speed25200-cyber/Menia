"""Numerical controls for retained generation state, without interventions.

Compare a cache with raw-token replay using either the generation's call
schedule or one full prefill. Equality at a matching schedule and differences
caused by changing the schedule are separate questions.
"""
import weakref

import torch
from transformers import DynamicCache

from research.confidence_cached_action_decode import PrefixSnapshot, _cache_hash, _model, _parameters
from research.confidence_prefix_interventions import tensor_hash
from research.iphone_coupling_report import require


def replay_generated_tokens(model, saved, prompt_tokens, *, model_state_id, schedule):
    """Teacher-force the exact raw IDs. Do not regenerate or render their text."""
    _model(model)
    require(isinstance(saved, PrefixSnapshot) and saved.owner() is model, 'Snapshot owner')
    require(saved.model_state_id == model_state_id and saved.parameters == _parameters(model), 'Model identity')
    require(_cache_hash(saved.kv) == saved.cache_hash, 'Changed source cache')
    require(type(prompt_tokens) is int and 0 < prompt_tokens < len(saved.prefix), 'Prompt boundary')
    require(schedule in ('generation', 'full'), 'Replay schedule')
    chunks = [saved.prefix] if schedule == 'full' else [saved.prefix[:prompt_tokens]] + [
        (token,) for token in saved.prefix[prompt_tokens:]]
    cache = DynamicCache(); previous = 0
    for tokens in chunks:
        total = previous + len(tokens)
        ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            output = model(input_ids=ids,
                attention_mask=torch.ones((1, total), dtype=torch.long, device=model.device),
                position_ids=torch.arange(previous, total, device=model.device)[None, :],
                cache_position=torch.arange(previous, total, device=model.device),
                past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        require(isinstance(cache, DynamicCache) and cache.get_seq_length() == total, 'Replay cache length')
        previous = total
    require(previous == len(saved.prefix) and saved.parameters == _parameters(model), 'Replay changed weights')
    require(_cache_hash(saved.kv) == saved.cache_hash, 'Replay changed source cache')
    kv = tuple(tuple(v.detach().clone() for v in pair) for pair in cache.to_legacy_cache())
    metadata = dict(prefixHash=tensor_hash(torch.tensor(saved.prefix, dtype=torch.long)),
        origin='raw-token-replay', schedule=schedule, forwardInputLengths=[len(c) for c in chunks],
        promptTokens=prompt_tokens, textWasRetokenized=False)
    return PrefixSnapshot(saved.prefix, kv, metadata, weakref.ref(model), saved.parameters,
                          model_state_id, _cache_hash(kv))


def compare_caches(reference, candidate):
    """Numerical diagnostics, not a tolerance chosen to obtain a passing result."""
    require(reference.prefix == candidate.prefix and reference.model_state_id == candidate.model_state_id,
            'Same token history and model identity required')
    require(_cache_hash(reference.kv) == reference.cache_hash and
            _cache_hash(candidate.kv) == candidate.cache_hash, 'Changed cache')
    require(len(reference.kv) == len(candidate.kv), 'Cache layer count')
    layers = []
    for layer, (a, b) in enumerate(zip(reference.kv, candidate.kv)):
        require(len(a) == len(b) == 2, 'K/V pairs')
        for name, x, y in zip(('key', 'value'), a, b):
            require(x.shape == y.shape and x.dtype == y.dtype and x.device == y.device, 'Cache tensor identity')
            require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()), 'Nonfinite cache')
            x64 = x.double(); difference = y.double() - x64
            norm = float(x64.norm()); error_norm = float(difference.norm())
            layers.append(dict(layer=layer, kind=name, elements=x.numel(), exact=bool(torch.equal(x, y)),
                maximumAbsoluteDifference=float(difference.abs().max()), absoluteL2=error_norm,
                referenceL2=norm, relativeL2=error_norm/norm if norm else None))
    require(bool(layers), 'Empty cache')
    return dict(exact=all(item['exact'] for item in layers),
        maximumAbsoluteDifference=max(item['maximumAbsoluteDifference'] for item in layers), layers=layers)


def compare_branches(reference, candidate):
    """Keep native decisions separate from conditional candidate probabilities."""
    require(reference['candidateTokenIds'] == candidate['candidateTokenIds'] and
            reference['candidateMeanings'] == candidate['candidateMeanings'], 'Branch code mapping')
    return dict(tokenIdsEqual=reference['tokenIds'] == candidate['tokenIds'],
        bothValidNative=reference['validNativeResponse'] and candidate['validNativeResponse'],
        decisionEqual=reference['decision'] == candidate['decision'],
        maximumCandidateLogitDifference=max(abs(a-b) for a,b in zip(reference['candidateLogits'], candidate['candidateLogits'])),
        conditionalProbabilityDifference=abs(reference['conditionalPositive']-candidate['conditionalPositive']),
        candidateMassDifference=abs(reference['candidateMass']-candidate['candidateMass']))
