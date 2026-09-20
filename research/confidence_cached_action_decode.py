"""Fork native decisions from a retained Qwen prefix, with fixed model weights.

The optional intervention occurs once, during prefill. Each continuation gets
private KV copies; neither confidence tokens nor another branch's tokens enter
its context. This is instrumentation, not evidence of self-monitoring.
"""
from dataclasses import dataclass
import hashlib
import weakref

import torch
from transformers import DynamicCache

from research.answer_confidence_data import confidence_from_logits
from research.confidence_action_decode import parse_native_tokens
from research.confidence_prefix_interventions import projected_interchange, tensor_hash
from research.iphone_coupling_report import require


def _parameters(model):
    # Detect ordinary registered-parameter replacement/update. This is not a
    # cryptographic attestation and cannot detect arbitrary .data/hook changes.
    return tuple((name, id(p), p._version, str(p.device), str(p.dtype), tuple(p.shape))
                 for name, p in model.named_parameters())


def _cache_hash(kv):
    return hashlib.sha256(''.join(tensor_hash(v) for pair in kv for v in pair).encode()).hexdigest()


def _model(model):
    require(not model.training, 'Evaluation mode required')
    require(model.config.model_type == 'qwen3', 'Qwen3 cache layout required')
    require(all(t == 'full_attention' for t in model.config.layer_types), 'Full attention only')
    require(len({p.device for p in model.parameters()}) == 1, 'One model device required')


@dataclass(frozen=True)
class PrefixSnapshot:
    prefix: tuple
    kv: tuple
    capture: dict
    owner: object
    parameters: tuple
    model_state_id: str
    cache_hash: str


def prefill_prefix(model, prefix_ids, layer, *, model_state_id, donor=None, basis=None):
    """Capture before either query, optionally replacing one residual subspace.

    model_state_id is a caller-declared checkpoint/adapter identity. Callers
    must keep weights, adapter selection, model config and hooks fixed for all
    forks. The parameter version guard alone does not establish that identity.
    """
    _model(model)
    require(isinstance(model_state_id, str) and bool(model_state_id.strip()), 'Explicit model state ID required')
    require(type(prefix_ids) in (list, tuple) and len(prefix_ids) > 0 and
            all(type(i) is int and 0 <= i < model.config.vocab_size for i in prefix_ids), 'Nonempty vocabulary IDs required')
    require(type(layer) is int and 0 <= layer < len(model.model.layers), 'Layer outside model')
    require((donor is None) == (basis is None), 'Donor and basis must be supplied together')
    versions = _parameters(model)
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=model.device)
    capture = {}

    def hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        require(not capture and value.ndim == 3 and value.shape[:2] == ids.shape, 'One complete prefill required')
        before = value[0, -1].detach().clone()
        after = before.clone() if donor is None else projected_interchange(before, donor, basis)
        capture.update(before=before, after=after.detach().clone(), layer=layer, position=len(prefix_ids)-1,
                       rank=0 if donor is None else basis.shape[1], prefixHash=tensor_hash(ids[0]),
                       displacementNorm=float((after.double()-before.double()).norm()))
        if donor is None:
            capture['untouchedTokensEqual'] = True
            return None
        changed = value.clone(); changed[0, -1] = after
        capture['untouchedTokensEqual'] = bool(torch.equal(changed[:, :-1], value[:, :-1]))
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1)
    finally:
        handle.remove()
    require(bool(capture) and _parameters(model) == versions, 'Missing capture or changed model')
    require(isinstance(result.past_key_values, DynamicCache), 'Dynamic KV cache required')
    kv = tuple(tuple(v.detach().clone() for v in pair) for pair in result.past_key_values.to_legacy_cache())
    require(len(kv) == len(model.model.layers) and all(len(pair) == 2 for pair in kv), 'One K/V pair per layer required')
    require(all(v.ndim == 4 and v.shape[0] == 1 and v.shape[-2] == len(prefix_ids) and
                bool(torch.isfinite(v).all()) for pair in kv for v in pair), 'Invalid prefix cache')
    return PrefixSnapshot(tuple(prefix_ids), kv, capture, weakref.ref(model), versions, model_state_id, _cache_hash(kv))


def decode_cached_branch(model, tokenizer, branch, saved, *, model_state_id,
                         max_new_tokens=2, max_input_tokens=1792):
    """Generate code/EOS from a private copy of a previously captured prefix."""
    _model(model)
    require(isinstance(saved, PrefixSnapshot) and saved.owner() is model, 'Snapshot belongs to another model')
    require(saved.model_state_id == model_state_id and saved.parameters == _parameters(model), 'Snapshot model state changed')
    require(_cache_hash(saved.kv) == saved.cache_hash, 'Snapshot cache was modified')
    require(type(max_new_tokens) is int and max_new_tokens == 2, 'Fixed code/EOS budget required')
    require(type(max_input_tokens) is int and max_input_tokens > 0, 'Positive input budget')
    original = list(branch['inputIds']); length = len(saved.prefix)
    require(all(type(i) is int and 0 <= i < model.config.vocab_size for i in original), 'Vocabulary IDs required')
    require(original[:length] == list(saved.prefix) and len(original) > length, 'Pre-query shared prefix required')
    require(len(original)+1 <= max_input_tokens, 'Decoding would exceed budget; no truncation')
    candidates = branch['candidateTokenIds']; meanings = [branch['negative'], branch['positive']]
    require(all(type(c) is int and 0 <= c < model.config.vocab_size for c in candidates), 'Candidate vocabulary IDs')
    eos = model.generation_config.eos_token_id
    eos = [eos] if type(eos) is int else eos
    require(type(eos) is list and tokenizer.eos_token_id in eos, 'Configured EOS must include tokenizer EOS')
    parse_native_tokens([], candidates, meanings, eos)
    cache = DynamicCache.from_legacy_cache(tuple(tuple(v.clone() for v in pair) for pair in saved.kv))
    decoded = []; passes = []; first_scores = None; first_logits = None
    for step in range(max_new_tokens):
        tokens = original[length:] if step == 0 else [decoded[-1]]
        previous = cache.get_seq_length(); total = previous+len(tokens)
        ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=torch.ones((1, total), dtype=torch.long, device=model.device),
                           position_ids=torch.arange(previous, total, device=model.device)[None, :],
                           cache_position=torch.arange(previous, total, device=model.device),
                           past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = result.past_key_values
        require(cache.get_seq_length() == total, 'Unexpected cache length')
        logits = result.logits[0, -1].detach()
        require(bool(torch.isfinite(logits).all()), 'Nonfinite vocabulary logits')
        if step == 0:
            values = logits.double().cpu().tolist()
            first_scores = confidence_from_logits(values, candidates[0], candidates[1])
            first_logits = [float(values[c]) for c in candidates]
        decoded.append(int(logits.argmax()))
        passes.append(dict(step=step, inputTokens=len(tokens), cacheTokensBefore=previous, cacheTokensAfter=total))
        if decoded[-1] in eos:
            break
    require(_cache_hash(saved.kv) == saved.cache_hash and saved.parameters == _parameters(model), 'Snapshot or model changed')
    parsed = parse_native_tokens(decoded, candidates, meanings, eos)
    return dict(schema='menia-cached-native-branch-greedy-decode-v1', tokenIds=decoded,
        text=tokenizer.decode(decoded, skip_special_tokens=True), stoppedAtEOS=decoded[-1] in eos,
        eosTokenIds=list(eos), reachedTokenLimit=len(decoded) == 2 and decoded[-1] not in eos,
        **parsed, candidateTokenIds=list(candidates), candidateMeanings=meanings,
        conditionalPositive=first_scores['conditionalCorrect'], candidateMass=first_scores['candidateMass'],
        candidateLogits=first_logits, firstTopIsCode=decoded[0] in candidates, passes=passes,
        prefixHash=saved.capture['prefixHash'], snapshotCacheHash=saved.cache_hash, modelStateId=model_state_id,
        snapshotUnchanged=True, settings=dict(do_sample=False, use_cache=True, candidateRestriction=False,
                                            max_new_tokens=2, interventionDuringBranch=False),
        scope='Native decoding from private copies of one retained prefix. No confidence report is inserted '
              'into an action branch. This does not establish self-monitoring or measured task utility.')
