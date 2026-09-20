"""Retain the actual KV trajectory of one Qwen generation, without text replay.

Generation leaves its last sampled token unconsumed in the cache. Consume that
token once, after the optional generation-only intervention has ended, before
forking subsequent queries. Raw prompt/completion token IDs are never replaced
with IDs obtained by decoding and re-rendering the answer.
"""
from contextlib import nullcontext
import weakref

import torch
from transformers import DynamicCache

from research.confidence_cached_action_decode import PrefixSnapshot, _cache_hash, _model, _parameters
from research.confidence_prefix_interventions import tensor_hash
from research.cross_model_gpu import generate_text
from research.cross_model_prediction import SETTINGS, digest
from research.iphone_coupling_report import require


class _ObservedTokenizer:
    def __init__(self,original,prompt_budget):
        self.original = original; self.prompt_budget = prompt_budget
        self.prompt_ids = None; self.generated_ids = None

    def __getattr__(self,key): return getattr(self.original,key)

    def __call__(self,*args,**kwargs):
        require(self.prompt_ids is None, 'One tokenizer input call required')
        encoded = self.original(*args,**kwargs); ids = encoded['input_ids']
        require(ids.ndim == 2 and ids.shape[0] == 1 and ids.shape[1] <= self.prompt_budget, 'Context budget or batch')
        require(bool(torch.all(encoded['attention_mask'] == 1)), 'Unpadded prompt required')
        self.prompt_ids = ids[0].detach().cpu().tolist()
        return encoded

    def decode(self,ids,*args,**kwargs):
        require(self.generated_ids is None and ids.ndim == 1, 'One completion decode required')
        self.generated_ids = ids.detach().cpu().tolist()
        return self.original.decode(ids,*args,**kwargs)


def capture_generated_prefix(model,tokenizer,messages,seed,*,model_state_id,settings=SETTINGS,
                             max_context_tokens=2048,generation_context=None):
    """Return text, original generation metrics, trajectory metadata and a snapshot.

generation_context, when supplied, is a context manager whose interventions
end before consuming the final token. No unrecorded intervention may remain
active during the closing forward or later branches. Callers freeze weights,
configuration and adapter selection as for prefill_prefix. Completed generation
is required; EOS can be any configured stop token, which is retained explicitly.
The snapshot alone does not assert that this EOS closes a valid chat turn.
"""
    _model(model)
    require(isinstance(model_state_id,str) and bool(model_state_id.strip()), 'Explicit model state ID required')
    require(settings['use_cache'] is True, 'Cached generation required')
    require(type(max_context_tokens) is int and type(settings['max_new_tokens']) is int and
            0 < settings['max_new_tokens'] < max_context_tokens, 'Context budget')
    versions = _parameters(model); observed = _ObservedTokenizer(tokenizer,max_context_tokens-settings['max_new_tokens'])
    processed = []; lengths = []; last_cache = None

    def capture(module,args,kwargs,result):
        nonlocal last_cache
        ids = kwargs.get('input_ids')
        require(ids is not None and ids.ndim == 2 and ids.shape[0] == 1, 'One token-ID sequence required')
        require(isinstance(result.past_key_values,DynamicCache), 'Dynamic cache required')
        processed.extend(ids[0].detach().cpu().tolist())
        lengths.append(ids.shape[1]); last_cache = result.past_key_values

    handle = model.register_forward_hook(capture,with_kwargs=True)
    try:
        with generation_context if generation_context is not None else nullcontext():
            text,metrics = generate_text(model,observed,messages,seed,settings=settings)
    finally:
        handle.remove()
    require(_parameters(model) == versions, 'Generation changed model parameters')
    prompt = observed.prompt_ids; generated = observed.generated_ids
    eos = model.generation_config.eos_token_id; eos = [eos] if type(eos) is int else eos
    require(type(eos) is list and tokenizer.eos_token_id in eos, 'EOS configuration')
    require(prompt is not None and generated is not None and len(generated)>0 and generated[-1] in eos, 'Completed EOS-terminated generation required')
    require(len(generated) == metrics['outputTokens'] == len(lengths) and metrics['inputTokens'] == len(prompt), 'Generation trace count')
    require(lengths == [len(prompt)]+[1]*(len(generated)-1) and processed == prompt+generated[:-1], 'Unexpected cache trajectory')
    require(last_cache is not None and last_cache.get_seq_length() == len(processed), 'Last sampled token must be unconsumed')
    before = tuple(tuple(v.detach().clone() for v in pair) for pair in last_cache.to_legacy_cache())
    before_hash = _cache_hash(before)
    cache = DynamicCache.from_legacy_cache(tuple(tuple(v.clone() for v in pair) for pair in before))
    prefix = prompt+generated; previous = len(processed)
    ids = torch.tensor([[generated[-1]]],dtype=torch.long,device=model.device)
    with torch.inference_mode():
        result = model(input_ids=ids,attention_mask=torch.ones((1,len(prefix)),dtype=torch.long,device=model.device),
            position_ids=torch.tensor([[previous]],device=model.device),cache_position=torch.tensor([previous],device=model.device),
            past_key_values=cache,use_cache=True,logits_to_keep=1)
    require(_parameters(model) == versions and _cache_hash(before) == before_hash, 'Closing forward changed original state')
    require(result.past_key_values.get_seq_length() == len(prefix), 'Completed prefix length')
    kv = tuple(tuple(v.detach().clone() for v in pair) for pair in result.past_key_values.to_legacy_cache())
    require(len(kv) == len(model.model.layers) and all(len(pair)==2 and all(
        v.ndim==4 and v.shape[0]==1 and v.shape[-2]==len(prefix) and bool(torch.isfinite(v).all()) for v in pair) for pair in kv), 'Invalid retained cache')
    capture_metadata = dict(prefixHash=tensor_hash(torch.tensor(prefix,dtype=torch.long)),
        origin='actual-generation',promptTokens=len(prompt),generatedTokens=len(generated),
        finalTokenConsumedOnce=True,generationInterventionEndedBeforeFinalToken=True)
    saved = PrefixSnapshot(tuple(prefix),kv,capture_metadata,weakref.ref(model),versions,model_state_id,_cache_hash(kv))
    trace = dict(schema='menia-generated-prefix-continuity-v1',promptTokenIds=prompt,generatedTokenIds=generated,
        promptHash=digest(prompt),generatedHash=digest(generated),generationForwardInputLengths=lengths,
        cacheTokensBeforeFinalToken=previous,cacheTokensAfterFinalToken=len(prefix),finalTokenId=generated[-1],
        eosTokenIds=list(eos),snapshotCacheHash=saved.cache_hash,modelStateId=model_state_id,
        textWasRetokenized=False,scope='Actual generation cache plus one unconsumed terminal token, processed after generation-only interventions end. Not a self-monitoring result or a guarantee of valid chat framing.')
    return text,metrics,trace,saved


def compile_raw_continuations(tokenizer,prefix_ids,queries):
    """Append Qwen user queries to caller-supplied raw, im_end-terminated IDs.

Retain the generation's raw thinking markers and every emitted token. A second
EOS such as endoftext is not silently converted to im_end. This changes the
    history relative to the answer-text-reconstructed compile_fork by design.
    The caller provides saved.prefix from the actual capture, or explicitly
    identifies a synthetic token fixture. This compiler cannot attest provenance.
    """
    require(type(prefix_ids) in (list,tuple) and bool(prefix_ids) and all(type(i) is int and i>=0 for i in prefix_ids), 'Nonempty raw token IDs required')
    require(tokenizer.encode('<|im_end|>',add_special_tokens=False) == [tokenizer.eos_token_id] and
            prefix_ids[-1] == tokenizer.eos_token_id, 'Actual im_end chat closure required')
    require(type(queries) is dict and len(queries)>=2 and len(set(queries.values()))==len(queries) and
            all(type(k) is str and k and type(v) is str and v for k,v in queries.items()), 'Distinct named queries required')
    result = {}
    for name,query in queries.items():
        require(not any(m in query for m in ('<|im_start|>','<|im_end|>','<tool_response>','</tool_response>')), 'Reserved framing marker')
        rendered = tokenizer.apply_chat_template([dict(role='user',content=query)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        require(rendered.startswith('<|im_start|>user\n'+query+'<|im_end|>\n<|im_start|>assistant\n'), 'Unsupported Qwen continuation framing')
        suffix = tokenizer.encode('\n'+rendered,add_special_tokens=False)
        require(bool(suffix), 'Empty continuation')
        result[name] = dict(inputIds=list(prefix_ids)+suffix,query=query,suffixText='\n'+rendered,
            originalPrefixTokenHash=digest(prefix_ids),prefixTokens=len(prefix_ids),prefixRetokenized=False)
    return result
