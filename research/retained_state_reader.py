"""An explicit frozen producer / suffix-adapted reader, following aLoRA's boundary.

The producer creates the historical KV states. Only the reader's query suffix
and generated report use its trainable Q/V adapters. This is not a full-prefix
LoRA, and the reader is not claimed to have generated the retained history.
"""
import copy
import weakref

import torch
from transformers import DynamicCache

from research.confidence_cached_action_decode import (
    PrefixSnapshot, _cache_hash, _model, _parameters, decode_cached_branch,
)
from research.confidence_prefix_interventions import tensor_hash
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require
from research.native_localization_gpu import LowRank


class RetainedStateReader:
    """Own a separate reader, leaving the producer and its version guards intact."""

    def __init__(self, producer, *, producer_state_id, rank=32, scale=1.0, seed=2026092070):
        _model(producer)
        require(not any(p.requires_grad for p in producer.parameters()), 'Frozen producer required')
        require(type(rank) is int and rank > 0 and scale > 0, 'Positive adapter capacity')
        require(isinstance(producer_state_id, str) and producer_state_id.strip(), 'Producer identity required')
        require(not any(m._forward_hooks or m._forward_pre_hooks for m in producer.modules()), 'Unhooked producer required')
        self.producer = weakref.ref(producer)
        self.producer_state_id = producer_state_id
        self.producer_versions = _parameters(producer)
        self.model = copy.deepcopy(producer).eval()
        require(all(na == nb and torch.equal(a, b) for (na,a),(nb,b)
                    in zip(producer.named_parameters(), self.model.named_parameters())), 'Exact initial backbone copy')
        self.modules = {}
        devices = sorted({p.device.index for p in producer.parameters() if p.device.type == 'cuda'})
        # Adapter initialization must not change subsequent producer sampling.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for name, module in list(self.model.named_modules()):
                if name.endswith(('.q_proj', '.v_proj')) and isinstance(module, torch.nn.Linear):
                    parent, field = name.rsplit('.', 1)
                    replacement = LowRank(module, rank, scale)
                    setattr(self.model.get_submodule(parent), field, replacement)
                    self.modules[name] = replacement
        require(len(self.modules) == 2*len(producer.model.layers), 'All Q/V projections required')
        self.model.eval()
        self.adapter_parameters = tuple(p for m in self.modules.values() for p in (m.a, m.b))
        self._adapter_ids = {id(p) for p in self.adapter_parameters}
        self.backbone_versions = self._backbone_versions()
        self.config = dict(rank=rank, scale=scale, seed=seed, targets=['q_proj','v_proj'],
                           activation='first token after the retained prefix; entire query suffix and report')
        self._cached_adapter_versions = None
        self._cached_adapter_hash = None

    def _backbone_versions(self):
        return tuple(row for row in _parameters(self.model) if row[1] not in self._adapter_ids)

    def guard(self, saved):
        producer = self.producer()
        require(producer is not None and saved.owner() is producer, 'Original producer required')
        _model(producer); _model(self.model)
        require(saved.model_state_id == self.producer_state_id and
                saved.parameters == self.producer_versions == _parameters(producer), 'Producer changed')
        require(self._backbone_versions() == self.backbone_versions, 'Reader backbone changed')
        require({id(p) for p in self.model.parameters() if p.requires_grad} == self._adapter_ids,
                'Only declared adapters may learn')
        require(all(m.enabled and m.scale == self.config['scale'] for m in self.modules.values()), 'Adapter mode changed')
        require(not any(m._forward_hooks or m._forward_pre_hooks for m in self.model.modules()), 'Unrecorded reader hook')
        require(not any(m._forward_hooks or m._forward_pre_hooks for m in producer.modules()), 'Unrecorded producer hook')
        require(_cache_hash(saved.kv) == saved.cache_hash, 'Producer cache changed')

    def adapter_state(self):
        return {name+'.'+part: getattr(m,part).detach().cpu().clone().contiguous()
                for name,m in self.modules.items() for part in ('a','b')}

    def adapter_hash(self):
        versions = tuple((id(p), p._version, str(p.device), str(p.dtype)) for p in self.adapter_parameters)
        if versions != self._cached_adapter_versions:
            self._cached_adapter_hash = digest({name:tensor_hash(t) for name,t in self.adapter_state().items()})
            self._cached_adapter_versions = versions
        return self._cached_adapter_hash

    def load_adapter(self, state):
        expected = {name+'.'+part:getattr(m,part) for name,m in self.modules.items() for part in ('a','b')}
        require(state.keys() == expected.keys(), 'Adapter tensor names')
        require(all(state[n].shape == p.shape and state[n].dtype == torch.float32 and
                    bool(torch.isfinite(state[n]).all()) for n,p in expected.items()), 'Finite FP32 adapter tensors with exact shapes')
        with torch.no_grad():
            for name, parameter in expected.items(): parameter.copy_(state[name])

    def _suffix(self, saved, branch, max_input_tokens):
        self.guard(saved)
        ids = list(branch['inputIds']); length = len(saved.prefix)
        require(ids[:length] == list(saved.prefix) and len(ids) > length, 'Exact producer history required')
        require(all(type(t) is int and 0 <= t < self.model.config.vocab_size for t in ids), 'Vocabulary IDs')
        require(len(ids)+1 <= max_input_tokens, 'No truncation at the adapter boundary')
        return ids[length:]

    def teacher_forced_logits(self, saved, branch, target, *, max_input_tokens=2048):
        """Two causal positions: predict target, then EOS after seeing target.

        Loss labels are appended only in this training method. Prediction calls
        accept no label. Historical tensors are detached private ordinary tensors,
        even when the source snapshot was captured under inference_mode.
        """
        suffix = self._suffix(saved, branch, max_input_tokens)
        require(type(target) is int and target in branch['candidateTokenIds'], 'Supervised code required')
        with torch.inference_mode(False), torch.enable_grad():
            kv = tuple(tuple(v.detach().clone() for v in pair) for pair in saved.kv)
            require(all(not v.requires_grad and not v.is_inference() for pair in kv for v in pair), 'Detached normal KV copies')
            cache = DynamicCache.from_legacy_cache(kv)
            length = len(saved.prefix); total = length+len(suffix)+1
            ids = torch.tensor([suffix+[target]],dtype=torch.long,device=self.model.device)
            out = self.model(input_ids=ids, attention_mask=torch.ones((1,total),dtype=torch.long,device=self.model.device),
                position_ids=torch.arange(length,total,device=self.model.device)[None,:],
                cache_position=torch.arange(length,total,device=self.model.device),
                past_key_values=cache,use_cache=True,logits_to_keep=2)
            logits = out.logits[0].float()
            require(logits.shape == (2,self.model.config.vocab_size) and bool(torch.isfinite(logits).all()), 'Two finite causal positions')
        self.guard(saved)
        return logits

    def loss(self, saved, branch, target, eos, *, max_input_tokens=2048):
        require(type(eos) is int and 0 <= eos < self.model.config.vocab_size, 'EOS vocabulary ID')
        logits = self.teacher_forced_logits(saved,branch,target,max_input_tokens=max_input_tokens)
        return torch.nn.functional.cross_entropy(logits,torch.tensor([target,eos],device=logits.device))

    def decode(self, tokenizer, saved, branch, *, max_input_tokens=2048):
        self._suffix(saved,branch,max_input_tokens)
        adapter_hash = self.adapter_hash()
        identity = self.producer_state_id+'/suffix-reader/'+adapter_hash
        # Explicit memory import into a DIFFERENT reader, not relabelling the
        # generation's provenance. The outer record retains both identities.
        imported = PrefixSnapshot(saved.prefix,saved.kv,dict(saved.capture),weakref.ref(self.model),
                                  _parameters(self.model),identity,saved.cache_hash)
        decoded = decode_cached_branch(self.model,tokenizer,branch,imported,
            model_state_id=identity,max_input_tokens=max_input_tokens)
        self.guard(saved)
        require(self.adapter_hash() == adapter_hash, 'Adapter changed during report')
        return dict(schema='menia-retained-state-reader-decode-v1',producerStateId=saved.model_state_id,
                    sourceCacheHash=saved.cache_hash,adapterHash=adapter_hash,config=self.config,decoded=decoded,
                    scope='Frozen producer history imported into a suffix-adapted reader. No task answer or future outcome is supplied.')
