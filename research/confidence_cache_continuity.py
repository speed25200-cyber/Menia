"""Architectural control: retained KV state versus replay of identical tokens.

Only a tiny randomly initialized, full-attention Qwen is used here. No learned
confidence direction, native self-report, sampled answer or performance result.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from research.confidence_prefix_interventions import forward_at_shared_prefix, projected_interchange, tensor_hash


def snapshot(model, prefix, layer, donor=None, basis=None):
    """Run a single prefill and copy its cache; always remove the temporary hook."""
    ids = torch.tensor([prefix], dtype=torch.long)
    capture = {}
    def hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        before = value[0, -1].detach().clone()
        after = before if donor is None else projected_interchange(before, donor, basis)
        capture.update(before=before, displacement=float((after.double()-before.double()).norm()))
        if donor is None: return None
        changed = value.clone(); changed[0, -1] = after
        return (changed, *output[1:]) if isinstance(output, tuple) else changed
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
    finally:
        handle.remove()
    cache = tuple(tuple(value.detach().clone() for value in pair) for pair in result.past_key_values.to_legacy_cache())
    return dict(prefix=tuple(prefix), kv=cache, capture=capture, logits=result.logits[0, -1].detach().clone())


def cache_hash(saved):
    return hashlib.sha256(''.join(tensor_hash(value) for pair in saved['kv'] for value in pair).encode()).hexdigest()


def continue_snapshot(model, saved, suffix):
    """Fork copies for each continuation; never append into the saved cache."""
    cache = DynamicCache.from_legacy_cache(tuple(tuple(value.clone() for value in pair) for pair in saved['kv']))
    prefix_length = len(saved['prefix']); total = prefix_length+len(suffix)
    ids = torch.tensor([suffix], dtype=torch.long)
    with torch.inference_mode():
        result = model(input_ids=ids, attention_mask=torch.ones((1, total), dtype=torch.long),
                       position_ids=torch.arange(prefix_length, total)[None, :],
                       cache_position=torch.arange(prefix_length, total), past_key_values=cache, use_cache=True)
    assert result.past_key_values.get_seq_length() == total
    return result.logits[0, -1].detach().clone()


def run():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(242024)
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, attention_dropout=0.,
            bos_token_id=1, eos_token_id=2, pad_token_id=0, use_sliding_window=False)
        config._attn_implementation = 'eager'
        model = Qwen3ForCausalLM(config).eval()
        weights = {name: tensor_hash(value) for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        prefix = [1, 3, 4]; suffixes = [[5, 6], [7, 8, 9]]
        basis = torch.eye(32)[:, :2]; cases = []
        for layer in range(2):
            original = snapshot(model, prefix, layer)
            donor = original['capture']['before'].clone(); donor[:2] += 1.
            altered = snapshot(model, prefix, layer, donor, basis)
            hashes_before = [cache_hash(original), cache_hash(altered)]
            for suffix in suffixes:
                full_ids = torch.tensor([prefix+suffix], dtype=torch.long)
                inputs = dict(input_ids=full_ids, attention_mask=torch.ones_like(full_ids))
                replay, _ = forward_at_shared_prefix(model, inputs, prefix, layer)
                full_patched, _ = forward_at_shared_prefix(model, inputs, prefix, layer, donor=donor, basis=basis)
                baseline = continue_snapshot(model, original, suffix)
                retained = continue_snapshot(model, altered, suffix)
                restored = continue_snapshot(model, original, suffix)
                replay_after, _ = forward_at_shared_prefix(model, inputs, prefix, layer)
                maxdiff = lambda a, b: float((a.double()-b.double()).abs().max())
                assert torch.allclose(baseline, replay, atol=1e-6, rtol=1e-6)
                assert torch.allclose(retained, full_patched, atol=1e-6, rtol=1e-6)
                assert torch.equal(baseline, restored) and torch.equal(replay, replay_after)
                assert [cache_hash(original), cache_hash(altered)] == hashes_before
                assert all(t.shape[-2] == len(prefix) for pair in original['kv'] for t in pair)
                effect = maxdiff(retained, baseline)
                assert effect > 1e-6 if layer == 0 else effect == 0.
                assert altered['capture']['displacement'] > 0
                assert maxdiff(altered['logits'], original['logits']) > 1e-6
                cases.append(dict(layer=layer, prefixIds=prefix, suffixIds=suffix,
                    stateDisplacement=altered['capture']['displacement'],
                    prefillLogitDifference=maxdiff(altered['logits'], original['logits']),
                    retainedStateSuffixLogitDifference=effect, replayHistoryDifference=maxdiff(replay, replay_after),
                    baselineCacheVersusReplay=maxdiff(baseline, replay),
                    retainedCacheVersusReappliedPatch=maxdiff(retained, full_patched),
                    restoredDifference=maxdiff(baseline, restored), cacheChanged=hashes_before[0] != hashes_before[1],
                    branchSnapshotsUnchanged=True))
        assert all(tensor_hash(value) == weights[name] for name, value in model.state_dict().items())
        assert torch.equal(rng, torch.get_rng_state())
        assert all(not block._forward_hooks for block in model.model.layers)
    return dict(schema='menia-cache-continuity-control-v1', origin='random_qwen_cpu_architectural_control',
        transformers=transformers.__version__, torch=torch.__version__, seed=242024,
        model=dict(layers=2, hiddenSize=32, vocabulary=64, attention='eager', dtype='float32'),
        cases=cases, weightsUnchanged=True, samplingRNGUnchanged=True, hooksRemoved=True,
        pretrainedModelLoaded=False, learnedDirection=False, generatedAnswers=0,
        sourceSHA256=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest(),
        scope='Fixed arbitrary token sequences. A hidden intervention can survive in later-layer KV state; '
              'replaying tokens without that state removes its historical effect. No self-monitoring, '
              'native-action, phenomenal continuity or consciousness result, and no claim that KV is necessary for consciousness.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); report = run()
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        assert args.output.read_text(encoding='utf-8') == content, 'Existing different receipt'
    else:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream: stream.write(content)
    print(json.dumps(report, ensure_ascii=False))
