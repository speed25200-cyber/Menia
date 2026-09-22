"""Post-hoc numerical diagnosis of the training-only case 1008 K/V flip.

Never changes the learning collector or its labels. FP32 here widens the
already rounded BF16 weights and cache; it does not reconstruct an FP32 past.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import weakref

from research import prospective_reader_learning as learning
from research.cross_model_prediction import digest
from research.iphone_coupling_report import require

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / 'artifacts/prospective-reader-learning/training-prefix/prospective-reader-learning-20260920-v1.jsonl'
PREFIX_HASH = '9dfd743dcaad85d333b7cdd51d807f513d8430bb43df6fdd1fe5af4a35c8a2d5'
PREPARATION = ROOT / 'artifacts/reader-permutation-numerics-preparation/design.json'
CONDITIONS = ('actual', 'joint_permuted', 'values_permuted', 'restored')
MODES = ('bf16-default', 'bf16-math', 'fp32-math')


def plan():
    return dict(schema='menia-reader-permutation-numerics-plan-v1', case=1008,
        selection='Post-hoc: the only joint K/V native change in the 768 observed training tasks',
        sourcePrefixSHA256=PREFIX_HASH, parentPlanHash=digest(learning.plan()),
        parentSourceHash=learning.source_hash(), model=learning.plan()['model'],
        conditions=list(CONDITIONS), modes=list(MODES), targets=list(range(8)), repeats=2,
        decodes=192, scope='Numerical diagnosis of one selected training case, not an independent replication or consciousness evidence',
        precision='FP32 suffix computation on exactly widened BF16 parameters and retained cache; no new FP32 prefill',
        gates=['Parent execution completed and original process terminal before launch',
               'Exact BF16 generation/cache reproduction before interpreting the diagnostic',
               'All default BF16 native decodes must reproduce the original training journal',
               'No replacement of parent labels or primary criteria'])


def preparation():
    p = plan()
    files = ['research/reader_permutation_numerics.py', 'tests_language/test_reader_permutation_numerics.py']
    return dict(plan=p, planHash=digest(p), sourceSHA256={
        name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files})


def widen_snapshot(model, saved, identity):
    """Explicit import after widening this owner's BF16 model to FP32."""
    import torch
    from research.confidence_cached_action_decode import PrefixSnapshot, _cache_hash, _parameters
    require(saved.owner() is model and _cache_hash(saved.kv) == saved.cache_hash, 'Unchanged source owned by this model')
    require(model.dtype == torch.float32 and all(v.dtype == torch.bfloat16 for pair in saved.kv for v in pair),
            'Only BF16 cache to FP32 compute is supported')
    require(identity != saved.model_state_id, 'Precision conversion requires a new identity')
    kv = tuple(tuple(v.to(dtype=torch.float32).detach().clone() for v in pair) for pair in saved.kv)
    require(all(torch.equal(a, b.to(torch.bfloat16)) for before, after in zip(saved.kv, kv)
                for a, b in zip(before, after)), 'Lossless cache widening')
    capture = dict(saved.capture, precisionImport=dict(sourceIdentity=saved.model_state_id,
        sourceCacheHash=saved.cache_hash, originalCacheRecomputed=False, operation='BF16 values widened to FP32'))
    return PrefixSnapshot(saved.prefix, kv, capture, weakref.ref(model), _parameters(model), identity, _cache_hash(kv))


def widen_model(model, source):
    """Record and verify exact numerical preservation of parameters and buffers."""
    import torch
    from research.confidence_cached_action_decode import _parameters, _cache_hash
    from research.confidence_prefix_interventions import tensor_hash
    require(source.owner() is model and source.parameters == _parameters(model) and
            source.cache_hash == _cache_hash(source.kv), 'Original model and cache must still match before widening')
    before = {}
    for kind, items in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
        for name, tensor in items:
            require(not tensor.is_floating_point() or tensor.dtype in (torch.bfloat16, torch.float32), 'Unexpected original precision')
            before[kind, name] = (tensor.dtype, tensor_hash(tensor))
    model.float()
    for kind, items in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
        for name, tensor in items:
            old_dtype, old_hash = before[kind, name]
            restored = tensor.to(old_dtype)
            require(tensor_hash(restored) == old_hash and torch.equal(tensor, restored.to(tensor.dtype)),
                    'Values changed while widening the model')
    return dict(parameters=len(list(model.parameters())), buffers=len(list(model.buffers())),
        originalValueHash=digest({kind+'/'+name: h for (kind,name), (_,h) in before.items()}),
        allValuesPreserved=True, sourcePrecision='bfloat16 parameters; existing buffers retained', targetPrecision='float32')


def summarize(records):
    """Report every key, including repeat instability; no pass criterion is fitted."""
    from research.confidence_generation_numerics import compare_branches
    expected = {(m,r,c,t) for m in MODES for r in range(2) for c in CONDITIONS for t in range(8)}
    keyed = {}
    for row in records:
        key = tuple(row[k] for k in ('mode','repeat','condition','target'))
        require(key not in keyed, 'Duplicate diagnostic branch')
        keyed[key] = row['decoded']
    require(set(keyed) == expected, 'Incomplete diagnostic grid')
    contrasts = []
    for m in MODES:
        for t in range(8):
            actual = keyed[m,0,'actual',t]
            contrasts.append(dict(mode=m,target=t,
                joint=compare_branches(actual,keyed[m,0,'joint_permuted',t]),
                values=compare_branches(actual,keyed[m,0,'values_permuted',t]),
                restoredExactlyEqual=actual == keyed[m,0,'restored',t],
                repeatExactlyEqual=all(keyed[m,0,c,t] == keyed[m,1,c,t] for c in CONDITIONS)))
    return dict(schema='menia-reader-permutation-numerics-summary-v1', decodes=len(records), contrasts=contrasts,
        scope=plan()['scope'], precision=plan()['precision'],
        warning='A smaller joint discrepancy in another numerical mode is not proof of a specific faulty kernel or a consciousness mechanism.')


def run(path, parent_results):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from research.prospective_reader_learning_audit import read_journal
    from research import prospective_pair_discovery as parent
    from research.confidence_generation_continuity import capture_generated_prefix
    from research.confidence_cached_action_decode import decode_cached_branch, _parameters
    from research.prospective_cache_interventions import permute_prefix_cache
    from research.cross_model_gpu import environment
    from research.natural_error_journal import Writer

    path = Path(path)
    require(not path.exists(), 'Preserve every existing attempt')
    execution = json.loads((Path(parent_results)/'execution.json').read_text(encoding='utf-8'))
    require(execution['status'] == 'completed', 'Wait for the parent experiment to complete')
    require(execution['revision'] == '97ca234a2e4f2f37f24f78ee60fd8a925a0a7404', 'Exact parent attempt')
    prepared = preparation()
    require(json.loads(PREPARATION.read_text(encoding='utf-8')) == prepared, 'Frozen numerical diagnostic preparation')
    require(hashlib.sha256(PREFIX.read_bytes()).hexdigest() == PREFIX_HASH, 'Training-only input identity')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Set workspace before process startup')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    w = Writer(path)
    metadata = environment()
    metadata['numericalFlags'] = dict(
        bf16ReducedPrecisionReduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        mathSDPAReducedPrecision=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        float32MatmulPrecision=torch.get_float32_matmul_precision())
    w.write(dict(event='header', **prepared, origin='transformers_gpu', metadata=metadata), create=True)
    try:
        spec = prepared['plan']['model']
        tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], trust_remote_code=False)
        reference = read_journal(PREFIX, tokenizer)
        states = [e['result'] for e in reference['events'] if e['event'] == 'state' and e['call']['case'] == 1008]
        require(len(states) == 1 and states[0]['case']['split'] == 'train', 'One original training state')
        original = states[0]
        tasks = {(e['call']['condition'], e['call']['target']): e['result']['decoded']
                 for e in reference['events'] if e['event'] == 'task' and e['call']['case'] == 1008}
        p = learning.plan()
        identity = spec['id']+'@'+spec['revision']+'/base-bfloat16-sdpa'
        model = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'], torch_dtype=torch.bfloat16,
            device_map={'':'cuda:0'}, attn_implementation='sdpa', use_safetensors=True, trust_remote_code=False).eval().requires_grad_(False)
        require(model.config._commit_hash == spec['revision'], 'Pinned model revision')
        text, metrics, trace, saved = capture_generated_prefix(model, tokenizer, original['case']['messages'], original['case']['seed'],
            model_state_id=identity, settings=p['settings'], max_context_tokens=p['maxContextTokens'])
        reproduction = dict(text=text == original['text'], metrics=metrics == original['metrics'],
                            trace=trace == original['trace'], cache=saved.cache_hash == original['cacheHashes']['actual'])
        w.write(dict(event='reproduction', checks=reproduction, actualCacheHash=saved.cache_hash))
        require(all(reproduction.values()), 'Original state did not reproduce; preserve and stop this diagnostic')
        branches = parent.compile_branches(tokenizer, saved.prefix, original['case'], p)
        records = []
        for mode in MODES:
            if mode != 'bf16-default':
                torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
            if mode == 'fp32-math':
                conversion = widen_model(model, saved)
                old_saved = saved
                identity += '/exact-value-fp32-suffix'
                saved = widen_snapshot(model, old_saved, identity)
                w.write(dict(event='precision_import', conversion=conversion, capture=saved.capture, cacheHash=saved.cache_hash))
            source_versions = _parameters(model)
            permutation = original['bindingMask']['permutation']
            args = dict(model_state_id=identity, layers=list(range(len(model.model.layers))), permutation=permutation)
            joint, _ = permute_prefix_cache(model, saved, mode='keys_and_values', **args)
            values, _ = permute_prefix_cache(model, saved, mode='values_only', **args)
            restored, _ = permute_prefix_cache(model, values, mode='values_only', **args)
            views = dict(actual=saved, joint_permuted=joint, values_permuted=values, restored=restored)
            require(restored.cache_hash == saved.cache_hash, 'Exact restoration')
            if mode == 'bf16-default':
                require(all(views[c].cache_hash == original['cacheHashes'][c] for c in CONDITIONS), 'Original transformed cache hashes')
            for repeat in range(2):
                for condition in CONDITIONS:
                    for target in range(8):
                        call = dict(mode=mode,repeat=repeat,condition=condition,target=target)
                        w.write(dict(event='request', **call))
                        context = nullcontext() if mode == 'bf16-default' else sdpa_kernel(SDPBackend.MATH)
                        with context:
                            decoded = decode_cached_branch(model,tokenizer,branches[learning.branch_name('task',dict(target=target))],
                                views[condition],model_state_id=identity,max_input_tokens=p['maxContextTokens'])
                        original_decode = tasks['actual' if condition == 'restored' else condition,target]
                        row = dict(**call, decoded=decoded)
                        w.write(dict(event='decode', **row, originalExactlyEqual=decoded == original_decode if mode == 'bf16-default' else None))
                        records.append(row)
                        if mode == 'bf16-default':
                            require(decoded == original_decode, 'Default BF16 task did not reproduce; stop this diagnostic')
            require(_parameters(model) == source_versions, 'Weights changed inside a numerical mode')
            print(json.dumps(dict(completedMode=mode,decodes=len(records))),flush=True)
        summary = summarize(records)
        w.write(dict(event='complete', summary=summary))
        with path.with_suffix('.summary.json').open('x',encoding='utf-8') as f:
            json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False)
    except BaseException as error:
        w.write(dict(event='failure',errorType=type(error).__name__,error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--journal', type=Path)
    parser.add_argument('--parent-results', type=Path)
    args = parser.parse_args()
    if args.prepare:
        PREPARATION.parent.mkdir(parents=True,exist_ok=True)
        with PREPARATION.open('x',encoding='utf-8') as f:
            json.dump(preparation(),f,ensure_ascii=False,indent=2)
    else:
        require(args.journal is not None and args.parent_results is not None, 'Journal and completed parent results required')
        run(args.journal,args.parent_results)
