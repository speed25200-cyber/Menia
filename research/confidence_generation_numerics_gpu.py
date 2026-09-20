"""Six fixed numerical controls on pretrained Qwen3-4B; no learning or steering."""
import argparse
import json
import os
from pathlib import Path

from research.cross_model_prediction import MODELS, SETTINGS, SOLVE, digest

PREPARATION = Path(__file__).resolve().parents[1]/'artifacts/generation-numerics-preparation/design.json'
FILES = ('confidence_generation_numerics.py', 'confidence_generation_numerics_gpu.py',
    'confidence_generation_continuity.py', 'confidence_cached_action_decode.py',
    'confidence_action_decode.py', 'confidence_prefix_interventions.py', 'answer_confidence_data.py',
    'cross_model_gpu.py', 'cross_model_prediction.py', 'iphone_coupling_report.py',
    'iphone_capability_learning_report.py')


def source_hash():
    return digest({n:Path(__file__).with_name(n).read_text(encoding='utf-8') for n in FILES})


def plan():
    questions = (
        'Combien vaut 7 + 10 ?',
        'Combien vaut 4 - 7 ?',
        'Compte les lettres A dans cette suite : A B A A B B A B',
        'Calcule 18 - 7 + 4 - 12.',
        'Calcule 12 - 8 + 19 - 6 + 4 - 11 + 7 - 3.',
        'Compte les lettres A dans cette suite : B A B B A A B A B A A B B A B A A A B B A B A B',
    )
    return dict(schema='menia-generation-numerics-plan-v1', model=MODELS['A'],
        settings=dict(SETTINGS, max_new_tokens=64), maxContextTokens=2048,
        cases=[dict(id=i, seed=2026092000+i, messages=[dict(role='system',content=SOLVE),
            dict(role='user',content=q)]) for i,q in enumerate(questions)],
        branches=dict(report=dict(query='Évalue ta réponse précédente. Réponds uniquement par 0 si elle est incorrecte, ou par 1 si elle est correcte.',
            codes=['0','1'], negative='incorrect', positive='correct'),
            action=dict(query='Choisis une action pour ta réponse précédente. Réponds uniquement par 2 pour demander une vérification, ou par 3 pour garder cette réponse.',
            codes=['2','3'], negative='verify', positive='keep')),
        histories=['actual_generation', 'same_schedule_replay', 'full_prefill_replay'],
        plannedGenerations=12, plannedBranchDecodes=60, weightUpdates=0, interventions=0,
        strictControls=['repeated_generation_cache_exact', 'same_schedule_cache_exact',
            'same_schedule_branch_outputs_exact', 'reverse_order_actual_branches_exact', 'parameters_unchanged'],
        descriptiveControls=['full_prefill_cache_error_by_layer', 'full_prefill_branch_error',
            'full_sequence_branch_error', 'native_code_and_EOS_validity'],
        failurePolicy='Stop and preserve the attempt on an exception or a failed strict control. No retry, fallback, changed tolerance or case selection.',
        scope='Technical equivalence and numerical floor on six fixed prompts. No accuracy, calibration, self-monitoring, utility, consciousness or novelty criterion.')


def run(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from research.cross_model_gpu import environment
    from research.confidence_generation_continuity import capture_generated_prefix, compile_raw_continuations
    from research.confidence_cached_action_decode import decode_cached_branch, _parameters
    from research.confidence_action_decode import decode_branch
    from research.confidence_generation_numerics import replay_generated_tokens, compare_caches, compare_branches
    from research.natural_error_journal import Writer
    from research.iphone_coupling_report import require

    path = Path(path); require(not path.exists(), 'Existing attempt')
    prepared = json.loads(PREPARATION.read_text(encoding='utf-8')); p = plan()
    require(prepared == dict(plan=p, planHash=digest(p), sourceHash=source_hash()), 'Frozen preparation changed')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Set CUDA workspace before starting the process')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    metadata = environment(); metadata.update(model=p['model'], settings=p['settings'], weightUpdates=0, interventions=0)
    path.parent.mkdir(parents=True, exist_ok=True); writer = Writer(path)
    writer.write(dict(event='header', **prepared, metadata=metadata), create=True)
    try:
        spec = p['model']; identity = spec['id']+'@'+spec['revision']+'/base-bfloat16-sdpa'
        tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'],
            torch_dtype=torch.bfloat16, device_map={'':'cuda:0'}, attn_implementation='sdpa',
            use_safetensors=True, trust_remote_code=False).eval().requires_grad_(False)
        require(model.config._commit_hash == spec['revision'], 'Model revision')
        parameters = _parameters(model); cases = []; all_passed = True
        for case in p['cases']:
            writer.write(dict(event='case_start', case=case))
            args = dict(model_state_id=identity, settings=p['settings'], max_context_tokens=p['maxContextTokens'])
            text, metrics, trace, actual = capture_generated_prefix(model, tokenizer, case['messages'], case['seed'], **args)
            text2, metrics2, trace2, repeated = capture_generated_prefix(model, tokenizer, case['messages'], case['seed'], **args)
            repeats = compare_caches(actual, repeated)
            repeat_exact = text == text2 and metrics == metrics2 and trace == trace2 and repeats['exact']
            del repeated
            scheduled = replay_generated_tokens(model, actual, len(trace['promptTokenIds']),
                model_state_id=identity, schedule='generation')
            full = replay_generated_tokens(model, actual, len(trace['promptTokenIds']),
                model_state_id=identity, schedule='full')
            caches = dict(same_schedule_replay=compare_caches(actual, scheduled), full_prefill_replay=compare_caches(actual, full))
            branches = compile_raw_continuations(tokenizer, actual.prefix,
                {name:b['query'] for name,b in p['branches'].items()})
            for name, branch in branches.items():
                config = p['branches'][name]; codes = [tokenizer.encode(c, add_special_tokens=False) for c in config['codes']]
                require(all(len(c)==1 for c in codes) and codes[0] != codes[1], 'Single distinct code tokens')
                branch.update(candidateTokenIds=[c[0] for c in codes], negative=config['negative'], positive=config['positive'])
            decoded = {}
            for name, snapshot in (('actual_generation', actual), ('same_schedule_replay', scheduled), ('full_prefill_replay', full)):
                decoded[name] = {key:decode_cached_branch(model,tokenizer,branch,snapshot,model_state_id=identity)
                    for key,branch in branches.items()}
            reverse = {key:decode_cached_branch(model,tokenizer,branches[key],actual,model_state_id=identity)
                for key in reversed(branches)}
            full_sequence = {key:decode_branch(model,tokenizer,branch,actual.prefix,0) for key,branch in branches.items()}
            # Different decoder schemas are compared on the scientific quantities,
            # not on their necessarily different pass/cache metadata.
            comparisons = {name:{key:compare_branches(decoded['actual_generation'][key], result[key]) for key in branches}
                for name,result in list(decoded.items())[1:]+[('full_sequence',full_sequence)]}
            scheduled_exact = all(decoded['actual_generation'][k] == decoded['same_schedule_replay'][k] for k in branches)
            checks = dict(repeated_generation_cache_exact=repeat_exact, same_schedule_cache_exact=caches['same_schedule_replay']['exact'],
                same_schedule_branch_outputs_exact=scheduled_exact,
                reverse_order_actual_branches_exact=reverse == decoded['actual_generation'], parameters_unchanged=parameters == _parameters(model))
            record = dict(event='case_result', id=case['id'], text=text, metrics=metrics, trace=trace,
                prefixHash=digest(actual.prefix), branchInputHashes={k:digest(b['inputIds']) for k,b in branches.items()},
                replaySchedules={name:s.capture['forwardInputLengths'] for name,s in (('same_schedule_replay',scheduled),('full_prefill_replay',full))},
                checks=checks, repeatedCache=repeats, cacheComparisons=caches, branchComparisons=comparisons,
                decoded=decoded, fullSequenceDecoded=full_sequence, allStrictControlsPassed=all(checks.values()))
            writer.write(record); cases.append(record); all_passed = all_passed and record['allStrictControlsPassed']
            print(json.dumps(dict(case=case['id'], checks=checks,
                fullPrefillMaximumCacheDifference=caches['full_prefill_replay']['maximumAbsoluteDifference'],
                branchComparisons=comparisons)), flush=True)
            require(record['allStrictControlsPassed'], 'Numerical strict control failed; preserve results and investigate')
            del actual, scheduled, full
        summary = dict(schema='menia-generation-numerics-summary-v1', planHash=digest(p), sourceHash=source_hash(),
            cases=len(cases), generations=2*len(cases), branchDecodes=10*len(cases), allStrictControlsPassed=all_passed,
            fullPrefillMaximumCacheDifference=max(c['cacheComparisons']['full_prefill_replay']['maximumAbsoluteDifference'] for c in cases),
            fullPrefillNativeTokenAgreements=sum(c['branchComparisons']['full_prefill_replay'][k]['tokenIdsEqual'] for c in cases for k in p['branches']),
            fullSequenceNativeTokenAgreements=sum(c['branchComparisons']['full_sequence'][k]['tokenIdsEqual'] for c in cases for k in p['branches']),
            actualValidNativeResponses=sum(c['decoded']['actual_generation'][k]['validNativeResponse'] for c in cases for k in p['branches']),
            branchComparisonsPerReference=2*len(cases), scope=p['scope'])
        with path.with_suffix('.summary.json').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        writer.write(dict(event='complete', summary=summary))
        return summary
    except BaseException as error:
        writer.write(dict(event='failure', errorType=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--journal', type=Path)
    args = parser.parse_args()
    if args.prepare:
        p = plan(); report = dict(plan=p, planHash=digest(p), sourceHash=source_hash())
        PREPARATION.parent.mkdir(parents=True, exist_ok=True)
        with PREPARATION.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print(json.dumps({k:report[k] for k in ('planHash','sourceHash')}))
    else:
        if args.journal is None: parser.error('--journal is required without --prepare')
        print(json.dumps(run(args.journal), ensure_ascii=False))
