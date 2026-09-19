"""Four fixed technical comparisons of joint capture against unchanged generation."""
import argparse
import json
import os
from pathlib import Path

from research.activation_monitor_gpu import sample_with_states
from research.cross_model_gpu import environment, generate_text
from research.cross_model_prediction import MODELS, SOLVE, digest
from research.joint_prediction_capture import sample_with_joint_capture
from research.output_confidence_trace import _ObservedTokenizer, trace_generation
from research.output_confidence_validation import QUESTIONS, SEEDS, VALIDATION_SETTINGS, reset_seed, rng_state, rng_equal, timestamp
from research.prospective_confidence_features import NAMES, features, validate_capture


def validate(model, tokenizer, *, settings=VALIDATION_SETTINGS):
    cases = []
    for question in QUESTIONS:
        for seed in SEEDS:
            task = dict(question=question, seed=seed, family='countA' if question.startswith('Combien') else 'alternatingSum',
                        level=8 if question.startswith('Combien') else 4)
            messages = [dict(role='system', content=SOLVE), dict(role='user', content=question)]
            record = dict(question=question, seed=seed, startedUTC=timestamp())
            try:
                reset_seed(model, seed)
                initial_rng = rng_state(model)
                seen = _ObservedTokenizer(tokenizer)
                plain = generate_text(model, seen, messages, seed, settings=settings)
                plain_rng = rng_state(model)
                old_states = []
                state_only = sample_with_states(model, tokenizer, task, old_states.append, settings=settings)
                state_rng = rng_state(model)
                confidence_only = trace_generation(model, tokenizer, messages, seed, settings=settings)
                confidence_rng = rng_state(model)
                before_hooks = [len(m._forward_hooks) for m in model.modules()]
                captured = []
                def capture(value):
                    validate_capture(value)
                    captured.append(dict(value=value, beforeSampling=rng_equal(initial_rng, rng_state(model))))
                actual = sample_with_joint_capture(model, tokenizer, messages, seed, capture, settings=settings)
                actual_rng = rng_state(model)
                after_hooks = [len(m._forward_hooks) for m in model.modules()]
                dimensions = {name: len(features(task, captured[0]['value'], name)) for name in NAMES} if captured else {}
                checks = dict(plainGenerationUnchanged=actual[:2] == plain,
                    tokensUnchanged=actual[2]['completion']['tokenIds'] == seen.generated_ids,
                    traceMatchesStandalone=actual[2] == confidence_only[2],
                    statesMatchStandalone=len(captured) == len(old_states) == 1 and captured[0]['value']['state'] == old_states[0],
                    stateOnlyGenerationUnchanged=state_only == plain,
                    confidenceOnlyGenerationUnchanged=confidence_only[:2] == plain,
                    rngAllPathsUnchanged=all(rng_equal(plain_rng, r) for r in (state_rng, confidence_rng, actual_rng)),
                    callbackOnceBeforeSampling=len(captured) == 1 and captured[0]['beforeSampling'],
                    callbackConfidenceMatchesTrace=len(captured) == 1 and captured[0]['value']['preAnswer'] == actual[2]['preAnswer'],
                    allHooksRemoved=before_hooks == after_hooks,
                    nestedFeatureDimensions=dimensions == dict(inputOnly=390, outputConfidence=9, inputConfidence=393, finalControl=521, internal=649),
                    noFutureFields=len(captured) == 1 and set(captured[0]['value']) == {'state', 'preAnswer'})
                record.update(status='completed', passed=all(checks.values()), checks=checks,
                              capture=captured[0]['value'], captureHash=digest(captured[0]['value']),
                              tokenIds=seen.generated_ids, outputTokens=plain[1]['outputTokens'],
                              reachedTokenLimit=plain[1]['reachedTokenLimit'], trace=actual[2])
            except Exception as error:
                record.update(status='error', passed=False, errorType=type(error).__name__, error=str(error))
            record['completedUTC'] = timestamp()
            cases.append(record)
    invariant = {}
    for question in QUESTIONS:
        matching = [c for c in cases if c['question'] == question]
        invariant[question] = all(c['status'] == 'completed' for c in matching) and len({c.get('captureHash') for c in matching}) == 1
    return dict(schema='menia-joint-capture-validation-v1', passed=all(c['passed'] for c in cases) and all(invariant.values()),
                plannedCases=4, cases=cases, seedInvariantCaptures=invariant, settings=settings,
                scope='Fixed short technical cases, same base Qwen and shared projection/tracing code. No error-prediction accuracy, native introspection, training or consciousness test.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep the earlier validation result')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = environment()
    free, _ = torch.cuda.mem_get_info()
    if free < 14 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory; leave existing work running')
    spec = MODELS['A']
    metadata.update(models={'A': spec}, settings=VALIDATION_SETTINGS)
    started = timestamp()
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'], torch_dtype=torch.bfloat16,
                device_map={'': 'cuda:0'}, attn_implementation='sdpa', use_safetensors=True, trust_remote_code=False).eval()
    if model.config._commit_hash != spec['revision']:
        raise RuntimeError('Model revision mismatch')
    result = validate(model, tokenizer)
    names = ('joint_capture_validation.py', 'joint_prediction_capture.py', 'prospective_confidence_features.py',
             'activation_monitor.py', 'activation_monitor_gpu.py', 'output_confidence_trace.py',
             'output_confidence_validation.py', 'cross_model_gpu.py', 'cross_model_prediction.py')
    result.update(startedUTC=started, completedUTC=timestamp(), origin='transformers_gpu', model=spec,
                  metadata=metadata, freeBytesBeforeLoad=free,
                  sourceHash=digest({name: Path(__file__).with_name(name).read_text(encoding='utf-8') for name in names}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(path=str(args.output), passed=result['passed'], cases=len(result['cases']))), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
