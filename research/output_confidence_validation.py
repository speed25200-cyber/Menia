"""Technical equivalence check on fixed prompts, not a test of metacognitive accuracy."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from research.cross_model_gpu import environment, generate_text
from research.cross_model_prediction import MODELS, SETTINGS, SOLVE, digest
from research.output_confidence_trace import _ObservedTokenizer, trace_generation

QUESTIONS = ('Combien de lettres A contient cette chaîne : ABACADAA ?',
             'Calcule 81 - 27 + 43 - 16.')
SEEDS = (421, 972)
VALIDATION_SETTINGS = dict(SETTINGS, max_new_tokens=32)
TOLERANCE = 1e-10


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def rng_state(model):
    import torch
    return [torch.random.get_rng_state().clone()] + (
        [s.clone() for s in torch.cuda.get_rng_state_all()] if model.device.type == 'cuda' else [])


def rng_equal(a, b):
    import torch
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def reset_seed(model, seed):
    import torch
    torch.manual_seed(seed)
    if model.device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()


def reference_generation(model, tokenizer, messages, seed, generation_config, settings):
    """Use Transformers' raw-logit return path, without the tracing hook."""
    import torch
    from transformers import GenerationConfig
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=settings['enable_thinking'])
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False,
                       return_token_type_ids=False).to(model.device)
    config = GenerationConfig(**generation_config)
    config.return_dict_in_generate = True
    config.output_logits = True
    reset_seed(model, seed)
    with torch.inference_mode():
        result = model.generate(**inputs, generation_config=config)
    if model.device.type == 'cuda':
        torch.cuda.synchronize()
    ids = result.sequences[0, inputs.input_ids.shape[-1]:].detach().cpu().tolist()
    logits = [row[0].detach().to(device='cpu', dtype=torch.float64).numpy() for row in result.logits]
    return ids, logits


def numerical_differences(trace, ids, logits):
    """NumPy/math reconstruction from Transformers outputs, not the trace helper."""
    if len(ids) != len(logits) or len(ids) != len(trace['completion']['tokenIds']):
        raise ValueError('Reference steps differ from captured steps')
    differences = []
    logps = []
    for i, (token, values) in enumerate(zip(ids, logits)):
        shifted = values - values.max()
        logp = shifted - math.log(math.fsum(np.exp(shifted)))
        probabilities = np.exp(logp)
        entropy = -math.fsum(probabilities * logp)
        logps.append(float(logp[token]))
        differences.extend((abs(logps[-1] - trace['completion']['tokenLogProbabilities'][i]),
                            abs(entropy - trace['completion']['entropyNats'][i])))
        if i == 0:
            top = np.sort(probabilities)[-2:]
            expected = dict(maxProbability=float(top[-1]), topTwoMargin=float(top[-1] - top[-2]),
                            entropyNats=entropy, normalizedEntropy=entropy / math.log(len(values)))
            differences.extend(abs(v - trace['preAnswer'][k]) for k, v in expected.items())
            if trace['preAnswer']['vocabularySize'] != len(values):
                raise ValueError('Reference vocabulary size changed')
            if values[trace['preAnswer']['topTokenId']] != values.max():
                raise ValueError('Captured top token is not a reference maximum')
    differences.extend((abs(math.fsum(logps) - trace['completion']['sumLogProbability']),
                        abs(math.fsum(logps)/len(logps) - trace['completion']['meanLogProbability'])))
    return max(differences)


def validate(model, tokenizer, *, settings=VALIDATION_SETTINGS):
    cases = []
    for question in QUESTIONS:
        for seed in SEEDS:
            messages = [dict(role='system', content=SOLVE), dict(role='user', content=question)]
            record = dict(question=question, seed=seed, startedUTC=timestamp())
            try:
                observed = _ObservedTokenizer(tokenizer)
                reset_seed(model, seed)
                initial_rng = rng_state(model)
                start = time.perf_counter()
                plain_text, plain_metrics = generate_text(model, observed, messages, seed, settings=settings)
                plain_seconds = time.perf_counter() - start
                plain_rng = rng_state(model)
                callbacks = []
                def capture(value):
                    callbacks.append(dict(summary=value, beforeSampling=rng_equal(rng_state(model), initial_rng)))
                hooks_before = len(model.lm_head._forward_hooks)
                start = time.perf_counter()
                text, metrics, trace = trace_generation(model, tokenizer, messages, seed, settings=settings, on_prefill=capture)
                traced_seconds = time.perf_counter() - start
                traced_rng = rng_state(model)
                hooks_after = len(model.lm_head._forward_hooks)
                ids, logits = reference_generation(model, tokenizer, messages, seed, plain_metrics['effectiveGeneration'], settings)
                reference_rng = rng_state(model)
                maximum = numerical_differences(trace, ids, logits)
                checks = dict(
                    textUnchanged=text == plain_text,
                    tokensUnchanged=trace['completion']['tokenIds'] == observed.generated_ids,
                    metadataUnchanged=metrics == plain_metrics,
                    rngUnchanged=rng_equal(plain_rng, traced_rng),
                    oneCallbackBeforeSampling=len(callbacks) == 1 and callbacks[0]['beforeSampling'],
                    callbackMatchesPrefill=len(callbacks) == 1 and callbacks[0]['summary'] == trace['preAnswer'],
                    hookRemoved=hooks_after == hooks_before,
                    referenceTokensEqual=ids == observed.generated_ids,
                    referenceRngEqual=rng_equal(reference_rng, plain_rng),
                    rawNumericsMatch=maximum <= TOLERANCE)
                record.update(status='completed', passed=all(checks.values()), checks=checks,
                              maxAbsoluteDifference=maximum, outputTokens=len(ids), tokenIds=ids,
                              textSHA256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                              plainSeconds=plain_seconds, tracedSeconds=traced_seconds,
                              reachedTokenLimit=metrics['reachedTokenLimit'], trace=trace)
            except Exception as error:
                record.update(status='error', passed=False, errorType=type(error).__name__, error=str(error))
            record['completedUTC'] = timestamp()
            cases.append(record)
    return dict(schema='menia-output-confidence-validation-v1', passed=all(c['passed'] for c in cases),
                plannedCases=len(QUESTIONS)*len(SEEDS), cases=cases, settings=settings,
                numericalTolerance=TOLERANCE,
                scope='Four fixed technical cases, base model, no training or correctness prediction. Three generation paths share Transformers and the same model. Timings include warm-up and shared GPU contention; not a performance benchmark.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Retain each technical validation; do not overwrite it')
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
        raise RuntimeError('Less than 14 GiB free; preserve the existing GPU job')
    spec = MODELS['A']
    metadata['models'] = {'A': spec}
    metadata['settings'] = VALIDATION_SETTINGS
    started = timestamp()
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(spec['id'], revision=spec['revision'], torch_dtype=torch.bfloat16,
                device_map={'': 'cuda:0'}, attn_implementation='sdpa', use_safetensors=True, trust_remote_code=False).eval()
    if model.config._commit_hash != spec['revision']:
        raise RuntimeError('Model revision mismatch')
    result = validate(model, tokenizer)
    result.update(startedUTC=started, completedUTC=timestamp(), origin='transformers_gpu', model=spec,
                  metadata=metadata, freeBytesBeforeLoad=free,
                  sourceHash=digest({name: Path(__file__).with_name(name).read_text(encoding='utf-8') for name in
                                    ('output_confidence_validation.py', 'output_confidence_trace.py', 'cross_model_gpu.py', 'cross_model_prediction.py')}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(path=str(args.output), passed=result['passed'], cases=len(result['cases']))), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
