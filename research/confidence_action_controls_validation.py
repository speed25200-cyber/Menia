"""Validate explicit-cost control framing with the pinned Qwen tokenizer only."""
import argparse
import hashlib
import json
from pathlib import Path

import transformers
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from research.confidence_action_controls import compile_explicit_controls, control_grid, grade_decision
from research.cross_model_prediction import MODELS, digest


def run():
    spec = MODELS['A']
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], trust_remote_code=False)
    files = {}
    for name in ('tokenizer_config.json', 'tokenizer.json'):
        path = Path(hf_hub_download(spec['id'], name, revision=spec['revision']))
        assert spec['revision'] in path.parts
        raw = path.read_bytes()
        files[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    cases = []; prefix_hashes = set()
    for case in control_grid():
        fork = compile_explicit_controls(tokenizer, 'Exemple technique pour contrôler les branches de décision.', '17',
                                         p_percent=case['pPercent'], verification_cost=case['verificationCost'])
        prefix_hashes.add(fork['prefixTokenHash'])
        branches = {}
        for name, branch in fork['branches'].items():
            assert branch['inputIds'][:len(fork['prefixIds'])] == fork['prefixIds']
            assert len(branch['inputIds'])+1 <= 1792
            branches[name] = {k: branch[k] for k in ('inputHash', 'information', 'codeToMeaning',
                              'optionOrder', 'candidateTokenIds', 'stipulatedExpectedCosts')}
            branches[name].update(inputTokens=len(branch['inputIds']), sharedPrefixExact=True)
        grading = {decision: grade_decision(decision, p_percent=case['pPercent'], verification_cost=case['verificationCost'])
                   for decision in ('accept', 'verify')}
        cases.append(dict(case, prefixHash=fork['prefixTokenHash'], prefixTokens=len(fork['prefixIds']),
                          branches=branches, stipulatedGrading=grading))
    assert len(prefix_hashes) == 1
    sources = {name: Path(__file__).with_name(name).read_text(encoding='utf-8') for name in (
        'confidence_action_controls.py', 'confidence_action_controls_validation.py',
        'confidence_action_branches.py', 'confidence_prompt_fork.py', 'risk_presentation.py')}
    return dict(schema='menia-explicit-action-controls-tokenizer-check-v1', origin='pinned_qwen_tokenizer_only',
                model=spec, transformers=transformers.__version__, files=files, sources=list(sources), sourceHash=digest(sources),
                cases=cases, branchesChecked=sum(len(case['branches']) for case in cases),
                pretrainedModelLoaded=False, generations=0, trainingUpdates=0,
                scope='Token framing and exact stipulated-cost arithmetic only. No model decision, '
                      'empirical self-estimate, measured utility or consciousness result.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); report = run()
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text(encoding='utf-8') != content:
            raise ValueError('Existing different validation receipt')
    else:
        args.output.write_text(content, encoding='utf-8')
    print(json.dumps(dict(origin=report['origin'], cases=len(report['cases']), branches=report['branchesChecked'],
                         sourceHash=report['sourceHash'], pretrainedModelLoaded=False, generations=0)))
