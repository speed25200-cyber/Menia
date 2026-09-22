"""Check nine-branch framing and stop/code tokens with the real pinned tokenizer.

No pretrained model is loaded and no task performance is measured.
"""
import argparse
import hashlib
import json
from pathlib import Path

import transformers
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from research.confidence_action_branches import compile_action_branches
from research.confidence_action_decode import parse_native_tokens
from research.cross_model_prediction import MODELS, digest


def run():
    spec = MODELS['A']
    tokenizer = AutoTokenizer.from_pretrained(spec['id'], revision=spec['revision'], trust_remote_code=False)
    files = {}; generation = None
    for name in ('tokenizer_config.json', 'tokenizer.json', 'generation_config.json'):
        path = Path(hf_hub_download(spec['id'], name, revision=spec['revision']))
        assert spec['revision'] in path.parts
        raw = path.read_bytes(); files[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if name == 'generation_config.json': generation = json.loads(raw)
    eos = generation['eos_token_id']; assert tokenizer.eos_token_id in eos
    question = 'Exemple technique pour contrôler les branches de décision.'
    cases = []
    for answer in ('17', '-3', 'réponse technique\nsur deux lignes', '\n17'):
        for cost in (20, 80):
            fork = compile_action_branches(tokenizer, question, answer, verification_cost=cost)
            assert len(fork['branches']) == 9
            summaries = {}
            for name, branch in fork['branches'].items():
                assert branch['inputIds'][:len(fork['prefixIds'])] == fork['prefixIds']
                assert len(branch['inputIds'])+1 <= 1792
                assert set(branch['candidateTokenIds']).isdisjoint(eos)
                meanings = [branch['negative'], branch['positive']]
                for stop in eos:
                    parsed = parse_native_tokens([branch['candidateTokenIds'][1], stop], branch['candidateTokenIds'], meanings, eos)
                    assert parsed['validNativeResponse'] and parsed['decision'] == branch['positive']
                summaries[name] = {key: branch[key] for key in ('kind', 'inputHash', 'codeToMeaning', 'candidateTokenIds')}
                summaries[name].update(inputTokens=len(branch['inputIds']), sharedPrefixExact=True,
                                       optionOrder=branch.get('optionOrder'), verificationCost=branch.get('verificationCost'))
            actions = {i for b in fork['branches'].values() if b['kind'] == 'action' for i in b['candidateTokenIds']}
            assert actions.isdisjoint(fork['branches']['judgment']['candidateTokenIds'])
            cases.append(dict(question=question, answer=answer, verificationCost=cost,
                              prefixTokens=len(fork['prefixIds']), prefixHash=fork['prefixTokenHash'],
                              position=fork['position'], branches=summaries))
    sources = {name: Path(__file__).with_name(name).read_text(encoding='utf-8') for name in (
        'confidence_prompt_fork.py', 'confidence_prefix_interventions.py', 'confidence_action_branches.py',
        'confidence_action_decode.py', 'confidence_action_validation.py')}
    return dict(schema='menia-native-action-tokenizer-check-v1', origin='pinned_qwen_tokenizer_only',
                model=spec, transformers=transformers.__version__, files=files, sourceHash=digest(sources),
                eosTokenIds=eos, cases=cases, branchesChecked=sum(len(c['branches']) for c in cases),
                pretrainedModelLoaded=False, generations=0, trainingUpdates=0,
                scope='Token framing, code mapping, input budget and declared stop tokens only. '
                      'Synthetic token sequences validate parsing; no Menia decision or consciousness result.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); report = run()
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_text(encoding='utf-8') != content: raise ValueError('Existing different validation receipt')
    else:
        args.output.write_text(content, encoding='utf-8')
    print(json.dumps(dict(origin=report['origin'], cases=len(report['cases']), branches=report['branchesChecked'],
                         eosTokenIds=report['eosTokenIds'], sourceHash=report['sourceHash'],
                         pretrainedModelLoaded=False, generations=0)))
