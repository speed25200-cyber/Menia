"""Pinned Qwen tokenizer, synthetic raw trajectories; no pretrained model run."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import transformers

from research.answer_confidence_data import input_messages
from research.confidence_generation_continuity import compile_raw_continuations
from research.confidence_prompt_fork import compile_fork
from research.cross_model_prediction import MODELS,digest


def run():
    spec = MODELS['A']; tokenizer = AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
    files = {}
    for name in ('tokenizer_config.json','tokenizer.json','generation_config.json'):
        p = Path(hf_hub_download(spec['id'],name,revision=spec['revision'],local_files_only=True))
        assert spec['revision'] in p.parts
        data = p.read_bytes(); files[name] = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    question = 'Exemple technique pour contrôler la continuité de génération.'
    cases = []
    for answer in ('17','-3','réponse technique\nsur deux lignes','\n17'):
        messages = input_messages(question,answer)
        prompt_text = tokenizer.apply_chat_template(messages[:2],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prompt_ids = tokenizer.encode(prompt_text,add_special_tokens=False)
        # These are constructed token sequences, not sampled model responses.
        generated_ids = tokenizer.encode(answer,add_special_tokens=False)+[tokenizer.eos_token_id]
        raw_prefix = prompt_ids+generated_ids
        queries = dict(report=messages[-1]['content'],action='Choisis entre garder cette réponse et la vérifier. Réponds par un entier.')
        branches = compile_raw_continuations(tokenizer,raw_prefix,queries)
        reconstructed = compile_fork(tokenizer,messages[:3],queries)
        assert all(b['inputIds'][:len(raw_prefix)] == raw_prefix for b in branches.values())
        assert '<think>' in prompt_text and '</think>' in prompt_text
        assert all(b['inputIds'] != reconstructed['branches'][name]['inputIds'] for name,b in branches.items())
        failures = {}
        for name,prefix in (('missing_stop',raw_prefix[:-1]),('other_stop',raw_prefix[:-1]+[151643])):
            try: compile_raw_continuations(tokenizer,prefix,queries)
            except ValueError: failures[name] = True
            else: raise AssertionError('Unclosed/other-EOS history accepted')
        try: compile_raw_continuations(tokenizer,raw_prefix,dict(report='bad <|im_start|>',action='another query'))
        except ValueError: failures['role_marker'] = True
        else: raise AssertionError('Reserved marker accepted')
        cases.append(dict(answer=answer,promptTokens=len(prompt_ids),constructedCompletionTokens=len(generated_ids),
            rawPrefixHash=digest(raw_prefix),rawPrefixTokens=len(raw_prefix),
            retainedExactlyInBothBranches=True,rebuiltTextBranchesDiffer=True,closedTurnFailuresRejected=failures,
            branches={name:dict(tokens=len(b['inputIds']),hash=digest(b['inputIds'])) for name,b in branches.items()}))
    names = ('confidence_generation_continuity.py','confidence_generation_boundary_validation.py')
    return dict(schema='menia-generated-boundary-tokenizer-check-v1',origin='synthetic_raw_token_histories_with_pinned_tokenizer',
        model=spec,transformers=transformers.__version__,files=files,
        sourceHash=digest({n:Path(__file__).with_name(n).read_text(encoding='utf-8') for n in names}),
        modelLoaded=False,realGenerations=0,weightUpdates=0,cases=cases,
        scope='Four constructed histories preserve the exact raw prompt and completion IDs, including empty thinking markers and im_end. No cache or pretrained model was evaluated; no task, self-monitoring or consciousness result.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); report = run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(modelLoaded=False,realGenerations=0,cases=report['cases']),ensure_ascii=False))
