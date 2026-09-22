"""Check the pinned real Qwen tokenizer on technical strings, without a model."""
import argparse
import hashlib
import json
from pathlib import Path

import transformers
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from research.answer_confidence_data import input_messages
from research.confidence_prompt_fork import compile_fork
from research.cross_model_prediction import MODELS,digest


def run():
    spec=MODELS['A']
    tokenizer=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],trust_remote_code=False)
    files={}
    for name in ('tokenizer_config.json','tokenizer.json'):
        p=Path(hf_hub_download(spec['id'],name,revision=spec['revision'],local_files_only=True))
        if spec['revision'] not in p.parts:raise ValueError('Tokenizer cache revision differs')
        raw=p.read_bytes();files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    question='Exemple technique pour contrôler la frontière du dialogue.'
    rows=[]
    for answer in ('17','-3','réponse technique\nsur deux lignes','\n17'):
        messages=input_messages(question,answer)
        queries=dict(report=messages[-1]['content'],
                     action='Choisis entre garder cette réponse et la vérifier. Réponds uniquement par garder ou vérifier.')
        fork=compile_fork(tokenizer,messages[:3],queries)
        assert fork['standaloneDiffers'] and all(not b['standaloneIsPrefix'] for b in fork['branches'].values())
        assert all(b['inputIds'][:len(fork['prefixIds'])]==fork['prefixIds'] for b in fork['branches'].values())
        assert all(q not in fork['prefixText'] for q in queries.values())
        rows.append(dict(answer=answer,**fork))
    sources={name:Path(__file__).with_name(name).read_text(encoding='utf-8') for name in
             ('confidence_prompt_fork.py','confidence_prompt_fork_validation.py')}
    return dict(schema='menia-confidence-real-tokenizer-fork-check-v1',origin='pinned_qwen_tokenizer_only',
                model=spec,transformers=transformers.__version__,files=files,sourceHash=digest(sources),
                modelLoaded=False,generations=0,trainingUpdates=0,cases=rows,
                scope='Four technical strings, actual pinned tokenizer. Corrected shared boundary only; no learned self-state, error prediction or action result.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();report=run();text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        if args.output.read_text(encoding='utf-8')!=text:raise ValueError('Existing different tokenizer receipt')
    else:args.output.write_text(text,encoding='utf-8')
    print(json.dumps(dict(model=report['model'],modelLoaded=False,generations=0,files=report['files'],
                         cases=[dict(prefixTokens=len(r['prefixIds']),position=r['position'],
                                     standaloneDiffers=r['standaloneDiffers'],
                                     allBranchesSharePrefix=all(b['inputIds'][:len(r['prefixIds'])]==r['prefixIds'] for b in r['branches'].values())) for r in report['cases']]),indent=2))
