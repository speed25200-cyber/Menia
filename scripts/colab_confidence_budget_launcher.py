"""One fixed exploratory continuation; preserve all existing attempts and parent weights."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile


def launch_confidence_budget(revision,content=Path('/content')):
    if re.fullmatch('[a-f0-9]{40}',revision) is None: raise ValueError('Immutable revision required')
    content = Path(content); python = content/'menia-activation-env-v1/bin/python'
    parent = content/'menia-results/confidence-ranking-v1'; previous = content/'menia-results/final-training-fit-v1'
    if not python.is_file(): raise RuntimeError('Existing Menia environment required')
    if any(json.loads((p/'execution.json').read_text())['status'] != 'completed' for p in (parent,previous)):
        raise RuntimeError('Completed parent experiments required')
    root = content/'menia-results/confidence-budget-v1'; root.mkdir(parents=True,exist_ok=True)
    journal = root/'confidence-budget-20260920-v1.jsonl'; status = root/'execution.json'
    archive = content/'menia-budget-apprentissage-v1.zip'; repo = content/('menia-confidence-budget-source-'+revision[:12])
    if journal.exists() or status.exists() or archive.exists(): raise RuntimeError('Existing attempt; no automatic rerun')
    manifest = dict(startedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=revision,status='preparing',journal=str(journal))
    with status.open('x',encoding='utf-8') as stream: json.dump(manifest,stream,indent=2)
    env = dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def command(args,cwd=None):
        with (root/'execution.log').open('ab',buffering=0) as output:
            subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    try:
        hardware = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
        manifest['hardware'] = hardware
        if 'A100' not in hardware or int(hardware.split(',')[1].strip().split()[0])>=1000: raise RuntimeError('Free A100 required')
        if not repo.exists():
            command(['git','init',repo]); command(['git','-C',repo,'remote','add','origin','https://github.com/speed25200-cyber/Menia.git'])
        if subprocess.check_output(['git','-C',repo,'status','--porcelain'],text=True).strip(): raise RuntimeError('Preserve source edits')
        command(['git','-C',repo,'fetch','--depth','1','origin',revision]); command(['git','-C',repo,'checkout','--detach',revision])
        if subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip() != revision: raise RuntimeError('Revision mismatch')
        command([python,'-m','unittest','tests_research.test_confidence_budget_diagnostic',
            'tests_language.test_confidence_ranking_gpu','tests_language.test_answer_confidence_gpu','-v'],repo)
        manifest['status'] = 'running'; status.write_text(json.dumps(manifest,indent=2)+'\n')
        command([python,'-m','research.confidence_budget_gpu',journal,parent],repo)
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest.update(status='failed',errorType=type(error).__name__,error=str(error)); raise
    finally:
        manifest['updatedUTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted(root.iterdir()):
                if file.is_file() and file.suffix in ('.json','.jsonl','.log','.safetensors'): z.write(file,'confidence-budget/'+file.name)
        print(json.dumps(dict(manifest,archive=str(archive)),ensure_ascii=False),flush=True)
    return manifest
