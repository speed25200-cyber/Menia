"""Launch the fixed numerical control only after the budget experiment finishes."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile


def launch_generation_numerics(revision, content=Path('/content')):
    if re.fullmatch('[a-f0-9]{40}',revision) is None: raise ValueError('Immutable revision required')
    content = Path(content); python = content/'menia-activation-env-v1/bin/python'
    previous = content/'menia-results/confidence-budget-v1/execution.json'
    if not python.is_file() or not previous.is_file(): raise RuntimeError('Existing experiment environment required')
    if json.loads(previous.read_text())['status'] != 'completed': raise RuntimeError('Budget experiment must finish first')
    hardware = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
    if '\n' in hardware or 'A100' not in hardware or int(hardware.split(',')[1].strip().split()[0]) >= 1000:
        raise RuntimeError('Free single A100 required; preserve active experiments')
    root = content/'menia-results/generation-numerics-v1'; root.mkdir(parents=True,exist_ok=True)
    journal = root/'generation-numerics-20260920-v1.jsonl'; status = root/'execution.json'
    archive = content/'menia-continuite-numerique-v1.zip'; repo = content/('menia-generation-numerics-source-'+revision[:12])
    if journal.exists() or status.exists() or archive.exists(): raise RuntimeError('Existing attempt; no automatic retry')
    manifest = dict(startedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=revision,
        status='preparing',journal=str(journal),hardware=hardware)
    with status.open('x',encoding='utf-8') as stream: json.dump(manifest,stream,indent=2)
    env = dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def command(args,cwd=None):
        with (root/'execution.log').open('ab',buffering=0) as output:
            subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    try:
        if not repo.exists():
            command(['git','init',repo]); command(['git','-C',repo,'remote','add','origin','https://github.com/speed25200-cyber/Menia.git'])
        if subprocess.check_output(['git','-C',repo,'status','--porcelain'],text=True).strip(): raise RuntimeError('Preserve source edits')
        command(['git','-C',repo,'fetch','--depth','1','origin',revision]); command(['git','-C',repo,'checkout','--detach',revision])
        if subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip() != revision: raise RuntimeError('Revision mismatch')
        command([python,'-m','unittest','tests_language.test_confidence_generation_numerics',
            'tests_language.test_confidence_generation_continuity','-v'],repo)
        manifest['status'] = 'running'; status.write_text(json.dumps(manifest,indent=2)+'\n')
        command([python,'-m','research.confidence_generation_numerics_gpu','--journal',journal],repo)
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest.update(status='failed',errorType=type(error).__name__,error=str(error)); raise
    finally:
        manifest['updatedUTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted(root.iterdir()):
                if file.is_file() and file.suffix in ('.json','.jsonl','.log'): z.write(file,'generation-numerics/'+file.name)
        print(json.dumps(dict(manifest,archive=str(archive)),ensure_ascii=False),flush=True)
    return manifest
