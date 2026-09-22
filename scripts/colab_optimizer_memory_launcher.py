"""Common-prefix optimizer intervention; uses the existing Colab 10/12 runtime."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile


def launch_optimizer_memory(revision,content=Path('/content')):
    if re.fullmatch('[a-f0-9]{40}',revision) is None:raise ValueError('Immutable Git revision required')
    content=Path(content);python=content/'menia-activation-env-v1/bin/python'
    parent=content/'menia-results/presence-detection-v1/20260919T100527Z-2508e757.jsonl'
    composition=content/'menia-results/state-composition-v1/composition-20260919-v1.jsonl'
    if not all(p.is_file() for p in (python,parent,composition)):
        raise RuntimeError('Continuer dans le runtime existant avec les journaux et poids 10/12. Aucun réentraînement des parents.')
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
    if 'A100' not in hardware:raise RuntimeError('A100 required for exact reproduction')
    root=content/'menia-results/optimizer-memory-v1';root.mkdir(parents=True,exist_ok=True)
    repo=content/('menia-optimizer-memory-source-'+revision[:12])
    journal=root/'memory-20260919-v1.jsonl';archive=content/'menia-memoire-optimiseur.zip'
    status=root/'execution.json'
    if status.exists():
        old=json.loads(status.read_text())
        if old['revision']!=revision:raise RuntimeError('Existing experiment uses another source revision')
    manifest=dict(startedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=revision,
        parentJournal=str(parent),compositionJournal=str(composition),journal=str(journal),hardware=hardware,status='preparing')
    status.write_text(json.dumps(manifest,indent=2)+'\n')
    env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    log=root/'execution.log'
    def command(args,cwd=None):
        print('Running: '+' '.join(map(str,args)),flush=True)
        with log.open('ab',buffering=0) as stream:
            subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    try:
        if not repo.exists():
            command(['git','init',repo]);command(['git','-C',repo,'remote','add','origin','https://github.com/speed25200-cyber/Menia.git'])
        command(['git','-C',repo,'fetch','--depth','1','origin',revision]);command(['git','-C',repo,'checkout','--detach',revision])
        head=subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',repo,'status','--porcelain'],text=True).strip()
        if head!=revision or dirty:raise RuntimeError('Unexpected source state')
        command([python,'-c',"import importlib.metadata as m; expected={'torch':'2.8.0','numpy':'2.2.6','transformers':'4.56.2','accelerate':'1.10.1','safetensors':'0.6.2','huggingface-hub':'0.36.2','tokenizers':'0.22.2'}; actual={k:m.version(k).split('+')[0] for k in expected}; assert actual==expected,(actual,expected); print(actual)"],repo)
        command([python,'-m','unittest','tests_research.test_optimizer_memory','tests_language.test_optimizer_memory_ops',
            'tests_language.test_optimizer_memory_gpu','tests_research.test_optimization_loss_diagnostic','-v'],repo)
        manifest['status']='running';status.write_text(json.dumps(manifest,indent=2)+'\n')
        command([python,'-m','research.optimizer_memory_gpu',journal,'--parent-journal',parent,'--composition-journal',composition,
            *(['--resume'] if journal.exists() else [])],repo)
        manifest['status']='completed'
    except BaseException as exc:
        manifest.update(status='failed',errorType=type(exc).__name__,error=str(exc));raise
    finally:
        manifest['updatedUTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        status.write_text(json.dumps(manifest,indent=2)+'\n')
        # Small audit archive for immediate transfer; learned tensors and fork states remain in the runtime.
        weights={p.name:dict(bytes=p.stat().st_size,sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest())
                 for p in root.iterdir() if p.suffix in ('.safetensors','.pt')}
        (root/'checkpoint-manifest.json').write_text(json.dumps(weights,indent=2)+'\n')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted(root.iterdir()):
                if file.is_file() and file.suffix in ('.json','.jsonl','.log'):z.write(file,'optimizer-memory/'+file.name)
        print(json.dumps(dict(manifest,archive=str(archive),checkpointFiles=len(weights)),ensure_ascii=False),flush=True)
    return manifest
