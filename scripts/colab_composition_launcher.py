"""Train counterbalanced state/question composition from preserved experiment 10 weights."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile


def launch_composition(revision, content=Path('/content')):
    if re.fullmatch('[a-f0-9]{40}',revision) is None:raise ValueError('An immutable Git revision is required')
    content=Path(content);python=content/'menia-activation-env-v1/bin/python'
    parent=content/'menia-results/presence-detection-v1/20260919T100527Z-2508e757.jsonl'
    if not python.is_file() or not parent.is_file():
        raise RuntimeError('Continuer dans le runtime du Colab 10 avec son environnement et ses neuf checkpoints. Restaurer les fichiers existants si nécessaire ; ne pas réentraîner les parents.')
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
    if 'A100' not in hardware:raise RuntimeError('A100 expected; no automatic hardware fallback')
    root=content/'menia-results/state-composition-v1';root.mkdir(parents=True,exist_ok=True)
    repo=content/('menia-composition-source-'+revision[:12]);journal=root/'composition-20260919-v1.jsonl'
    archive=content/'menia-composition-etat.zip'
    manifest=dict(startedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=revision,
                  parentJournal=str(parent),journal=str(journal),hardware=hardware,status='preparing')
    status=root/'execution.json';status.write_text(json.dumps(manifest,indent=2)+'\n')
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
        if head!=revision or dirty:raise RuntimeError('Source revision mismatch or modified files')
        command([python,'-c',"import importlib.metadata as m; expected={'torch':'2.8.0','numpy':'2.2.6','transformers':'4.56.2','accelerate':'1.10.1','safetensors':'0.6.2','huggingface-hub':'0.36.2','tokenizers':'0.22.2'}; actual={k:m.version(k).split('+')[0] for k in expected}; assert actual==expected,(actual,expected); print(actual)"],repo)
        command([python,'-m','unittest','tests_research.test_state_composition','tests_language.test_state_composition_gpu','-v'],repo)
        manifest['status']='running';status.write_text(json.dumps(manifest,indent=2)+'\n')
        command([python,'-m','research.state_composition_gpu',journal,'--parent-journal',parent,*(['--resume'] if journal.exists() else [])],repo)
        manifest['status']='completed'
    except BaseException as exc:
        manifest.update(status='failed',errorType=type(exc).__name__,error=str(exc));raise
    finally:
        manifest['updatedUTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();status.write_text(json.dumps(manifest,indent=2)+'\n')
        # Include every new checkpoint; never replace or duplicate the parent files.
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted(root.iterdir()):
                if file.is_file():z.write(file,'composition/'+file.name)
        print(json.dumps(dict(manifest,archive=str(archive)),ensure_ascii=False),flush=True)
    return manifest
