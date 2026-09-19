"""Pinned, resumable Colab-16 execution using the existing audited Colab-14 parents."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile


def launch_cross_task_interchange(revision,content=Path('/content')):
    if re.fullmatch('[a-f0-9]{40}',revision) is None:raise ValueError('Immutable source revision required')
    content=Path(content);python=content/'menia-activation-env-v1/bin/python'
    parent=content/'menia-results/optimizer-memory-v1/memory-20260919-v1.jsonl'
    interchange=content/'menia-results/state-interchange-v1/interchange-20260919-v1.jsonl'
    if not python.is_file() or not parent.is_file() or not interchange.is_file():raise RuntimeError('Keep the audited runtime, Colab-14 weights and Colab-15 journal; no automatic retraining')
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
    if 'A100' not in hardware:raise RuntimeError('A100 required')
    root=content/'menia-results/cross-task-interchange-v1';root.mkdir(parents=True,exist_ok=True)
    repo=content/('menia-cross-task-source-'+revision[:12]);status=root/'execution.json'
    journal=root/'cross-task-20260919-v1.jsonl';archive=content/'menia-transfert-entre-taches.zip'
    if status.exists() and json.loads(status.read_text())['revision']!=revision:raise RuntimeError('Source revision changed')
    manifest=dict(startedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=revision,
                  parentJournal=str(parent),interchangeJournal=str(interchange),journal=str(journal),hardware=hardware,status='preparing')
    status.write_text(json.dumps(manifest,indent=2)+'\n')
    env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def command(args,cwd=None):
        with (root/'execution.log').open('ab',buffering=0) as output:
            subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    try:
        if not repo.exists():
            command(['git','init',repo]);command(['git','-C',repo,'remote','add','origin','https://github.com/speed25200-cyber/Menia.git'])
        command(['git','-C',repo,'fetch','--depth','1','origin',revision]);command(['git','-C',repo,'checkout','--detach',revision])
        head=subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()
        if head!=revision or subprocess.check_output(['git','-C',repo,'status','--porcelain'],text=True).strip():raise RuntimeError('Unexpected source tree')
        command([python,'-m','unittest','tests_research.test_cross_task_hypotheses','tests_research.test_cross_task_interchange',
                 'tests_language.test_cross_task_interchange_gpu','-v'],repo)
        manifest['status']='running';status.write_text(json.dumps(manifest,indent=2)+'\n')
        command([python,'-m','research.cross_task_interchange_gpu',journal,'--parent-journal',parent,'--interchange-journal',interchange,
                 *(['--resume'] if journal.exists() else [])],repo)
        manifest['status']='completed'
    except BaseException as exc:
        manifest.update(status='failed',errorType=type(exc).__name__,error=str(exc));raise
    finally:
        manifest['updatedUTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        status.write_text(json.dumps(manifest,indent=2)+'\n')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted(root.iterdir()):
                if file.is_file() and file.suffix in ('.json','.jsonl','.log'):z.write(file,'cross-task/'+file.name)
        print(json.dumps(dict(manifest,archive=str(archive)),ensure_ascii=False),flush=True)
    return manifest
