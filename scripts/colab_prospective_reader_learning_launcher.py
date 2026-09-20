"""Run the frozen matched reader experiment once; preserve every failed attempt."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile


def launch_prospective_reader_learning(revision,content=Path('/content')):
    if re.fullmatch('[a-f0-9]{40}',revision) is None:raise ValueError('Immutable revision required')
    content=Path(content);python=content/'menia-activation-env-v1/bin/python'
    previous=content/'menia-results/retained-reader-control-v1/execution.json'
    if not python.is_file() or not previous.is_file() or json.loads(previous.read_text())['status']!='completed':
        raise RuntimeError('Completed retained reader control and existing environment required')
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
    if '\n' in hardware or 'A100' not in hardware or int(hardware.split(',')[1].strip().split()[0])>=1000:
        raise RuntimeError('Free single A100 required')
    root=content/'menia-results/prospective-reader-learning-v1';root.mkdir(parents=True,exist_ok=True)
    journal=root/'prospective-reader-learning-20260920-v1.jsonl';status=root/'execution.json'
    archive=content/'menia-apprentissage-lecteur-v1.zip';repo=content/('menia-prospective-reader-learning-source-'+revision[:12])
    if journal.exists() or status.exists() or archive.exists():raise RuntimeError('Existing attempt; preserve it')
    manifest=dict(startedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=revision,status='preparing',journal=str(journal),hardware=hardware)
    with status.open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2)
    env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def command(args,cwd=None):
        with (root/'execution.log').open('ab',buffering=0) as output:
            subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    try:
        if not repo.exists():
            command(['git','init',repo]);command(['git','-C',repo,'remote','add','origin','https://github.com/speed25200-cyber/Menia.git'])
        if subprocess.check_output(['git','-C',repo,'status','--porcelain'],text=True).strip():raise RuntimeError('Preserve source edits')
        command(['git','-C',repo,'fetch','--depth','1','origin',revision]);command(['git','-C',repo,'checkout','--detach',revision])
        if subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()!=revision:raise RuntimeError('Revision mismatch')
        command([python,'-m','unittest','tests_language.test_prospective_reader_learning',
            'tests_language.test_retained_state_reader','tests_language.test_prospective_learning_data',
            'tests_language.test_prospective_cache_interventions','tests_language.test_confidence_cached_action_decode','-v'],repo)
        manifest['status']='running';status.write_text(json.dumps(manifest,indent=2)+'\n')
        command([python,'-m','research.prospective_reader_learning_gpu','--journal',journal],repo)
        manifest['status']='auditing';status.write_text(json.dumps(manifest,indent=2)+'\n')
        command([python,'-m','research.prospective_reader_learning_audit',journal],repo)
        manifest['status']='completed'
    except BaseException as error:
        manifest.update(status='failed',errorType=type(error).__name__,error=str(error));raise
    finally:
        manifest['updatedUTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        status.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted(root.iterdir()):
                if file.is_file() and file.suffix in ('.json','.jsonl','.log','.safetensors'):z.write(file,'prospective-reader-learning/'+file.name)
        print(json.dumps(dict(manifest,archive=str(archive)),ensure_ascii=False),flush=True)
    return manifest
