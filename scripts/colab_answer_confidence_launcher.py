"""One frozen Colab23 attempt; retain failed attempts and previous experiments."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile


TESTS = ('tests_research.test_answer_confidence_data',
         'tests_research.test_answer_confidence_plan',
         'tests_research.test_answer_confidence_baselines',
         'tests_research.test_answer_confidence_crossed',
         'tests_research.test_answer_confidence_study',
         'tests_language.test_answer_confidence_gpu',
         'tests_language.test_output_confidence_trace')


def launch_answer_confidence(revision, content=Path('/content')):
    if re.fullmatch('[a-f0-9]{40}',revision) is None:raise ValueError('Immutable revision required')
    content=Path(content);python=content/'menia-activation-env-v1/bin/python'
    if not python.is_file():raise RuntimeError('Conserver le runtime Menia A100 et son environnement Python.')
    root=content/'menia-results/native-answer-confidence-v1'
    root.mkdir(parents=True,exist_ok=True)
    status=root/'execution.json';journal=root/'native-answer-confidence-20260920-v1.jsonl'
    archive=content/'menia-confiance-native-v1.zip'
    repo=content/('menia-answer-confidence-source-'+revision[:12])
    if status.exists() or journal.exists() or archive.exists():raise RuntimeError('Tentative existante : conserver et auditer ; aucun nouveau lancement automatique.')
    manifest=dict(startedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=revision,journal=str(journal),status='preparing')
    with status.open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2)
    env=dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def command(args,cwd=None):
        with (root/'execution.log').open('ab',buffering=0) as output:
            subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    try:
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
        manifest['hardware']=hardware
        if 'A100' not in hardware or int(hardware.split(',')[1].strip().split()[0])>=1000:raise RuntimeError('Free A100 required')
        if not repo.exists():
            command(['git','init',repo])
            command(['git','-C',repo,'remote','add','origin','https://github.com/speed25200-cyber/Menia.git'])
        if subprocess.check_output(['git','-C',repo,'status','--porcelain'],text=True).strip():raise RuntimeError('Source edits must be preserved')
        command(['git','-C',repo,'fetch','--depth','1','origin',revision])
        command(['git','-C',repo,'checkout','--detach',revision])
        if subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()!=revision:raise RuntimeError('Revision mismatch')
        command([python,'-m','unittest',*TESTS,'-v'],repo)
        manifest['status']='running';status.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        command([python,'-m','research.answer_confidence_learning_gpu',journal,repo/'artifacts/answer-confidence-training-data'],repo)
        manifest['status']='completed'
    except BaseException as error:
        manifest.update(status='failed',errorType=type(error).__name__,error=str(error));raise
    finally:
        manifest['updatedUTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        status.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(root.iterdir()):
                if path.is_file() and path.suffix in ('.json','.jsonl','.log','.safetensors'):z.write(path,'native-answer-confidence/'+path.name)
        print(json.dumps(dict(manifest,archive=str(archive)),ensure_ascii=False),flush=True)
    return manifest
