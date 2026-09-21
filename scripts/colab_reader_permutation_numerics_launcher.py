"""One isolated numerical diagnostic, only after the learning process exits."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile


def launch_reader_permutation_numerics(revision, parent_pid, content=Path('/content')):
    if re.fullmatch('[a-f0-9]{40}',revision) is None:
        raise ValueError('Immutable revision required')
    if type(parent_pid) is not int or parent_pid <= 0 or Path(f'/proc/{parent_pid}').exists():
        raise RuntimeError('The previous process must have exited; never stop or restart it here')
    content = Path(content)
    python = content/'menia-activation-env-v1/bin/python'
    parent = content/'menia-results/prospective-reader-learning-v1'
    previous = json.loads((parent/'execution.json').read_text(encoding='utf-8'))
    if previous['status'] != 'completed' or previous['revision'] != '97ca234a2e4f2f37f24f78ee60fd8a925a0a7404':
        raise RuntimeError('Completed parent experiment required')
    if not python.is_file() or not (parent/'prospective-reader-learning-20260920-v1.audit.json').is_file():
        raise RuntimeError('Existing environment and completed parent audit required')
    hardware = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
    if '\n' in hardware or 'A100' not in hardware or int(hardware.split(',')[1].strip().split()[0]) >= 1000:
        raise RuntimeError('A free single A100 is required; leave current jobs untouched')
    root = content/'menia-results/reader-permutation-numerics-v1'
    archive = content/'menia-diagnostic-numerique-lecteur-v1.zip'
    if root.exists() or archive.exists():
        raise RuntimeError('Preserve any previous numerical diagnostic attempt')
    root.mkdir(parents=True,exist_ok=False)
    journal = root/'reader-permutation-numerics-20260920-v1.jsonl'
    repo = content/('menia-reader-permutation-numerics-source-'+revision[:12])
    status = root/'execution.json'
    manifest = dict(startedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),revision=revision,
        status='preparing',parentPID=parent_pid,parentRevision=previous['revision'],hardware=hardware,journal=str(journal))
    status.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    env = dict(os.environ,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    def command(args,cwd=None):
        with (root/'execution.log').open('ab',buffering=0) as output:
            subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    try:
        if not repo.exists():
            command(['git','init',repo])
            command(['git','-C',repo,'remote','add','origin','https://github.com/speed25200-cyber/Menia.git'])
        if subprocess.check_output(['git','-C',repo,'status','--porcelain'],text=True).strip():
            raise RuntimeError('Preserve source edits')
        command(['git','-C',repo,'fetch','--depth','1','origin',revision])
        command(['git','-C',repo,'checkout','--detach',revision])
        if subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip() != revision:
            raise RuntimeError('Source revision mismatch')
        command([python,'-m','unittest','tests_language.test_reader_permutation_numerics',
                 'tests_language.test_audit_reader_permutation_numerics','-v'],repo)
        manifest['status'] = 'running'; status.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        command([python,'-m','research.reader_permutation_numerics','--journal',journal,'--parent-results',parent],repo)
        manifest['status'] = 'auditing'; status.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        command([python,'-m','research.audit_reader_permutation_numerics',journal,'--output',journal.with_suffix('.audit.json')],repo)
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest.update(status='failed',errorType=type(error).__name__,error=str(error)); raise
    finally:
        manifest['updatedUTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted(root.iterdir()):
                if file.is_file() and file.suffix in ('.json','.jsonl','.log'):
                    z.write(file,'reader-permutation-numerics/'+file.name)
        print(json.dumps(dict(manifest,archive=str(archive))),flush=True)
    return manifest
