"""Generate the canonical notebook from the published experiment-16 launcher."""
import hashlib
import json
from pathlib import Path
import subprocess

REVISION='b97753c1cca1f936e7319f971e1c1a4058df0a01'
ROOT=Path(__file__).resolve().parents[1]


def main():
    data=subprocess.check_output(['git','show',REVISION+':scripts/colab_cross_task_launcher.py'],cwd=ROOT)
    sha=hashlib.sha256(data).hexdigest()
    code=f'''import hashlib, importlib.util, pathlib, urllib.request
REVISION = "{REVISION}"
url = f"https://raw.githubusercontent.com/speed25200-cyber/Menia/{{REVISION}}/scripts/colab_cross_task_launcher.py"
data = urllib.request.urlopen(url, timeout=60).read()
assert hashlib.sha256(data).hexdigest() == "{sha}", "Launcher content changed"
path = pathlib.Path("/content/menia-cross-task-launcher.py")
path.write_bytes(data)
spec = importlib.util.spec_from_file_location("menia_cross_task_launcher", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.launch_cross_task_interchange(REVISION)
print(result)
'''
    def cell(kind,source):
        r=dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True))
        if kind=='code':r.update(execution_count=None,outputs=[])
        return r
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[cell('markdown','# Menia — transfert de contenu entre questions\n\n'
            'Colab 16 : 44 544 passages, aucun entraînement. Conserver le runtime A100 et les poids audités du Colab 14 et le journal du Colab 15. '
            'Les huit tests logiciels sont exécutés avant les inférences. Le journal complet et les contrôles sont conservés. '
            'Ce test ne mesure pas la conscience subjective.\n\n'
            '[Protocole fixé](https://github.com/speed25200-cyber/Menia/blob/'+REVISION+'/docs/CROSS_TASK_INTERCHANGE_PROTOCOL.md).'),
            cell('code',code),cell('code','from google.colab import files\nfiles.download("/content/menia-transfert-entre-taches.zip")\n')])
    (ROOT/'notebooks/16_cross_task_interchange_colab.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(scienceRevision=REVISION,launcherSHA256=sha)))


if __name__=='__main__':main()
