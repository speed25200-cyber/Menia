"""Generate the canonical notebook from the published experiment-15 launcher."""
import hashlib
import json
from pathlib import Path
import subprocess

REVISION='0a86c18796c788400492f0c45036595057760e60'
ROOT=Path(__file__).resolve().parents[1]


def main():
    data=subprocess.check_output(['git','show',REVISION+':scripts/colab_state_interchange_launcher.py'],cwd=ROOT)
    sha=hashlib.sha256(data).hexdigest()
    code=f'''import hashlib, importlib.util, pathlib, urllib.request
REVISION = "{REVISION}"
url = f"https://raw.githubusercontent.com/speed25200-cyber/Menia/{{REVISION}}/scripts/colab_state_interchange_launcher.py"
data = urllib.request.urlopen(url, timeout=60).read()
assert hashlib.sha256(data).hexdigest() == "{sha}", "Launcher content changed"
path = pathlib.Path("/content/menia-state-interchange-launcher.py")
path.write_bytes(data)
spec = importlib.util.spec_from_file_location("menia_interchange_launcher", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.launch_state_interchange(REVISION)
print(result)
'''
    def cell(kind,source):
        r=dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True))
        if kind=='code':r.update(execution_count=None,outputs=[])
        return r
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[cell('markdown','# Menia — transfert d’état ou copie de réponse\n\n'
            'Colab 15 : 19 584 passages, aucun entraînement. Conserver le runtime A100 et les poids audités du Colab 14. '
            'Les six tests logiciels sont exécutés avant les inférences. Le journal complet et les contrôles sont conservés. '
            'Ce test ne mesure pas la conscience subjective.\n\n'
            '[Protocole fixé](https://github.com/speed25200-cyber/Menia/blob/'+REVISION+'/docs/STATE_INTERCHANGE_PROTOCOL.md).'),
            cell('code',code),cell('code','from google.colab import files\nfiles.download("/content/menia-transfert-etat.zip")\n')])
    (ROOT/'notebooks/15_state_interchange_colab.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(scienceRevision=REVISION,launcherSHA256=sha)))


if __name__=='__main__':main()
