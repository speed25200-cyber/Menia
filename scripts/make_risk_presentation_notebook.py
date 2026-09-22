"""Publish a notebook pointing to an immutable, hash-checked Colab19 launcher."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('revision');revision=parser.parse_args().revision
    if re.fullmatch('[a-f0-9]{40}',revision) is None:raise ValueError('Immutable revision required')
    raw=subprocess.check_output(['git','show',revision+':scripts/colab_risk_presentation_launcher.py'],cwd=ROOT)
    sha=hashlib.sha256(raw).hexdigest()
    code=f'''import hashlib, importlib.util, pathlib, urllib.request
REVISION = "{revision}"
url = f"https://raw.githubusercontent.com/speed25200-cyber/Menia/{{REVISION}}/scripts/colab_risk_presentation_launcher.py"
data = urllib.request.urlopen(url, timeout=60).read()
assert hashlib.sha256(data).hexdigest() == "{sha}", "Launcher changed"
path = pathlib.Path("/content/menia-risk-presentation-launcher.py")
path.write_bytes(data)
spec = importlib.util.spec_from_file_location("menia_risk_presentation_launcher", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.launch_risk_presentation(REVISION)
print(result)
'''
    def cell(kind,source):
        result=dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True))
        if kind=='code':result.update(execution_count=None,outputs=[])
        return result
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),
        cells=[cell('markdown','# Menia — ordre des options et choix sous risque\n\n'
            'Colab 19 : 864 décisions, facteurs indépendants, cinq tests avant collecte. '
            'Conserver le runtime A100 et l’environnement Menia. Aucun entraînement ; une seule tentative. '
            'Le choix sous risque public est un contrôle fonctionnel, pas une preuve de conscience.\n\n'
            f'[Protocole fixé](https://github.com/speed25200-cyber/Menia/blob/{revision}/docs/RISK_PRESENTATION_PROTOCOL.md).\n'),
            cell('code',code),cell('code','from google.colab import files\nfiles.download("/content/menia-presentation-risque.zip")\n')])
    path=ROOT/'notebooks/19_risk_presentation_colab.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(scienceRevision=revision,launcherSHA256=sha)))


if __name__=='__main__':main()
