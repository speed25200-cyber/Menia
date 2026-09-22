"""Publish a notebook pointing to an immutable, hash-checked Colab22 launcher."""
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
    raw=subprocess.check_output(['git','show',revision+':scripts/colab_value_action_learning_launcher.py'],cwd=ROOT)
    sha=hashlib.sha256(raw).hexdigest()
    code=f'''import hashlib, importlib.util, pathlib, urllib.request
REVISION = "{revision}"
url = f"https://raw.githubusercontent.com/speed25200-cyber/Menia/{{REVISION}}/scripts/colab_value_action_learning_launcher.py"
data = urllib.request.urlopen(url, timeout=60).read()
assert hashlib.sha256(data).hexdigest() == "{sha}", "Launcher changed"
path = pathlib.Path("/content/menia-value-action-learning-v2-launcher.py")
path.write_bytes(data)
spec = importlib.util.spec_from_file_location("menia_value_action_learning_launcher", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.launch_value_action_learning(REVISION)
print(result)
'''
    def cell(kind,source):
        result=dict(cell_type=kind,metadata={},source=source.splitlines(keepends=True))
        if kind=='code':result.update(execution_count=None,outputs=[])
        return result
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),
        cells=[cell('markdown','# Menia — apprentissage des associations valeur-action (v2)\n\n'
            'Colab 22 : neuf adaptateurs, 576 mises à jour, 9 216 appels réservés au test. '
            'Conserver le runtime A100 et l’environnement Menia. Trois entraînements appariés sur trois initialisations ; une seule tentative. '
            'Le choix sous risque public est un contrôle fonctionnel, pas une preuve de conscience.\n\n'
            f'[Protocole fixé](https://github.com/speed25200-cyber/Menia/blob/{revision}/docs/VALUE_ACTION_LEARNING_PROTOCOL.md).\n'),
            cell('code',code),cell('code','from google.colab import files\nfiles.download("/content/menia-apprentissage-valeurs-actions-v2.zip")\n')])
    path=ROOT/'notebooks/22_value_action_learning_colab.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(scienceRevision=revision,launcherSHA256=sha)))


if __name__=='__main__':main()
