"""Build the single-attempt Colab18 entry point from immutable published sources."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('revision')
    revision=parser.parse_args().revision
    if re.fullmatch('[a-f0-9]{40}',revision) is None:raise ValueError('Immutable revision required')
    data=subprocess.check_output(['git','show',revision+':scripts/colab_native_choice_launcher.py'],cwd=ROOT)
    sha=hashlib.sha256(data).hexdigest()
    code=f'''import hashlib, importlib.util, pathlib, urllib.request
REVISION = "{revision}"
url = f"https://raw.githubusercontent.com/speed25200-cyber/Menia/{{REVISION}}/scripts/colab_native_choice_launcher.py"
data = urllib.request.urlopen(url, timeout=60).read()
assert hashlib.sha256(data).hexdigest() == "{sha}", "Launcher changed"
path = pathlib.Path("/content/menia-native-choice-launcher.py")
path.write_bytes(data)
spec = importlib.util.spec_from_file_location("menia_native_choice_launcher", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result = module.launch_native_choices(REVISION)
print(result)
'''
    def cell(kind,text):
        value=dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True))
        if kind=='code':value.update(execution_count=None,outputs=[])
        return value
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),
        cells=[cell('markdown','# Menia — choix natifs et coût de vérification\n\n'
            'Colab 18 : 96 questions nouvelles, 24 conditions par question et 180 contrôles de risque explicite. '
            'Qwen3-4B sans mise à jour. Conserver le runtime A100, l’environnement Menia et le journal complet du Colab 17. '
            'Le lanceur vérifie les comptes historiques et refuse une tentative existante. '
            'Sept tests sont exécutés avant collecte. Les demandes de vérification appellent un outil exact. '
            'Les points du jeu et les durées mesurées sont distincts. Ce test n’établit pas une conscience.\n\n'
            f'[Protocole fixé](https://github.com/speed25200-cyber/Menia/blob/{revision}/docs/NATIVE_CHOICE_PROTOCOL.md).\n'),
            cell('code',code),cell('code','from google.colab import files\nfiles.download("/content/menia-choix-natifs.zip")\n')])
    path=ROOT/'notebooks/18_native_choice_colab.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(scienceRevision=revision,launcherSHA256=sha)))


if __name__=='__main__':main()
