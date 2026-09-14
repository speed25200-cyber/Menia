"""Rebuild the checked-in notebook without notebook dependencies."""
import json
from pathlib import Path

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(True)}
cells = [
md('''# Menia — Colab A100 80 Go
Adaptation LoRA d’un modèle existant pour des épreuves fonctionnelles.
**Ce notebook ne crée ni ne prouve une conscience subjective.**

Les cellules lancent un entraînement réel lorsqu’elles sont exécutées ici.
Le dépôt ne contient pas encore de poids Menia entraînés.
Démarre par 20 steps ; ce run ne valide que le fonctionnement technique.
Le corpus synthétique n’est pas suffisant pour un assistant généraliste.

Choisir Exécution → Modifier le type d’exécution → GPU A100.
Les 80 Go seront vérifiés. Si Colab alloue 40 Go, ce profil s’arrête.
'''),
code('''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys
ROOT = Path('/content/drive/MyDrive/Menia')
ROOT.mkdir(parents=True, exist_ok=True)
REPO = Path('/content/Menia')
if not REPO.exists():
    subprocess.run(['git', 'clone', 'https://github.com/speed25200-cyber/Menia.git', str(REPO)], check=True)
print(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(REPO/'requirements-train.txt')], check=True)
'''),
md('''Si Colab demande un redémarrage après installation, redémarrer la session puis reprendre à la cellule suivante. Ne pas installer vLLM/Unsloth en plus dans ce runtime.'''),
code('''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import os, subprocess, sys, datetime, json
ROOT = Path('/content/drive/MyDrive/Menia')
REPO = Path('/content/Menia')
os.chdir(REPO)
import torch
assert torch.cuda.is_available(), 'Activer le GPU'
gpu = torch.cuda.get_device_properties(0)
print(gpu.name, gpu.total_memory / 1024**3, 'GiB')
assert 'A100' in gpu.name and gpu.total_memory >= 70 * 1024**3, 'Ce profil exige A100 80 Go'
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
DATA = ROOT/'data-format-v1'
subprocess.run([sys.executable, '-m', 'menia.data', '--out', str(DATA)], check=True)
RUN = ROOT/'runs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
print('Run:', RUN)
'''),
md('''## Entraînement technique
20 steps pour vérifier chargement, loss et sauvegarde. Augmenter uniquement après revue du corpus et de la validation.
La confiance cible 1.0 du corpus jouet n’est pas une calibration de confiance.
Pour un corpus réel, fournir des réponses relues dans `assistant_text` et conserver des jeux séparés.
'''),
code('''STEPS = 20
command = [sys.executable, '-m', 'menia.train', '--data', str(DATA), '--out', str(RUN), '--steps', str(STEPS)]
subprocess.run(command, check=True)
print((RUN/'metrics.json').read_text())
'''),
md('''## Reprise après interruption
Ne pas réexécuter la cellule qui crée RUN. Renseigner les chemins exacts sauvegardés.
Les checkpoints contiennent les états d’optimiseur ; `adapter/` est destiné à l’inférence.
Exemple à adapter :
```python
RUN = ROOT/'runs'/'IDENTIFIANT_EXISTANT'
subprocess.run([sys.executable, '-m', 'menia.train', '--data', str(DATA),
 '--out', str(RUN), '--steps', '100', '--resume', str(RUN/'checkpoint-20')], check=True)
```
'''),
md('''## Évaluation tenue à l’écart
Comparer base/adapté sur les mêmes entrées et budget. Chaque invocation libère son modèle à la fin.
La variante sans état conserve la réponse attendue originale : c’est une ablation de dépendance à l’information, pas un test de calibration.
Les données de test partagent des gabarits avec le train ; prévoir des gabarits nouveaux relus avant toute conclusion de généralisation.
'''),
code('''REPORTS = RUN/'reports'
REPORTS.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, '-m', 'menia.evaluate', '--data', str(DATA/'test.jsonl'), '--out', str(REPORTS/'baselines.json')], check=True)
for name, adapter, ablation in [('base', None, 'none'), ('adapted', RUN/'adapter', 'none'),
                                 ('no_system', RUN/'adapter', 'no-system'), ('no_state', RUN/'adapter', 'no-state')]:
    predictions = REPORTS/f'{name}.jsonl'
    cmd = [sys.executable, '-m', 'menia.predict', '--data', str(DATA/'test.jsonl'), '--out', str(predictions), '--ablation', ablation]
    if adapter:
        cmd += ['--adapter', str(adapter)]
    else:
        cmd += ['--revision', json.loads((RUN/'run.json').read_text())['revision']]
    subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, '-m', 'menia.evaluate', '--data', str(DATA/'test.jsonl'), '--predictions', str(predictions), '--out', str(REPORTS/f'{name}-scores.json')], check=True)
'''),
md('''## Export HF fusionné
L’export fusionne les adaptateurs avec la révision exacte de la base. Aucun upload automatique des poids ou des données.
'''),
code('''EXPORT = RUN/'merged-hf'
subprocess.run([sys.executable, '-m', 'menia.export', '--run', str(RUN), '--out', str(EXPORT)], check=True)
print('Récupérer ce dossier sur le Mac:', EXPORT)
'''),
md('''## Mac puis iPhone
Sur Mac Apple Silicon, dans un environnement Python dédié :
```bash
pip install mlx-lm
pip freeze > export-environment.txt
python -m mlx_lm.convert --hf-path /chemin/merged-hf --mlx-path /chemin/menia-4bit -q --q-bits 4
python -m mlx_lm.generate --model /chemin/menia-4bit --prompt "Présente-toi brièvement. /no_think" --max-tokens 128
```
La conversion est une quantification après entraînement (PTQ), pas QAT.
Rejouer les épreuves après quantification avant de conclure à une qualité conservée.
Puis suivre `docs/IPHONE.md` pour compiler, importer le dossier et mesurer sur l’appareil.
Ne pas publier de score de conscience ni de débit extrapolé depuis Colab.
''')]
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"name": "Menia A100 80 Go", "provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
for i, cell in enumerate(cells): cell['id'] = f'menia-{i:02d}'
Path('notebooks/01_colab_a100.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n')
