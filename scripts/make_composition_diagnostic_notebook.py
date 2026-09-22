"""Generate a pinned, no-training follow-up notebook."""
import argparse
import json
from pathlib import Path
from make_cross_model_notebook import make_notebook, md, code


def make_diagnostic_notebook(revision):
    nb = make_notebook(revision)
    launcher = Path(__file__).with_name('colab_composition_diagnostic_launcher.py').read_text(encoding='utf-8')
    nb['cells'] = [md('''# Menia — diagnostic des consignes et de la décision

Continuer dans le runtime A100 des Colab 10/12, avec leurs journaux et poids.
**Aucun entraînement : 13 824 évaluations, trois répétitions, quatre bras.**

Ce test compare la base, les parents et les variantes sur deux formulations.
Il teste aussi six seuils externes fixés uniquement sur l'ancien apprentissage,
en les comparant aux réponses effectivement produites par le modèle.
Les nouvelles phrases ne réutilisent aucun ancien exemple ; le vocabulaire
et les formulations de consigne restent ceux des expériences précédentes.

Exécuter l'unique cellule. Environ 25 à 35 minutes estimées, non garanties.
Le protocole complet est `docs/COMPOSITION_DIAGNOSTIC_PROTOCOL.md` dans la
révision scientifique figée. Une interruption peut reprendre dans ce runtime.
Conserver `/content/menia-diagnostic-composition.zip`. Le test ne prouve pas
la conscience et ne modifie aucun poids installé sur l'iPhone.
'''), code(launcher+'\nMENIA_DIAGNOSTIC_RESULT = launch_diagnostic('+repr(revision)+')\n')]
    for i, c in enumerate(nb['cells']): c['id'] = f'menia-composition-diagnostic-{i:02d}'
    nb['metadata']['colab']['name'] = 'Menia — diagnostic des consignes et de la décision'
    return nb


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('revision'); a = p.parse_args()
    Path('notebooks/13_composition_diagnostic_colab.ipynb').write_text(json.dumps(make_diagnostic_notebook(a.revision), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
