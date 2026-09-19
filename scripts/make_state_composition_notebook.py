"""Generate a notebook pinned to the frozen state/question composition protocol."""
import argparse
import json
from pathlib import Path
from make_cross_model_notebook import make_notebook,md,code


def make_composition_notebook(revision):
    nb=make_notebook(revision)
    launcher=Path(__file__).with_name('colab_composition_launcher.py').read_text(encoding='utf-8')
    nb['cells']=[md('''# Menia — composer état interne et consigne

**Continuer dans le runtime A100 du Colab 10**, avec son environnement isolé,
son journal et ses neuf checkpoints conservés. Ne pas réentraîner les parents.

Le Colab 11 détectait une perturbation mais ne renversait pas correctement
le sens du signal avec la consigne. Cette expérience séparée compare trois
apprentissages, tous issus des mêmes parents : supervision interne correcte,
supervision interne mélangée, consignes visibles sans nouvelles étiquettes
internes. Tous apprennent aussi à préserver la lecture d'un repère public.

**Neuf adaptateurs, 576 mises à jour, 25 920 évaluations.** Les formulations
reformulées et les réponses 2/3 sont réservées au test. Aucun checkpoint n'est
sélectionné selon ses scores. Les critères sont publiés dans
`docs/STATE_COMPOSITION_PROTOCOL.md` avant la collecte.

Exécuter l'unique cellule. Environ une heure de calcul estimée à partir des
expériences précédentes, durée non mesurée pour ce nouveau protocole.
Conserver `/content/menia-composition-etat.zip` : journal, neuf nouveaux
adaptateurs et diagnostics. Une interruption peut reprendre dans ce même
runtime ; une tentative complète ne refait pas les inférences.

Cette expérience teste une capacité fonctionnelle limitée. Elle ne valide
pas la conscience et n'installe aucun poids sur l'iPhone.
'''),code(launcher+'\nMENIA_COMPOSITION_RESULT = launch_composition('+repr(revision)+')\n')]
    for i,c in enumerate(nb['cells']):c['id']=f'menia-composition-{i:02d}'
    nb['metadata']['colab']['name']='Menia — composition état et consigne — continuation du Colab 10'
    return nb


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('revision');a=p.parse_args()
    Path('notebooks/12_state_composition_colab.ipynb').write_text(json.dumps(make_composition_notebook(a.revision),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
