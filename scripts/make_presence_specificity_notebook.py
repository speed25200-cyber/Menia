"""One-cell continuation notebook, pinned to the question-specificity science revision."""
import argparse
import json
from pathlib import Path
from make_cross_model_notebook import make_notebook,md,code


def make_specificity_notebook(revision):
    nb=make_notebook(revision)
    launcher=Path(__file__).with_name('colab_specificity_launcher.py').read_text(encoding='utf-8')
    nb['cells']=[md('''# Menia — le signal suit-il la question ?

**Continuer dans l’environnement A100 du Colab 10.** Les neuf adaptateurs et
le journal parent doivent y être encore présents. Aucun entraînement n’est
relancé. Ce notebook vérifie leurs empreintes, la version du code et les tests,
puis exécute **9 216 évaluations** sur de nouvelles phrases.

Quatre questions : présence ; absence avec réponses inversées ; deux lectures
publiques dont la cible ne dépend pas de la perturbation. Base et trois
adaptateurs, familles visible et cachée, trois répétitions, copies témoins.
Le protocole et sa règle de lecture sont fixés avant les inférences.

Exécuter l’unique cellule. Conserver `/content/menia-specificite-questions.zip`.
L’archive contient résultats et diagnostics ; les poids déjà conservés dans
`menia-detection-presence.zip` ne sont pas dupliqués. Après une interruption,
la même tentative peut reprendre. Une tentative complète ne refait pas les
inférences. Estimation à partir du Colab 10 : environ 13 minutes de calcul,
auxquelles s’ajoutent chargement et vérifications. Durée non encore mesurée.

Le résultat portera sur la spécificité du signal à la question, sans verdict
sur la conscience ni installation sur iPhone.
'''),code(launcher+'\nMENIA_SPECIFICITY_RESULT = launch_specificity('+repr(revision)+')\n')]
    for i,c in enumerate(nb['cells']):c['id']=f'menia-specificity-{i:02d}'
    nb['metadata']['colab']['name']='Menia — spécificité des questions — continuation du Colab 10'
    return nb


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('revision');args=parser.parse_args()
    Path('notebooks/11_presence_specificity_colab.ipynb').write_text(json.dumps(make_specificity_notebook(args.revision),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
