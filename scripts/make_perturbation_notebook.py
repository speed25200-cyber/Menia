"""Generate the separate paired-intervention Colab with the tested launcher."""
import argparse
import json
from pathlib import Path

from make_cross_model_notebook import make_notebook,md,code


def make_perturbation_notebook(revision):
    notebook=make_notebook(revision)
    bootstrap=Path(__file__).with_name('colab_bootstrap.py').read_text(encoding='utf-8')
    launcher=Path(__file__).with_name('colab_activation_launcher.py').read_text(encoding='utf-8')
    notebook['cells']=[md('''# Menia — capacités perturbées : expérience appariée

**Un seul bloc à exécuter. A100 de 40 ou 80 Go.**

1. Choisir **Exécution → Modifier le type d'exécution → GPU A100**.
2. Cliquer sur **▶** ci-dessous ou sur **Tout exécuter**.
3. Autoriser Drive, puis laisser le programme aller au bout.
4. Transmettre **`menia-perturbations.zip`**, même si le programme s'arrête.

Le calcul normal est comparé à une manipulation sans effet et à deux perturbations temporaires.
Le texte et la graine de génération sont identiques pour les quatre versions de chaque question.
Les poids de Qwen3-4B restent inchangés. Les perturbations disparaissent après chaque appel.

**32 appels de contrôle technique**, puis **672 appels d'étude** : 384 d'apprentissage,
96 de validation et 192 de test, soit 48 questions de test présentées sous quatre conditions.
Les questions sont nouvelles. Les prévisions et décisions de test sont enregistrées avant le premier token.
Les routes « réponse directe » ou « outil de vérification » sont ensuite exécutées sur les mêmes candidats.

Les témoins incluent un calcul connaissant le type de perturbation et des états d'un autre problème
soumis à la même perturbation. Ils permettent de distinguer la détection d'une panne de la prévision
de ses conséquences. Un résultat positif concernerait un moniteur ajouté, pas une preuve de conscience.

L'installation réutilise la procédure corrigée. Le diagnostic est exportable dès le départ.
Les tentatives sont conservées dans **`MyDrive/Menia/perturbation-monitor-v1`**, séparément des anciennes.
Si le téléchargement est bloqué, retrouver l'archive dans le panneau **Fichiers** de Colab.
Environ 8 Go de poids, plus les bibliothèques, sont nécessaires. Ce lancement consomme le quota Colab.
La durée de cette nouvelle expérience sur A100 n'est pas encore mesurée.
'''),code('# Lancer ou reprendre le nouveau protocole\n'+launcher+'\nBOOTSTRAP_SOURCE = '+repr(bootstrap)+
          '\nMENIA_LAUNCH_RESULT = launch_activation('+repr(revision)+", BOOTSTRAP_SOURCE, profile='perturbation')\n")]
    for i,cell in enumerate(notebook['cells']):
        cell['id']=f'menia-perturbation-{i:02d}'
    notebook['metadata']['colab']['name']='Menia — capacités perturbées — lancement autonome'
    return notebook


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('revision')
    args=parser.parse_args()
    Path('notebooks/05_perturbation_monitor_colab.ipynb').write_text(
        json.dumps(make_perturbation_notebook(args.revision),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
