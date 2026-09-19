"""Generate a one-block Colab for prospective replay-based routing."""
import argparse
import json
from pathlib import Path
from make_cross_model_notebook import make_notebook, md, code


def make_replay_notebook(revision):
    notebook=make_notebook(revision)
    bootstrap=Path(__file__).with_name('colab_bootstrap.py').read_text(encoding='utf-8')
    launcher=Path(__file__).with_name('colab_activation_launcher.py').read_text(encoding='utf-8')
    notebook['cells']=[md('''# Menia — apprendre quand répondre, vérifier ou s'abstenir

**Un seul bloc. A100 40 ou 80 Go. Sans montage Google Drive.**
Choisir A100 dans les paramètres d'exécution, puis lancer le bloc ci-dessous.
Le ZIP **menia-strategies-rejeu.zip** est téléchargé à la fin, avec un diagnostic
même en cas d'erreur. Conserver le ZIP avant que Colab efface sa session.

Le test produit **720 réponses Qwen3-4B** : 240 pour apprendre les moniteurs,
trois séries de 96 pour améliorer le contrôleur par rejeu, puis 192 réservées au
test final. Les poids de Qwen restent fixes. Les stratégies choisissent entre
réponse directe, outil de vérification et abstention ; aucune relance imaginaire.

Chaque série utilise une stratégie figée avant ses réponses. Après chaque série,
les stratégies candidates sont comparées sur les données déjà reçues. Le test
final ne modifie ni moniteurs ni contrôleurs. Une stratégie sans états internes
et un contrôle à étiquettes mélangées sont évalués sur les mêmes réponses.

Une vérification déterministe supplémentaire, explicitement comptée comme audit
lorsqu'elle n'est pas choisie, complète le tableau des conséquences rejouables.
Le coût de décision est exprimé en points ; appels et durées sont séparés.
Le nombre obtenu et le respect du format sont mesurés séparément.

Il s'agit d'une adaptation limitée de l'idée de Dream-RSI, pas d'une reproduction
de son orchestrateur ni d'un test de conscience. Le contrôleur exporté est
expérimental et n'est pas installé sur l'iPhone. Le Colab 06 reste inchangé.
La durée sur A100 et les gains de ce nouveau pilote ne sont pas encore mesurés.
Le lancement consomme le quota Colab. Relancer une tentative complète n'exécute
pas de nouveaux appels ; une interruption conserve les échecs avant reprise.
'''),code(launcher+'\nBOOTSTRAP_SOURCE = '+repr(bootstrap)+
           '\nMENIA_LAUNCH_RESULT = launch_activation('+repr(revision)+
           ", BOOTSTRAP_SOURCE, profile='replay', storage='local')\n")]
    for i,cell in enumerate(notebook['cells']): cell['id']=f'menia-replay-{i:02d}'
    notebook['metadata']['colab']['name']='Menia — stratégies par rejeu — sans Drive'
    return notebook


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('revision'); args=parser.parse_args()
    Path('notebooks/07_replay_controller_colab.ipynb').write_text(json.dumps(make_replay_notebook(args.revision),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
