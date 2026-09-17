"""Single-block, no-Drive notebook for the fixed presence-detection protocol."""
import argparse
import json
from pathlib import Path
from make_cross_model_notebook import make_notebook, md, code


def make_presence_notebook(revision):
    notebook=make_notebook(revision)
    bootstrap=Path(__file__).with_name('colab_bootstrap.py').read_text(encoding='utf-8')
    launcher=Path(__file__).with_name('colab_activation_launcher.py').read_text(encoding='utf-8')
    notebook['cells']=[md('''# Menia — détecter une perturbation interne

**Un seul bloc. A100 40 ou 80 Go. Sans Google Drive ni fichier à importer.**

Choisir **A100**, puis **Tout exécuter**. Le fichier à conserver et transmettre
est **menia-detection-presence.zip**. Si le téléchargement automatique
échoue, il se trouve à la racine `/content` du panneau Fichiers.

La réplication précédente n'a pas reproduit la localisation. Un diagnostic
ajouté après coup suggérait toutefois une information de **présence** de la
perturbation. Cette expérience teste cette seule question, avec une mesure sans
seuil et une règle de lecture **fixées avant la collecte**, sur **trois
initialisations et trois jeux de phrases nouveaux**.

Neuf adaptateurs : pour chaque répétition, un contrôle visible, un entraînement
sur perturbation forte et un témoin à cibles mélangées. L'entraînement n'utilise
que la couche 17 à pleine intensité. Le test ajoute une intensité réduite et
deux autres couches jamais vues, ainsi qu'une présentation inversée.
Tous les entraînements sont terminés avant l'évaluation. Aucune sélection.

**576 mises à jour, 2 880 passages avec rétropropagation, 12 864 évaluations.**
Le calcul seul est estimé autour de 30 minutes à partir de la réplication
précédente ; ce nouveau budget n'a pas encore été chronométré sur A100.
Prévoir davantage avec l'installation. Le calcul consomme le quota Colab.

L'archive contient les diagnostics, tous les résultats et les neuf adaptateurs
(environ 106 Mo de poids avant compression). Une interruption conserve ce qui
est terminé ; seul l'adaptateur incomplet repart de son initialisation fixe.
Relancer une tentative complète ne refait aucune inférence. Le stockage Colab
est temporaire : télécharger l'archive avant la suppression de l'environnement.

Même réussi, ce protocole établirait un détecteur entraîné de perturbations
artificielles. Il n'établit pas la conscience et n'installe aucun poids dans
l'application iPhone.
'''), code(launcher+'\nBOOTSTRAP_SOURCE = '+repr(bootstrap)+
          '\nMENIA_LAUNCH_RESULT = launch_activation('+repr(revision)+
          ", BOOTSTRAP_SOURCE, profile='presence', storage='local')\n")]
    for i,cell in enumerate(notebook['cells']): cell['id']=f'menia-presence-{i:02d}'
    notebook['metadata']['colab']['name']='Menia — détection de présence — sans Drive'
    return notebook


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('revision'); args=parser.parse_args()
    Path('notebooks/10_presence_detection_colab.ipynb').write_text(json.dumps(make_presence_notebook(args.revision),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
