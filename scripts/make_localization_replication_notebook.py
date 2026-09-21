"""Single-block, no-Drive notebook for a fixed prospective replication."""
import argparse
import json
from pathlib import Path
from make_cross_model_notebook import make_notebook, md, code


def make_replication_notebook(revision):
    notebook=make_notebook(revision)
    bootstrap=Path(__file__).with_name('colab_bootstrap.py').read_text(encoding='utf-8')
    launcher=Path(__file__).with_name('colab_activation_launcher.py').read_text(encoding='utf-8')
    notebook['cells']=[md('''# Menia — reproduire le signal interne

**Un seul bloc. A100 40 ou 80 Go. Sans Google Drive ni fichier à importer.**

Choisir **A100**, puis **Tout exécuter**. Le fichier à conserver et transmettre
est **menia-replication-localisation.zip**. Si le téléchargement automatique
échoue, il se trouve à la racine `/content` du panneau Fichiers.

Le diagnostic précédent a montré un signal partiel. Cette expérience vérifie
sa répétition avec **trois initialisations et trois jeux de phrases différents**.
Elle croise l'ordre des phrases et les numéros affichés, afin d'examiner les biais
de position. Le signal visible et la lecture servent de contrôles.

Neuf adaptateurs : pour chaque répétition, un contrôle visible, un entraînement
sur perturbation forte et un témoin à cibles mélangées. Tous les entraînements
sont terminés avant l'évaluation. Aucune sélection du meilleur lancement.

**288 mises à jour, 1 440 passages avec rétropropagation, 12 480 évaluations.**
Prévoir plusieurs dizaines de minutes, installation comprise. Le calcul seul
est estimé autour de 25 minutes à partir du pilote antérieur ; ce nouveau
budget n'a pas encore été chronométré sur A100. Le calcul consomme le quota Colab.

L'archive contient les diagnostics, tous les résultats et les neuf adaptateurs
(environ 106 Mo de poids avant compression). Une interruption conserve ce qui
est terminé ; seul l'adaptateur incomplet repart de son initialisation fixe.
Relancer une tentative complète ne refait aucune inférence. Le stockage Colab
est temporaire : télécharger l'archive avant la suppression de l'environnement.

Ce protocole teste la reproductibilité d'une capacité limitée. Il n'établit pas
la conscience et n'installe aucun poids dans l'application iPhone.
'''), code(launcher+'\nBOOTSTRAP_SOURCE = '+repr(bootstrap)+
          '\nMENIA_LAUNCH_RESULT = launch_activation('+repr(revision)+
          ", BOOTSTRAP_SOURCE, profile='replication', storage='local')\n")]
    for i,cell in enumerate(notebook['cells']): cell['id']=f'menia-replication-{i:02d}'
    notebook['metadata']['colab']['name']='Menia — réplication localisation — sans Drive'
    return notebook


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('revision'); args=parser.parse_args()
    Path('notebooks/09_localization_replication_colab.ipynb').write_text(json.dumps(make_replication_notebook(args.revision),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
