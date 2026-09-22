"""One-block diagnostic notebook with fixed scientific revision."""
import argparse
import json
from pathlib import Path
from make_cross_model_notebook import make_notebook,md,code


def make_diagnostic_notebook(revision):
    notebook=make_notebook(revision)
    bootstrap=Path(__file__).with_name('colab_bootstrap.py').read_text(encoding='utf-8')
    launcher=Path(__file__).with_name('colab_activation_launcher.py').read_text(encoding='utf-8')
    notebook['cells']=[md('''# Menia — diagnostic de l'apprentissage interne

**Un seul bloc. A100 40 ou 80 Go. Sans Google Drive ni fichier à importer.**

Choisir **A100**, puis **Tout exécuter**. À la fin, conserver et transmettre
**menia-diagnostic-apprentissage.zip**. Le ZIP reste dans le panneau Fichiers,
à la racine `/content`, si son téléchargement automatique échoue.

Le Colab précédent a produit une réponse constante « 2 ». Ce diagnostic utilise
deux phrases et les réponses 0, 1 ou 2 pour distinguer un problème d'apprentissage
d'une difficulté à exploiter la modification interne.

Quatre adaptateurs séparés : repère visible (contrôle positif), perturbation
faible, perturbation forte, perturbation forte avec cibles mélangées.
Chaque mise à jour équilibre les trois réponses et inclut de la lecture ordinaire.
Le modèle de base est également évalué. Tous les résultats restent exportés,
y compris si un contrôle échoue ; aucun meilleur lancement n'est sélectionné.

**640 passages avec rétropropagation, répartis en 128 mises à jour**, puis
**1 792 évaluations**. Huit paires de phrases servent à l'apprentissage répété ;
24 paires nouvelles servent au test. Les deux ensembles sont évalués séparément.
La durée sur A100 de ce nouveau protocole n'est pas encore mesurée.

Ce test mesure une capacité fonctionnelle limitée. Il ne prouve pas la conscience
et n'installe aucun adaptateur sur l'iPhone. Les quatre adaptateurs sont inclus
dans le ZIP. Le calcul consomme le quota Colab.

Une interruption conserve les diagnostics. Un adaptateur incomplet repart de
son initialisation fixe, avec la reprise inscrite dans le journal ; les adaptateurs
et évaluations terminés sont conservés. Une tentative complète n'est pas rejouée.
Télécharger le ZIP avant la suppression de l'environnement temporaire.
'''),code(launcher+'\nBOOTSTRAP_SOURCE = '+repr(bootstrap)+
        '\nMENIA_LAUNCH_RESULT = launch_activation('+repr(revision)+
        ", BOOTSTRAP_SOURCE, profile='diagnostic', storage='local')\n")]
    for i,cell in enumerate(notebook['cells']): cell['id']=f'menia-diagnostic-{i:02d}'
    notebook['metadata']['colab']['name']='Menia — diagnostic apprentissage — sans Drive'
    return notebook


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('revision');args=p.parse_args()
    Path('notebooks/08_learning_diagnostic_colab.ipynb').write_text(json.dumps(make_diagnostic_notebook(args.revision),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
