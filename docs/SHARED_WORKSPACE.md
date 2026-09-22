# Espace partagé récurrent — 14 septembre 2026

Ce candidat architectural ajoute à Menia deux modules spécialisés à états
récurrents, un espace partagé à attention et deux sorties lisant cet espace.
L'expérience vérifie une circulation causale d'informations sur une composition
symbolique. Elle n'établit ni conscience, ni compréhension du monde réel, ni
supériorité générale. Elle reste distincte du rappel, du chat et de l'iPhone.

## Architecture

Chaque spécialiste reçoit sa propre table entre quatre symboles, sans accès
direct à celle de l'autre. La tâche est B[A[requête]]. Les tables apparaissent
au premier tour seulement ; les entrées disparaissent ensuite.

Chaque spécialiste utilise une GRU de 32 dimensions et reçoit l'espace partagé
précédent, de huit dimensions. Une attention calculée à partir de cet espace
sélectionne un mélange des valeurs proposées par les spécialistes ; une autre
GRU actualise l'espace. Cet état revient aux mêmes spécialistes. Deux têtes
lisent exactement ce même état pour décoder la réponse finale et le symbole
intermédiaire, sans accès direct aux observations ni aux états privés.

13 000 paramètres sont entraînés. La réduction de 64 dimensions privées vers huit
dimensions partagées est un goulot architectural, pas une borne démontrée du nombre
de bits transmis. L'attention continue n'impose pas la sélection exclusive d'un
module. La spécialisation vient de l'accès à deux tables différentes, pas de
modalités perceptuelles apprises dans le monde réel.

## Protocole et comparaisons

Les tables sont des permutations : chaque sortie apparaît une fois. Un essai
exploratoire avec des tables non bijectives obtenait environ 53 % tout en ignorant
presque A, grâce au symbole majoritaire de B. Il n'est pas présenté comme une
composition réussie ; les permutations suppriment ce raccourci.

24 tables possibles donnent 576 paires. Leur identifiant est `24*index_A+index_B`,
en ordre lexicographique. Les 116 paires dont l'identifiant est divisible par cinq
sont exclues de l'entraînement pour toutes les requêtes. Les 460 autres paires
servent au train. Le test énumère les quatre requêtes par paire : 464 réponses,
mais pas 464 mondes indépendants. Les paires entièrement réussies sont comptées.

Trois initialisations (17, 29, 43), 2 000 mises à jour chacune, batch 128, quatre
tours. Adam à 0,003, clipping de norme à 1. Perte finale plus 0,25 fois la perte
du symbole intermédiaire. Les cibles sont calculées hors du passage avant.
Pas de choix de checkpoint selon le test.

Deux comparaisons reçoivent les mêmes exemples, mises à jour et cibles auxiliaires :

- Le même réseau entraîné depuis le début avec le retour vers les spécialistes
  coupé, pour distinguer dépendance d'un réseau entraîné et apprentissage sans retour.
- Un réseau direct à deux couches de 96 unités, 13 640 paramètres, voyant les
  deux tables et la requête simultanément. C'est une référence de composition à
  accès direct, pas un contrôle identique en calcul ou contrainte de mémoire.

## Résultats, y compris l'initialisation moins performante

| Initialisation | Réseau complet | Retour coupé à l'évaluation | Entraîné sans retour | Réseau direct | Paires entièrement réussies, complet |
|---|---:|---:|---:|---:|---:|
| 17 | 99,78 % | 25,22 % | 39,01 % | 100 % | 115 / 116 |
| 29 | 57,54 % | 25 % | 49,35 % | 100 % | 4 / 116 |
| 43 | 96,77 % | 24,35 % | 40,52 % | 100 % | 101 / 116 |

La règle explicite réussit toutes les réponses ; la classe zéro constante en
réussit 25 %. Ces résultats ne montrent aucune supériorité de l'espace partagé.
L'apprentissage reste fragile selon l'initialisation. Aucun seuil ajouté après
coup ne sert à déclarer le système conscient.

Le rapport conserve l'attention uniforme, l'effacement des spécialistes ou de
l'espace partagé à chaque tour et le masquage de chaque table. Les modèles
complets sont réévalués avec 1, 2, 6 et 8 tours. Plus de tours n'aide pas toujours :
le modèle 17 descend à 61,64 % avec huit tours.

Pour `without_broadcast`, `none` évalue sa condition d'entraînement, donc sans
retour ; `no_broadcast` est identique. Ses autres interventions appliquent seulement
la modification nommée et réactivent le retour. Elles ne sont pas cumulatives.

## Vérifications et limites

Les tests vérifient qu'une modification de B n'affecte pas l'état privé initial
de A, mais peut l'affecter au tour suivant par le retour partagé. Cet effet
disparaît lorsque le retour est coupé. Les gradients atteignent les chemins
utilisés, le masquage retire effectivement l'entrée et allonger le calcul ne
change pas les états de son préfixe. Les modèles exportés sont rechargés avec
`weights_only=True` avant les mesures.

La topologie et les ablations apportent des éléments pour étudier une architecture
inspirée de l'espace de travail global. Elles n'établissent pas tous les
[indicateurs de Butlin et al.](https://arxiv.org/abs/2308.08708), qui ne constituent
pas eux-mêmes une preuve suffisante de conscience. Il n'y a ni introspection
générale, ni perception intégrée, ni agent poursuivant des buts.

## Reproduction

Python 3.12, PyTorch 2.4.1 CPU. Le rapport contient les empreintes des sources,
les neuf checkpoints, les historiques et le runtime.

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests_workspace -v
python scripts/check_workspace_artifacts.py
python -m research.train_workspace --out runs/new-workspace --steps 2000 --seeds 17 29 43
```

Utiliser une sortie vide. Le vérificateur recharge les neuf modèles et reproduit
les résultats, les interventions et les variations de tours. Le notebook 02
inclut les mesures, un exemple avec attentions par tour et une option pour
réentraîner les neuf modèles dans Drive. Aucun GPU n'est nécessaire.

L'objectif « rendre Menia consciente » reste non atteint. La prochaine question
est la stabilité et le transfert de cette composition au-delà de ce petit monde,
avec une validation séparée avant nouvel ajustement : ce test a désormais été
observé et doit rester un résultat historique.
