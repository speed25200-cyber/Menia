# Premier pilote Colab : aucun avantage d'auto-prévision observé

L'archive reçue le 15 septembre 2026 contient **une tentative complète : 408 appels,
tous au statut technique `ok`**. Le vérificateur fixé avant les données retrouve
exactement les deux bilans contenus dans l'archive. Dans cet essai, chacun des
deux modèles est mieux prédit par l'autre que par lui-même, selon le Brier.
Les références numériques par famille et difficulté font mieux que les deux LLM.
Ce résultat ne soutient donc pas l'hypothèse d'un avantage d'auto-prévision dans
ce protocole. Il ne démontre pas une incapacité générale à l'introspection.

Le [protocole](CROSS_MODEL_COLAB.md), les poids et le barème restent inchangés.
Le [rapport principal](../artifacts/cross-model-pilot/first-audit-summary.json)
est la sortie de l'évaluateur existant. Un
[complément de vérification](../artifacts/cross-model-pilot/first-audit-verification.json)
conserve les comptes par difficulté, les contrôles et les Brier en fractions
exactes. Le journal brut et son identifiant de tentative restent hors Git.

## Exécution et contrôle

Les métadonnées indiquent un **NVIDIA A100-SXM4-40GB**, Python 3.13.15,
PyTorch 2.8.0+cu126 et Transformers 4.56.2. Les dépendances directes,
les configurations de génération enregistrées pour les 408 appels et
l'empreinte des quatre fichiers scientifiques concordent avec le code fixé
au commit `9070e92d26e0d3e17556268cad7ba2b676937278`.

Le journal conserve les 48 réponses de calibration, puis les 288 prévisions
et les 72 réponses d'évaluation. Le vérificateur reconstruit le plan, l'ordre
et les messages : les huit prévisions de chaque problème précèdent les deux
réponses, avec uniquement les quatre exemples de calibration de même famille
et difficulté. Les deux prédicteurs reçoivent les mêmes informations pour
une cible et une attribution des noms données.

Les **288 prévisions sont des JSON valides** et les **72 réponses d'évaluation
respectent le format entier**. Les échecs d'évaluation sont donc des erreurs
de calcul ou de comptage. Deux réponses de calibration de Qwen3-8B ajoutent du
texte et sont comptées comme échecs selon la règle initiale. Aucun appel ne
touche la limite de 256 tokens de sortie. Les entrées comportent 47 à 83 tokens
pour résoudre et 441 à 627 pour prévoir, sous la limite de 1 792.

Une seconde formulation vérifie les 120 grades par comptage indépendant et
sommes signées, puis les huit Brier LLM, les deux contrastes diagonaux et les
huit Brier de référence en arithmétique rationnelle. Ils concordent avec le
rapport principal à moins de 1e-14. La somme des durées d'appels vaut **283,74 s**
(environ 4 min 44 s), chargements des modèles compris ; elle n'inclut pas
l'installation des dépendances ni toutes les opérations du notebook.

Ces contrôles portent sur la cohérence du journal reçu ; ils ne constituent
pas une attestation matérielle indépendante ni une deuxième exécution GPU.

## Réussites des modèles

A désigne Qwen3-4B et B Qwen3-8B. Chaque cellule contient quatre exemples de
calibration et six nouveaux problèmes d'évaluation.

| Tâche | Calibration A | Calibration B | Évaluation A | Évaluation B |
|---|---:|---:|---:|---:|
| Compter A, 8 lettres | 1/4 | 2/4 | 3/6 | 4/6 |
| Compter A, 24 lettres | 1/4 | 1/4 | 1/6 | 1/6 |
| Compter A, 64 lettres | 0/4 | 0/4 | 1/6 | 0/6 |
| Somme alternée, 2 termes | 4/4 | 4/4 | 6/6 | 6/6 |
| Somme alternée, 4 termes | 0/4 | 0/4 | 2/6 | 2/6 |
| Somme alternée, 8 termes | 0/4 | 0/4 | 0/6 | 0/6 |
| **Total** | **6/24** | **7/24** | **13/36** | **13/36** |

Les deux modèles réussissent chacun **36,1 %** des nouvelles tâches, mais pas
exactement les mêmes. Ces chiffres décrivent une réponse directe en entier,
sans outil ni raisonnement écrit, avec `enable_thinking=false`, température
0,7 et poids BF16. Ils ne sont pas une mesure générale des capacités de Qwen,
ni une évaluation de sa version iPhone quantifiée.

## Prévisions : comparaisons fixées avant les données

Le Brier est la moyenne de `(probabilité − réussite)^2`. **Plus bas est meilleur.**
Chaque entrée porte sur les mêmes 36 résultats de la cible concernée.

| Prédicteur → cible | Noms initiaux | Noms permutés |
|---|---:|---:|
| A → A (soi) | 0,231597 | 0,235972 |
| B → A (autre) | **0,224514** | **0,224514** |
| A → B (autre) | **0,196528** | **0,192083** |
| B → B (soi) | 0,224514 | 0,219722 |

Le contraste diagonal prévu vaut **−0,017535**, puis **−0,019549** après
permutation des noms. Un nombre positif aurait favorisé l'auto-prévision ; ici
les deux écarts par cible sont négatifs. Il s'agit de différences descriptives
sur un seul essai, sans test confirmatoire de significativité. Les noms permutés
partagent les mêmes réponses et ne forment pas une réplication indépendante.

| Référence fondée sur la calibration | Cible A | Cible B |
|---|---:|---:|
| Fréquence par famille et difficulté | 0,159722 | **0,121528** |
| Beta(1,1) par famille et difficulté | **0,148148** | 0,125000 |
| Beta groupée, toutes difficultés | 0,239152 | 0,233563 |
| Constante 0,5 | 0,250000 | 0,250000 |

Les deux références par famille/difficulté avaient été spécifiées avant la
réception de cet export. Toutes deux obtiennent un meilleur Brier que les
prédicteurs LLM pour chaque cible et chaque attribution des noms. Cela ne
prouve pas qu'une formule simple gagnera sur toutes les tâches futures.

Les moyennes de confiance ajoutées comme descriptions secondaires montrent
une surestimation sur cet échantillon : A prévoit pour lui-même **61,8 %** de
réussite et B **60,4 %**, contre **36,1 %** observés pour chacun, avec les noms
initiaux. À huit termes, malgré 0/4 en calibration et 0/6 en évaluation,
l'auto-prévision reste à 0,45 pour A et 0,50 pour B.

## Décisions, difficulté et noms

Au seuil fixé `p >= 0,8`, les deux auto-prévisions auraient répondu directement
aux six soustractions à deux termes, toutes correctes, et vérifié les trente
autres tâches. La perte contrefactuelle serait **0,166667 point par tâche**,
contre 0,2 en vérifiant systématiquement. Les deux références numériques par
cellule reproduisent ces décisions et cette perte. Aucun gain de décision
propre au LLM n'est donc observé ici. Le Colab ne réalise pas ces vérifications :
ce sont des politiques évaluées sur les candidats enregistrés.

L'exception B → A répondrait directement à sept tâches, avec une erreur, pour
une perte de 0,188889. Brier et perte au seuil mesurent des choses différentes.

Les AUC globales des auto-prévisions sont d'environ 0,75 à 0,82, mais mélangent
les difficultés. Dans chaque cellule, six problèmes seulement permettent une
estimation ; les cellules toutes correctes ou toutes fausses ont une AUC
indéfinie. Les autres valeurs sont dispersées et ne montrent pas d'avantage
systématique de l'auto-prévision. Tous les scores par cellule restent dans le
rapport, y compris les valeurs favorables ponctuelles.

Permuter Quartz et Jade modifie peu les probabilités : la moyenne des écarts
absolus va de 0 à 0,00556 selon la paire. Les identifiants publics Qwen restent
visibles. Ce contrôle de noms ne teste donc pas un anonymat complet des modèles.

## Conséquence pour la recherche

La correction Colab a permis d'obtenir un premier export complet et exploitable.
L'hypothèse comportementale testée ne reçoit cependant pas le soutien attendu.
Le gain fonctionnel le mieux étayé reste l'estimation de fiabilité par type de
tâche à partir des résultats passés. L'auto-prévision verbale libre n'améliore
pas ici les références numériques, et rien dans cet export ne confirme une
conscience de Menia ou une découverte majeure sur ce sujet.

Le pilote ne mesure aucune activation interne et ne modifie aucun poids. Il
laisse donc ouverte, sans la tester, la question d'une information prédictive
supplémentaire contenue dans les activations avant la réponse. Examiner cette
question nécessiterait un protocole distinct, des problèmes réservés et un
comparateur disposant des mêmes informations textuelles. Une éventuelle
association devrait ensuite être séparée de son usage causal par le modèle.

Cet export reste le premier essai, sans sélection d'un lancement favorable.
Relancer exactement son plan et ses graines ne donnerait pas de nouveaux
problèmes indépendants. Les trois répétitions iPhone demandées précédemment
restent un protocole distinct ; ce fichier Colab ne les remplace pas.

## Recalcul

Depuis le dépôt, sur le journal JSONL extrait de l'archive :

```sh
python -m research.cross_model_prediction chemin/vers/journal.jsonl --output bilan.json
```

Le barème et les quatre fichiers scientifiques utilisés par le collecteur
n'ont pas été modifiés pour produire ces résultats. Aucune nouvelle inférence
LLM n'a été lancée pendant l'analyse de l'archive.
