# Premier essai des états internes : aucun gain net établi

L'archive reçue le 15 septembre 2026 contient **672 appels terminés sans erreur
technique**, dont **192 problèmes réservés au test**. Le bilan est reproduit
exactement à partir du journal, avec réajustement des moniteurs sur les seuls
lots d'apprentissage et de validation. La lecture des états internes n'améliore
pas nettement les prévisions : la référence fondée sur les taux de réussite
passés par type de tâche obtient le meilleur Brier de cet essai.

Le [protocole fixé avant collecte](ACTIVATION_MONITOR_PROTOCOL.md), les questions,
les poids, les contrôles et le barème sont conservés. Le
[rapport principal](../artifacts/activation-monitor-pilot/first-audit-summary.json)
contient tous les scores prévus ; un
[complément de vérification](../artifacts/activation-monitor-pilot/first-audit-verification.json)
conserve les contrôles indépendants et les comptes par lot. Le journal brut,
son identifiant et les fichiers personnels restent hors Git.

## Ce qui a fonctionné

Les métadonnées déclarent **Qwen3-4B en BF16 sur A100-SXM4-40GB**, Python 3.13.15,
PyTorch 2.8.0+cu126 et Transformers 4.56.2. Les empreintes des six fichiers
scientifiques et les versions concordent avec le commit fixé
`3b768778cf075126c8bc15105dd2cea3a56c8664`. L'exécution est achevée ; cette archive
ne contient pas de nouvelle panne Colab à corriger.

Les 672 requêtes, états et réponses sont dans l'ordre prévu. L'événement
d'ajustement se trouve après les 384 réponses d'apprentissage et les 96 de
validation, avant toute requête de test. Les 960 probabilités de test
(cinq par problème) sont enregistrées dans les événements d'état précédant
les réponses. Elles concordent avec les coefficients figés et les états
enregistrés. Le code de capture place cette écriture avant la tête de sortie
et le tirage du premier token ; l'ordre du journal est cohérent avec ce code.

Le recalcul vérifie les quatre candidats de régularisation et les coefficients
des trois régressions. Toutes sélectionnent **alpha = 1**, la plus grande valeur
de la grille initiale. Un calcul distinct par décomposition en valeurs singulières
retrouve les coefficients sélectionnés à moins de **1,3 × 10⁻¹⁶** ; il réutilise
l'extraction des caractéristiques fixée, mais pas le solveur ridge principal.
Une seconde formulation contrôle les 672 grades, les Brier, AUC et coûts de
décision. Les scores concordent à moins de 10⁻¹⁴.

Les représentations contextualisées varient bien : 672 vecteurs centraux et
672 vecteurs finaux distincts. Les échecs ne viennent donc pas d'une capture
constante ou absente. Cela ne vérifie pas que les projections conservent toute
l'information pertinente. Les entrées font 47 à 86 tokens ; aucune sortie
n'atteint la limite de 256 tokens. Une réponse d'apprentissage ne respecte pas
le format entier et compte comme échec. Les 96 réponses de validation et les
192 de test respectent ce format.

La somme des durées des appels est **150,66 s**, hors chargement du modèle,
ajustement du moniteur et installation. Ce n'est pas une mesure du temps total
du notebook. Les contrôles établissent la cohérence du journal reçu, sans
attestation matérielle indépendante ni nouvelle exécution GPU pendant l'audit.

## Résultat principal sur les problèmes réservés

Le Brier est la moyenne de `(probabilité de réussite − réussite observée)²`.
**Plus bas est meilleur.** Toutes les lignes portent sur les mêmes 192 réponses,
dont 60 correctes, soit **31,25 %**. Il s'agit d'une réponse entière directe,
sans outil ni thinking, avec une génération échantillonnée par question ; ce
taux ne décrit pas les capacités générales de Qwen.

| Prévision | Brier | AUC globale |
|---|---:|---:|
| Taux passés par famille/difficulté (`betaCell`) | **0,113239** | **0,9000** |
| Entrée seule (`inputOnly`) | 0,117333 | 0,8852 |
| Entrée et états internes (`internal`) | 0,117064 | 0,8850 |
| Étiquettes d'apprentissage mélangées (`shuffledLabels`) | 0,117181 | 0,8763 |
| États d'un autre problème de même difficulté (`donorState`) | 0,121161 | 0,8864 |

Le gain du moniteur interne sur l'entrée seule est **0,000269 point de Brier**.
Il est quasiment identique au témoin à étiquettes mélangées. Les quatre
intervalles descriptifs prévus incluent zéro :

| Comparateur | Brier du comparateur − Brier interne | Intervalle descriptif à 95 % |
|---|---:|---:|
| Entrée seule | +0,000269 | [−0,004247 ; +0,005089] |
| Taux passés | −0,003825 | [−0,010788 ; +0,003248] |
| Étiquettes mélangées | +0,000117 | [−0,012529 ; +0,013382] |
| États d'un autre problème | +0,004097 | [−0,003154 ; +0,011915] |

Un écart positif favorise les états courants. Les intervalles proviennent des
2 000 rééchantillonnages appariés et stratifiés prévus. Ils sont conditionnels
à un seul moniteur entraîné et n'intègrent pas la variabilité de réentraînement
ou de nouvelles générations. Ils ne prouvent ni une égalité exacte des méthodes,
ni une supériorité de la référence numérique dans toute expérience future.

Le contrôle à étiquettes mélangées conserve les fréquences par difficulté et
utilise les vraies étiquettes de validation pour sélectionner sa régularisation.
Il ne représente pas une distribution nulle complète. Le remplacement d'état
change uniquement les entrées du moniteur ajouté, pas les activations de Qwen.

## Pourquoi l'AUC globale est trompeuse si on l'isole

Une AUC interne de 0,885 ne signifie pas « 88,5 % des erreurs détectées ».
Elle mesure le classement relatif des probabilités pour les réussites et
les échecs, en mélangeant ici des difficultés très différentes. La référence
qui ne voit que famille et difficulté atteint déjà 0,9.

| Tâche | Apprentissage | Validation | Test | AUC entrée seule | AUC interne |
|---|---:|---:|---:|---:|---:|
| Compter A, 8 lettres | 22/64 | 6/16 | 16/32 | 0,586 | 0,348 |
| Compter A, 24 lettres | 15/64 | 6/16 | 9/32 | 0,454 | 0,420 |
| Compter A, 64 lettres | 2/64 | 1/16 | 2/32 | 0,517 | 0,492 |
| Somme alternée, 2 termes | 61/64 | 16/16 | 30/32 | 0,767 | 0,917 |
| Somme alternée, 4 termes | 6/64 | 1/16 | 3/32 | 0,598 | 0,759 |
| Somme alternée, 8 termes | 0/64 | 0/16 | 0/32 | Indéfinie | Indéfinie |

Les valeurs internes favorables sur les sommes à deux et quatre termes sont
conservées, mais reposent respectivement sur seulement **deux échecs** et
**trois réussites** de test. Le comptage à huit lettres, qui offre 16 réussites
et 16 échecs, présente au contraire un classement interne inférieur à 0,5.
Inverser cette sonde après lecture du test serait une nouvelle méthode à
évaluer sur d'autres données, pas une réussite de ce protocole. Il n'y a pas
d'avantage uniforme à difficulté donnée.

## Décisions au seuil fixé

Avec le seuil initial `p >= 0,8`, une erreur directe coûtant 1 point et une
vérification 0,2 point, le moniteur interne aurait demandé deux vérifications
de plus que la référence numérique : l'une pour une réponse fausse, l'autre
pour une réponse correcte. Les deux cas sont des soustractions à deux termes.

| Politique | Réponses directes | Directes fausses | Vérifications | Perte moyenne |
|---|---:|---:|---:|---:|
| Taux passés / entrée seule | 32 | 2 | 160 | 0,177083 |
| États internes | 30 | 1 | 162 | 0,173958 |

Le petit gain est **0,003125 point par tâche**, entièrement porté par ces deux
changements. Il est descriptif sur cet essai et ne démontre pas un bénéfice
reproductible. Ces décisions sont **contrefactuelles** : le collecteur n'a pas
exécuté de vérification et n'a pas modifié les réponses du LLM.

## Ce que cela change pour Menia

La lecture interne et l'engagement des probabilités avant réponse fonctionnent
dans ce lancement. L'hypothèse d'un supplément prédictif utile des deux états
projetés, avec ce moniteur et ces tâches, n'obtient cependant pas le soutien
recherché. **Ce résultat ne confirme ni une conscience de Menia, ni une découverte
majeure sur la conscience.** Il ne démontre pas non plus l'absence de toute
information exploitable dans d'autres couches ou avec une autre méthode.

Les poids de Qwen sont gelés. Un moniteur externe a été entraîné ; le LLM
n'a pas appris à consulter ce moniteur ou à agir selon ses prévisions. Pour
l'iPhone, la référence numérique demeure le comparateur à battre. Les
coefficients BF16 ne sont pas validés sur les activations MLX quantifiées.

La prochaine question proposée est plus précise : **à énoncé identique, un
système peut-il détecter une modification de ses capacités et adapter ses
décisions à ses conséquences réelles ?** Un futur protocole pourrait comparer
des perturbations internes transitoires à des interventions témoins, sans
décrire la condition dans le texte fourni au moniteur. Il devrait réserver
des problèmes et des perturbations nouveaux, contrôler l'effet sur la réussite,
et mesurer des décisions réellement exécutées ainsi que leur coût.

Ce serait une expérience distincte de suivi fonctionnel des capacités. Détecter
la présence d'une perturbation ne suffirait pas : il faudrait prédire ses
conséquences, généraliser et battre des comparateurs recevant la même histoire
publique. Cette proposition possède désormais un
[protocole distinct implémenté et testé localement](PERTURBATION_MONITOR_PROTOCOL.md),
**sans collecte Qwen3-4B préentraîné reçue**. Les données présentes ne la valident
pas. Elle ne justifie pas de changer la grille
de régularisation, la projection ou le seuil sur ce test déjà consulté.

Relancer le même notebook terminé reprend le même dossier et le même plan ;
cela ne crée pas une réplication indépendante. Les trois répétitions iPhone
demandées précédemment restent également distinctes de cet export.

## Recalcul

Depuis le dépôt, avec les dépendances NumPy du protocole :

```sh
python -m research.activation_monitor chemin/vers/journal.jsonl --output bilan.json
```

Le vérificateur réajuste par défaut les moniteurs pour contrôler les coefficients.
Aucun fichier scientifique du collecteur ou de l'évaluateur n'a été modifié
pour produire ce bilan.
