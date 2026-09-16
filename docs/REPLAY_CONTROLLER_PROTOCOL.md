# Menia : améliorer le choix de vérification par rejeu

**État : boucle implémentée, contrôles logiciels exécutables, premier rejeu
rétrospectif réalisé. Le nouvel essai Qwen3-4B sur A100 reste à exécuter.**

Le [Colab 07, un seul bloc sans Drive](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/codex/recall-reliability/notebooks/07_replay_controller_colab.ipynb)
produit `menia-strategies-rejeu.zip`. Le Colab 06 de localisation interne est
conservé. Les deux expériences posent des questions différentes.

## L'idée reprise et sa portée

[Dream-RSI, version 1](https://arxiv.org/html/2609.14858v1) utilise des explorations
enregistrées pour évaluer et améliorer le code d'un contrôleur d'exploration,
avant de le redéployer. Les modèles restent fixes. Le rejeu n'invente pas les
résultats de branches absentes de l'historique ; une amélioration du score sur
l'historique ne garantit pas le transfert. Le [dépôt officiel](https://github.com/zhengkid/Dream-RSI)
annonce encore la préparation du code complet et des scripts de reproduction.

Notre réalisation adapte le principe à des **décisions indépendantes, en une
étape**. Elle sélectionne des paramètres dans une famille fixée de contrôleurs.
Elle ne réécrit pas du Python avec un LLM, ne simule pas une exploration à long
horizon et ne reproduit pas l'architecture Dream-RSI. Son objet mesuré est la
qualité des décisions de vérification, pas une expérience subjective.

## Ce qui s'exécute

`menia/replay_control.py` fournit des politiques sérialisables, le rejeu,
la sélection et l'exécution des actions. Le programme
`research/replay_controller.py` collecte les réponses, apprend des moniteurs,
gèle le contrôleur pendant chaque série, puis réévalue les politiques après
la série. `research/replay_controller_gpu.py` relie effectivement ce collecteur
au backend Qwen et aux hooks déjà vérifiés.

Une politique reçoit uniquement des probabilités établies avant le premier
token. Elle ne reçoit ni la réponse attendue ni l'issue de la tâche. Elle peut :

- répondre avec le candidat du LLM ;
- exécuter un vérificateur déterministe à partir de la question publique ;
- s'abstenir, avec une sortie finale nulle enregistrée comme telle.

Une relance du LLM est **hors du domaine pris en charge** : nous ne possédons
pas sa conséquence dans ce tableau. Le moteur refuse une action sans résultat
disponible au lieu de lui inventer une récompense.

Le candidat Qwen est produit dans tous les cas pour permettre une comparaison
appariée. Ce pilote n'économise donc pas ces 720 générations. Après la décision,
le vérificateur est exécuté une fois par question : comme action sélectionnée,
ou comme **audit supplémentaire de couverture** si la décision était autre.
Ce résultat n'entre pas dans le contexte du LLM. Les comptes et durées des audits
restent séparés des vérifications choisies. Seule la route `learned` est la route
principale exécutée ; les autres décisions sont évaluées sur les mêmes candidats
et résultats de vérification, sans prétendre avoir huit générations différentes.

## Plan fixé avant les nouvelles réponses

Qwen3-4B, révision `1cfa9a7208912126459214e8b04321603b3df60c`, BF16, SDPA,
A100 40 ou 80 Go. Aucun entraînement de ses poids. Paramètres de génération et
projections identiques au pilote d'activations, sans contexte entre les questions.
La graine du plan est `202609161`.

| Série | Questions | Apprentissage autorisé après la série |
|---|---:|---|
| Calibration | 240 | Moniteurs de réussite |
| Round 1 | 96 | Sélection sur les 96 épisodes reçus |
| Round 2 | 96 | Sélection sur les 192 épisodes reçus |
| Round 3 | 96 | Sélection sur les 288 épisodes reçus |
| Test final | 192 | Aucun |

Total : **720 générations prévues**, 720 appels du vérificateur et trois mises à
jour de politique. Les échecs et reprises éventuels sont comptés en plus.
Chaque bloc contient six cellules : comptage de A de longueurs 8, 24 et 64,
et sommes alternées de 2, 4 et 8 nombres. Les 720 questions sont distinctes,
et excluent les questions des trois pilotes Colab précédents comparés ici.
Elles restent du même domaine arithmétique et lexical : aucune généralisation
à la conversation libre n'est impliquée.

Les moniteurs utilisent **seulement la calibration**, avec pénalisation ridge
fixée à 1. Leurs paramètres restent figés pendant les trois rounds et le test.
Il y a une référence Beta par famille/difficulté, un moniteur lisant l'entrée,
un moniteur ajoutant les états centraux et finaux, et un témoin dont les cibles
de calibration sont mélangées au sein de chaque cellule.

Trois contrôleurs évoluent séparément depuis la même règle initiale
`betaCell >= 0,8 → direct ; sinon verify` :

| Contrôleur | Signaux disponibles | Politiques candidates |
|---|---|---:|
| `learned` | Beta, entrée, états internes | 75 |
| `public` | Beta, entrée | 51 |
| `shuffled` | Beta, entrée, moniteur à cibles mélangées | 75 |

Chaque famille comprend trois actions constantes, puis des seuils 0, 0,1, …, 1
et 1,01, avec vérification ou abstention sous le seuil. Le seuil 1,01 permet de
ne jamais accepter directement un signal donné. Les anciens contrôleurs font
toujours partie de la sélection et gagnent les égalités exactes. La sélection
minimise uniquement la perte moyenne des épisodes déjà terminés. Tous les
candidats et scores sont journalisés ; aucun candidat n'est choisi sur le test.

Les références fixes sont Beta au seuil 0,8, moniteur interne au seuil 0,8,
toujours répondre, toujours vérifier et toujours s'abstenir. Le témoin mélangé
utilise autant de candidats que `learned`, pour contrôler en partie l'effet
d'une recherche plus large que `public`.

## Barème et coûts

Le nouveau barème primaire utilise un extracteur numérique fixé : entier seul,
`Réponse: entier`, `Answer: entier`, ou expression numérique suivie de
`= entier`. Il ne consulte jamais la valeur attendue pour choisir quoi extraire.
Un texte ambigu ou hors de ces formes compte comme non interprétable et échoue.
Ce barème mesure donc une **exactitude selon un parseur restreint**, pas toute
la sémantique possible d'une réponse. Les réussites sous format strict et les
sorties non interprétables sont également rapportées. Les anciens barèmes et
rapports restent inchangés.

La perte vaut 1 par réponse fausse, 0,2 par vérification, 0,35 par abstention.
Une vérification fausse coûterait donc 1,2. L'abstention n'est jamais comptée
comme une réponse correcte. Dans ce domaine, l'outil déterministe rend
l'abstention dominée en perte par la vérification ; elle reste un contrôle.
Ces coefficients sont des points déclarés, **pas des euros, des secondes ou de
l'énergie électrique**. Les durées de génération avec capture et de vérification
sont exportées séparément. Chargement, installation et ajustement ne sont pas
inclus dans la somme des durées de génération.

Le test final compare `learned` à `public`, au témoin mélangé et aux références
fixes. Les intervalles descriptifs rééchantillonnent 2 000 fois les 32 blocs de
six questions. Ils sont conditionnels à cette calibration et cette trajectoire
d'apprentissage uniques ; ils ne remplacent pas des répétitions indépendantes.

## Pourquoi ce rejeu est valide dans son domaine

Les questions sont indépendantes, les générations ont leur propre graine, les
poids et moniteurs restent fixes et les outils n'altèrent pas un environnement
partagé. Pour les trois actions autorisées, nous disposons donc du candidat et
du résultat effectivement calculé par l'outil. Modifier la décision finale
d'un épisode ne change pas les entrées du suivant. Ce raisonnement ne s'étend
pas automatiquement à un agent dont les actions changent ses observations ou
sa mémoire future : celui-ci nécessiterait des trajectoires et une couverture
des branches appropriées.

## Premier contrôle sur les données déjà reçues

Le [rejeu rétrospectif](../artifacts/replay-controller-pilot/historical-backtest.json)
valide le journal de perturbations antérieur, sélectionne sur ses 96 variantes
de validation et examine les 192 variantes de test déjà connues, correspondant
à 48 questions. Les moniteurs historiques prédisent encore le score strict ;
le parseur numérique et la sélection sont ajoutés après réception.

| Stratégie | Perte diagnostique moyenne | Vérifications |
|---|---:|---:|
| Sélection pouvant lire les états internes | 0,182292 | 170 |
| Sélection avec données publiques | 0,130208 | 120 |
| Référence Beta fixe | 0,105208 | 96 |
| Toujours vérifier | 0,200000 | 192 |

**Le meilleur score sur l'historique ne se transfère pas ici en meilleur score
diagnostique.** La sélection interne choisit un seuil de 1,0 et vérifie davantage,
avec une perte supérieure à Beta. Ce résultat défavorable est conservé. Il
n'est pas confirmatoire : les réponses du test avaient déjà été examinées avant
la conception de cette méthode. Aucun nouvel appel LLM n'a eu lieu ; les 288
vérifications contrefactuelles ont été recalculées à partir des questions.
Le nouveau Colab utilisera les données nouvelles prévues ci-dessus.

## Persistance, exécution et livraison

Le journal JSONL est écrit durablement : plan, empreintes des sources,
requête, état et décisions avant le premier token, résultat et outil, ajustement,
sélections. L'analyse reconstruit les moniteurs, probabilités et sélections.
Elle refuse les décisions incohérentes et l'accès au test avant gel de la
politique. Une interruption conserve la tentative, puis peut rejouer la même
requête avec sa graine initiale ; les réponses terminées ne sont pas rappelées.
Une tentative complète ne recharge pas Qwen lors d'une relance du programme.
La reprise exige le même environnement déclaré et les mêmes sources.

```sh
python -m research.replay_controller_gpu chemin/tentative.jsonl
python -m research.replay_controller_gpu chemin/tentative.jsonl --resume
python -m research.replay_controller chemin/tentative.jsonl --output bilan.json
```

Le ZIP contient le journal, le bilan, les diagnostics et, après le dernier round,
un fichier `.policy.json` avec les politiques et les moniteurs figés. Ce fichier
est un contrôleur expérimental ; il n'est ni un adaptateur de poids Qwen ni un
module déjà installé dans l'application iPhone.

Les tests incluent un signal utile, un signal nul, une inversion du signal dans
le futur, le refus d'actions non observées, les fuites de test, les falsifications,
la reprise et l'export. Un petit Qwen aléatoire exécute réellement les hooks et
la boucle complète sans modifier ses poids. Ces tests sont des contrôles
logiciels ; ils ne mesurent pas les performances du Qwen3-4B préentraîné.

Validation locale avant publication : 54 tests du noyau, 162 tests de recherche,
16 tests du moteur de langage et 21 tests d'installation/lancement passent.
Les 17 tests des quatre profils de lancement passent aussi sous Ubuntu/Python
3.12. Le vrai tokenizer Qwen3-4B traite les 720 préfixes, de 47 à 86 tokens,
sans charger les poids préentraînés. Un parcours synthétique complet des 720
épisodes réalise les trois mises à jour et l'export ; un calcul séparé retrouve
les huit pertes du test. Aucun résultat A100 de ce nouveau pilote n'est acquis.
