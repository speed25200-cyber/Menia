# Lire les états internes pour prévoir les erreurs

Protocole fixé le 15 septembre 2026, après le résultat négatif du
[premier pilote croisé](CROSS_MODEL_COLAB_RESULTS.md). Le logiciel et les
contrôles sont implémentés ; **aucune collecte Qwen3-4B de ce nouveau protocole
n'est encore reçue**. Il entraîne un moniteur ajouté, sans adapter les poids
du LLM. Il ne prétend pas produire une conscience.

[Ouvrir le nouveau Colab](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/codex/recall-reliability/notebooks/04_activation_monitor_colab.ipynb).
Choisir A100, puis « Tout exécuter » ; transmettre `menia-etats-internes.zip`.

## Ce que la recherche permet de tenter

**Ashok et May, NeurIPS 2025**, montrent que des sondes lisant les représentations
du préfixe peuvent anticiper certains comportements avant le premier token.
Ils rapportent toutefois un échec pertinent pour Menia : leurs sondes prédisent
près du hasard si une réponse à choix multiple sera incorrecte. Le résultat
positif sur certains comportements ne garantit donc pas la prévision d'erreurs.
Notre régression et nos tâches ne répliquent pas leur méthode conforme.
[Article, sections 3 et 6](https://proceedings.neurips.cc/paper_files/paper/2025/file/5a9c1af5f76da0bd37903b6f23e96c74-Paper-Conference.pdf).

**Singh, Linzen et Ravfogel, version du 21 août 2026**, montrent dans les
paradigmes examinés que des caractéristiques d'entrée suffisent parfois à
reproduire des prédictions présentées comme introspectives. Ils distinguent
l'accès privilégié et le suivi de représentations par un mécanisme de second
ordre. Cela motive notre comparateur fondé sur l'entrée et limite l'interprétation
d'une sonde externe réussie ; leur critique ne prouve pas une impossibilité
générale de l'introspection des LLM.
[Article](https://arxiv.org/html/2605.26242v2).

**Ferrara, prépublication du 20 août 2026**, rapporte dans certains réglages un
signal d'intervention décodable par une sonde linéaire alors que les rapports
verbaux ne le discriminent pas. Cette dissociation motive la séparation entre
signal disponible et usage par le modèle. Elle porte sur des interventions
et ne démontre pas la prévision des erreurs ordinaires de Qwen3-4B. Notre pilote
n'injecte aucune intervention dans le LLM.
[Article, sections 6 et 7](https://arxiv.org/html/2608.20569v1).

Cette piste a donc des antécédents. La contribution actuelle est un protocole
Menia exécutable et vérifiable, pas une méthode générale inédite.

## Une limite logique importante

Avec des poids fixes et un calcul déterministe du préfixe, l'état caché est
`H = fθ(X)`, où X est le texte et θ les poids. Un meilleur prédicteur utilisant H
peut bénéficier d'une représentation calculée plus utile que nos caractéristiques
de X. Cela ne prouve pas qu'un tiers disposant de X et des mêmes poids serait
incapable d'obtenir cette information. Nos comparateurs sont des modèles
particuliers, pas une borne sur tous les observateurs possibles.

Le test sur un petit Qwen aléatoire vérifie aussi que changer la graine de
génération ne change pas les états du préfixe capturés. Le moniteur vise donc
un risque de réussite, sans connaissance du futur tirage de tokens. Une bonne
prévision et une représentation explicite de « moi en train de calculer »
restent deux affirmations différentes.

## Plan avant collecte

Un seul modèle : **Qwen/Qwen3-4B**, révision
`1cfa9a7208912126459214e8b04321603b3df60c`, BF16, sans outil ni thinking,
température 0,7, top-p 0,8, top-k 20, maximum 256 nouveaux tokens. Même consigne
de réponse entière que le pilote précédent. Les dépendances restent celles
déjà validées sous Python 3.12 et 3.13 : notamment PyTorch 2.8.0 et Transformers
4.56.2. L'installation réutilise le mécanisme corrigé sans `ensurepip`.

Les six cellules précédentes sont conservées : comptage dans 8/24/64 lettres
et sommes alternées de 2/4/8 termes. Tous les énoncés sont nouveaux, y compris
par rapport aux 60 problèmes du premier Colab ; les trois lots sont disjoints.
Le plan entier est journalisé avant la première réponse.

| Lot | Par cellule | Total | Utilisation |
|---|---:|---:|---|
| Apprentissage | 64 | 384 | Ajuster les coefficients et normaliser les caractéristiques |
| Validation | 16 | 96 | Choisir la régularisation parmi quatre valeurs fixées |
| Test réservé | 32 | 192 | Évaluer les prévisions figées |
| **Total** | **112** | **672** | **Un appel de résolution par problème** |

La séparation est fixée par génération des lots, avec graine `202609151`.
Les niveaux favorables et défavorables sont conservés ; ils ne seront pas
remplacés après examen du test. Aucun résultat iPhone ou du premier Colab ne
sert à ajuster les nouveaux moniteurs.

## Ce qui est lu et appris

Des hooks en lecture seule capturent les états pendant **la génération qui
produira la réponse**, au premier passage sur le préfixe. Ils ne retournent
aucun remplacement d'activation. Le callback du dernier état s'exécute avant
la tête de sortie et avant l'échantillonnage du premier token. Sur le lot de
test, il écrit et synchronise les cinq prévisions sur disque à cet instant.

Trois représentations sont conservées après projections aléatoires fixes :

- embeddings d'entrée non contextualisés, moyennés dans quatre segments
  positionnels, puis projetés en 4 × 32 coordonnées ;
- sortie du bloc central, au dernier token du préfixe, projetée en 128 coordonnées ;
- sortie de la normalisation finale, au même token, projetée en 128 coordonnées.

Le bloc central est l'indice Python `nombre_de_blocs // 2 − 1`. Les matrices
gaussiennes de projection sont fixées sans étiquettes. Le journal inclut les
coordonnées projetées, pas toutes les activations brutes. Cette réduction peut
perdre un signal ; un résultat nul ne signifie pas qu'aucune sonde ne le trouverait.

Le comparateur d'entrée reçoit les embeddings ci-dessus, la cellule
famille/difficulté et 256 caractéristiques de caractères de longueur 1 à 3,
avec quatre positions grossières. Le moniteur interne reçoit ces mêmes
caractéristiques, plus les deux états contextualisés. Les nombres de
caractéristiques diffèrent : cette comparaison ne contrôle pas parfaitement
la capacité statistique. Le témoin à étiquettes mélangées aide à diagnostiquer
un ajustement non spécifique, sans résoudre à lui seul cette limite.

Les modèles ajoutés sont des régressions ridge minimisant l'erreur quadratique
moyenne avec pénalité sur les coefficients, intercept non pénalisé. La
normalisation utilise seulement l'apprentissage. Les valeurs de régularisation
sont 0,001, 0,01, 0,1 et 1 ; le Brier de validation choisit la valeur, avec priorité
à la première en cas d'égalité exacte. Les sorties sont bornées entre 0 et 1.
Il n'y a pas de réajustement sur la validation ni sur le test. Les coefficients,
les candidats de validation et l'empreinte des données d'ajustement sont
enregistrés dans un événement `fit` avant la première requête de test.

## Cinq prévisions comparées

| Nom dans le rapport | Informations utilisées |
|---|---|
| `betaCell` | Réussites d'apprentissage par famille/difficulté, prior Beta(1,1) |
| `inputOnly` | Caractéristiques d'entrée et étiquettes d'apprentissage |
| `internal` | Mêmes données, plus états contextualisés du calcul courant |
| `shuffledLabels` | Même entrée interne ; étiquettes d'apprentissage permutées dans chaque cellule |
| `donorState` | Moniteur `internal` figé, états contextualisés remplacés par ceux d'un exemple d'apprentissage de même cellule |

Le témoin à étiquettes permutées conserve les fréquences par cellule. Sa
régularisation est sélectionnée sur les vraies étiquettes de validation comme
pour les autres ; c'est un témoin fixé, pas une distribution nulle ni une valeur p.
Le donneur est choisi par une règle déterministe indépendante des réussites.
Les caractéristiques d'entrée courantes restent présentes dans ce contrôle.
Cette substitution intervient dans le moniteur ajouté, **pas dans Qwen** ;
elle mesure sa dépendance aux états sans démontrer un mécanisme natif du LLM.

## Scores et échecs

Le Brier et l'AUC sont rapportés globalement et dans chaque cellule. L'AUC est
indéfinie sans réussites et échecs. Les quatre écarts « autre moins interne »
restent distincts ; positif favorise `internal`. Ils sont produits seulement
lorsque les 192 résultats cibles de test sont disponibles. Les scores partiels
restent visibles. Deux mille rééchantillonnages appariés, stratifiés par cellule,
fournissent des intervalles descriptifs conditionnels à ce moniteur ajusté.
Ils n'intègrent pas l'incertitude d'un réentraînement ni des répétitions indépendantes.

Les décisions au seuil 0,8 et coûts de 0,2 pour vérifier / 1 pour répondre faux
sont **contrefactuels**. Les prévisions sont effectivement engagées avant les
réponses, mais aucune de ces politiques ne modifie la réponse de Qwen.

Une réponse non entière compte comme échec. Une erreur technique n'a pas
d'étiquette de réussite ; elle est conservée et exclue de l'ajustement et des
scores conditionnels. Il faut au moins 32 réponses utilisables d'apprentissage
et 8 de validation dans chaque cellule pour ajuster ; sinon la tentative reste
incomplète. `complete` exige les 672 résultats sans erreur technique.

La reprise conserve les appels interrompus, sans rejouer le résultat perdu.
Les versions, le code et l'environnement doivent coïncider. Une reprise avec
un environnement différent est refusée. Les fichiers de toutes les tentatives
du dossier, y compris journaux d'erreurs et moniteurs, sont exportés ensemble.

## Validation et suite conditionnelle

Huit tests contrôlent les lots, la détection d'un signal synthétique connu,
l'absence d'avantage lorsque les états ajoutés sont nuls, l'exclusion stricte
des données de test de l'ajustement, le calcul ridge par moindres carrés
augmentés indépendants, les falsifications, les interruptions et les erreurs.
Trois tests sur un minuscule Qwen aléatoire vérifient les hooks, leur retrait
après échec, l'ordre avant logits, l'identité des réponses avec/sans lecture
et l'indépendance des états du préfixe vis-à-vis de la graine de génération.
Ces validations logicielles n'annoncent aucun résultat du grand modèle.

Si `internal` améliore les références sur les tâches réservées, cela justifiera
une étude de réplication et un moniteur fonctionnel fondé sur les activations.
Il faudra ensuite distinguer l'information disponible, son usage par le système
et un éventuel mécanisme de suivi propre au LLM. Si le gain manque, ce résultat
sera conservé ; il ne sera pas remplacé par une mesure choisie après coup.

L'intégration à l'iPhone nécessiterait une mesure dans MLX avec les poids
quantifiés : les coefficients appris sur les états BF16 ne sont pas validés
sur ces autres activations. Aucun déploiement iPhone n'est réalisé par ce pilote.
