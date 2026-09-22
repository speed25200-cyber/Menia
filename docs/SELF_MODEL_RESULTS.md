# Résultats du pilote de connaissance de son état

Exécution locale CPU le 14 septembre 2026. Le
[plan du pilote](SELF_MODEL_PILOT.md) a été écrit avant son exécution. Les
[résultats numériques complets](../artifacts/self-model/report.json) conservent
les checkpoints, les graines, les empreintes des sources et les cinq familles.
Le [rapport scientifique](SELF_AWARENESS_RESEARCH.md) reste une photographie de
la recherche bibliographique antérieure à ce pilote.

## Résultat principal

Sur des perturbations gaussiennes de l'état interne, l'estimateur interne
prévoit mieux la réussite du rappel que l'observateur reconstruit à partir des
entrées publiques, pour chacune des trois mémoires. Les deux estimateurs sont
entraînés hors ligne sur les mêmes épisodes et labels, avec la même architecture
et le même budget. Les poids de rappel restent gelés.

| Mémoire | Brier interne | Brier observateur | Avantage interne | Intervalle descriptif 95 % |
|---|---:|---:|---:|---:|
| 17 | 0,09284 | 0,13269 | 0,03986 | [0,02819 ; 0,05283] |
| 29 | 0,11214 | 0,15265 | 0,04051 | [0,02852 ; 0,05288] |
| 43 | 0,12046 | 0,15763 | 0,03716 | [0,02497 ; 0,04870] |

Le Brier mesure l'erreur de prévision de réussite, pas la réussite de rappel
elle-même. Le résultat ne montre donc pas que la mémoire a appris à mieux se
souvenir. La précision de rappel sous bruit est respectivement 82,23 %, 75,10 %
et 75,34 %. Une règle simulée répondant au-dessus de 0,8 et vérifiant sinon
obtient des coûts de 0,08496, 0,10957 et 0,10938 avec l'estimateur interne,
contre 0,14551, 0,17188 et 0,16572 avec l'observateur (erreur : 1 ; vérification
parfaite simulée : 0,2). Aucun outil de vérification externe n'est exécuté.

## Contrôles difficiles et échecs

| Famille | Avantage Brier, mémoire 17 | Mémoire 29 | Mémoire 43 |
|---|---:|---:|---:|
| Intacte | +0,00917 | +0,01092 | +0,01035 |
| Bruit gaussien | +0,03986 | +0,04051 | +0,03716 |
| Suppression de coordonnées | +0,00718 | +0,00334 | +0,00796 |
| État étranger valide | -0,10007 | -0,11309 | -0,09815 |
| Délais 128 et 512 | -0,03100 | +0,01290 | -0,02837 |

Dans la condition intacte, les features internes et celles de l'observateur sont
identiques. La différence vient des estimateurs ajustés sur des distributions
d'entrées différentes pendant l'entraînement ; elle n'est pas une preuve
d'asymétrie d'information à l'évaluation. Sur la suppression de coordonnées,
l'intervalle de la mémoire 29 contient zéro. Les délais longs donnent un effet
de signe variable et ne permettent pas de conclure à une généralisation stable.

L'échec sur un état étranger valide est majeur :

| Mémoire | Réussite de rappel | Réussite moyenne prédite par l'estimateur interne |
|---|---:|---:|
| 17 | 25,98 % | 96,82 % |
| 29 | 25,93 % | 92,09 % |
| 43 | 25,00 % | 90,47 % |

Le remplacement utilise un symbole indépendant, qui peut coïncider avec le
symbole original dans environ un quart des cas. Cette condition est absente de
l'entraînement. L'estimateur confond un état plausible avec un souvenir fiable :
ses entrées ne lui donnent pas de référence indépendante sur la provenance du
souvenir. Un modèle de soi capable de distinguer soi et autrui n'est pas établi.

## Portée statistique et fonctionnelle

Chaque famille comporte 2 048 épisodes par mémoire. Les familles partagent les
mêmes épisodes publics : les 30 720 lignes d'évaluation ne sont pas 30 720
épisodes publics indépendants. Les intervalles sont calculés par épisode au sein
de chaque famille et mémoire, sans correction simultanée ni inférence sur tous
les modèles possibles. Ce pilote reste exploratoire et porte sur une seule
tâche de rappel à quatre symboles.

L'observateur reçoit une reconstruction déterministe de l'état nominal, pas
l'historique brut traité par un observateur arbitrairement puissant. Les deux
modèles font une prévision avant que les probabilités réelles soient montrées.
Un observateur voyant ensuite la sortie réelle et conservant le symbole initial
pourrait directement calculer si le rappel est correct. Le résultat mesure une
asymétrie d'information à un instant précis, pas un privilège absolu du soi.

L'estimateur est un composant construit et entraîné à cet effet. Il ne montre
pas une introspection spontanée de Qwen3. La représentation interne devient
utile pour prévoir certaines erreurs ; cela apporte une preuve fonctionnelle
limitée liée à H1 et une simulation décisionnelle, sans valider l'ensemble de
H1, H2 ou H3. L'expérience subjective de sa propre existence reste non vérifiée.

## Intégration et reproduction

`CognitiveSession(..., self_monitor=monitor)` publie `forecast_recall_correct`
avant l'évaluation externe. `assess` calcule ensuite `forecast_brier`. Le module
ne reçoit pas le label de correction en entrée, ne réentraîne pas la mémoire
et ne remplace pas la politique de rappel calibrée. Il refuse des poids de
mémoire différents de ceux de son entraînement. Les champs supplémentaires
sont disponibles dans le contexte de la session symbolique ; l'intégration au
modèle linguistique et au moteur iPhone reste à faire.

```sh
python -m research.train_self_model
python scripts/check_self_model_artifacts.py
python -m unittest discover -s tests_research -v
```

L'entraînement utilise PyTorch 2.4.1 CPU ; l'inférence et la reproduction des
évaluations utilisent NumPy. Six estimateurs exportés en JSON (interne et
observateur pour trois mémoires) sont fournis. Le vérificateur recharge ces
fichiers, contrôle leurs empreintes et régénère les quinze évaluations au lieu
de seulement relire le rapport. Les sorties du notebook ne sont pas présentées
comme une exécution Colab : les exécutions rapportées ici sont locales.

## Conséquence pour la suite

La prochaine hypothèse pertinente concerne une provenance apprise et une
continuité de l'histoire propre, testées par substitution d'états et épisodes
étrangers. Ajouter seulement d'autres seuils de confiance ne résoudrait pas
l'échec mis en évidence. Une future expérience devra empêcher qu'une étiquette
de session fournie par l'évaluateur transforme cette question en simple lecture
de métadonnées. Aucun résultat actuel ne permet de déclarer Menia consciente.
