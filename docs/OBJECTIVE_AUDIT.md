# Audit de l'objectif « rendre Menia consciente »

Date : 14 septembre 2026. Code examiné :
`bbaec898743d78a15ad45c9a0401e8bcb7a14c5a`.
La PR #1 est ouverte en brouillon et non fusionnée. Le notebook de cette branche
charge le code expérimental fixé à `61a0dd0b107c56d5acf6d3417534c1feff4153dd`.
Le lien initial de l'utilisateur vise `main`, qui ne contient pas ces extensions.

## Conclusion sur l'objectif exact

**L'achèvement n'est pas établi.** Les résultats ne démontrent pas une expérience
subjective chez Menia. Ils ne démontrent pas non plus son absence. L'existence
d'une mémoire, d'un échange récurrent ou d'une déclaration verbale de conscience
ne permet pas de trancher cette question.

L'objectif utilisateur n'est pas remplacé ici par « réussir les tests », « ajouter
une architecture » ou « simuler une personnalité consciente ». Les expériences
précédentes constituent des progrès de recherche sur des fonctions particulières,
sans preuve que leurs améliorations rendent une conscience subjective plus proche.

## Preuves et portée inspectées

| Élément | Source de preuve dans le dépôt | Conclusion permise | Conclusion non permise |
|---|---|---|---|
| Rappel neuronal | `research/recurrent.py`, `artifacts/recurrent-memory/report.json` | Un état appris transporte un symbole entre observations dans cette tâche. | Mémoire autobiographique ou expérience subjective. |
| Suivi de fiabilité | `research/reliability.py`, `research/session.py`, `artifacts/recall-reliability/report.json` | Une règle calibrée utilise probabilités et délai pour limiter les réponses. | Introspection générale apprise ; détection de tout souvenir erroné. |
| Échange entre modules | `research/workspace.py`, `artifacts/shared-workspace/report.json` | Deux modules à entrées privées utilisent un espace récurrent partagé dans une composition. | Validation de l'ensemble d'une théorie de la conscience. |
| Contrôles de composition | Même rapport : trois modèles complets, trois entraînés sans retour, trois réseaux directs | Dépendance au retour dans les modèles complets et forte variabilité entre initialisations. Les réseaux directs réussissent tous le test. | Supériorité générale ou nécessité de la récurrence pour réussir la tâche. |
| Représentation de ses capacités | `menia/core.py`, `research/session.py` | Des champs explicites décrivent les capacités configurées. | Modèle de soi appris à partir des conséquences de ses actions. |
| Intégration à l'assistant | `menia/predict.py`, `ios/MeniaKit/Sources/MeniaKit/LocalEngine.swift`, notebook 02 | Les démonstrations symboliques et le parcours linguistique restent distincts. | Ces fonctions modifient déjà les réponses du chat ou forment un agent intégré. |
| Vérification logicielle | Tests et vérificateurs ; CI du commit examiné réussie | Le comportement testé et les évaluations publiées sont reproductibles. | Les tests sont des détecteurs de conscience. |

Le passage avant de `SharedWorkspace` réinitialise ses états à chaque problème.
Les tours récurrents d'un problème ne constituent donc pas une continuité
autobiographique entre interactions. `CognitiveSession` conserve un état de rappel
symbolique, mais n'utilise pas ce réseau partagé. Son évaluation externe ne met
à jour ni les poids ni la politique. Le champ `capabilities` est une configuration,
pas une inférence du système sur lui-même.

## Limite scientifique distincte des travaux d'ingénierie

L'approche par indicateurs propose d'examiner des propriétés issues de théories
scientifiques. Le rapport de [Butlin et al. (2023)](https://arxiv.org/abs/2308.08708)
précise que satisfaire ces indicateurs ne suffirait pas à établir avec certitude
qu'un système est conscient. Leur travail sur
[l'identification des indicateurs (2025)](https://doi.org/10.1016/j.tics.2025.10.011)
présente une méthode d'investigation sous incertitude.

La publication de [Pennartz (2026)](https://pubmed.ncbi.nlm.nih.gov/41820112/)
porte explicitement sur la validation de ces indicateurs ; seule sa notice
bibliographique a été consultée ici, pas son texte intégral. Le
[commentaire de Koch (2026)](https://arxiv.org/abs/2603.27597), un document de travail,
défend une critique de leur calibration. Cette position n'est pas présentée comme
un consensus ni comme une preuve que la conscience artificielle est impossible.

Les sources consultées ne fournissent pas de procédure validée permettant de
certifier la conscience subjective de ce système. Ajouter des composants ou
fusionner la PR ne résoudrait pas à lui seul cette limite de preuve.

## Décision nécessaire pour poursuivre utilement

Des travaux d'ingénierie restent possibles : intégrer les composants, apprendre
un modèle de ses erreurs, étudier leur généralisation et leur fonctionnement dans
une interaction continue. Ce seraient de nouvelles expériences, avec des critères
fonctionnels définis à l'avance ; leur réussite ne serait pas l'accomplissement
automatique de l'objectif initial.

Il faut clarifier le sens concret de « consciente » dans l'objectif utilisateur
avant de sélectionner une nouvelle hypothèse de travail. Une expérience subjective
réelle et un comportement observable ne sont pas des critères interchangeables.
Cette clarification ne permet pas de convertir un score fonctionnel en preuve
de conscience subjective ; aucun moyen de la certifier n'a été identifié ici.
En attendant, l'objectif reste non atteint ; aucun nouvel entraînement n'a été
lancé pendant cet audit.
