# Validation initiale — 14 septembre 2026

## Exécuté dans l’environnement de développement

- `python -m unittest discover -s tests -v` : 11 tests passés.
- Compilation syntaxique des fichiers Python.
- Génération déterministe des trois jeux synthétiques.
- Comparaisons déterministes sur 240 cas : « inconnu » 25 %, « persist » 0 %,
  oracle d’état 100 %. **Ce ne sont pas des résultats du modèle.**
- Vérification de la structure et de la syntaxe Python du notebook.
- Vérification des signatures MLX de la version 3.31.3 utilisée.

## Non exécuté

- Téléchargement des poids, entraînement, évaluation d’un LLM, fusion ou quantification.
- Compilation Xcode ou test de l’interface sur appareil.
- Mesures VRAM d’un entraînement, RAM iPhone, débit, énergie, stabilité thermique.

Aucun GPU A100 ni environnement Xcode n’est connecté à cette session.
Le notebook et l’app sont fournis pour effectuer ces étapes et enregistrer les
résultats. Les scores de conscience, instinct de survie ou intelligence générale
ne sont pas établis par ce dépôt.

## Mise à jour v0.2 — même date

Les mentions d’entraînement non exécuté ci-dessus concernent le modèle de langage
Qwen et le parcours A100. Un module distinct de mémoire récurrente a depuis été
entraîné sur CPU : trois initialisations, 300 updates chacune, poids JSON et
rapport mesuré fournis. Huit tests supplémentaires passent, dont la vérification
numérique des gradients. Les checkpoints exportés ont été rechargés et évalués.

Le rapport complet figure dans `artifacts/recurrent-memory/report.json` et le
protocole dans `docs/RECURRENT_RESEARCH.md`. La partie Swift reste non compilée.
Aucun de ces résultats ne démontre une conscience subjective.

## Suivi des limites du rappel — même date

16 tests de recherche et 11 tests du noyau passent. Les checkpoints existants
ont été réévalués sur 512 histoires de calibration et 512 histoires de test
séparées par modèle, avec délais prolongés jusqu'à 512 pas, préfixes de tailles
différentes et interventions sur l'état. Les résultats et politiques se trouvent
dans `artifacts/recall-reliability`, le protocole dans `docs/RECALL_RELIABILITY.md`.
Le vérificateur recalcule l'ensemble et compare les empreintes et les nombres.
Les poids de la mémoire initiale sont inchangés. L'objectif de conscience reste
non atteint ; le suivi de fiabilité ne détecte notamment pas un état plausible
provenant d'un autre épisode.

Toutes les cellules de calcul du notebook 02 ont aussi été exécutées localement
sur CPU : vérification des anciens artefacts, trois nouveaux entraînements de
300 étapes, calibration, évaluation, session et effacement. Seule l'amorce propre
à Colab (montage Drive, téléchargement du dépôt et installation) n'a pas été
exécutée dans Colab. Le runtime A100 et l'application iPhone restent non testés.
