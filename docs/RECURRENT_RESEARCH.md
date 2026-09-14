# Menia v0.2 — module récurrent expérimental

## Ce qui a réellement été construit

Un petit réseau neuronal de 1 540 paramètres, entraîné par rétropropagation à
travers le temps sur CPU. Il reçoit un symbole parmi quatre, ou aucune observation,
et doit rappeler le dernier symbole vu. Les poids sont disponibles dans
`artifacts/recurrent-memory/memory-seed-{17,29,43}.json`.

Ce modèle est un composant de mémoire, pas un assistant généraliste ni un modèle
conscient. Qwen3-1.7B reste le composant linguistique prévu. Aucune intégration
ne transforme automatiquement ces deux composants en un système conscient.

## Architecture calculable

Pour chaque pas, x a cinq composantes (quatre symboles et validité), h en a 32 :

```
g = sigmoid(x G + bg)
c = tanh(x W + h_précédent U + b)
h = (1-g) h_précédent + g c
p = softmax(h V + bo)
```

Tous les paramètres sont appris. Cross-entropie sur les pas sans observation,
BPTT, Adam, clipping de norme à 1. Les cibles ne sont jamais données au réseau.
Les probabilités portent uniquement sur quatre symboles : elles ne sont pas
une confiance sur la qualité générale de l’assistant ni une mesure de conscience.

`CognitiveSession` ajoute une trace bornée d’épisodes, des capacités explicites,
une prédiction enregistrée avant son évaluation et une sauvegarde sur demande.
Le contexte distingue observation et inférence. Les entrées sont symboliques et
fournies par l’utilisateur ; ni caméra ni microphone ne sont déclenchés.

## Résultats exécutés

Train : 300 updates par initialisation, batch 64, séquences de 12 pas.
Test : 256 épisodes de 32 pas, seed 910001 séparée du train,
6 597 pas sans observation évalués pour chaque initialisation.

| Initialisation | Réseau entraîné | État effacé à chaque pas | Sans observations | Règle du dernier symbole |
|---|---:|---:|---:|---:|
| 17 | 100 % | 25,83 % | 24,75 % | 100 % |
| 29 | 100 % | 25,83 % | 25,83 % | 100 % |
| 43 | 99,92 % | 24,59 % | 24,59 % | 100 % |

Résultats exacts et Brier multiclasse dans `report.json`. Les pourcentages ne
sont pas des scores de conscience. L’ablation indique que l’état récurrent aide
sur ce rappel différé. La règle déterministe réussit tout : le réseau n’établit
pas une supériorité sur une mémoire programmée. Les épisodes de test appartiennent
à la même famille synthétique. Les pas d’un épisode sont corrélés ; ne pas traiter
6 597 pas comme autant d’expériences indépendantes.

## Reproduction

```bash
pip install -r requirements-research.txt
python -m unittest discover -s tests_research -v
python scripts/check_recurrent_artifacts.py
python -m research.train_recurrent --out /tmp/menia-reproduction --steps 300 --seeds 17 29 43
```

Les gradients ont été vérifiés par différences finies. Tests de causalité
(absence d’influence d’entrées futures), validation des poids, round-trip,
annulation de session, effacement et autorisation de sauvegarde.
Les hashes des sources et checkpoints sont inscrits dans le rapport livré.
Le temps d’exécution est une mesure de cette machine CPU, pas une prévision Colab.

## iPhone

`RecurrentMemoryModule.swift` implémente le calcul des poids JSON sans dépendance
supplémentaire. Charger un checkpoint dans `RecurrentMemoryModule(data:)`, appeler
`step(symbol:)`, et `reset()` à l’effacement. Le fichier `mobile-fixture.json`
contient une séquence et les probabilités Python attendues pour vérifier la parité.
**Cette implémentation Swift n’a pas été compilée ni mesurée sur iPhone.**
Elle n’est pas encore raccordée au chat, et ne modifie pas ses réponses.
Le petit nombre de paramètres concerne ce module, pas Qwen ni la mémoire totale.

## Hypothèse de recherche et limite

La récurrence et l’exploitation d’informations internes font partie des mécanismes
que la littérature discute au sujet de la conscience artificielle. Le rapport de
[Butlin et al. (2023)](https://arxiv.org/abs/2308.08708) propose des indicateurs
dérivés de plusieurs théories ; il ne donne pas une recette qui établit la
conscience à partir d’un tel module. Nous ne prétendons pas implémenter ou valider
l’ensemble de ces indicateurs.

L’hypothèse testée ici est beaucoup plus étroite : « le réseau utilise un état
interne pour rappeler une information qui n’est plus visible ». Il s’agit d’une
première brique vérifiée. **L’existence d’une expérience subjective reste inconnue.**
