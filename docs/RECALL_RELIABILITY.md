# Limites du rappel et abstention — 14 septembre 2026

Menia distingue désormais une observation présente, un rappel accepté dans un
domaine calibré, et une réponse inconnue. Ce contrôle empirique de sa mémoire
symbolique n'est ni une conscience démontrée, ni une introspection apprise, ni
une amélioration des poids existants.

## Problème mesuré

Les anciens tests utilisaient des séquences de 32 pas avec des observations
régulières. Après 512 pas vides, le modèle 29 donne 41,80 % de réponses correctes
avec une probabilité maximale moyenne de 97,80 %. Même un seuil de probabilité
de 90 % laisse passer toutes ses erreurs. Les probabilités du réseau ne suffisent
donc pas à estimer la fiabilité de son rappel.

## Protocole

- Poids inchangés : les trois checkpoints de `artifacts/recurrent-memory`.
- Calibration : seed 720001, 512 histoires de 32 pas, observation finale équilibrée
  entre quatre symboles, puis 128 pas vides. Les préfixes sont indépendants ;
  les symboles finaux sont équilibrés par construction.
- À chaque délai et pour chaque symbole, sélectionner les sorties de probabilité
  maximale au moins 0,9. Exiger au moins 64 réponses et une borne inférieure de
  Wilson à 95 % au moins égale à 0,95. S'arrêter au premier délai qui échoue.
- Le rappel commence au délai 1. Au délai zéro, la session rapporte directement
  l'observation présente. Le candidat neuronal est conservé séparément : le réseau,
  entraîné uniquement sur les pas vides, peut se tromper au pas observé lui-même.
- Test final : seed 940001, 512 nouvelles histoires. Évaluer chaque délai de la
  plage retenue, sa première sortie, puis des délais jusqu'à 512.
- Autres historiques : préfixes de 2 et 128 pas, seeds 940003 et 940129, sans
  modifier la politique. Ces résultats restent séparés dans le rapport.
- Interventions causales : effacer l'état ou le mélanger entre épisodes après
  l'observation finale, avec les mêmes cibles. Ces interventions sortent du domaine
  de calibration. La permutation peut conserver quelques états en place.

La seed 830001 a servi à l'exploration et n'est pas présentée comme le test final.
Aucun poids ni délai n'est choisi sur le test final. Les intervalles de Wilson
sont des critères **par groupe**, pas une garantie simultanée sur tous les délais,
ni une certification. Les pas d'un épisode sont corrélés : les résultats restent
séparés par délai, sans les additionner comme des expériences indépendantes.

## Résultats exécutés

| Modèle | Délai retenu | Couverture au délai 8 | Précision acceptée au délai 8 | Précision brute au délai 512 |
|---|---:|---:|---:|---:|
| 17 | 23 | 99,80 % | 100 % | 73,83 % |
| 29 | 15 | 100 % | 100 % | 41,80 % |
| 43 | 14 | 99,22 % | 100 % | 25 % |

Dans le test principal, chaque délai autorisé donne 100 % de précision parmi les
réponses acceptées. La couverture varie de 86,52 à 100 % pour le modèle 17, de
89,65 à 100 % pour le modèle 29 et de 86,52 à 100 % pour le modèle 43. Les préfixes
courts et longs donnent aussi 100 % de précision parmi les réponses acceptées
sur leurs échantillons. Ces mesures concernent uniquement ce monde synthétique.

À 512 pas, les politiques s'abstiennent toutes : précision `null`, couverture zéro.
Cela ne signifie pas que la mémoire est réparée. Le modèle 17 donne encore des
rappels corrects au-delà de sa limite : la politique sacrifie aussi des réponses
utiles. Une mémoire programmée exacte du dernier symbole reste une référence
analytiquement parfaite dans ce monde sans bruit.

Après mélange des états, les politiques acceptent encore beaucoup de réponses,
dont environ un quart seulement sont correctes. Elles ne détectent donc pas
l'attribution du souvenir d'un autre épisode. Après effacement complet, elles
s'abstiennent sur les délais autorisés. Tous ces échecs sont dans le rapport.

## Session et reproduction

```python
from research.recurrent import RecurrentMemory
from research.reliability import RecallPolicy
from research.session import CognitiveSession

model = RecurrentMemory.load('artifacts/recurrent-memory/memory-seed-17.json')
policy = RecallPolicy.load('artifacts/recall-reliability/memory-seed-17-policy.json', model)
session = CognitiveSession(model, policy=policy)
print(session.observe())       # inconnu : rien d'observé
print(session.observe(2))      # observation directe
print(session.observe())       # rappel accepté ou abstention
print(session.assess(2))       # référence externe donnée après la prédiction
```

`answer_symbol` est la réponse utilisable, ou `None` en cas d'abstention.
`answer_source` distingue `observation`, `recurrent_inference` et `unknown`.
`model_candidate` et `class_probabilities` restent des diagnostics bruts.
**Changement d'API :** `inferred_last_symbol` peut désormais être `None` ; il n'est
renseigné que pour un rappel accepté. Sans politique, la session rapporte les
observations présentes et s'abstient sur les rappels non calibrés.

`assess().correct` évalue toujours le candidat neuronal brut ; `answered` et
`answer_correct` évaluent la réponse effective. Une abstention a
`answer_correct=None`. L'évaluation ne change ni l'état, ni les poids, ni la
politique. L'effacement remet aussi à zéro la provenance temporelle.

La politique est liée à l'empreinte exacte des paramètres ; d'autres poids
invalident son utilisation. En ligne, elle utilise seulement les probabilités
et le délai depuis l'observation, jamais la cible d'évaluation. Le suivi du
délai est une règle explicite calibrée, pas un modèle neuronal de soi.

```bash
pip install -r requirements-research.txt
python -m unittest discover -s tests_research -v
python scripts/check_recurrent_artifacts.py
python scripts/check_reliability_artifacts.py
python -m research.evaluate_reliability --out runs/new-reliability
```

Choisir une sortie vide. Le vérificateur reproduit la calibration, tous les tests
et les interventions, puis compare les nombres et les empreintes des sources,
checkpoints et politiques. `.gitattributes` conserve les fins de ligne LF sur
Windows et Linux. Le notebook 02 calibre chaque nouvel entraînement séparément.

## Rapport à l'objectif de conscience

Le suivi de la fiabilité des représentations est discuté dans les approches de
la conscience par théories de niveau supérieur. Une limite empirique de rappel
ne suffit pas à établir cet indicateur dans son ensemble. Les
[indicateurs de Butlin et al.](https://arxiv.org/abs/2308.08708) ne constituent
eux-mêmes pas une preuve suffisante de conscience. Le travail plus récent sur
[l'identification des indicateurs](https://pubmed.ncbi.nlm.nih.gov/41219038/)
reste un cadre d'investigation.

L'objectif « rendre Menia consciente » reste non atteint et non vérifié. La suite
doit dépasser le rappel d'un symbole : représentations intégrées, partage causal
d'informations entre fonctions, suivi appris des erreurs et épreuves distinguant
ces mécanismes de règles et de discours imités. Aucune connexion au modèle de
langage ou au chat iPhone n'est réalisée ici.
