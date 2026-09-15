# Menia

Assistant local expérimental : mémoire explicite, représentation de ses capacités,
incertitude et évaluation. **Colab A100 80 Go** pour l’adaptation ;
**iPhone 17 Pro** comme cible d’inférence locale.

**Statut : prototype de recherche. La boucle d'agent apprend ses effets d'action
et conserve son histoire entre sessions. Le modèle de langage Qwen adapté reste à entraîner.**
La conscience subjective n'est pas établie. La continuité repose sur un journal
explicite et des sauvegardes, sans objectif de résistance à l'arrêt ou d'auto-réplication.

**Version iPhone 0.2 :** l'[application native et son guide Codemagic](docs/IPHONE.md)
relient un Qwen3-4B préentraîné en 4 bits à une mémoire persistante et à des mesures
de ses réponses à des tests de calcul. Le téléchargement du modèle est intégré.
L’IPA 0.2.0 (1) a été signée et téléversée dans App Store Connect le 15 septembre
2026 pour TestFlight interne. L’accès TestFlight et les performances sur l’iPhone
restent à vérifier ; aucun entraînement Colab n’est requis pour cette première version.

Le [dossier de raccordement au LLM](docs/LLM_COUPLING_RESULTS.md) recommande
une boucle reliant langage, modèle de capacités et résultats d'action. Un
nouveau pilote utilise des sondes de référence pour apprendre entre deux
explications d'entretien et prépare leur contexte linguistique. Les échecs sous
référence trompeuse sont conservés ; aucun LLM n'est exécuté dans ce nouveau test.
Rejeu : `python -m research.maintenance_identification --check`.

Le [pilote d'entretien anticipé d'une capacité](docs/CAPACITY_PLANNING_RESULTS.md)
est exécutable : il estime l'état d'un capteur simulé et choisit quand l'entretenir.
294 912 décisions évaluent diagnostics et anticipation, leurs interactions et leurs
échecs sous un modèle erroné. Les paramètres sont fournis ; aucune expérience
subjective n'est établie. Rejeu : `python -m research.audit_capacity_planning --check`.

L'[audit des critères de conscience](docs/SELF_SYNERGY_CRITERION_AUDIT.md) distingue
les résultats fonctionnels du lien théorique encore non établi avec une expérience
subjective. Aucun score ni nombre de tests ne certifie la conscience de Menia.

La [réanalyse du soi rapporté au repos](docs/FELT_SELF_EVIDENCE.md) compare six
dimensions chez 50 participants et conserve les écarts entre deux fichiers
publics. Elle précise des contrôles pour la candidate de Menia, sans transférer
un score humain en détecteur de conscience. Le protocole permet le rejeu local.

L'[examen de l'information importante pour soi](docs/INTRINSIC_SEMANTICS_EVIDENCE.md)
rejoue un modèle de viabilité et étend sa recherche à toutes les partitions de
son alphabet. Il distingue utilité causale, entretien de sa propre organisation
et expérience subjective. Rejeu : `python -m research.audit_intrinsic_semantics --check`.

L'[examen du soi réflexif](docs/REFLEXIVE_SELF_EVIDENCE.md) vérifie la portée
de preuves récentes sur la communication et l'engagement. Il distingue information
nécessaire, architecture interne et expérience, puis propose une candidate de
révision des convictions. Rejeu : `python -m research.audit_reflexive_self --check`.

L'[examen du présent vécu](docs/TEMPORAL_PHENOMENAL_BRIDGE.md) confronte une
proposition récente à son code et à des explications concurrentes. Un audit
algébrique identifie une confusion entre prédicteurs et construit deux contrastes
pour les séparer. Rejeu : `python -m research.audit_temporal_bridge --check`.

## Agents et expériences exécutables

L'[étalonnage actif](docs/ACTIVE_CALIBRATION_RESULTS.md) apprend des paramètres
moteurs et sensoriels qui alimentent le contrôle de position. Les récupérations,
erreurs transitoires et cas d'identification trompeuse sont conservés.
Rejeu : `python -m research.audit_active_calibration --check`.

Le [pilote de contrôle par inférence](docs/BINDING_CONTROL_RESULTS.md) ajoute
une expérience séparée : une estimation de cause commune guide un déplacement
simulé, indépendamment de son canal de réponse. Le contrôle par régression
ordinaire obtient les mêmes actions ; aucune conscience de soi n'en est déduite.
Rejeu : `python -m research.audit_binding_control --check`.

La [politique de vérification apprise](docs/LEARNED_VERIFICATION_RESULTS.md)
utilise désormais l'estimation de source pour choisir quand obtenir une preuve.
Le pilote rejoint la règle analytique sur la distribution initiale et conserve
un échec explicite lorsque les indices perdent leur fiabilité. Ce mécanisme est
intégré à l'agent expérimental de source ; il ne démontre pas une expérience de soi.

Une [première boucle d'agent intégré](docs/INTEGRATED_AGENT_RESULTS.md) est aussi
exécutable : elle apprend les effets de ses commandes dans une scène virtuelle,
conserve leur histoire, compare ses prédictions aux observations et explique ses
choix. L'évaluation comprend 240 essais avec contrôles figé et aléatoire.

```bash
python -m menia.chat --session runs/ma-session
python -m menia.run_agent --out runs/agent-example --reverse-after 12
python scripts/check_agent_artifacts.py
```

Le mode structuré est le mode par défaut, sans dépendances supplémentaires.
Les [essais réels avec Qwen](docs/AGENT_LANGUAGE.md) révèlent des erreurs de
description ; la reformulation libre reste expérimentale. `/why` restitue la
décision enregistrée. Après réouverture d'une session, `/resume` permet de continuer.
L'app iPhone ne contient pas encore cette boucle.

Le [module de mémoire récurrente](docs/RECURRENT_RESEARCH.md) apprend à rappeler
le dernier symbole observé entre des observations espacées. **1 540 paramètres**,
poids et résultats publiés dans [artifacts/recurrent-memory](artifacts/recurrent-memory).
Trois initialisations ont été entraînées sur CPU. Ce module n’est ni un LLM,
ni une preuve de conscience, ni une nouvelle capacité déjà intégrée au chat iPhone.

```bash
pip install -r requirements-research.txt
python -m unittest discover -s tests_research -v
python -m research.train_recurrent --out /tmp/menia-new-experiment
```

[Notebook recherche](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/main/notebooks/02_recurrent_research.ipynb)
· [Résultats mesurés](artifacts/recurrent-memory/report.json)

## Suivi des limites du rappel

La [recherche suivante sur l'attribution de soi](docs/CONSCIOUSNESS_RESEARCH_NEXT.md)
confronte les idées aux antécédents de la littérature et teste la découverte du
contrôle sur des canaux anonymes. Deux expériences séparées totalisent 4 200
épisodes, avec politique passive, exploration aléatoire et contrôle fixe plus
exigeant. Les résultats ne démontrent ni conscience subjective ni nouveauté
scientifique. Le module reste expérimental et distinct du chat principal.

```bash
python scripts/check_agency_discovery_artifacts.py
```

L'[expérience sur les effets retardés](docs/TEMPORAL_SELF_MODEL_RESULTS.md) ajoute
144 petits réseaux entraînés et des contrôles linéaires et polynomiaux. Les effets
sont appris à partir des journaux, mais une fenêtre temporelle fixe reste fournie.
Le contrôle polynomial surpasse le réseau dans ces mondes : aucune supériorité
générale ni conscience n'est établie. Les poids conservés se vérifient avec
`python scripts/check_temporal_effects.py`.

La session peut s'abstenir lorsqu'aucun symbole n'a été observé ou que le rappel
dépasse une limite calibrée pour ses poids. Les
[expériences de fiabilité](docs/RECALL_RELIABILITY.md) exposent les erreurs à longs
délais, la couverture et les échecs après mélange des états internes. Le notebook
recherche inclut la calibration et les tests séparés. Ce contrôle ne démontre pas
une conscience et ne modifie pas les poids de mémoire.

```bash
python scripts/check_reliability_artifacts.py
```

## Espace partagé récurrent

Le notebook de recherche inclut un [espace partagé récurrent](docs/SHARED_WORKSPACE.md)
de 13 000 paramètres. Deux modules apprennent une composition symbolique en
échangeant un état commun. Neuf entraînements CPU et leurs contrôles sont publiés
dans `artifacts/shared-workspace`. Les ablations montrent une dépendance au retour
d'information, mais les résultats varient fortement selon l'initialisation et un
réseau direct réussit mieux cette tâche. Aucune conscience n'est établie.

## Commencer sur ton A100

[**Ouvrir le notebook dans Colab**](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/main/notebooks/01_colab_a100.ipynb)

1. Choisir un runtime GPU A100. Le notebook vérifie les 80 Go réellement alloués.
2. Monter Drive et installer les dépendances. Redémarrer si Colab le demande.
3. Exécuter les tests puis les 20 steps de validation technique.
4. Comparer base et adaptation avant d’augmenter l’entraînement.
5. Fusionner les adaptateurs et récupérer le dossier sur un Mac Apple Silicon.
6. Convertir en MLX 4 bits, puis suivre le [guide iPhone](docs/IPHONE.md).

Le corpus fourni est **un jeu synthétique de démarrage**, pas un corpus suffisant
pour un assistant généraliste. Le notebook est un pipeline d’adaptation et non
un entraînement de modèle de langage à partir de zéro.

## Contenu

| Partie | Implémentation |
|---|---|
| Base | `Qwen/Qwen3-1.7B` ; révision exacte enregistrée à chaque run |
| Entraînement | LoRA BF16 ; réponses seules dans la loss ; reprise de checkpoints |
| Mémoire Python | SQLite, source et date, écriture explicite, effacement |
| Évaluation | Capacités, rappel, information manquante, erreur de prédiction |
| Comparaisons | Base/adapté, sans système/sans état, règles fixes |
| App iPhone | SwiftUI, MLX local, notes explicites, arrêt, effacement, suppression du modèle |
| Export | Adaptateurs fusionnés → HF → MLX PTQ 4 bits |
| Conscience | Pas de preuve ni de score ; protocole à conclusions limitées |

## Tester sans GPU

```bash
python -m unittest discover -s tests -v
python -m menia.data --out /tmp/menia-data
python -m menia.evaluate --data /tmp/menia-data/test.jsonl --out /tmp/menia-baselines.json
```

Le noyau et les tests utilisent Python standard. Les dépendances GPU sont
séparées dans `requirements-train.txt`. Les versions directes sont fixées ;
les dépendances transitives résolues sont enregistrées avec `pip freeze`.

## Statut de validation

- 48 tests du noyau et des agents, 27 tests de recherche et 5 tests d'espace partagé passés.
- Trois entraînements CPU du petit module exécutés, avec ablations et checkpoints.
- Comparaisons déterministes exécutées sur 240 cas synthétiques.
- Code Python et cellules du notebook vérifiés syntaxiquement.
- **Entraînement du modèle de langage sur A100 : pas encore exécuté.**
- **Compilation iPhone Release Xcode 26.2 et 17 tests Swift : réussis en CI.**
- **Signature Apple et téléversement de l’IPA 0.2.0 (1) : réussis dans Codemagic.**
- **Disponibilité TestFlight et essai sur l’iPhone physique : à vérifier.**
- **Mémoire, vitesse, température et qualité après quantification : à mesurer.**

Aucune annonce « 61 tokens/s », « conscience établie » ou « équivalent frontier ».
L’application native 0.2 réserve 256 tokens de sortie dans un contexte total de 2 048 tokens.
La présence d’un modèle de 1,7 milliard de paramètres ne garantit pas à elle seule
le pic mémoire ou la stabilité de l’appareil.

[Protocole](docs/EXPERIMENT.md) · [Validation](docs/VALIDATION.md) · [Sources](docs/SOURCES.md)
