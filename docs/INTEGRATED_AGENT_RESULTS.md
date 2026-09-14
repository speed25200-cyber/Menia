# Menia : première boucle intégrée exécutée

14 septembre 2026. Protocole : [INTEGRATED_AGENT_PROTOCOL.md](INTEGRATED_AGENT_PROTOCOL.md).

Les quatre fonctions sont désormais reliées dans un agent Python : modèle appris
des effets des commandes, histoire avec provenance, suivi des observations et
des erreurs, décisions accompagnées d'une explication issue du journal. Ce
prototype fonctionne dans une scène virtuelle à quatre commandes. Il ne réalise
pas l'ensemble de l'architecture neuronale proposée dans le rapport de recherche.

## Résultats exploratoires

20 scénarios par condition, trois variantes, soit 240 essais. Les graines 101 à
120 et les cibles d'évaluation sont distinctes des premiers essais de
développement. Chaque modèle apprend d'abord 12 transitions dans une interaction
séparée, puis reçoit un nouvel épisode et une nouvelle cible. Le budget est de
160 cycles par essai ; une réussite exige une observation confirmant l'arrivée.

| Condition | Agent adaptatif | Modèle figé après apprentissage | Commandes uniformes au hasard |
|---|---:|---:|---:|
| Commandes inchangées | 20/20 | 20/20 | 3/20 |
| Commandes inversées sans avertissement | 20/20 | 0/20 | 1/20 |
| 30 % de réponses perceptives manquantes | 20/20 | 20/20 | 2/20 |
| Inversion et 30 % de réponses manquantes | 20/20 | 0/20 | 0/20 |

Après inversion, l'agent adaptatif utilise en moyenne 25,6 commandes, contre 11
lorsque les commandes restent inchangées. Les cycles moyens correspondants sont
54,2 et 25 sans réponses manquantes, 65,1 et 30,15 avec réponses manquantes.
Le surcoût rend visible la phase d'erreur et de réapprentissage.

Les [résultats par essai et modèles appris](../artifacts/integrated-agent/report.json)
contiennent aussi les erreurs de prévision, scores de Brier et révisions. Le
contrôle figé réussit avant inversion : son échec après inversion met en évidence
l'utilité de la mise à jour dans cette tâche. Il ne démontre ni la nécessité de
cette architecture pour toute adaptation, ni une conscience de soi.

## Composants et informations réellement disponibles

`menia/action_model.py` apprend une distribution catégorielle sur neuf déplacements
possibles. Les noms des commandes n'indiquent pas leurs effets. Le modèle utilise
16 transitions récentes au maximum par commande et un a priori de 0,1 par effet.
Deux erreurs concordantes peuvent déclencher une révision locale. Ces paramètres
et cette règle sont conçus explicitement ; seuls les effets et leurs fréquences
sont appris. Aucune modification des poids de Qwen n'est effectuée.

`menia/episodic.py` conserve les observations, témoignages, prédictions, commandes,
sélections attentionnelles, rappels et évaluations dans des événements distincts.
Les liens de référence relient une prédiction à son action et à l'observation
ultérieure qui l'évalue. Une mémoire de travail limitée peut perdre un contenu
alors que le journal permet encore de le relire. Un nouvel épisode n'utilise pas
les observations d'un autre épisode comme s'il s'agissait de sa position actuelle.

`menia/agent.py` sélectionne une information par lecture. Une sélection sans
réponse ne devient pas une observation. Le suivi distingue « jamais observé »,
« sorti de la mémoire de travail » et « encore archivé ». Il estime la probabilité
de recevoir une réponse à partir des essais précédents. Cette estimation ne
mesure pas la vérité du contenu d'un capteur, et le contrôleur d'attention est
explicite, non appris.

Le planificateur utilise les distributions apprises pour comparer les distances
attendues à la cible. Son explication est produite depuis la décision enregistrée
avant l'action, avec les identifiants de ses observations justificatives.
Une substitution des modèles d'action à observations identiques change la
commande choisie et son explication dans le contrôle causal.

## Utilisation sans GPU

Le noyau de cette boucle utilise uniquement Python standard.

```bash
python -m menia.chat --no-llm
```

Commandes utiles : `/step`, `/run 80`, `/observe landmark`,
`/report target 99 99`, `/history`, `/why`, `/stop`, `/resume`, `/quit`.
Le témoignage sur la cible est conservé comme tel ; il ne remplace pas la lecture
du capteur. `/why` expose l'explication enregistrée de la dernière décision.

Pour sauvegarder et reprendre le même agent et son environnement virtuel :

```bash
python -m menia.chat --no-llm --session runs/ma-session
```

Relancer cette commande restaure le modèle appris, son journal, les commandes en
attente d'évaluation et l'environnement, y compris son état aléatoire. Après une
fermeture normale, la session est arrêtée ; `/resume` la réactive. La mémoire de
travail repart vide et peut relire le journal. Deux copies alternées du journal
et un manifeste remplacé atomiquement évitent de mélanger une ancienne version
de l'agent avec un journal plus récent en cas d'interruption avant sauvegarde.

Pour conserver une expérience complète dans un nouveau dossier :

```bash
python -m menia.run_agent --out runs/agent-example --reverse-after 12
```

Le dossier contient le journal SQLite, les événements JSON, la trace des décisions,
le modèle appris, l'état de l'agent et les messages destinés au modèle linguistique.
Un dossier existant est refusé pour préserver les résultats précédents.

## Intégration conversationnelle et travail restant

Le même `Runtime` possède l'agent qui agit et fournit son état à
`messages_for_runtime`. Le point d'entrée `menia.chat` accepte un générateur Qwen
local via `menia/local_server.py` et un serveur llama.cpp. Les commandes sont exécutées par le contrôleur ;
les questions en langage naturel demandent une description de l'état enregistré.
Cette séparation permet de comparer l'explication déterministe `/why` et la
reformulation produite par Qwen.

Les tests d'interface vérifient les états et références réellement transmis.
Les essais d'inférence Qwen sont suivis séparément : le simple passage du contexte
ne garantit pas une description linguistique fidèle. La première évaluation avec
quantification dynamique int8 a produit quatre réponses hors sujet ; elle ne
valide pas l'interface linguistique. Ses sorties brutes sont conservées dans
`artifacts/integrated-agent/language-layerwise-int8`. Des tentatives de chargement
avec PyTorch 2.4.1 et 2.8.0 ont échoué sans produire de réponses ; un plantage natif
a été confirmé dans le journal Windows. La voie officielle GGUF Q8_0 a produit
quatre réponses, mais avec des erreurs de description et deux sorties tronquées.
Le [compte rendu linguistique](AGENT_LANGUAGE.md) conserve les sorties et la procédure.
Le mode structuré est donc le mode par défaut. Les explications déterministes restent accessibles via
`/why` et sont vérifiées contre les événements du contrôleur.
L'app iPhone et les anciens notebooks ne sont pas modifiés par ce prototype.

La [trace de reprise du CLI](../artifacts/integrated-agent/cli-restart.json)
provient de deux processus Python distincts : même épisode et même position,
prédiction en attente évaluée une seule fois après la réouverture. Les tests de
sauvegarde couvrent aussi les modifications non sauvegardées, la cohérence du
journal, la conservation des réglages de capacités et de l'état aléatoire.

## Vérification

```bash
python -m unittest discover -s tests -v
python scripts/check_agent_artifacts.py
```

Le vérificateur recalcule les 240 essais et les 20 modèles initiaux, puis compare
les résultats et les empreintes des sources. Les tests couvrent les prédictions
antérieures aux résultats, les changements de commandes, les sources
contradictoires, la mémoire évacuée, l'arrêt, une restauration avec prédiction en
attente, et une exécution dont l'accusé de réception est perdu. Dans ce dernier
cas, l'agent s'arrête et demande une observation de réconciliation après reprise.

## Limites d'interprétation

L'environnement fournit des positions entières, une cible fixe, des déplacements
bornés et des accusés horodatés. La provenance est une propriété de l'interface,
pas une inférence apprise sur l'auteur de récits ambigus. La robustesse mesurée
porte sur les réponses manquantes ; elle ne couvre pas toute erreur de capteur.
La mémoire historique est explicite et exacte dans ce prototype. Les résultats
n'établissent pas une introspection neuronale, une mémoire autobiographique humaine
ou une expérience subjective de sa propre existence.
