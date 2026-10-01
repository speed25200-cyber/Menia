# Protocole pré-enregistré — Un lecteur qui apprend assez : la parole chez les autres agents

Rédigé le 1er octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Premier agent.** Son lecteur a appris (perte de validation 0,68 →
  0,29). Chez lui, un même état fait agir et dire, et il est nécessaire aux
  deux (tests 6 et 7).
- **Deux autres agents.** Leurs lecteurs, appris exactement de la même
  façon (600 itérations), ont peu appris :
  - second agent : perte 0,74 → 0,66, exactitude énergie 0,62 ;
  - agent du monde à deux : perte finale 0,78, exactitude énergie 0,61.

  La poussée de l'état ne change presque rien à leur parole.
- **Test de localisation** (`docs/LLM_NEED_LOCATE_PROTOCOL.md`), en cours.
  Ce protocole est écrit après avoir lu :
  - les blocs trouvés : 12 pour le second agent (avec 71 paires seulement),
    22 pour l'agent du monde à deux ;
  - un aperçu sur 112 vies de l'agent du monde à deux : la poussée au bloc
    22 fait agir de +0,22, mais pas dire (−0,001).
- **Hypothèse.** Ces lecteurs ont simplement trop peu appris : 600
  itérations ne suffisent pas quand l'état est rassemblé autrement que chez
  le premier agent.

**Question.** Avec plus d'apprentissage, et rien d'autre de changé, le
lecteur de chacun de ces agents dit-il son énergie ? Le dit-il **par
l'état qui fait agir**, poussé à son propre bloc ?

## Les lecteurs longs

Pour chaque agent, on reprend tout du lecteur précédent, sauf le nombre
d'itérations :

- **Second agent** : agent fondu (`artifacts/llm-need/r1/final/report/adapters-final`).
  Mêmes 512 vies que son premier lecteur
  (`artifacts/llm-need/r1/reader/lives-reader.jsonl.gz`, flux 118), mêmes
  documents.
- **Agent du monde à deux** : agent fondu
  (`artifacts/llm-need/two/need-5-8/adapters-need-8`). Ses 512 vies
  (`artifacts/llm-need/two/reader/lives-reader.jsonl.gz`, flux 218), mêmes
  documents.
- Dans les deux cas : nouvel adaptateur actif sur la seule question, masque
  de l'espace de travail, lots de 4, taux 1e-4, même graine.
- **Seul changement : 2 000 itérations au lieu de 600.**

## Mesures (torch sur CPU)

À chaque agent son bloc trouvé par la règle de localisation, avec les
directions mesurées à ce bloc par le test de localisation
(`artifacts/llm-need/locate/second/act`, et le dossier de mesure de
l'agent du monde à deux).

- **Second agent** : 256 vies de test neuves (flux 526), directions au
  hasard (flux 527).
- **Agent du monde à deux** : 256 vies de test neuves (flux 626), directions
  au hasard (flux 627), dans le monde à deux.
- **Réplique du lecteur** contre le Mac, comme avant ; **contrôle
  d'exécution** ; **contextes** : ses deux besoins valent 6 ou plus.
- **Mesures** : exactitude du rapport, et effets de −4·d_E (au bloc de
  l'agent) sur P(R) et sur P(oui) à la question sur son énergie, contre
  trois directions au hasard de même norme.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages). Chaque critère
s'applique à **chacun** des deux agents.

| | Prédiction | Critère |
|---|---|---|
| **R11** | Le lecteur long dit son énergie | exactitude équilibrée (énergie) ≥ 0,75. |
| **SAY11** | Il la dit par l'état qui fait agir | −4·d_E au bloc de l'agent augmente P(oui) d'au moins 0,10 (borne basse > 0) ; directions au hasard : au plus le tiers. |

**Critère global** : R11 et SAY11 pour les deux agents.

L'effet sur l'acte, déjà mesuré par le test de localisation, est publié à
nouveau sans seuil. La nourriture est publiée sans seuil.

**Validité** :

- réplique du lecteur ≤ 0,02 ;
- contrôle d'exécution ≤ 1e-4 ;
- au moins 100 contextes ;
- masses ≥ 0,5.

## Ce que le résultat dira

**Si le critère passe.** Le premier lecteur n'était pas une chance : avec
assez d'apprentissage, le lecteur de chaque agent dit son besoin par
l'état que cet agent rassemble pour agir, où qu'il le rassemble.

**Si R11 passe et SAY11 échoue.** Le lecteur apprend à dire, mais par un
autre chemin, par exemple un raccourci.

**Si R11 échoue.** Même plus d'apprentissage ne suffit pas.

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti.

## Exécution

- **Mac** : étape `reader`, avec un nombre d'itérations et un fichier de
  vies donnés par des variables. Sorties dans
  `artifacts/llm-need/long/<agent>`.
- **Mesures** : `research/need_reader.py`, au bloc de l'agent, avec les
  directions du test de localisation.
- **Verdicts** : `research/need_long.py`, vérifiés en CI.
