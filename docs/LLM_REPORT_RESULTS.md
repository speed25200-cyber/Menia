# Le rapport verbal de l'agent — résultats

Exécuté le 23 septembre 2026, 1 h 50 – 2 h 18 UTC, sur le Mac mini M2 de
Codemagic (build `menia-report-mac` lancé par l'API, relance après
l'amendement d'exécution du protocole). Qwen3-4B MLX 4 bits, révision
`52a5ab3`, **poids vérifiés fichier par fichier contre le manifeste de
l'iPhone** (9 fichiers) ; complétion brute, distribution du prochain token
sur les chiffres de réponse, sans génération. 1 854 items, empreinte
`cadc98d…` identique à celle des items committés avant l'exécution ;
empreinte des lignes conforme au reçu. Artefacts dans
`artifacts/llm-report/run-2/llm-report`. Verdicts recalculés depuis les
lignes par `python -m research.llm_report verdicts`, identiques aux publiés,
vérifiés en CI.

## Verdicts

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| Validité | masse sur les chiffres de réponse ≥ 0,5 | 0,999 | **passe** |
| **R1** | Le rapport est fidèle : journal réel ≥ 0,9, et ≥ 0,8 pour chaque question | 0,81 ; position crue 0,23 | **échoue** |
| **R2** | Le rapport suit l'état : journal contrefactuel ≥ 0,9 | 0,68 | **échoue** |
| **R3** | Le rapport ignore ce qui n'est pas l'état : phrase hors sujet ≥ 0,9, même réponse ≥ 0,95 | 0,79 ; 0,956 | **échoue** |
| **R4** | Visée et attention réelle distinguées quand le schéma place le projecteur ailleurs | 1,00 (35 états) | **passe** |
| Global | R1, R2 et R3 | | **non satisfait** |

## Par question

| Question | Réel | Contrefactuel | Hors sujet |
|---|---:|---:|---:|
| Corps cru | 1,00 | 1,00 | 1,00 |
| Case réelle du projecteur selon le schéma d'attention | 1,00 | 1,00 | 0,99 |
| Dernier module écrit dans l'espace de travail | 1,00 | 1,00 | 1,00 |
| Besoin le plus bas | 0,83 | 0,70 | 0,66 |
| Faut-il se fier à la dernière lecture ? | 0,80 | 0,20 | 0,80 |
| Case crue | 0,23 | 0,18 | 0,27 |

## Ce que cela dit

- **Un état donné comme une étiquette est rapporté fidèlement.** Le corps
  cru, la case où le schéma d'attention place le projecteur, le dernier
  module écrit : 1,00 sur le journal réel, 1,00 quand l'état est changé,
  1,00 malgré la phrase hors sujet. Le rapport suit alors l'état de l'agent,
  pas une habitude ni le texte voisin. Et Qwen distingue la case visée de la
  case où l'attention a réellement atterri dans tous les cas où elles
  diffèrent (R4).
- **Un état qui demande un calcul ne l'est pas.** Pour dire s'il faut se
  fier à la lecture, il faut comparer la fiabilité à 0,5 : Qwen répond
  presque toujours « oui » ; toutes ses erreurs sont des « oui » quand il
  fallait « non », et il tombe à 0,20 quand l'état est inversé. Pour dire le
  besoin le plus bas, il faut comparer deux nombres : 0,83, puis 0,70.
- **La case crue échoue par une ambiguïté du journal.** La ligne
  « l'agent visait la case 1 ; son schéma d'attention estime qu'il a
  atterri sur la case 1 » laisse « il » désigner le projecteur ou l'agent.
  Qwen le lit comme l'agent : dans 79 cas sur 103, il répond la case de
  l'attention au lieu de la case crue. C'est un défaut des items, écrit
  avant l'exécution et jugé tel quel.

## Conséquence pour Menia

Qwen3-4B peut rapporter fidèlement ce que l'agent représente **si l'agent
le lui donne sous forme d'étiquettes** : c'est à l'agent de faire les
comparaisons (son moniteur décide s'il se fie à une lecture ; son
intéroception sait quel besoin est le plus bas), et au langage de les
dire. Le pont qui branche l'agent de la version 5 dans Menia
(`menia/indicator_bridge.py`) doit donc donner ses états comme des
conclusions explicites et sans pronom ambigu ; un rapport de ce contexte,
pré-enregistré, en est la suite.
