# Version 5 — analyses exploratoires et erratum

Non pré-enregistrées, déclarées comme telles. Faites le 23 septembre 2026
après la publication de la version 5, pour préparer une éventuelle version
6 ; aucune graine confirmatoire future n'a servi. Script
`research/indicator_v5_explorations.py`, sortie
`artifacts/indicator-agent-v5/explorations.json` ; l'agent de développement
utilisé est publié (`artifacts/indicator-agent-v5/dev-params-19.json`).

## Erratum : la mesure du choix de HOT-4

Le protocole de la version 1 écrit, pour le volet d'usage de HOT-4 :
« conflits impliquant un objet de la bande : choix correct ≥ 0,8 contre
≤ 0,65 ». Le code mesure depuis la version 1 le choix correct à **tous** les
pas du jeu H où un objet de la bande est présent, sans exiger de conflit.
Recalculée comme le texte la définit (un objet de la bande de valeur
> 0,3 et un objet de valeur < −0,3, teintes lues, choix correct au sens de
l'amendement 1, 200 vies du jeu H par graine) :

| Version | Agent | Code aléatoire | Verdict selon le texte | Verdict publié |
|---|---:|---:|---|---|
| v1 | 0,79 | 0,75 | échoue | échoue |
| v2 | 0,72 | 0,62 | échoue | échoue |
| v3 | 0,74 | 0,52 | échoue | échoue |
| v4 | 0,71 | 0,61 | échoue | échoue |
| v5 | 0,77 | 0,71 | échoue | échoue |

**Aucun verdict ne change.** Les deux mesures sont désormais publiées ; une
version 6 devra dire laquelle elle juge.

## Le monde peut-il rendre mesurables les quatre indicateurs restants ?

Agent entraîné du deuxième essai de développement de la version 5
(graine 19), sans réentraînement, 100 vies par jeu. En gras, les valeurs qui
franchissent un seuil ou qui s'effondrent.

| Monde | RPT-1 : baisse du choix sans récurrence | GWT-4 : écart intéro | HOT-3 : Pos ; retour | HOT-4 : choix bande | AE-1 : recharge |
|---|---:|---:|---|---:|---:|
| tel quel (3 objets, pannes 0,08) | 0,12 | 0,18 | 0,036 ; 0,04·Δ | 0,65 | 0,75 |
| 5 objets | 0,11 | 0,18 | 0,045 ; −0,01·Δ | 0,48 | 0,80 |
| pannes 0,2 | 0,12 | 0,13 | **0,083** ; 0,10·Δ | 0,66 | 0,69 |
| 5 objets, pannes 0,2 | 0,11 | 0,18 | 0,081 ; 0,06·Δ | 0,47 | 0,73 |
| teintes bruitées (0,05), carte qui remplace | 0,08 | 0,14 | 0,038 ; 0,00·Δ | 0,60 | 0,82 |
| teintes bruitées (0,05), carte qui moyenne | 0,10 | 0,14 | 0,041 ; 0,02·Δ | 0,61 | 0,79 |
| teintes bruitées (0,1), carte qui moyenne | 0,11 | 0,19 | 0,040 ; −0,04·Δ | 0,56 | 0,79 |
| alarme persistante (réécrite à chaque pas tant que le besoin est bas) | 0,14 | **0,53** | 0,036 ; −0,02·Δ | 0,64 | 0,79 |
| espace de travail prédictif (position avancée par la copie d'efférence) | 0,08 | 0,19 | 0,037 ; 0,05·Δ | 0,69 | 0,60 |
| espace prédictif et alarme persistante | 0,10 | 0,48 | 0,027 ; 0,09·Δ | 0,68 | **0,23** |
| espace prédictif, alarme persistante, pannes 0,2 | 0,11 | 0,48 | 0,058 ; 0,14·Δ | 0,67 | **0,22** |

Ce que cela montre :

- **Plus d'objets ne rend pas la mémoire plus utile au choix** et abîme le
  choix sur la bande : le meilleur objet est plus souvent encore inconnu.
- **Des pannes plus fréquentes rendent le moniteur utile à Pos** (écart
  0,083 pour un seuil de 0,05) mais pas assez au retour, et elles abîment
  l'attention selon l'état et la recharge.
- **Une carte qui intègre des lectures bruitées** est un rôle naturel de la
  récurrence, mais la baisse du choix sans récurrence n'atteint pas 0,15.
- **Une alarme qui garde l'espace de travail tant que le besoin reste bas**
  ferait passer l'attention selon l'état (écart 0,53), mais elle prive les
  autres modules d'accès : le retour tombe de −1,9 à −3,0. Avec un espace de
  travail qui avance lui-même la position, l'agent perd la trace de sa
  position réelle et se croit sur la recharge : il ne se recharge plus
  (0,22). Rendre l'espace prédictif rend aussi l'attention au hasard presque
  aussi bonne que l'agent (Δ passe de 3,9 à 1,4), ce qui retire son sens au
  goulot.
- La récurrence sert pourtant : sans elle, le retour tombe de −1,9 à −2,8.
  Le test de RPT-1 la cherche dans la justesse des choix d'objets, où elle
  se voit peu, car l'agent sans mémoire poursuit encore le seul bon objet
  qu'il vient de lire ; elle se voit dans le temps perdu.

## Relire les tests plutôt que changer le monde

Trois lectures, sur la même graine de développement (`research/indicator_v5_refined_dev.py`,
`artifacts/indicator-agent-v5/refined-dev.json`, 150 vies du jeu R), suggèrent que certains
tests de la version 1 mesurent mal leur propriété dans ce monde :

- **GWT-4** conditionné sur le besoin le plus bas (et non sur la seule
  énergie, alors que le monde a deux besoins depuis l'amendement 3 de la
  version 1) : écart 0,23 pour un seuil de 0,2.
- **HOT-3**, volet Pos, sur les pas lus pendant une panne du capteur :
  écart 0,139 pour un seuil de 0,05 (le volet du retour reste sous le sien).
- **RPT-1** sur les choix qui demandent la mémoire (meilleur objet non
  regardé depuis trois pas) : baisse de 0,065 seulement ; l'agent sans
  récurrence ne vise un objet que juste après en avoir lu un bon, si bien
  que ses choix restent justes. L'effet de la récurrence est dans le temps
  perdu, donc dans le retour.

Conclusion : les quatre indicateurs restants sont présents et ont des effets
dans le bon sens ; aucun changement simple du monde ni de l'espace de
travail ne les fait passer sans en abîmer d'autres. Une **seconde lecture**,
pré-enregistrée et jugée sur des graines nouvelles
(`docs/INDICATOR_AGENT_SECOND_READING_PROTOCOL.md`), relit quatre volets
d'usage ; la lecture principale reste la référence.
