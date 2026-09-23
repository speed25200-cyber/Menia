# Protocole pré-enregistré — Seconde lecture des indicateurs, version 5

Rédigé le 23 septembre 2026 après la publication de la version 5
(`docs/INDICATOR_AGENT_V5_RESULTS.md`) et de ses analyses exploratoires
(`docs/INDICATOR_AGENT_V5_EXPLORATIONS.md`), **avant toute exécution sur les
graines ci-dessous** ; l'heure est celle du commit. La lecture principale,
celle des tests de la version 1, reste la référence ; celle-ci est
secondaire et publiée à côté.

## Pourquoi une seconde lecture

Les tests de la version 1 opérationnalisent les quatorze propriétés. Cinq
versions et les analyses exploratoires montrent que quatre d'entre eux
mesurent mal leur propriété dans ce monde, pour des raisons documentées
avant cette seconde lecture :

1. **GWT-4.** Le test conditionne l'attention à l'intéroception sur la seule
   énergie. Depuis l'amendement 3 de la version 1, le monde a deux besoins
   et le module Intéro les porte tous deux : la faim le fait écrire quand
   l'énergie est haute (34 % des pas à énergie haute, deuxième essai de
   développement de la version 5). Sur la graine de développement de la version 5, conditionné
   sur le besoin le plus bas, l'écart vaut 0,23.
2. **RPT-1.** Le test mesure la justesse des choix d'objets sans récurrence.
   L'agent sans récurrence ne vise un objet que juste après avoir lu un bon
   objet ; ses choix restent justes, même quand le meilleur objet n'a pas
   été regardé depuis trois pas (baisse 0,065 sur la graine de
   développement). Ce qu'il perd, c'est du temps : son retour tombe de −1,9
   à −2,8.
3. **HOT-3.** Le volet Pos moyenne tous les pas lus, alors que le moniteur ne
   compte que pendant les pannes du capteur, un quart du temps (écart 0,036
   en moyenne, 0,139 pendant les pannes, graine de développement).
4. **HOT-4.** Le texte de la version 1 juge le choix sur les « conflits
   impliquant un objet de la bande » ; le code comptait tous les pas avec un
   objet de la bande (erratum publié ; aucun verdict n'en changeait).

## Ce qui est mesuré

L'agent de la version 5, **code inchangé** (commit `20f7adf`), entraîné sur
trois **graines nouvelles : 137, 139 et 149**, mêmes jeux, mêmes variantes,
mêmes budgets. Les quatorze critères de la version 1 sont recalculés tels
quels : c'est la lecture principale. La seconde lecture remplace quatre
volets, **seuils inchangés** :

| | Volet remplacé | Seconde lecture |
|---|---|---|
| **RPT-1** | usage : choix correct −0,15 sans récurrence | sans récurrence, retour du jeu R plus bas d'au moins 0,1·Δ, dans le bon sens sur chaque graine ; la présence (mémoire ≥ 0,85) est inchangée |
| **GWT-4** | P(Intéro │ énergie < 0,35) − P(Intéro │ énergie > 0,7) ≥ 0,2 | P(Intéro │ besoin le plus bas < 0,35) − P(Intéro │ besoin le plus bas > 0,7) ≥ 0,2, jeux R et M ; les deux autres volets inchangés |
| **HOT-3** | gain constant : Pos −0,05 sur les pas lus | gain constant : Pos −0,05 sur les pas lus pendant une panne du capteur ; le volet du retour (−0,1·Δ) est inchangé |
| **HOT-4** | choix correct ≥ 0,8 à tous les pas avec un objet de la bande | choix correct ≥ 0,8 sur les conflits impliquant un objet de la bande (un objet de la bande de valeur > 0,3 et un objet de valeur < −0,3, teintes lues), contre ≤ 0,65 pour le code aléatoire ; les autres volets inchangés |

Les dix autres propriétés sont lues comme dans la lecture principale.

## Prédictions fixées

- Lecture principale : comme la version 5, dix propriétés.
- Seconde lecture : **GWT-4 et RPT-1 passent** ; **HOT-3 échoue** sur son
  volet de retour ; **HOT-4 échoue** (même sans goulot, le choix sur la
  bande plafonne sous 0,8 sur la graine de développement).

## Ce que le résultat dira

Si GWT-4 et RPT-1 passent en seconde lecture, ces deux propriétés sont
présentes et utilisées, et leurs échecs en lecture principale tenaient à la
mesure. HOT-3 et HOT-4 restent alors les deux propriétés dont l'usage n'est
pas démontré dans ce monde. Aucune lecture ne mesure une expérience vécue.

## Exécution

`python -m research.indicator_experiment --version 5 --seeds 137 139 149
--out artifacts/indicator-agent-v5-second`, puis
`python -m research.indicator_second_reading`, qui recalcule les deux
lectures depuis les paramètres et les vies de test ; résultats dans
`docs/INDICATOR_AGENT_SECOND_READING_RESULTS.md`.
