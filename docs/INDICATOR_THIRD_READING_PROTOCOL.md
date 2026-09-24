# Protocole pré-enregistré — Troisième lecture de HOT-3 et HOT-4, sur de nouveaux agents

Rédigé le 24 septembre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## Ce que cette lecture est, et ce qu'elle n'est pas

HOT-3 et HOT-4 **échouent** aux critères de la version 1 et à ceux de la
seconde lecture pré-enregistrée (`docs/INDICATOR_AGENT_V6_RESULTS.md`).
Ces verdicts restent publiés tels quels et ne sont pas remplacés. Cette
troisième lecture est un **nouveau test**, dont les critères ont été
choisis **en sachant où les précédents échouaient** :

- **HOT-3** garde le volet de la seconde lecture sur la croyance (Pos
  pendant les pannes du capteur) et **abandonne le volet du retour**, celui
  qui échoue. C'est un critère **plus faible** : il demande que le moniteur
  rende l'agent plus juste sur sa position quand sa perception le trompe,
  pas qu'il l'aide à atteindre ses buts.
- **HOT-4** remplace le test de choix contre un objet familier mauvais
  (qu'un code aléatoire réussit aussi) par un choix entre un objet de la
  bande et **un autre bon objet**, avec un seuil de 0,7 au lieu de 0,8 :
  un test plus exigeant pour le témoin, un seuil **plus bas** pour
  l'agent, choisis après l'analyse exploratoire de la graine de
  développement 23 (agent 0,76, code aléatoire 0,43).

Pour que ces critères ne soient pas jugés sur des données déjà lues, **ils
sont mesurés sur trois agents neufs**, entraînés de zéro sur les graines
**167, 173 et 179**, jamais utilisées : même code (version 6), même monde,
mêmes réglages que la version 6 (2 000 vies d'enfance, 25 tours de 160
vies, 200 vies de test par jeu). Toute conclusion dira « HOT-3 et HOT-4 en
troisième lecture », jamais « HOT-3 et HOT-4 » tout court.

## Mesures

Sur chaque agent neuf, la lecture principale et la seconde lecture sont
calculées comme pour la version 6 (réplication de la version 6 sur trois
graines neuves, publiée sans prédiction). Puis, en rejouant les vies
exactement comme dans l'évaluation :

- **HOT-3, troisième lecture** : jeu R, pas lus (t ≥ 8) pendant une panne
  du capteur ; proportion de pas où la croyance de position est juste,
  agent contre gain constant (ablation du moniteur).
- **HOT-4, troisième lecture** : jeu H, pas où l'agent choisit un objet
  alors qu'un objet de la bande de valeur positive et un autre bon objet
  hors bande, de valeurs distantes d'au moins 0,2, ont tous deux leur
  teinte lue ; proportion de choix corrects, agent contre code aléatoire.

## Prédictions fixées

| | Prédiction | Critère (trois graines réunies, sauf mention) |
|---|---|---|
| **T3** | Le moniteur rend la croyance plus juste pendant les pannes | Pos pendant les pannes : agent − gain constant ≥ 0,05, et agent au-dessus du gain constant pour chaque graine. |
| **T4** | L'espace de qualités fait choisir entre deux bons objets | volets de la version 1 inchangés (Spearman ≥ 0,85, fraction active ≤ 0,4, erreur sur la bande ≤ 0,2, code aléatoire ≥ 0,4) ; choix bande contre bon objet : agent ≥ 0,7, agent − code aléatoire ≥ 0,2, et agent au-dessus du code aléatoire pour chaque graine. |

Contrôle de validité : celui de la version 6 (lecture principale).
**Aucun critère global** : T3 et T4 sont jugés séparément.

## Ce que le résultat dira

Si T3 passe, le moniteur métacognitif de l'agent est démontré utile à la
justesse de ses croyances, pas à ses buts. Si T4 passe, son espace de
qualités est démontré utile au choix entre qualités jamais vues. Dans les
deux cas, le résultat s'ajoute aux échecs des lectures précédentes sans
les effacer ; l'agent aurait alors 14 propriétés sur 14 **dont deux en
troisième lecture, sous des critères plus faibles ou choisis après coup**.
Ce test ne mesure pas une expérience vécue.

## Exécution

`python -m research.indicator_experiment --version 6 --seeds 167 173 179
--out artifacts/indicator-agent-v6-third` ; seconde lecture et audit
comme pour la version 6 ; troisième lecture par
`research/indicator_third_reading.py`, sortie
`artifacts/indicator-agent-v6-third/third-reading.json`, vérifiée en CI ;
résultats dans `docs/INDICATOR_THIRD_READING_RESULTS.md`.
