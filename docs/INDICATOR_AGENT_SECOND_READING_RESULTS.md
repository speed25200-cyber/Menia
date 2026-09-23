# Seconde lecture des indicateurs, version 5 — résultats

Exécuté le 23 septembre 2026, 2 h 11 – 2 h 32 UTC, sur CPU. Code de la
version 5 inchangé (commit `20f7adf`), protocole
`docs/INDICATOR_AGENT_SECOND_READING_PROTOCOL.md` (commit `6508415`), calcul
de la seconde lecture écrit après lui (commit `f6b898b`). Graines nouvelles
137, 139 et 149 ; 27 000 vies de test. Artefacts dans
`artifacts/indicator-agent-v5-second` ; les deux lectures dans
`second-reading.json`, recalculées en CI ; audit complet dans
`verification.json`. Les analyses exploratoires citées par le protocole ont
été calculées avant lui dans le carnet de session et publiées après lui
(commit `b104339`), aux mêmes valeurs.

**Contrôle de validité : passé** (Δ = 4,12, 4,03 et 4,21).

## Verdicts

| | Propriété | Lecture principale (tests de la v1) | Seconde lecture |
|---|---|---|---|
| **RPT-1** | Récurrence | échoue : choix −0,10 sans récurrence | **passe** : retour −0,53 sans récurrence (seuil 0,41), sur chaque graine |
| **RPT-2** | Représentation intégrée | passe | passe |
| **GWT-1** | Modules spécialisés | passe | passe |
| **GWT-2** | Capacité limitée | passe | passe |
| **GWT-3** | Diffusion globale | passe | passe |
| **GWT-4** | Attention dépendante de l'état | échoue : énergie basse 0,49 contre haute 0,34 (écart 0,15) | **passe** : besoin le plus bas bas 0,49 contre haut 0,24 (écart 0,25) ; Corps 0,65 ; tour de rôle 1,04·Δ |
| **HOT-1** | Perception générative | passe | passe |
| **HOT-2** | Surveillance métacognitive | passe | passe |
| **HOT-3** | Croyances selon le moniteur | échoue | échoue : Pos −0,16 pendant les pannes, mais retour 0,01·Δ (seuil 0,1·Δ) |
| **HOT-4** | Espace de qualités | échoue | échoue : choix sur les conflits de bande 0,77 (seuil 0,8), code aléatoire 0,62 |
| **AST-1** | Schéma d'attention | passe | passe |
| **PP-1** | Codage prédictif | passe | passe |
| **AE-1** | Agence, buts concurrents | **échoue** : recharge 0,75 quand l'énergie est basse (seuil 0,8) ; objet 0,91, apprentissage 1,37, but unique +4,2 | échoue (même test) |
| **AE-2** | Incarnation | passe | passe |
| | **Total** | **9 sur 14** | **11 sur 14** |

## Prédictions

- **Lecture principale : réfutée.** Elle devait redonner les dix propriétés
  de la version 5 ; elle en donne neuf. AE-1 échoue sur ces graines.
- **Seconde lecture : confirmée.** GWT-4 et RPT-1 passent ; HOT-3 et HOT-4
  échouent.

## Ce que cela dit

- **AE-1 est à la limite.** Sur les six graines confirmatoires de la
  version 5 (113, 127, 131, puis 137, 139, 149), la recharge quand l'énergie
  est basse vaut 0,75, 0,87, 0,82, 0,80, 0,73 et 0,72 : 0,78 en moyenne,
  pour un seuil de 0,8. Les trois autres volets passent partout. L'agence à
  buts concurrents est donc démontrée sur un jeu de graines sur deux, pas de
  façon robuste.
- **Neuf propriétés sont robustes** sur les deux jeux de graines, en
  lecture principale : RPT-2, GWT-1, GWT-2, GWT-3, HOT-1, HOT-2, AST-1, PP-1
  et AE-2.
- **La récurrence et l'attention selon l'état sont présentes et utilisées**
  quand on les mesure là où elles agissent : sans récurrence, l'agent perd
  du temps et du retour ; son attention va à l'intéroception quand un
  besoin, quel qu'il soit, est bas.
- **HOT-3 et HOT-4 restent non démontrées.** Le moniteur rend la position
  plus juste pendant les pannes du capteur (0,16), mais ce gain ne se voit
  pas dans le retour. L'espace de qualités donne la bonne valeur aux teintes
  jamais vues (erreur 0,12 contre 1,22 pour un code aléatoire), mais le
  choix sur la bande reste sous 0,8, limité par les objets pas encore
  regardés.

La lecture principale reste la référence du programme ; la seconde dit ce
que ses tests manquaient. Aucune lecture ne mesure une expérience vécue.
