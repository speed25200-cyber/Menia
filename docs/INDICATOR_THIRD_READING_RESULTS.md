# Troisième lecture de HOT-3 et HOT-4, sur trois agents neufs — résultats

Exécuté le 24 septembre 2026, 22 h 11 – 22 h 29 UTC, sur CPU ; code de
l'agent inchangé depuis la version 6 (`b22773b`) ; protocole
`docs/INDICATOR_THIRD_READING_PROTOCOL.md` (commit `ea443a7`, avant tout
code), calcul de la troisième lecture écrit avant son exécution (commit
`a9161b8`). Graines neuves 167, 173 et 179 ; 27 000 vies de test.
Artefacts dans `artifacts/indicator-agent-v6-third` : critères dans
`criteria.json`, seconde lecture dans `second-reading.json`, troisième
lecture dans `third-reading.json`, recalculées en CI ; audit complet dans
`verification.json` (200 vies rejouées par jeu et par graine, toutes les
variantes réévaluées).

**Contrôle de validité : passé** (Δ = 4,04, 3,90 et 4,22).

## Troisième lecture (prédictions fixées)

| | Prédiction | Mesure (trois graines réunies) | Par graine (167, 173, 179) | Verdict |
|---|---|---|---|---|
| **T3** | Le moniteur rend la croyance plus juste pendant les pannes | Pos pendant les pannes : agent 0,941, gain constant 0,766 ; écart **0,175** (seuil 0,05) | 0,950/0,784 ; 0,938/0,758 ; 0,936/0,757 | **passe** |
| **T4** | L'espace de qualités fait choisir entre deux bons objets | Spearman 0,93, fraction active 0,18, erreur sur la bande 0,15 contre 0,96 ; choix bande contre bon objet : agent **0,768**, code aléatoire 0,534, écart **0,234** (seuils 0,7 et 0,2) | 0,736/0,542 ; **0,789/0,764** ; 0,776/0,390 | **passe** |

**Les deux prédictions sont confirmées**, sur des agents jamais lus.

## Ce qu'il faut en retenir, sans l'enjoliver

- **T3 est net, mais c'est un critère plus faible.** Sur chaque graine, le
  moniteur rend la croyance de position juste dans 94 % des pas lus pendant
  une panne du capteur, contre 76 % sans lui. L'écart (0,175) est celui
  déjà vu sur les graines 151, 157 et 163 (0,172) : l'effet est stable.
  Mais le volet du retour **échoue encore** (0,169 contre un seuil de
  0,405) : le moniteur sert à la justesse des croyances de l'agent, pas à
  ses buts.
- **T4 passe, mais tout juste sur une graine.** Pour la graine 173,
  l'agent (0,789) ne dépasse le code aléatoire (0,764) que de 0,025 ; le
  critère pré-enregistré (« au-dessus pour chaque graine ») est satisfait,
  sans marge. L'écart réuni tient au code aléatoire, dont la réussite dépend
  de sa table tirée au hasard (0,39 à 0,76 selon la graine). Le seuil de
  l'agent (0,7) a été choisi après la graine de développement 23 (0,76) ;
  les nouveaux agents donnent 0,74 à 0,79.

## Réplication de la version 6 sur trois graines neuves

Publiée sans prédiction, comme le protocole le prévoit.

| | Propriété | Lecture principale | Seconde lecture | Version 6 (151, 157, 163), principale / seconde |
|---|---|---|---|---|
| **RPT-1** | Récurrence | échoue (choix −0,096, seuil 0,15) | passe (retour −0,79, seuil 0,41) | échoue / passe |
| **RPT-2** | Représentation intégrée | passe | passe | passe / passe |
| **GWT-1** | Modules spécialisés | passe | passe | passe / passe |
| **GWT-2** | Capacité limitée | passe | passe | passe / passe |
| **GWT-3** | Diffusion globale | passe | passe | passe / passe |
| **GWT-4** | Attention selon l'état | échoue (écart 0,16) | passe (écart 0,25) | échoue / passe |
| **HOT-1** | Perception générative | passe | passe | passe / passe |
| **HOT-2** | Surveillance métacognitive | passe | passe | passe / passe |
| **HOT-3** | Croyances selon le moniteur | échoue (Pos −0,042, seuil 0,05) | échoue (retour 0,169, seuil 0,405) | échoue / échoue |
| **HOT-4** | Espace de qualités | **passe** (conflits 0,80, code aléatoire 0,63, plafond 0,65) | **passe** | échoue / échoue |
| **AST-1** | Schéma d'attention | passe | passe | passe / passe |
| **PP-1** | Codage prédictif | passe | passe | passe / passe |
| **AE-1** | Agence, buts concurrents | passe (recharge 0,87) | passe | passe / passe |
| **AE-2** | Incarnation | passe | passe | passe / passe |
| | **Total** | **11 sur 14** | **13 sur 14** | 10 / 12 |

Les douze propriétés de la version 6 en seconde lecture se répliquent sur
trois graines neuves. **HOT-4 passe ici les critères de la version 1, mais
par le tirage du code aléatoire, pas par l'agent** : l'agent fait 0,80 sur
les conflits de bande, comme sur les graines 151, 157 et 163 ; le code
aléatoire fait 0,74, 0,74 et 0,49 selon la graine (0,63 en tout, sous le
plafond de 0,65), contre 0,71 en tout sur les graines précédentes. Ce
passage ne remplace pas l'échec publié de la version 6 : sur six graines,
le test de la version 1 dépend de la table aléatoire.

## Bilan

- En seconde lecture pré-enregistrée, **les douze mêmes propriétés sur
  quatorze** sont démontrées sur deux jeux de trois graines (151, 157,
  163, puis 167, 173, 179).
- **HOT-3 et HOT-4 en troisième lecture passent** sur trois agents neufs.
  L'agent a ainsi **quatorze propriétés sur quatorze, dont deux en
  troisième lecture, sous des critères plus faibles (HOT-3 : la croyance,
  pas le but) ou choisis après coup (HOT-4 : seuil fixé après la graine de
  développement, marge de 0,025 sur une graine)**. Les échecs des lectures
  précédentes restent publiés.
- Aucun de ces tests ne mesure une expérience vécue. Ils montrent que des
  mécanismes que les théories de la conscience jugent nécessaires sont
  présents et utiles dans un même agent ; ils ne disent pas qu'il est
  conscient.
