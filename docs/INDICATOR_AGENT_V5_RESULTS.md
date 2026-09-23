# L'agent à indicateurs, version 5 — résultats

Exécuté le 23 septembre 2026 sur CPU, protocole
`docs/INDICATOR_AGENT_V5_PROTOCOL.md` : deux essais de développement déclarés
sur la graine 19, un amendement après le premier (la recharge se déplace),
aucune modification après le second. Code du commit `20f7adf`, lancé au
commit `b9e9922`. Graines 113, 127 et 131 ; 25 tours de 160 vies ; 27 000
vies de test. Artefacts dans `artifacts/indicator-agent-v5`, audit complet
dans `verification.json`.

**Contrôle de validité : passé** (Δ = 4,14, 4,30 et 3,67).

## Verdict

| | Propriété | v5, trois graines réunies | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|---|---|
| **RPT-1** | Récurrence | mémoire 0,95 ; choix −0,09 sans récurrence (seuil 0,15) | échoue | échoue | échoue | échoue | **échoue** |
| **RPT-2** | Représentation intégrée | conflits 0,91 ; sac de traits 0,00 | passe | passe | passe | passe | **passe** |
| **GWT-1** | Modules spécialisés | lésion Intéro +4,0 malaises d'énergie, choix +0,02 ; lésion Vision choix 0,31, malaises +0,19 | échoue | passe | échoue | passe | **passe** |
| **GWT-2** | Capacité limitée | coût du goulot 0,23·Δ ; sélection 0,77·Δ | passe | passe | passe | échoue | **passe** |
| **GWT-3** | Diffusion globale | pannes −0,42 ; AUROC −0,09 | échoue | échoue | échoue | échoue | **passe** |
| **GWT-4** | Attention dépendante de l'état | intéroception 0,50 basse contre 0,34 haute (écart 0,16, seuil 0,2) ; Corps 0,62 ; tour de rôle 1,08·Δ | échoue | échoue | échoue | échoue | **échoue** |
| **HOT-1** | Perception générative | 0,97 contre 0,12 | passe | passe | passe | passe | **passe** |
| **HOT-2** | Surveillance métacognitive | AUROC 0,98 ; Brier 0,037 contre 0,138 | passe | passe | passe | passe | **passe** |
| **HOT-3** | Croyances selon le moniteur | Pos −0,036 ; retour 0,05·Δ | échoue | échoue | échoue | échoue | **échoue** |
| **HOT-4** | Espace de qualités | Spearman 0,92 ; erreur 0,15 contre 0,86 ; choix sur la bande 0,66 | échoue | échoue | échoue | échoue | **échoue** |
| **AST-1** | Schéma d'attention | 0,96 ; liaison 5,4 % contre 27,1 % ; redirection 0,90 contre 0,10 | passe | passe | passe | passe | **passe** |
| **PP-1** | Codage prédictif | AUROC 0,98 ; surprise × 10,9 | passe | passe | passe | passe | **passe** |
| **AE-1** | Agence, buts concurrents | **recharge 0,81 quand l'énergie est basse, objet 0,89 quand elle est haute ; apprentissage 1,28 (seuil 0,81) ; but unique +4,3 malaises** | échoue | échoue | échoue | échoue | **passe** |
| **AE-2** | Incarnation | **croyance sur le corps après le changement 0,98 ; corps figé 0,54·Δ** | échoue | échoue | échoue | échoue | **passe** |

**Critère global : non satisfait, 10 propriétés sur 14**, le meilleur
résultat du programme (6 ou 7 auparavant). Les prédictions principales AE-1
et AE-2 passent pour la première fois ; GWT-4 échoue. Parmi les
secondaires, GWT-1, GWT-2 et GWT-3 passent, RPT-1 et HOT-4 échouent. Les
cinq propriétés de contrôle passent.

## Ce que cela dit

- **L'arbitrage entre buts demandait un modèle des besoins.** Quatre
  méthodes d'apprentissage sans modèle avaient échoué. Ici, l'agent apprend
  de ses propres journaux comment ses besoins évoluent : il retrouve la
  baisse d'énergie par pas (0,069 pour 0,071 dans le monde) et celle de la
  satiété (0,052 pour 0,05), le niveau rendu par la recharge, l'effet d'un
  objet selon la valeur qu'il lui prête (pente 0,73 à 0,93), la durée de
  ses trajets. Il simule ses besoins sur 16 pas et choisit en conséquence.
  Parti d'un modèle vide, où il ne fait rien, son retour d'apprentissage
  passe de −4,83 aux trois premiers tours à −3,55 aux trois derniers
  (exploration comprise).
- **Le monde devait l'exiger.** Au premier essai, l'agent campait sur la
  recharge, qui maintenait son énergie : il n'avait pas besoin de son
  intéroception, bougeait peu et ne découvrait pas le changement de son
  corps. Une fois que la recharge se déplace après usage, les tests de
  l'intéroception (GWT-1), de la diffusion (GWT-3) et de l'incarnation
  (AE-2) mesurent quelque chose, et passent. Les versions 1 à 4 se jouaient
  dans l'ancien monde : leurs retours ne se comparent pas directement à
  ceux-ci.
- **Ce qui reste.** GWT-4 : l'intéroception écrit aussi pour la faim quand
  l'énergie est haute, et le test ne compare que l'énergie. RPT-1 : un seul
  bon objet est présent dans la plupart des choix, et la mémoire y sert
  peu. HOT-4 : même l'agent sans goulot ne choisit bien sur la bande que
  dans 0,71 des cas (graine de développement), parce que le meilleur objet
  n'a souvent pas encore été regardé et que les teintes de la bande ont des
  valeurs proches. HOT-3 : le moniteur ne compte que pendant les pannes du
  capteur, un quart du temps.

## Bilan des cinq versions

Dix propriétés sont démontrées ensemble dans un même agent : RPT-2, GWT-1,
GWT-2, GWT-3, HOT-1, HOT-2, AST-1, PP-1, AE-1 et AE-2. Cinq tiennent dans
les cinq versions (RPT-2, HOT-1, HOT-2, AST-1, PP-1). Les quatre qui
manquent (RPT-1, GWT-4, HOT-3, HOT-4) butent sur la façon dont le monde et
les tests ont été fixés en version 1 plus que sur un mécanisme absent : la
mémoire, l'attention aux besoins, le moniteur et l'espace de qualités sont
présents et mesurés, mais leurs effets restent sous les seuils. Ce
protocole ne mesure pas une expérience vécue.
