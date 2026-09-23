# L'agent à indicateurs, version 4 — résultats

Exécuté le 23 septembre 2026 sur CPU, protocole
`docs/INDICATOR_AGENT_V4_PROTOCOL.md` : deux essais de développement déclarés
sur la graine 13, trois amendements après le premier, aucune modification
après le second. Code du commit `72df11f`, lancé au commit `d83240a`.
Graines 103, 107 et 109 ; mêmes budgets que les versions 2 et 3 ; 27 000
vies de test. Artefacts dans `artifacts/indicator-agent-v4`, audit complet
dans `verification.json`.

**Contrôle de validité : passé** (Δ = 2,77, 2,93 et 3,09).

## Verdict

| | Propriété | v4, trois graines réunies | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|---|
| **RPT-1** | Récurrence | mémoire 0,92 ; choix −0,07 sans récurrence | échoue | échoue | échoue | **échoue** |
| **RPT-2** | Représentation intégrée | conflits 0,90 ; sac de traits 0,00 | passe | passe | passe | **passe** |
| **GWT-1** | Modules spécialisés | lésion Intéro +4,2 malaises d'énergie, choix +0,04 ; lésion Vision choix 0,28, malaises −0,06 | échoue | passe | échoue | **passe** |
| **GWT-2** | Capacité limitée | **coût du goulot −0,19** (l'agent sans goulot fait moins bien) ; sélection 3,12 | passe | passe | passe | **échoue** |
| **GWT-3** | Diffusion globale | pannes −0,30 ; AUROC −0,045 (seuil 0,05) | échoue | échoue | échoue | **échoue** |
| **GWT-4** | Attention dépendante de l'état | intéroception 0,56 basse contre 0,37 haute (écart 0,19, seuil 0,2) ; Corps 0,57 | échoue | échoue | échoue | **échoue** |
| **HOT-1** | Perception générative | 0,97 contre 0,13 | passe | passe | passe | **passe** |
| **HOT-2** | Surveillance métacognitive | AUROC 0,99 ; Brier 0,025 contre 0,138 | passe | passe | passe | **passe** |
| **HOT-3** | Croyances selon le moniteur | Pos −0,032 ; retour 0,08·Δ | échoue | échoue | échoue | **échoue** |
| **HOT-4** | Espace de qualités | Spearman 0,93 ; erreur 0,17 contre 0,77 ; choix sur la bande 0,63 | échoue | échoue | échoue | **échoue** |
| **AST-1** | Schéma d'attention | 0,96 ; liaison 4,9 % contre 25,9 % ; redirection 0,90 contre 0,09 | passe | passe | passe | **passe** |
| **PP-1** | Codage prédictif | AUROC 0,99 ; surprise × 8,1 | passe | passe | passe | **passe** |
| **AE-1** | Agence, buts concurrents | **recharge 0,23 quand l'énergie est basse**, objet 0,55 quand elle est haute ; apprentissage 0,11 | échoue | échoue | échoue | **échoue** |
| **AE-2** | Incarnation | corps figé 0,36·Δ ; croyance après le changement 0,62 | échoue | échoue | échoue | **échoue** |

**Critère global : non satisfait, 6 propriétés sur 14. Les trois
prédictions principales (AE-1, AE-2, GWT-4) sont réfutées ; parmi les
secondaires, GWT-1 passe et GWT-3 échoue de peu ; GWT-2, propriété de
contrôle, est perdue.**

## Ce que cela dit

- **Les alarmes font ce qu'on attend d'elles.** L'intéroception prend
  l'espace de travail quand l'énergie baisse (0,56 contre 0,37), sa lésion
  coûte 4,2 malaises d'énergie par vie, et la lésion de la vision ne change
  plus les malaises d'énergie : GWT-1 passe pour la deuxième fois en quatre
  versions. GWT-4 et GWT-3 échouent de peu (écarts 0,19 et 0,045 pour des
  seuils de 0,2 et 0,05).
- **La pulsion n'a pas appris l'arbitrage.** Les retours d'apprentissage ne
  s'améliorent pas (−2,31 aux trois premiers tours, −2,20 aux trois
  derniers, graines réunies). L'agent reste immobile dans 67 % des pas
  (graine 103, 60 vies) et se recharge moins qu'en version 3 (0,23 des pas
  où l'énergie est basse). Le diagnostic des essais de développement vaut
  ici : pendant qu'un but est poursuivi, les alarmes le remettent en jeu tous
  les deux pas, si bien que presque aucun trajet vers la recharge n'aboutit
  dans la valeur d'un but ; et une valeur linéaire des besoins et des
  distances ne peut pas dire que la distance compte surtout quand le besoin
  est bas.
- **GWT-2 est perdu.** Dans la variante sans goulot, la vision et
  l'intéroception écrivent à chaque pas, donc le but est remis en jeu à
  chaque pas, avec des valeurs apprises pour des décisions espacées ; cette
  variante fait moins bien que l'agent (−1,76 contre −1,57 au jeu R).

## Bilan des quatre versions

Cinq propriétés tiennent dans les quatre versions : RPT-2, HOT-1, HOT-2,
AST-1 et PP-1. GWT-2 passe dans trois versions sur quatre. **L'arbitrage
entre buts (AE-1) échoue avec les quatre méthodes d'apprentissage sans
modèle** : gradient de politique, Monte-Carlo par tours, rejeu, pulsion. La
suite est donc un arbitrage qui prévoit l'évolution des besoins avec un
modèle appris par l'expérience, pré-enregistré comme version 5.
