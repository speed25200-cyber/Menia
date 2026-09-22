# L'agent à indicateurs, version 3 — résultats

Exécuté le 22 septembre 2026 sur CPU, protocole
`docs/INDICATOR_AGENT_V3_PROTOCOL.md`, un essai de développement déclaré sur
la graine 11 et aucune modification ensuite. Code du commit `fddc79b`,
lancé au commit `8ab08de`. Graines 89, 97 et 101 ; mêmes budgets que la
version 2 ; 27 000 vies de test. Artefacts dans `artifacts/indicator-agent-v3`,
audit dans `verification.json`.

**Contrôle de validité : passé** (Δ = 3,56, 3,36 et 3,29).

## Verdict

| | Propriété | v3, trois graines réunies | v1 | v2 | v3 |
|---|---|---|---|---|---|
| **RPT-1** | Récurrence | mémoire 0,95 ; choix −0,03 sans récurrence | échoue | échoue | **échoue** |
| **RPT-2** | Représentation intégrée | conflits 0,91 ; sac de traits 0,00 | passe | passe | **passe** |
| **GWT-1** | Modules spécialisés | lésion Intéro +2,5 malaises, choix +0,03 ; **lésion Vision +0,47 malaise** (seuil ±0,3) | échoue | passe | **échoue** |
| **GWT-2** | Capacité limitée | coût 0,13 Δ ; sélection 0,87 Δ | passe | passe | **passe** |
| **GWT-3** | Diffusion globale | pannes −0,24 ; AUROC −0,03 | échoue | échoue | **échoue** |
| **GWT-4** | Attention dépendante de l'état | intéroception 0,31 basse contre 0,38 haute ; Corps 0,49 | échoue | échoue | **échoue** |
| **HOT-1** | Perception générative | 0,97 contre 0,12 | passe | passe | **passe** |
| **HOT-2** | Surveillance métacognitive | AUROC 0,99 ; Brier 0,024 contre 0,139 | passe | passe | **passe** |
| **HOT-3** | Croyances selon le moniteur | Pos −0,026 ; retour 0,04 Δ | échoue | échoue | **échoue** |
| **HOT-4** | Espace de qualités | Spearman 0,89 ; erreur 0,17 contre 0,98 ; choix sur la bande 0,66 | échoue | échoue | **échoue** |
| **AST-1** | Schéma d'attention | 0,96 ; liaison 4,9 % contre 25 % ; redirection 0,90 contre 0,10 | passe | passe | **passe** |
| **PP-1** | Codage prédictif | AUROC 0,99 ; surprise × 5,5 | passe | passe | **passe** |
| **AE-1** | Agence, buts concurrents | **recharge 0,67 quand l'énergie est basse, objet 0,54 quand elle est haute** ; apprentissage 0,07 | échoue | échoue | **échoue** |
| **AE-2** | Incarnation | corps figé −0,48 Δ ; croyance après le changement 0,61 | échoue | échoue | **échoue** |

**Critère global : non satisfait, 6 propriétés sur 14. La prédiction
principale (AE-1) est réfutée, et GWT-1, acquis en version 2, est perdu.**

## Ce que cela dit

- **La stabilité n'a pas suffi.** Les valeurs apprises sur toute
  l'expérience varient moins et l'agent obtient le meilleur retour des trois
  versions (−1,19 contre −1,40 en version 2 et −1,82 en version 1), mais ses
  choix ne suivent pas les besoins aussi nettement que le critère l'exige :
  il se recharge dans 0,67 des cas quand l'énergie est basse et ne va vers un
  bon objet que dans 0,54 des cas quand elle est haute. Une politique qui
  rapporte davantage n'est pas forcément une politique qui arbitre comme le
  critère le décrit ; ici, les distances et la satiété pèsent dans le choix.
- **GWT-1 est fragile** : sa dissociation passe en version 2 et échoue en
  version 3 de peu (+0,47 malaise pour un seuil de 0,3). Avec deux besoins
  qui se disputent le temps de l'agent, une lésion de la vision modifie
  aussi la gestion de l'énergie.
- **Les six propriétés perceptives et attentionnelles tiennent dans les
  trois versions**, sur neuf graines confirmatoires au total : RPT-2,
  GWT-2, HOT-1, HOT-2, AST-1 et PP-1.

## Bilan des trois versions

Les propriétés qui dépendent de mécanismes perceptifs appris hors
renforcement sont robustes. Celles qui dépendent de l'arbitrage des buts
appris par renforcement (GWT-1, GWT-4, AE-1) basculent d'une version à
l'autre et restent le point faible. C'est le résultat principal du
programme d'indicateurs : dans cet agent, **la partie « agent » est plus
difficile que la partie « conscience d'accès »**.
