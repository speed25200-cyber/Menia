# L'agent à indicateurs, version 2 — résultats

Exécuté le 22 septembre 2026 sur CPU, protocole
`docs/INDICATOR_AGENT_V2_PROTOCOL.md` avec ses deux amendements et trois
essais de développement déclarés sur la graine 7. Code du commit `ca4e78b`,
lancé au commit `4608dc0`. Graines 53, 67 et 79 ; 2 000 vies d'enfance,
25 tours de Monte-Carlo de 160 vies, puis 15 variantes × 3 jeux × 200 vies :
27 000 vies de test. Artefacts dans `artifacts/indicator-agent-v2`, audit
dans `verification.json`.

**Contrôle de validité : passé** (Δ = 2,67, 3,28 et 3,12).

## Verdict

| | Propriété | Mesures, trois graines réunies | v1 | v2 |
|---|---|---|---|---|
| **RPT-1** | Récurrence | mémoire 0,93 ; sans récurrence, choix −0,06 (seuil −0,15) | échoue | **échoue** |
| **RPT-2** | Représentation intégrée | conflits 0,94 ; sac de traits 0,00 | passe | **passe** |
| **GWT-1** | Modules spécialisés | Pos 0,97, Corps 0,97, Intéro 0,95 ; **lésion Intéro +3,7 malaises d'énergie, choix ±0,02** ; lésion Vision : choix 0,28, malaises +0,23 | échoue | **passe** |
| **GWT-2** | Capacité limitée | coût du goulot 0,10 Δ ; gain de la sélection 0,90 Δ | passe | **passe** |
| **GWT-3** | Diffusion globale | Pos en panne −0,21 ; AUROC du moniteur −0,02 (seuil −0,05) | échoue | **échoue** |
| **GWT-4** | Attention dépendante de l'état | tour de rôle −0,88 Δ ; Corps après le changement 0,51 (seuil 0,6) ; **intéroception 0,30 quand l'énergie est basse contre 0,39 quand elle est haute** | échoue | **échoue** |
| **HOT-1** | Perception générative | pannes 0,95 ; sans prédiction 0,13 | passe | **passe** |
| **HOT-2** | Surveillance métacognitive | AUROC 0,99 ; Brier 0,024 contre 0,138 | passe | **passe** |
| **HOT-3** | Croyances selon le moniteur | Pos −0,022 (seuil −0,05) ; retour 0,05 Δ | échoue | **échoue** |
| **HOT-4** | Espace de qualités | Spearman 0,92, 17 % actives ; erreur 0,12 contre 0,71 ; choix sur la bande 0,63 (seuil 0,8) | échoue | **échoue** |
| **AST-1** | Schéma d'attention | 0,96 ; liaison 4,5 % contre 24 % ; choix −0,19 ; redirection 0,91 contre 0,11 | passe | **passe** |
| **PP-1** | Codage prédictif | AUROC 0,99 ; surprise × 5,0 ; sans prédiction −0,14 | passe | **passe** |
| **AE-1** | Agence, buts concurrents | but unique +4,3 malaises ; **recharge quand l'énergie est basse 0,48, objet quand elle est haute 0,42** ; apprentissage −0,20 | échoue | **échoue** |
| **AE-2** | Incarnation | corps figé −0,58 Δ au jeu M ; croyance après le changement 0,61 (seuil 0,85) | échoue | **échoue** |

**Critère global : non satisfait, 7 propriétés sur 14**, une de plus que la
version 1. **La prédiction principale est confirmée pour GWT-1 et réfutée
pour GWT-4 et AE-1.** Aucune propriété qui passait en version 1 n'est perdue.

## Ce que cela dit du diagnostic de la version 1

- **L'intéroception est désormais utilisée.** Sa lésion ajoute 3,2 à 4,5
  malaises d'énergie par vie selon la graine, sans changer le choix des
  objets, et la lésion de la vision dégrade le choix sans changer l'énergie :
  la double dissociation de GWT-1 tient. Le diagnostic de la version 1 est
  donc juste sur ce point : l'architecture réalisait la spécialisation, et
  c'est l'arbitrage appris qui l'empêchait de s'exprimer.
- **L'arbitrage par valeur reste instable.** Les valeurs sont réajustées à
  chaque tour sur les seules vies du tour : le retour moyen oscille entre
  les tours sans tendance (apprentissage −0,20), et la politique finale
  dépend du dernier ajustement. Sur la graine 53, elle ne se recharge que
  0,32 fois quand l'énergie est basse ; sur 67 et 79, 0,71 et 0,83.
- **L'attention par pertinence suit la décision, pas le besoin.** Elle
  écrit l'intéroception surtout quand une recharge vient de la remplir, car
  c'est là qu'elle change la décision (quitter la recharge). Le critère de
  GWT-4 demandait l'inverse. La variante à but unique, qui ne peut pas se
  recharger, montre l'effet attendu (0,49 contre 0,23) : l'urgence attire
  l'attention quand rien d'autre ne la retient.

## Ce que ce résultat dit

Sept des quatorze propriétés indicatrices sont désormais démontrées
présentes et utilisées dans un même agent, sur des graines nouvelles, avec
des seuils fixés à l'avance : liaison des traits (RPT-2), modules spécialisés
(GWT-1), espace de travail limité à sélection (GWT-2), perception générative
(HOT-1), surveillance métacognitive (HOT-2), schéma d'attention (AST-1),
codage prédictif (PP-1). Les sept autres sont présentes par construction
mais leurs tests échouent, et les journaux disent pourquoi. Ce n'est pas une
preuve de conscience ; c'est la mesure la plus précise que ce dépôt puisse
donner de l'écart entre un agent qui a les propriétés et un agent qui s'en
sert.
