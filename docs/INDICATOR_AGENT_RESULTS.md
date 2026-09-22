# L'agent à indicateurs — résultats

Exécuté le 22 septembre 2026 sur CPU, protocole `docs/INDICATOR_AGENT_PROTOCOL.md`
avec ses onze amendements, tous datés avant l'exécution confirmatoire, et trois
essais de développement déclarés sur la graine 5. Code du commit `eb368e8`,
lancé au commit `a71c5f3`. Trois graines d'agent (17, 29, 43), chacune avec
2 000 vies d'enfance, 800 mises à jour de renforcement de 16 vies, puis
15 variantes × 3 jeux × 200 vies de test : **27 000 vies de test**, 7 à
8 minutes par graine. Artefacts dans `artifacts/indicator-agent`, audit dans
`verification.json`.

**Contrôle de validité : passé.** Δ = R(illimité) − R(hasard) vaut 2,28,
2,49 et 2,65 selon la graine.

## Verdict des prédictions fixées

| | Propriété | Mesures, trois graines réunies | Verdict |
|---|---|---|---|
| **RPT-1** | Récurrence des modules d'entrée | mémoire : signe de la valeur correct à 0,96 après ≥ 3 pas sans regarder ; sans récurrence : choix correct −0,03 seulement (seuil −0,15) | **échoue** |
| **RPT-2** | Représentation intégrée | conflits : bon objet mieux valué à 0,97 ; sac de traits 0,00 | **passe** |
| **GWT-1** | Modules spécialisés | Pos 0,98, Corps 1,00, Intéro 0,94 ; lésion Vision : choix 0,36, malaises d'énergie +0,25 ; **lésion Intéro : +0,00 malaise** (seuil +1,0) | **échoue** |
| **GWT-2** | Capacité limitée, sélection | un écrivain par pas ; coût du goulot 0,36 = 0,15 Δ ; gain de la sélection 2,11 = 0,85 Δ | **passe** |
| **GWT-3** | Diffusion globale | sans diffusion : Pos en panne −0,26 ; **AUROC du moniteur −0,03** (seuil −0,05) | **échoue** |
| **GWT-4** | Attention dépendante de l'état | Corps écrit après le changement dans 0,75 des vies ; tour de rôle −3,24 = 1,31 Δ ; **attention à l'intéroception 0,09 quand l'énergie est basse contre 0,10 quand elle est haute** (seuil +0,2) | **échoue** |
| **HOT-1** | Perception générative | position perçue pendant les pannes 0,98 ; sans prédiction 0,13 | **passe** |
| **HOT-2** | Surveillance métacognitive | AUROC 0,99 ; Brier 0,020 contre 0,139 au taux constant | **passe** |
| **HOT-3** | Croyances selon le moniteur | gain constant : Pos −0,029 (seuil −0,05), retour −0,11 = 0,05 Δ (seuil 0,1 Δ) | **échoue** |
| **HOT-4** | Espace de qualités | Spearman 0,91, 18 % d'unités actives ; erreur sur la bande 0,13 contre 0,75 au code aléatoire ; **choix sur la bande 0,65** (seuil 0,8), code aléatoire 0,58 | **échoue** |
| **AST-1** | Schéma d'attention | projecteur localisé à 0,98 ; erreurs de liaison 2,2 % contre 13,8 % sans schéma ; choix −0,11 sans schéma ; redirection 0,92 contre 0,17 | **passe** |
| **PP-1** | Codage prédictif | AUROC de la surprise 0,99 ; surprise après le changement 7,3 × celle d'avant ; sans prédiction Pos −0,15 | **passe** |
| **AE-1** | Agence, buts concurrents | apprentissage +0,78 = 0,32 Δ ; but unique +4,4 malaises d'énergie ; **recharge quand l'énergie est basse 0,56, objet quand elle est haute 0,56** (seuils 0,8) | **échoue** |
| **AE-2** | Incarnation | corps figé : retour du jeu M −0,99 = 0,40 Δ ; **croyance sur le corps après le changement 0,50** (seuil 0,85) | **échoue** |

**Critère global : non satisfait, 6 propriétés sur 14.** Le même compte que
le troisième essai de développement, avec les mêmes propriétés : le
résultat se reproduit sur trois graines nouvelles. Le sens de chaque écart
d'ablation qui passe est le même sur les trois graines.

## Ce que montrent les échecs

Les six propriétés qui passent sont des **mécanismes perceptifs et
attentionnels** : liaison des traits en objets, perception qui complète
l'absence de lecture, erreur de prédiction qui signale l'inattendu,
moniteur qui sépare une lecture fiable d'un glitch, espace de travail
limité dont la sélection apprise vaut 0,85 Δ, schéma d'attention qui sait
où est allé le projecteur et le ramène. Chacun est présent **et** utilisé :
le retirer coûte, sur chaque graine.

Les huit échecs se regroupent en trois causes, lues dans les journaux :

1. **L'arbitrage appris ne lit pas l'intéroception** (GWT-1, GWT-4,
   AE-1). La politique entraînée par renforcement gère l'énergie par une
   routine, des recharges fréquentes : 0,01 à 0,03 malaise d'énergie par
   vie, mais le choix du but ne dépend pas de l'énergie (0,56 et 0,56), le
   contrôleur n'écrit pas l'intéroception davantage quand elle baisse, et la
   lésion de l'intéroception ne change rien. La propriété d'agence à buts
   concurrents n'est pas réalisée par ce que l'agent a appris, bien que
   l'architecture la permette.
2. **Le goulot masque l'usage de certaines propriétés** (RPT-1, HOT-4). La
   mémoire visuelle est là (0,96) et le code de teinte généralise (erreur
   0,13 contre 0,75), mais le choix correct de l'agent plafonne à 0,68 parce
   que la politique lit une vision souvent périmée dans l'espace de travail
   (0,94 sans goulot). Les écarts de choix restent alors petits.
3. **Des effets réels mais sous les seuils** (GWT-3, HOT-3, AE-2). Sans
   diffusion, le moniteur perd peu (0,97 contre 1,00) : son histoire des
   surprises compense. Le gain métacognitif aide (Pos +0,03, retour
   +0,11) sans atteindre le seuil. La révision du corps fonctionne — en
   développement, exacte à 0,95 dès deux mouvements après le changement —
   mais dans 40 % des vies l'agent bouge au plus une fois après le
   changement, faute d'objet à atteindre, et sa croyance reste l'ancienne.

## Histoire du développement, déclarée

| Essai | Code | Propriétés | Ce qui a été corrigé ensuite |
|---|---|---:|---|
| 1 | amendements 1 et 2 | 5 | l'agent campait sur la recharge ; fiabilité constante ; captures rares ; code de teinte biaisé |
| 2 | amendements 3 à 7 | 6 | planificateur immobile tant que le corps est inconnu ; but retiré au sort à chaque pas |
| 3 | amendements 8 à 11 | 6 | aucune correction : code figé |
| **Confirmatoire** | idem | **6** | — |

Aucun seuil n'a été modifié. Les amendements ont changé le monde et
l'agent pour que les tests mesurent les propriétés et non des défauts de
conception ; ils ont aussi pu rendre certains tests plus faciles. C'est
pourquoi chaque essai est publié.

## Ce que ce résultat dit

Un seul agent réunit **par construction** les quatorze propriétés : modules
récurrents à codage prédictif, espace de travail limité à diffusion globale,
contrôleur d'attention appris, moniteur métacognitif, code clairsemé et
lisse, schéma d'attention, buts concurrents, modèle du corps. **Six sont
démontrées présentes et utilisées** par des tests pré-enregistrés et des
ablations sur des graines nouvelles. Huit ne le sont pas, et la cause
principale est localisée : la couche d'arbitrage apprise par renforcement.

Ce que cela ne dit pas : que l'agent est conscient. Les indicateurs sont des
propriétés que des théories jugent pertinentes ; les réunir augmenterait,
selon ces théories, la plausibilité d'une conscience, sans rien établir. La
théorie de l'information intégrée, pour qui le substrat physique compte,
n'est pas touchée par un résultat logiciel.

## Suite possible

Une version 2, pré-enregistrée sur des graines nouvelles, viserait la cause
localisée : un arbitrage des buts appris par valeur à partir des besoins
lus dans l'espace de travail, et une exploration du corps après une
surprise motrice. Elle serait jugée avec les mêmes seuils.
