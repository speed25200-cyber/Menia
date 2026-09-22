# Protocole pré-enregistré — L'agent à indicateurs, version 3

Rédigé le 22 septembre 2026 à 22 h 05 UTC, après la publication de la
version 2 (`docs/INDICATOR_AGENT_V2_RESULTS.md`), **avant l'écriture du code
de la version 3 et avant toute exécution**. Mêmes règles que les versions 1
et 2.

## Question

La version 2 confirme le diagnostic pour GWT-1 : un arbitrage qui lit les
besoins fait apparaître la spécialisation des modules. Mais AE-1 échoue,
et les journaux l'attribuent à l'**instabilité de l'apprentissage des
valeurs** : réajustées à chaque tour sur les seules vies du tour, elles
oscillent, et la politique finale dépend du dernier tour (sur une graine,
recharge 0,32 quand l'énergie est basse). **Un apprentissage stable des
valeurs, sur toute l'expérience accumulée et avec une représentation des
besoins par cases, donne-t-il l'agence à buts concurrents (AE-1) ?**

## Ce qui change, et rien d'autre

- **Valeurs des buts** : Q(recharge), Q(objet), Q(rester), chacune linéaire
  sur une représentation **par cases** des besoins (énergie et satiété en
  quatre intervalles chacune, [0 ; 0,25), [0,25 ; 0,5), [0,5 ; 0,75),
  [0,75 ; 1], soit 16 cases indicatrices) et sur les distances et la valeur
  de l'objet comme en version 2. **Ajustées à chaque tour sur toutes les
  expériences de tous les tours précédents** (mémoire de rejeu), même
  régression ridge, mêmes retours escomptés (16 pas, γ = 0,9), même
  exploration ε = 0,2 pendant l'apprentissage, choix glouton ensuite.
- Monde (trois objets), modules, moniteur, schéma, code de teinte,
  planificateur, attention par pertinence avec urgence des besoins,
  engagement du but : **identiques à la version 2**.

## Évaluation et prédictions fixées

Mêmes jeux, mêmes quinze variantes, mêmes mesures, **mêmes quatorze
critères et mêmes seuils** que les versions 1 et 2 (AE-1 : retour moyen des
3 derniers tours − 3 premiers ≥ 0,2·Δ). Contrôle de validité : Δ > 0 sur
chaque graine.

**Prédiction principale** : AE-1 passe. **Prédiction de contrôle** : les
sept propriétés qui passaient en version 2 passent encore. **Critère
global** : les quatorze propriétés passent.

Graines d'agent : **89, 97, 101** ; graine de développement **11**. Au plus
trois essais de développement, déclarés ; après eux, plus aucune
modification.

## Premier essai de développement (graine 11), déclaré

Le 22 septembre à 22 h 20 UTC, code du commit `fddc79b` : 6 propriétés sur
14 (RPT-2, GWT-2, HOT-1, HOT-2, AST-1, PP-1). L'apprentissage des valeurs
est maintenant régulier (retour −1,97 au premier tour, autour de −1,6 aux
derniers) et l'agent obtient le meilleur retour des trois versions (−1,22).
AE-1 reste sous ses seuils : recharge 0,68 quand l'énergie est basse (0,78
quand l'énergie est aussi le besoin le plus bas), objet 0,53 quand
l'énergie est haute et qu'un bon objet est connu ; apprentissage 0,29
(seuil 0,73). GWT-1 échoue sur la dissociation (lésion de la vision, +0,59
malaise). **Aucune modification n'est faite** ; l'exécution confirmatoire
sur les graines 89, 97 et 101 est lancée avec ce code.

## Exécution

`python -m research.indicator_experiment --version 3`, artefacts dans
`artifacts/indicator-agent-v3`, audit branché en CI, résultats dans
`docs/INDICATOR_AGENT_V3_RESULTS.md`.
