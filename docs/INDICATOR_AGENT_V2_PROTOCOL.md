# Protocole pré-enregistré — L'agent à indicateurs, version 2

Rédigé le 22 septembre 2026 à 21 h 35 UTC, après la publication de la
version 1 (`docs/INDICATOR_AGENT_RESULTS.md`), **avant l'écriture du code de
la version 2 et avant toute exécution**. Mêmes règles : seuils fixés, tout
amendement daté avant l'exécution confirmatoire, toute lecture de
développement déclarée, un échec est un résultat.

## Question

La version 1 démontre six propriétés sur quatorze. Les journaux localisent
la cause principale des échecs dans la couche d'arbitrage apprise par
REINFORCE : le contrôleur d'attention n'écrit pas l'intéroception quand elle
compte, et le choix du but ne dépend pas des besoins, l'agent gérant son
énergie par routine. **Si l'on remplace cette seule couche par une
attention guidée par la pertinence pour la décision et par un arbitrage des
besoins appris par valeur, les propriétés qui échouaient passent-elles, sans
perdre celles qui passaient ?**

## Ce qui change, et rien d'autre

- **Monde** : l'Atelier des sens amendé de la version 1, avec **trois
  objets** au lieu de deux, pour que l'agent ait plus souvent un objet à
  atteindre. Tout le reste est identique.
- **Contrôleur d'attention** : pour chaque module, l'agent calcule la
  décision (but et action) qu'il prendrait si l'espace de travail recevait
  le contenu actuel de ce module ; un module dont le contenu **changerait la
  décision** l'emporte. Entre modules qui changent la décision, ou quand
  aucun ne la change, l'emporte la plus grande somme saillance + âge / 8
  (mêmes définitions qu'en version 1). Rien n'est appris dans le contrôleur.
- **Arbitrage des besoins** : trois buts, recharge, meilleur objet de valeur
  positive connue, rester. Deux fonctions de valeur linéaires, Q(recharge)
  et Q(objet), sur les besoins et les distances lus dans l'espace de travail
  (énergie, satiété, leurs carrés manquants, distances à la recharge et à
  l'objet, valeur de l'objet), apprises par **Monte-Carlo ajusté** : 25
  tours de 160 vies, retours escomptés sur 16 pas (γ = 0,9), exploration ε
  = 0,2 pendant l'entraînement, choix glouton ensuite ; rester vaut 0.
  L'engagement du but de la version 1 est conservé.
- Modules, moniteur, schéma d'attention, code de teinte, planificateur :
  **identiques** à la version 1.

## Évaluation et prédictions fixées

Mêmes jeux R, M et H (graines 930001 à 930003, 200 vies chacun), mêmes
quinze variantes, mêmes mesures, **mêmes quatorze critères et mêmes seuils
que la version 1**. Seule adaptation, imposée par l'apprentissage par tours :
dans AE-1, « retour moyen des 30 dernières mises à jour − 30 premières »
devient « retour moyen des 3 derniers tours − 3 premiers », toujours ≥ 0,2·Δ.
Pour la variante « but unique », Q(recharge) est ignorée. Contrôle de
validité inchangé : Δ > 0 sur chaque graine.

**Prédiction principale** : GWT-1, GWT-4 et AE-1, qui échouaient par
l'arbitrage, passent. **Critère global** : les quatorze propriétés passent.

Graines d'agent : **53, 67, 79** pour l'exécution confirmatoire ; graine de
développement **7**, jamais utilisée pour les verdicts. Au plus trois essais
de développement, déclarés ; après eux, plus aucune modification.

## Ce que le résultat dira

Si GWT-1, GWT-4 et AE-1 passent, le diagnostic de la version 1 est confirmé :
l'architecture réalisait ces propriétés, et c'est l'apprentissage de
l'arbitrage qui les empêchait de s'exprimer. Si elles échouent encore, la
cause est ailleurs. Dans tous les cas, ce protocole ne mesure pas une
expérience vécue.

## Premier essai de développement (graine 7), déclaré, et amendement 1

Le 22 septembre à 21 h 35 UTC : 4 propriétés sur 14 (HOT-1, HOT-2, AST-1,
PP-1) et **contrôle de validité échoué** (Δ = −0,03). Les valeurs apprises
ont dégradé l'agent au fil des tours (retour −1,8 au premier tour, −4,2 au
dernier) : il ne se recharge presque plus (recharge choisie 0,002 quand
l'énergie est basse, 4,2 malaises d'énergie par vie). Cause : le protocole
fixait la valeur de rester à 0, alors que les retours escomptés sont presque
toujours négatifs à cause de la faim ; tout but au retour négatif perdait
donc contre l'immobilité. L'attention par pertinence écrit déjà davantage
l'intéroception quand l'énergie est basse (0,43 contre 0,27).

**Amendement 1, daté du 22 septembre 2026, 21 h 40 UTC, avant toute exécution
sur les graines 53, 67 et 79** : la valeur de rester est apprise comme les
deux autres, Q(rester) linéaire sur les mêmes entrées, ajustée par le même
Monte-Carlo. Aucun seuil n'est modifié.

## Exécution

`python -m research.indicator_experiment --version 2`, artefacts dans
`artifacts/indicator-agent-v2`, audit `research/audit_indicator_agent.py`
avec la même option, branché en CI. Résultats dans
`docs/INDICATOR_AGENT_V2_RESULTS.md`.
