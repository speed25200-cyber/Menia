# Protocole pré-enregistré — L'agent à indicateurs, version 5

Rédigé le 23 septembre 2026, après la publication de la version 4
(`docs/INDICATOR_AGENT_V4_RESULTS.md`), **avant l'écriture du code de la
version 5 et avant toute exécution** ; l'heure est celle du commit. Mêmes
règles que les versions 1 à 4.

## Diagnostic des versions 1 à 4

1. **L'arbitrage entre buts échoue avec toute valeur apprise sans modèle.**
   Gradient de politique (v1), Monte-Carlo par tours (v2), rejeu (v3),
   pulsion (v4) : aucune de ces méthodes n'arbitre selon les besoins aussi
   nettement que le critère d'AE-1 l'exige. Les essais de la version 4 en
   donnent la raison : une valeur linéaire des besoins et des distances ne
   peut pas dire que la distance à la recharge compte surtout quand l'énergie
   est basse ; les buts sont remis en jeu avant d'aboutir ; les écarts entre
   buts se perdent dans le bruit des retours. Une règle fixe des besoins fait
   mieux sur l'objet (0,93) que toutes les valeurs apprises.
2. **Le choix des objets souffre de l'ordre d'accès.** Quand l'agent poursuit
   un objet qui n'est pas le meilleur alors que sa vision connaît le
   meilleur (19 % des pas à objet en version 3, graine 89), c'est le plus
   souvent que le contenu de la vision et celui de Pos auraient tous deux
   changé la décision, et que Pos a gagné l'accès (52 cas sur 98) : un
   contenu qui ne change que le trajet passe avant un contenu qui change le
   but. Sans goulot, le choix est juste dans 0,88 des pas à objet ; avec, dans
   0,69 à 0,74. C'est ce qui limite RPT-1 (le choix sans récurrence ne baisse
   que de 0,03 à 0,07) et HOT-4 (choix sur la bande 0,63 à 0,66).
3. **Un but tenu jusqu'à l'arrivée devient périmé.** L'engagement du but,
   introduit en version 1 parce que le but y était tiré au sort à chaque
   pas, garde un objet ou la recharge alors que l'espace de travail sait déjà
   mieux.

## Question

**Un arbitrage allostatique — l'agent apprend de son expérience un modèle de
l'évolution de ses besoins, et choisit à chaque pas le but dont il prévoit le
meilleur effet (régulation par anticipation, Sterling, 2012) — et un accès à
l'espace de travail qui sert d'abord le contenu qui change le but donnent-ils
l'agence à buts concurrents (AE-1), l'incarnation (AE-2), l'attention
dépendante de l'état (GWT-4), et un choix des objets assez juste pour que la
récurrence (RPT-1) et l'espace de qualités (HOT-4) servent ?**

## Ce qui change, et rien d'autre

1. **Modèle des besoins, appris.** De ses propres journaux (ses estimations
   d'énergie et de satiété, sa croyance de position, ses récompenses), l'agent
   estime :
   - la baisse par pas de l'énergie et de la satiété (moyenne des baisses de
     ses estimations d'un pas au suivant, quand elles ne montent pas et, pour
     l'énergie, quand il ne se croit pas sur la recharge) ;
   - le niveau d'énergie sur la case de recharge (moyenne de son estimation
     quand il s'y croit) ;
   - l'effet d'un objet atteint : à chaque arrivée sur la case de l'objet
     poursuivi, quand l'objet n'y est plus vu, la hausse de satiété (plus la
     baisse d'un pas) et la récompense du pas, chacune en régression linéaire
     sur la valeur qu'il prêtait à l'objet (arrivées où la satiété estimée
     plus cette valeur reste sous 0,95, pour la satiété) ;
   - le nombre de pas pour atteindre une cible, en régression linéaire sur sa
     distance attendue au premier pas de la poursuite (poursuites menées
     jusqu'à l'arrivée sans changer de but).

   Le modèle est réajusté après chaque tour sur toute l'expérience de la
   phase d'agence (mêmes 25 tours de 160 vies). Il est **vide au départ** :
   l'agent ne prévoit alors aucun effet.
2. **Arbitrage par anticipation, à chaque pas.** Les besoins lus dans
   l'espace de travail sont d'abord ramenés au pas présent (âge du contenu
   d'Intéro × baisse par pas). Pour chaque premier but possible (recharge,
   sauf s'il se croit dessus ; meilleur objet connu de valeur positive ;
   rester) suivi de chaque second but (un autre but, ou rester), l'agent
   simule avec son modèle ses besoins et ses récompenses sur 16 pas : trajets
   de la durée prévue, besoins qui baissent à chaque pas, énergie tenue au
   niveau de la recharge tant qu'il y est, satiété et récompense prévues à
   l'arrivée sur l'objet. Il compte la somme escomptée (γ = 0,9) des
   récompenses prévues moins la pulsion D = (1 − e)⁴ + (1 − f)⁴ de la
   version 4, et poursuit le premier but de la meilleure suite. À égalité, il
   reste. Le but est recalculé à chaque pas.
3. **Exploration pendant l'apprentissage.** À chaque pas sans but
   d'exploration en cours, avec une probabilité de 0,05, un but tiré au
   hasard parmi les buts possibles devient but d'exploration, tenu jusqu'à ce
   qu'il soit atteint ou que l'objet disparaisse. Rien de tel à l'évaluation.
4. **Accès au but d'abord.** Dans le contrôleur par pertinence de la
   version 2, un module dont le contenu changerait le but l'emporte sur un
   module qui ne changerait que l'action ; ensuite saillance et âge comme
   avant. La décision éprouvée est celle de l'anticipation. Les alarmes de la
   version 4 gardent leur priorité.

Monde (trois objets), modules, moniteur, schéma d'attention, code de teinte,
planificateur des trajets, alarmes et pulsion convexe : **identiques à la
version 4**.

## Évaluation et prédictions fixées

Mêmes jeux, mêmes quinze variantes, mêmes mesures, **mêmes quatorze critères
et mêmes seuils** que les versions 1 à 4 (AE-1 : retour moyen des 3 derniers
tours − 3 premiers ≥ 0,2·Δ, retours d'apprentissage exploration comprise).
Pour la variante « but unique », la recharge n'est jamais un but possible.
Contrôle de validité : Δ > 0 sur chaque graine.

- **Prédictions principales** : AE-1, AE-2 et GWT-4 passent.
- **Prédictions secondaires** : GWT-1, GWT-2 et GWT-3 passent ; le choix des
  objets devient assez juste pour que RPT-1 et HOT-4 passent.
- **Contrôle** : RPT-2, HOT-1, HOT-2, AST-1 et PP-1 passent encore.
- **Rien n'est prédit** pour HOT-3.
- **Critère global** : les quatorze propriétés passent.

Graines d'agent : **113, 127, 131** ; graine de développement **19**. Au plus
trois essais de développement, déclarés ; après eux, plus aucune
modification.

## Ce que le résultat dira

Si AE-1 passe, l'arbitrage entre buts concurrents demandait un modèle des
besoins, pas seulement une valeur apprise : c'est la thèse de l'allostasie,
mise à l'épreuve dans un agent qui porte aussi les autres indicateurs. Si
RPT-1 et HOT-4 passent, leurs échecs précédents tenaient à l'ordre d'accès à
l'espace de travail et non à la mémoire ou au code de teinte. Dans tous les
cas, ce protocole ne mesure pas une expérience vécue.

## Premier essai de développement (graine 19), déclaré

Le 23 septembre, de 0 h 42 à 0 h 54 UTC, code du commit `ed2e43f` : 6
propriétés sur 14 (RPT-2, GWT-2, HOT-1, HOT-2, AST-1, PP-1). **L'arbitrage
allostatique apprend** : retour −6,87 au premier tour (modèle vide),
−2,66 au deuxième, −1,70 au dernier ; apprentissage 1,93 (seuil
0,2·Δ = 0,89) ; objet quand l'énergie est haute 0,84 (seuil 0,8) ;
recharge quand elle est basse 0,72 (seuil 0,8) ; 0,01 malaise d'énergie
par vie. Le modèle appris retrouve la baisse d'énergie par pas (0,066 pour
0,071 dans le monde) et celle de la satiété (0,052 pour 0,05).

Mais **l'agent campe sur la recharge** (70 % des pas, 60 vies du jeu R) :
la case maintient l'énergie à 1 tant qu'il y reste, et quand aucun bon
objet n'est connu, rien ne vaut mieux. Un agent qui ne quitte pas la
recharge n'a pas besoin de son intéroception pour gérer son énergie
(lésion d'Intéro : +0,0 malaise), bouge peu (33 % des pas) et ne découvre
pas que son corps a changé (croyance après le changement 0,57) ; les
lectures qui suivent un mouvement restent rares (écart d'AUROC sans
diffusion 0,043). Le choix des objets s'améliore (0,77 contre 0,69 en
version 4), mais la mémoire y sert peu : dans 65 % des pas à objet, un seul
bon objet est présent.

### Amendement 1, daté du 23 septembre 2026 (heure du commit)

Avant toute exécution sur les graines 113, 127 et 131. **Aucun seuil n'est
modifié.** Comme l'amendement 3 de la version 1, il touche le monde, parce
que le monde permettait une stratégie qui rend trois tests muets.

1. **La recharge se déplace** : quand l'agent s'y recharge, la case de
   recharge part vers une case libre tirée au hasard (tirages préparés à
   l'avance, comme les apparitions d'objets, pour que toutes les variantes
   rencontrent les mêmes). L'agent voit toujours où elle est. Comme les
   objets, la recharge devient une ressource qu'il faut rejoindre. Le modèle
   des besoins apprend en conséquence le niveau d'énergie à l'arrivée sur la
   case de recharge du pas précédent, et la variation d'énergie quand
   l'agent reste sur une case de recharge qui ne part pas, s'il l'observe
   jamais ; sans observation, il ne prévoit aucun maintien.

Le monde garde trois objets ; les prédictions sur RPT-1 et HOT-4 restent
écrites et seront jugées telles quelles. Le deuxième essai de
développement sur la graine 19 suit ; il sera déclaré.

## Exécution

`python -m research.indicator_experiment --version 5`, artefacts dans
`artifacts/indicator-agent-v5`, audit `research/audit_indicator_agent.py`,
branché en CI. Résultats dans `docs/INDICATOR_AGENT_V5_RESULTS.md`.
