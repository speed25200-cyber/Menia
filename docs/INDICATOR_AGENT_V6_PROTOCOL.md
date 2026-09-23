# Protocole pré-enregistré — L'agent à indicateurs, version 6

Rédigé le 23 septembre 2026, après la seconde lecture de la version 5
(`docs/INDICATOR_AGENT_SECOND_READING_RESULTS.md`), **avant l'écriture du
code de la version 6 et avant toute exécution de la version 6** ; l'heure
est celle du commit. Mêmes règles que les versions 1 à 5.

## Diagnostic de la version 5

Analyses exploratoires, déclarées, sur l'agent du deuxième essai de
développement de la version 5 (graine 19, paramètres publiés), sans
réentraînement ; aucune graine confirmatoire n'a servi. Script
`research/indicator_v6_diagnostics.py`, sortie
`artifacts/indicator-agent-v5/diagnostics-v6.json`.

1. **AE-1 : l'agent attend une recharge qui ne vient pas.** Sur 200 vies du
   jeu R, quand l'énergie vraie est sous 0,3 sans but de recharge (37 pas
   sur 164), 20 pas viennent d'un espace de travail qui place l'agent sur la
   case de recharge alors qu'il est ailleurs : la recharge n'est alors pas un
   but possible, et l'agent reste sur place, parfois plusieurs pas, pendant
   que son énergie baisse. Les autres sont surtout un objet tout proche
   d'abord (12 pas), choix que son modèle justifie. Or une croyance a des
   conséquences prévisibles : s'il était sur la recharge, son modèle des
   besoins prévoit une recharge ; sur la case d'un objet, un repas.
2. **HOT-4 : les erreurs de choix sur la bande sont des erreurs de
   liaison.** Sur 200 vies du jeu H, dans les conflits qui impliquent un
   objet de la bande (définition du texte, erratum de la version 5), l'agent
   choisit le meilleur objet dans 0,75 des pas (code aléatoire 0,47). La
   tête de valeur généralise bien à la bande (erreur ≤ 0,13 sur toute la
   bande), mais quand l'agent poursuit un autre objet alors que la carte
   sous-estime le meilleur, c'est surtout que la teinte d'un autre objet a
   été écrite sur sa case (69 cas) ou celle d'un objet plus ancien (28 cas).
   6,8 % des lectures de teinte sont liées à une autre case que celle du
   projecteur, alors que le schéma d'attention donne la probabilité de son
   estimation.
3. **HOT-3 : le retour perdu est ailleurs.** L'agent perd surtout du retour
   en marchant sur des objets mauvais en chemin (4,1 par vie) ; le moniteur
   en épargne 0,5 par vie, soit un écart de retour de 0,05·Δ avec le gain
   constant, sous le seuil de 0,1·Δ. Un planificateur de trajets qui évite
   les objets mauvais aide l'agent sans goulot mais pas l'agent, et ne rend
   pas le moniteur plus utile : il n'est pas retenu.
4. **Prototypes**, mêmes vies : juger la position diffusée par ses
   conséquences porte la recharge quand l'énergie est basse de 0,77 à 0,85
   (les croyances fausses sur la recharge passent de 20 pas à 2) sans
   augmenter l'écart de retour avec le gain constant (0,16 contre 0,22) ;
   ne lier une teinte que si le schéma est sûr de lui (≥ 0,8) fait tomber
   les liaisons fausses de 6,8 % à 0,7 % des lectures et porte le choix sur
   les conflits de bande à 0,81 (code aléatoire 0,51).
5. **La position révisée ne doit pas passer avant le corps.** Si le contenu
   révisé de Pos prend l'espace de travail avec la priorité d'une alarme,
   il en prive le module Corps juste après le changement de corps : la part
   des vies du jeu M où Corps écrit aux pas 25 à 27 tombe de 0,68 à 0,60
   (0,57 avec la liaison contrôlée), près du seuil de GWT-4 (0,6). Sans
   priorité, la révision seule garde l'essentiel de l'effet (recharge 0,84)
   et laisse Corps intact (0,68). La liaison contrôlée coûte peu à Corps :
   sur 800 vies, 0,667 pour la version 5, 0,644 avec la liaison contrôlée,
   0,635 avec les deux changements retenus (recharge 0,84, retour −1,74
   contre −1,86).

## Question

**Un moniteur qui juge les croyances diffusées par leurs conséquences
prévues, et qui les fait réviser (HOT-3 : croyances mises à jour selon la
surveillance métacognitive), et un schéma d'attention qui contrôle la
liaison des perceptions (AST : le schéma sert à contrôler l'attention)
rendent-ils robustes l'agence à buts concurrents (AE-1) et utile l'espace de
qualités pour les choix (HOT-4), sans rien retirer aux autres propriétés ?**

## Ce qui change, et rien d'autre

1. **Surveillance de la position diffusée par ses conséquences.** Après
   chaque pas, l'agent retient la case où l'espace de travail le plaçait
   quand il a décidé. Si c'était la case de recharge et que son modèle des
   besoins connaît le niveau d'une recharge, il prévoit une recharge ; si un
   objet y était vu et que son modèle connaît l'effet d'un objet, il prévoit
   un repas. Au pas suivant, le moniteur compare : **recharge absente** si
   la lecture d'énergie reste sous le niveau appris moins 0,15 et que la
   recharge est toujours sur la même case ; **repas absent** si l'objet y
   est toujours vu, sans apparition nouvelle sur cette case, et que la
   récompense du pas est nulle. Une conséquence absente veut dire que la
   croyance était fausse : le module Pos multiplie sa croyance sur cette
   case par 0,02 et la renormalise. Le contenu révisé concourt pour
   l'espace de travail comme tout contenu : il l'emporte s'il change le
   but, sans priorité d'alarme (point 5 du diagnostic). Dans l'ablation
   « gain constant », le moniteur est remplacé par des constantes : la
   fiabilité des lectures par le taux moyen, et aucune croyance n'est jugée
   par ses conséquences.
2. **Le schéma d'attention contrôle la liaison.** Une lecture de teinte
   n'est liée à la case que le schéma désigne que si la probabilité de son
   estimation est au moins 0,8 ; sinon elle n'est liée à aucune case, et le
   projecteur revient à l'objet visé, encore inconnu, au pas suivant. Dans
   l'ablation « sans schéma », la teinte est liée à la case visée, comme
   avant.

Monde (trois objets, recharge qui se déplace), modules, moniteur des
lectures, code de teinte, modèle des besoins et son apprentissage,
arbitrage allostatique, alarmes, accès au but d'abord, enfance et phase
d'agence : **identiques à la version 5**.

## Évaluation et prédictions fixées

Mêmes jeux, mêmes quinze variantes, mêmes mesures, **mêmes quatorze critères
et mêmes seuils** que les versions 1 à 5 : c'est la lecture principale, qui
juge. Une seule précision, annoncée par l'erratum de la version 5 : le volet
de choix de HOT-4 est jugé **comme le texte de la version 1 le définit**
(conflits impliquant un objet de la bande de valeur > 0,3 et un objet de
valeur < −0,3, teintes lues) ; la mesure du code des versions 1 à 5 (tous
les pas avec un objet de la bande) est publiée à côté. La seconde lecture
pré-enregistrée (`docs/INDICATOR_AGENT_SECOND_READING_PROTOCOL.md`) est
aussi calculée, mêmes seuils. Contrôle de validité : Δ > 0 sur chaque graine.

- **Prédictions principales** : AE-1 et HOT-4 passent ; la recharge quand
  l'énergie est basse atteint 0,8 sur chacune des trois graines.
- **Prédictions secondaires** : les liaisons fausses de l'agent tombent
  sous 2 % des lectures ; AST-1 passe ; en seconde lecture, la part des
  vies où Corps écrit aux pas 25 à 27 reste au moins 0,6.
- **Contrôle** : RPT-2, GWT-1, GWT-2, GWT-3, HOT-1, HOT-2, PP-1 et AE-2
  passent encore.
- **Rien n'est prédit** pour HOT-3 (le prototype ne change pas l'écart de
  retour), ni pour RPT-1 et GWT-4 en lecture principale ; en seconde
  lecture, RPT-1 et GWT-4 devraient passer comme en version 5.
- **Critère global** : les quatorze propriétés passent en lecture
  principale.

Graines d'agent : **151, 157, 163** ; graine de développement **23**. Au
plus trois essais de développement, déclarés ; après eux, plus aucune
modification.

## Ce que le résultat dira

Si AE-1 passe sur chaque graine, l'agence à buts concurrents de la version 5
était limitée par des croyances fausses que l'agent pouvait détecter ; la
détection est une surveillance métacognitive de ses propres croyances. Si
HOT-4 passe, l'espace de qualités servait déjà, et ses choix étaient
masqués par des liaisons fausses que le schéma d'attention savait signaler.
Si HOT-3 échoue encore, le moniteur change les croyances sans que le retour
en dépende assez dans ce monde. Dans tous les cas, ce protocole ne mesure
pas une expérience vécue.

## Premier essai de développement (graine 23), déclaré

Le 23 septembre, de 3 h 34 à 3 h 50 UTC, code du commit `87bfce8` : **11
propriétés sur 14** (RPT-2, GWT-1, GWT-2, GWT-3, HOT-1, HOT-2, HOT-4,
AST-1, PP-1, AE-1, AE-2). AE-1 : recharge quand l'énergie est basse 0,88,
objet quand elle est haute 0,89, apprentissage 1,38, but unique +4,3
malaises. HOT-4 : choix sur les conflits de bande 0,81 contre 0,51 pour le
code aléatoire (0,67 et 0,45 sur tous les pas avec un objet de la bande) ;
Spearman du code 0,850, au seuil (le code de teinte, appris dans
l'enfance, n'a pas changé). Liaisons fausses : 0,8 % des lectures, 26 %
sans schéma ; 4,8 lectures de teinte refusées par vie sur 36. Le moniteur
signale 2,0 conséquences absentes par vie (1,7 repas, 0,3 recharge),
toutes à raison : sur 100 vies rejouées, l'agent n'était jamais sur la
case jugée. Restent sous leurs seuils : HOT-3 (écart de retour 0,092·Δ
pour 0,1·Δ ; Pos 0,038 pour 0,05), GWT-4 (écart d'attention 0,19 pour
0,2 ; Corps écrit aux pas 25 à 27 dans 0,595 des vies pour 0,6), RPT-1
(baisse du choix 0,12 pour 0,15).

**Aucune modification n'est faite** : les écarts restants sont au bord des
seuils, et aucun amendement fondé ne les vise sans viser le seuil. Les
deuxième et troisième essais ne sont pas utilisés ; l'exécution
confirmatoire sur les graines 151, 157 et 163 est lancée avec ce code.

## Exécution

`python -m research.indicator_experiment --version 6`, artefacts dans
`artifacts/indicator-agent-v6`, audit `research/audit_indicator_agent.py`
(complet, `verification.json`), branché en CI ; seconde lecture par
`research/indicator_second_reading.py`. Résultats dans
`docs/INDICATOR_AGENT_V6_RESULTS.md`.
