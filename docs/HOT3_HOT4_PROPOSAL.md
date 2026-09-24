# Proposition — Redéfinir les tests de HOT-3 et HOT-4

**Statut, 24 septembre 2026 (nuit) : remplacée.** Sans réponse du
propriétaire, les deux tests ont été pré-enregistrés comme une
**troisième lecture**, mesurée sur trois agents neufs (graines 167, 173,
179) plutôt que sur les graines 151, 157 et 163 déjà lues
(`docs/INDICATOR_THIRD_READING_PROTOCOL.md`) ; les verdicts précédents
restent publiés. Résultats : `docs/INDICATOR_THIRD_READING_RESULTS.md`.
Le texte ci-dessous est celui de la proposition.

**Statut d'origine : proposition, en attente de l'accord du
propriétaire.** Rien n'est exécuté ni pré-enregistré. Les règles du programme interdisent de
changer un test après en avoir lu les résultats sans cet accord : ce
document sert à le demander, en disant exactement ce qui changerait.

## Où en sont les deux propriétés

L'agent à indicateurs (version 6, `docs/INDICATOR_AGENT_V6_RESULTS.md`)
démontre 12 propriétés sur 14 en seconde lecture. Pour les deux qui
manquent, le mécanisme est présent et agit ; c'est le **test d'usage**
fixé en version 1 qui ne peut pas le montrer dans ce monde :

- **HOT-3 (moniteur métacognitif)** : sans moniteur, les croyances sur la
  position sont plus fausses (−0,041 ; seuil 0,05) mais le retour à la
  base ne perd que 0,055·Δ (seuil 0,1·Δ). Des pannes plus fréquentes ou
  bloquées rendent le moniteur plus utile à la croyance (0,061 avec des
  pannes bloquées), jamais assez au retour
  (`docs/INDICATOR_AGENT_V6_EXPLORATIONS.md`).
- **HOT-4 (espace de qualités)** : l'espace de qualités généralise aux
  teintes jamais vues (Spearman 0,88), mais le test de choix l'oppose à un
  objet familier nettement mauvais, qu'un code tiré au hasard écarte aussi
  (0,71 contre 0,80). Contre un **autre bon objet**, le test sépare
  nettement l'agent du code aléatoire (0,76 contre 0,43, graine de
  développement), mais l'agent reste sous 0,8, limité par l'espace de
  travail (0,82 sans goulot).

## Ce qui est proposé

### HOT-4 — recommandé

Remplacer le test de choix « objet de la bande contre objet familier
mauvais » par **« objet de la bande contre un autre bon objet (valeurs
distantes d'au moins 0,2) »**, avec le critère : **agent ≥ 0,7 et agent −
code aléatoire ≥ 0,2**. Mesuré sur les agents des graines confirmatoires
de la version 6 (151, 157, 163), pour qui ce test n'a jamais été calculé,
200 vies du jeu H chacun. Coût : quelques heures de calcul local, aucun
build.

*Limite honnête* : le nouveau test et son seuil sont inspirés d'une
analyse sur la graine de développement 23 ; seul le résultat sur les
graines confirmatoires compterait.

### HOT-3 — deux options

- **Option A (recommandée)** : juger le moniteur sur ce qu'il fait
  directement, **la justesse des croyances pendant les pannes**, et non
  plus sur le retour. Critère : sans moniteur, l'erreur de position sur
  les pas où le capteur est en panne augmente d'au moins 0,1 (valeur de la
  version 6 sur la graine 23 : 0,17). Mesuré sur les graines 151, 157,
  163, monde inchangé.
- **Option B** : garder le critère de retour, mais dans un monde à pannes
  bloquées (une illusion persistante), version 7 entraînée de zéro sur de
  nouvelles graines (167, 173, 179). Plus coûteux, et les explorations
  suggèrent qu'il échouerait encore au retour.

*Limite honnête de l'option A* : elle abandonne l'exigence que le moniteur
serve au but de l'agent (le retour), pour ne garder que son effet sur la
croyance. C'est un critère plus faible ; il faudrait le dire ainsi.

**Correction du 24 septembre 2026 (soir).** Telle qu'écrite, l'option A
n'est pas recevable : l'erreur de position pendant les pannes a **déjà été
lue** sur les graines 151, 157 et 163 (−0,17 sans moniteur,
`docs/INDICATOR_AGENT_V6_RESULTS.md`), et fixer un seuil de 0,1 après
l'avoir vue serait choisir le seuil d'après le résultat. Pour être
recevable, l'option A devrait être jugée sur des agents entraînés de zéro
sur de **nouvelles graines** (167, 173, 179), avec le seuil fixé avant ;
coût : quelques heures de calcul local, aucun build. Le test proposé pour
HOT-4 n'a pas ce défaut : le choix « bande contre bon objet » n'a été
calculé que sur la graine de développement 23.

## Ce qu'il faut répondre

« **Oui pour HOT-4** », « **oui pour HOT-3 option A** » (sur nouvelles graines) ou « B », ou « non ».
Avec un oui, le protocole est écrit et commité avant tout calcul, les
résultats sont publiés tels quels, et l'audit indépendant les vérifie en
CI.
