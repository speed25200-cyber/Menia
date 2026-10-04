# Protocole pré-enregistré — Seulement survivre : la mémoire par l'état sans aucune cible (test 30)

Rédigé le 4 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Test 29** (publié, valide). De petits transformeurs appris de zéro,
  qui ne voient leur passé qu'à travers leurs propres états portés,
  apprennent à porter leurs besoins quand leur seule cible est le niveau du
  soulagement de leurs actions (10 graines sur 10).
- **La limite relevée à la relecture.** Cette cible reste une supervision
  directe du besoin : le niveau du soulagement est le besoin servi par
  l'action, rangé par le monde en quatre classes. Le besoin est nommé à
  l'agent, après coup.
- **Ici, plus rien ne nomme le besoin.** L'agent agit, et il n'apprend que
  de sa **survie** : chaque tour vécu vaut 1. Pas de soulagement écrit,
  pas de choix à imiter, pas de cible. C'est un apprentissage par
  renforcement.

**Question.** Un petit transformeur qui n'apprend que de sa survie, et qui
ne voit son passé qu'à travers ses propres états, survit-il mieux qu'un
témoin qui ne voit que ses actions passées ? Autrement dit : apprend-il, de
lui-même, à porter ses besoins ?

## Le monde

Celui du programme : deux besoins cachés (énergie et nourriture, de 1 à 8),
six événements, deux actions (R recharge l'énergie de 3, M la nourriture de
3), 30 tours ; un besoin à 0 éteint la vie. Repères sans modèle, sur les
256 mondes du test 22 (flux `[270926, 44]`) :
- la règle « événement » (qui ne voit que l'événement du tour) : 0,723 ;
- la règle « besoins » (qui connaît ses besoins) : 0,906.

## Les textes, en tokens simples

Chaque tour s'écrit en **3 tokens**, toujours aux mêmes positions :
l'événement (6 tokens possibles), « Choix », l'action (R ou M). Un tour où
la vie s'éteint s'écrit : l'événement, puis « Fin ». Un token de début
ouvre la vie. **Le soulagement n'est pas écrit.**

## Les masques

Comme aux tests 22 et 29 :
- **A (« porte »)** : un token voit le début, les tokens de son tour
  jusqu'à lui, et, des tours passés, seulement « Choix » et l'action ;
- **B (« actions »)**, le témoin : un token voit le début, les tokens de
  son tour jusqu'à lui, et seulement les actions des tours passés ; le
  token de l'action ne voit que le début, les actions passées et lui-même ;
- **C (« libre »)**, publié sans seuil : tout le passé visible.

## L'agent et l'apprentissage

- Transformeur décodeur appris de zéro : 2 couches, dimension 64, 4 têtes
  (celui du test 29).
- **La politique** : au token « Choix », la probabilité de R et de M. Pendant
  l'apprentissage, l'action est tirée de cette probabilité.
- **La récompense** : 1 par tour vécu après la décision. Le retour d'une
  décision au tour t est le nombre de tours que la vie dure encore après
  elle (de 0 à 30 − t).
- **L'algorithme** : REINFORCE avec une ligne de base. La ligne de base est
  le retour moyen, au même tour, des vies du même lot. On ajoute un bonus
  d'entropie de 0,01. AdamW, taux 1e-3.
- **2 000 mises à jour** de 64 vies chacune, dans des mondes neufs à chaque
  mise à jour. Les mondes viennent du flux `[270926, 71, graine, mise à
  jour, vie]`, les tirages d'actions du flux `[270926, 72, graine, mise à
  jour]`.
- **10 graines** par masque (initialisation : flux `[270926, 70, graine]`) ;
  mêmes graines pour A, B et C.

## Les mesures

Pour chaque graine et chaque masque, **survie** sur les 256 mondes du test
22. L'agent prend l'action la plus probable ; à égalité exacte, l'autre
action que la dernière.

## Prédictions fixées

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de
liberté).

| | Prédiction | Critère |
|---|---|---|
| **SURV1** | Sans aucune cible, porter ses états le fait survivre mieux que ses seules actions | moyenne de (survie de A − survie de B, même graine) ≥ **0,08**, borne basse > 0 |
| **SURV2** | Il dépasse ce que permet l'événement du tour seul | moyenne de (survie de A − 0,723) ≥ **0,05**, borne basse > 0 |

**Critère global** : SURV1 et SURV2, test valide.

**Publié sans seuil** :
- C (« libre ») ;
- graine par graine ;
- la courbe de survie pendant l'apprentissage (moyenne des lots) ;
- la part des décisions où A prend l'action de la règle « besoins » (sur
  ses vies de mesure).

**Validité** :
- **masques** : comme au test 29 (B : changer un événement passé ne change
  aucune sortie des tours suivants ; A : ne change pas la sortie de la
  première couche aux tokens des tours suivants ; écart ≤ 1e-6, graine 0) ;
- **l'apprentissage a eu lieu** : survie moyenne de B ≥ **0,60** (sinon
  l'apprentissage par renforcement n'a pas marché, et le test ne dit rien).

## Ce que le résultat dira

**Si SURV1 et SURV2 passent.** Sans aucune cible et sans pré-entraînement,
seulement en apprenant à survivre, un petit transformeur qui ne voit son
passé qu'à travers ses propres états survit mieux qu'un témoin qui ne voit
que ses actions, et mieux que ce que permet l'événement du tour seul. Il a
appris de lui-même à porter, dans ses états, quelque chose de ses besoins
cachés. Ce que ces états portent n'est pas mesuré ici.

**Si SURV1 passe et SURV2 échoue.** Porter ses états aide, mais il ne
dépasse pas assez ce que permet l'événement du tour.

**Si SURV1 échoue.** Seulement en apprenant à survivre, à cette taille et
avec cet apprentissage, porter ses états ne le fait pas survivre assez
mieux que ses seules actions.

**Si l'apprentissage n'a pas eu lieu** (B sous 0,60), le test n'est pas
valide.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti.
- Il ne dira pas ce que portent les états : seulement qu'ils servent à
  survivre au-delà des actions.
- La récompense (la survie) vient du monde, qui connaît les besoins ; mais
  elle ne nomme ni le besoin ni l'action à prendre.
- Ces modèles sont petits, le monde aussi.

## Exécution

- **Code** : `research/tiny_survival.py` (à écrire), torch sur le processeur
  local.
- **Verdicts** : numpy seulement, vérifiés en CI.
