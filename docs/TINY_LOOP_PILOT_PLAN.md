# Plan pré-enregistré du pilote — Faire servir l'état transmis (avant le test 33)

Rédigé le 5 octobre 2026, après la publication du test 32 et **avant tout
code du pilote et toute exécution** ; l'heure est celle du commit.

## Pourquoi un pilote

Au test 32 (publié, valide, échoue), une vraie boucle — un petit
transformeur qui ne voit jamais les tours passés, seul un état passant d'un
tour au suivant — a appris par la seule survie à **porter** ses besoins
dans cet état (un décodeur linéaire y lit E et N à 0,68 et 0,69, contre
0,42 et 0,46 chez le témoin coupé), mais **pas à s'en servir** : greffé,
l'état ne déplace le choix que de 0,05 ; la boucle survit à 0,721, le
témoin à 0,733. Trois graines sur dix s'en servent en partie ; une s'est
effondrée.

Le réglage venait du pilote du test 31, choisi pour un autre réseau. Avant
de reposer la question (test 33), il faut un apprentissage qui fasse servir
l'état.

**Ce qui est réglé, et le risque.** Le témoin coupé n'a rien à régler : sans
mémoire, il apprend presque toujours la même table (0,730 ou 0,734 au test
32). Le pilote règle donc **l'apprentissage de la boucle**. C'est régler le
traitement, ce qui peut le favoriser. Pour limiter ce risque : graines et
mondes à part, choix par une règle fixée ici, et test 33 sur des graines
neuves où la boucle et le témoin reçoivent **le même réglage**.

## Ce que le pilote compare

Le monde, la récompense (1 par tour vécu, rien d'autre), le réseau (2
couches, dimension 64, 4 têtes ; 4 positions par tour : l'état, la dernière
action, l'événement, « Choix ») et l'acteur-critique (γ = 0,9, avantage
normalisé, valeur × 0,5, entropie 0,01, AdamW, 64 vies par mise à jour)
sont ceux du test 32. Quatre réglages :

- **L1** : le test 32, mais **12 000** mises à jour (au lieu de 4 000) ;
- **L2** : taux d'apprentissage **1e-3** (au lieu de 3e-4), 4 000 mises à
  jour ;
- **L3** : un **état à porte**, 4 000 mises à jour. L'état suivant mélange
  l'ancien et le nouveau : s_{t+1} = (1 − z) · s_t + z · n_t, où n_t est la
  sortie normalisée de « Choix » (l'état du test 32) et z = σ(W h + b), h
  la sortie de « Choix » (W et b appris, b initialisé à 0). L'état peut
  ainsi durer sans être réécrit à chaque tour. Le témoin coupé n'en est pas
  changé (son état est remplacé à chaque tour) ;
- **L4** : l'état à porte et 12 000 mises à jour.

## Règle de choix, fixée ici

- **Graines du pilote** : 100, 101, 102 (jamais utilisées par un test de la
  boucle). Flux d'initialisation, de mondes et de tirages : ceux du test 32
  (`[270926, 76/74/75, graine, …]`).
- **Mondes du pilote** : 256 mondes du flux `[270926, 73]`, distincts des
  256 mondes de mesure (flux `[270926, 44]`).
- Pour chaque réglage, on apprend **la boucle** sur les trois graines. On
  apprend aussi **le témoin coupé** une fois par graine, avec le réglage du
  test 32. On mesure la survie sur les mondes du pilote, en prenant l'action
  la plus probable.
- **Gain d'un réglage** = moyenne sur les trois graines de (survie de la
  boucle − survie du témoin coupé de la même graine).
- **On retient le réglage au plus grand gain**, s'il atteint au moins
  **0,04** et si **aucune graine** de la boucle ne survit à moins que son
  témoin − 0,05 (pas d'effondrement). À égalité à 0,005 près, on retient le
  plus simple, dans l'ordre L1, L2, L3, L4.
- **Si aucun réglage ne convient**, le pilote échoue. Il est publié, et le
  test 33 n'est pas lancé sous cette forme.

## Ensuite

Le protocole du test 33 sera écrit après le pilote, avec le réglage retenu,
pour la boucle **et** le témoin coupé. Il reprendra les critères du test 32
(LOOP1 à LOOP4, validité), les 256 mondes de mesure, les mêmes 128 vies de
greffe et le même tirage, sur des graines neuves (10 à 19).

Tout est publié : réglages, survies par graine, courbes, et le choix.
