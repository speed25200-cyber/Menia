# Plan pré-enregistré du second pilote — Rendre la boucle fiable (avant le test 34)

Rédigé le 6 octobre 2026, après la publication du test 33 et **avant tout
code de ce pilote et toute exécution** ; l'heure est celle du commit.

## Pourquoi

Au test 33 (publié, valide), la boucle à porte apprise par la seule survie
survit mieux que le témoin coupé pour 10 graines sur 10 (LOOP1, LOOP2
passent), et LOOP4 passe. Mais LOOP3 échoue : l'état greffé ne fait
choisir selon les besoins d'origine (m > 0,05) que pour 7 graines sur 10,
il en fallait 8. Les trois autres graines ont m à peu près nul. Les
apprentissages sont **instables** : quatre graines s'effondrent un moment
(jusqu'à 0,00) puis se rétablissent ; les courbes montent encore à
12 000 mises à jour.

La question devient : un apprentissage **plus stable** ou **plus long**
rend-il fiable, graine après graine, un état transmis qui porte les
besoins ?

## Ce qui est réglé, et le risque

On règle l'apprentissage de la boucle (le traitement), cette fois **sur la
greffe elle-même**, sur des graines et des vies à part. Cela peut favoriser
la boucle ; le test 34 se fera sur des graines neuves, avec des vies de
greffe neuves, la boucle et le témoin recevant le même réglage.

## Ce que le pilote compare

Le monde, la récompense (1 par tour vécu, rien d'autre), le réseau à porte
et l'acteur-critique du test 33 (γ = 0,9, avantage normalisé, valeur × 0,5,
entropie 0,01, AdamW 3e-4, 64 vies par mise à jour). Quatre réglages :

- **R1** : le test 33 tel quel (12 000 mises à jour) ;
- **R2** : R1 avec la norme du gradient **bornée à 1** à chaque mise à
  jour ;
- **R3** : R1 avec **24 000** mises à jour ;
- **R4** : R2 avec 24 000 mises à jour.

## Mesures et règle de choix, fixées ici

- **Graines du pilote** : 100 à 103 (flux d'initialisation, de mondes et de
  tirages du test 33 avec ces graines ; R1 sur les graines 100 à 102
  redonne L4 du premier pilote).
- **Calcul épargné, sans rien changer** : R1 sur les graines 100 à 102 est
  repris des points gardés à 12 000 mises à jour par le premier pilote (sa
  survie doit redonner celle publiée pour L4) ; R3 et R4 prolongent R1 et R2
  depuis leur point à 12 000. C'est le même apprentissage : chaque mise à
  jour tire ses mondes et ses actions de ses propres flux, et la reprise
  après un arrêt est exacte.
- **Mondes du pilote** : les 256 mondes du flux `[270926, 73]`.
- **Vies de greffe du pilote** : 128 vies écrites par la règle
  « événement » avec 30 % d'actions au hasard, flux `[270926, 79]` ; tirage
  des greffes comme au test 32, flux `[270926, 80, 0]`. Elles ne servent
  pas au test 34.
- Pour chaque réglage et chaque graine : la survie de la boucle sur les
  mondes du pilote, et **m** = moyenne sur les greffes (a) de ΔP(R) × e.
  Le témoin coupé est appris une fois par graine avec le réglage R1.
- **Un réglage est fiable** si, pour **les quatre graines**, m > 0,05 et la
  survie de la boucle dépasse celle du témoin coupé de la même graine d'au
  moins 0,04.
- **On retient le réglage fiable au plus grand m moyen.** À 0,01 près, le
  plus simple, dans l'ordre R1, R2, R3, R4.
- **Si aucun réglage n'est fiable**, le pilote échoue ; il est publié, et le
  test 34 n'est pas lancé sous cette forme.

## Ensuite

Le protocole du test 34 sera écrit après le pilote, avec le réglage retenu,
pour la boucle et le témoin coupé, sur des graines neuves (20 à 29), avec
les critères du test 33 (LOOP1 à LOOP4, dont huit graines sur dix pour
LOOP3), les 256 mondes de mesure et de nouvelles vies de greffe.

Tout est publié : réglages, survies, m par graine, courbes, et le choix.
