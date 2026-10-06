# Protocole pré-enregistré — La boucle fiable : un état transmis qui porte les besoins, graine après graine (test 34)

Rédigé le 6 octobre 2026, **après le second pilote**
(`docs/TINY_LOOP_PILOT2_PLAN.md`) et **avant tout apprentissage** sur les
graines de ce test ; l'heure est celle du commit. Seuils fixés, un échec est
un résultat.

## D'où vient ce test (dit tel quel)

- **Test 33** (publié, valide, échoue sur LOOP3). La boucle à porte apprise
  par la seule survie survit mieux que le témoin coupé (10 graines sur 10),
  et dans 7 graines sur 10 l'état greffé fait choisir selon les besoins de
  la vie d'origine (moyenne +0,321) ; il en fallait 8. Les apprentissages
  sont instables.
- **Le second pilote** (publié, `99f3dbd`). Quatre réglages, choisis sur
  la greffe elle-même, sur les graines 100 à 103 et des mondes et des vies
  de greffe à part ; le témoin coupé une fois par graine. Fiables (sur les
  quatre graines, m > 0,05 et un gain sur le témoin d'au moins 0,04) : R1
  (le test 33 tel quel ; m moyen +0,432, gain +0,164), R3 (24 000 mises à
  jour ; +0,499, +0,203) et R4 (R3 avec le gradient borné à 1 ; +0,472,
  +0,191) ; R2 (gradient borné, 12 000 mises à jour) ne l'est pas. Selon la
  règle fixée d'avance, le réglage retenu est **R3**. Ce pilote départage
  peu : avec R1, les quatre graines portaient déjà les besoins, alors
  qu'au test 33 trois graines sur dix ne les portaient pas.

**Question (celle des tests 32 et 33).** Un petit transformeur dont le passé
ne passe que par un état transmis d'un tour au suivant, et qui n'apprend que
de sa survie, apprend-il à porter ses besoins cachés dans cet état et à
s'en servir — **de façon fiable, graine après graine** ?

## Ce qui change par rapport au test 33

- L'apprentissage : **24 000 mises à jour** de 64 vies (au lieu de
  12 000), pour la boucle **et** le témoin coupé ; tout le reste de
  l'apprentissage est celui du test 33 (taux 3e-4, pas de borne sur le
  gradient).
- **Graines neuves : 20 à 29** (flux d'initialisation, de mondes et de
  tirages du test 32 avec ces graines).
- **Vies de greffe neuves** : 128 vies écrites par la règle « événement »
  avec 30 % d'actions au hasard, flux `[270926, 81]` ; tirage des greffes
  comme au test 32, flux `[270926, 82, 0]` : 400 (a), 200 (a0), 200 (b),
  dont 24 greffes (b) sans autre histoire (comptés avant ce protocole, sans
  modèle).

Tout le reste est repris du test 33 (`docs/TINY_LOOP_GATE_PROTOCOL.md`) : le
monde, la récompense (1 par tour vécu, rien d'autre), le réseau à porte,
l'acteur-critique, la mesure de survie sur les 256 mondes du test 22, l'effet
de la règle e et les trois sortes de greffes.

## Prédictions fixées (celles du test 33, inchangées)

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de
liberté).

| | Prédiction | Critère |
|---|---|---|
| **LOOP1** | L'état transmis le fait survivre mieux que le témoin coupé | moyenne de (survie de A − survie de B, même graine) ≥ **0,08**, borne basse > 0 |
| **LOOP2** | Il dépasse ce que permet l'événement du tour seul | moyenne de (survie de A − 0,723) ≥ **0,05**, borne basse > 0 |
| **LOOP3** | L'état transmis porte les besoins : greffé, il fait choisir selon les besoins de la vie d'origine | m = moyenne sur les greffes (a) de ΔP(R) × e, par graine ; moyenne des m ≥ **0,10**, borne basse > 0, et m > 0,05 pour au moins **8 graines sur 10** |
| **LOOP4** | C'est le besoin, pas l'histoire | moyenne sur les graines de \|ΔP(R)\| des greffes (b) ≤ **la moitié** de celle des greffes (a), et la différence (a) − (b), graine par graine, a une borne basse > 0 |

**Critère global** : LOOP1 à LOOP4, test valide.

**Publié sans seuil** :
- comme au test 33 : graine par graine ; courbes ; part des décisions qui
  suivent la règle « besoins » ; \|ΔP\| des greffes (a0) ; résultats
  selon g ; le décodeur linéaire de E et N dans l'état transmis, et sur le
  réseau non appris ; à événement égal au tour j ; LOOP4 sans les 24
  greffes (b) sans autre histoire ; la valeur moyenne de z ;
- le plus bas de la courbe d'apprentissage après 2 000 mises à jour, par
  graine (les effondrements) ;
- **à 12 000 mises à jour** (le réglage du test 33 : c'est le même
  apprentissage jusque-là, mesuré en chemin) : la survie de A et de B, m
  par graine et le nombre de graines où m > 0,05. Cela dira si apprendre
  plus longtemps change, sur des graines neuves, ce qui a manqué au
  test 33.

**Validité** (celle du test 33) : B ≥ 0,60 ; coupure ≤ 1e-6 ; greffe de
soi ≤ 1e-6 ; au moins 100 greffes (a) et (b) ; les 10 graines apprises.

## Ce que le résultat dira

**Si LOOP1 à LOOP4 passent.** Seulement en apprenant à survivre, un petit
transformeur dont le passé ne passe que par un état transmis d'un tour au
suivant apprend, de façon fiable, à y porter ses besoins cachés et à s'en
servir. Greffé dans une autre vie, cet état fait choisir selon les besoins
de la vie d'origine ; venu d'une vie aux mêmes besoins mais à l'histoire
différente, il change peu le choix. C'est une récurrence au sens strict, et
ce qu'elle porte est surtout le niveau de ses besoins, que rien ne lui a
nommé.

**Si LOOP1 et LOOP2 passent, mais LOOP3 ou LOOP4 échoue.** L'état sert à
survivre, mais la greffe ne montre pas, de façon fiable, qu'il porte
surtout les besoins.

**Si LOOP1 échoue.** Même avec ce réglage, l'état transmis ne le fait pas
survivre assez mieux que le témoin coupé.

**Si l'apprentissage n'a pas eu lieu** (B sous 0,60), le test n'est pas
valide.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti.
- Le réglage a été choisi en regardant la boucle et la greffe (au pilote,
  sur d'autres graines, d'autres mondes et d'autres vies de greffe) : cela
  peut la favoriser ; le témoin reçoit le même réglage.
- La récompense (la survie) vient du monde, qui connaît les besoins ; elle
  ne nomme ni le besoin ni l'action.
- La greffe lit des vies écrites par une règle, pas des vies vécues par
  l'agent ; elle ne sépare pas le niveau des besoins de ce qui en commande
  le choix.
- Ces modèles sont petits, le monde aussi. Des agents récurrents qui
  suivent un niveau interne **donné en entrée**, et dont on force l'état,
  sont connus (`docs/LITERATURE_CHECK_LOOP_2026-10-06.md`) ; l'apport
  revendiqué se limite à ce montage (besoins jamais observés), à son témoin,
  à la mesure causale par greffe et au pré-enregistrement.

## Exécution

- **Code** : `research/tiny_loop_reliable.py` (à écrire), qui reprend
  `research/tiny_loop_gate.py` avec le réglage retenu, les graines 20 à 29
  et les vies de greffe neuves ; torch sur le processeur local, un fil par
  processus, plusieurs processus se partageant les apprentissages.
- **Verdicts** : numpy seulement, vérifiés en CI.
