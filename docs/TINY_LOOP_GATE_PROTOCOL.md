# Protocole pré-enregistré — La boucle qui s'en sert : un état transmis, appris par la seule survie (test 33)

Rédigé le 6 octobre 2026, **après le pilote** (`docs/TINY_LOOP_PILOT_PLAN.md`)
et **avant tout apprentissage** sur les graines de ce test ; l'heure est
celle du commit. Seuils fixés, un échec est un résultat.

## D'où vient ce test (dit tel quel)

- **Test 32** (publié, valide, échoue). Une vraie boucle — un petit
  transformeur qui ne voit jamais les tours passés, seul un état passant
  d'un tour au suivant — apprise par la seule survie garde dans son état
  une mémoire courte (en partie présente sans apprentissage), mais s'en
  sert peu : elle ne survit pas mieux que le témoin coupé (0,721 contre
  0,733), et l'état greffé ne déplace le choix que de 0,05.
- **Le pilote** (publié). Quatre façons d'apprendre la boucle, sur les
  graines 100 à 102 et 256 mondes à part ; le témoin coupé une fois par
  graine. Gain moyen sur le témoin : L1 (12 000 mises à jour) +0,069 ;
  L2 (taux 1e-3) −0,004 ; L3 (état à porte) +0,089 ; **L4 (état à porte,
  12 000 mises à jour) +0,161** (0,848, 0,816 et 0,867 contre 0,688, 0,688
  et 0,672). Selon la règle fixée d'avance, le réglage retenu est **L4**
  (`7327b20`).

**Question (celle du test 32).** Un petit transformeur dont le passé ne
passe que par un état transmis d'un tour au suivant, et qui n'apprend que
de sa survie, apprend-il à porter ses besoins cachés dans cet état et à
s'en servir ?

## Ce qui change par rapport au test 32

Le réglage L4 du pilote :
- **un état à porte** : l'état suivant mélange l'ancien et le nouveau,
  s_{t+1} = (1 − z) · s_t + z · n_t, où n_t est la sortie normalisée de
  « Choix » (l'état du test 32) et z = σ(W h + b), h la sortie de « Choix »
  (W et b appris, b initialisé à 0) ;
- **12 000 mises à jour** de 64 vies (au lieu de 4 000), taux 3e-4.

**Le témoin coupé reçoit le même réglage** (pour lui, la porte ne sert à
rien : son état est remplacé à chaque tour par le vecteur initial appris).

Tout le reste est repris du test 32 (`docs/TINY_LOOP_PROTOCOL.md`) : le
monde, la récompense (1 par tour vécu, rien d'autre), le réseau (2 couches,
dimension 64, 4 têtes ; l'état, la dernière action, l'événement, « Choix »),
l'acteur-critique (γ = 0,9, avantage normalisé, valeur × 0,5, entropie
0,01, AdamW), la mesure de survie sur les 256 mondes du test 22, les 128
vies de greffe (flux `[270926, 77]`), le même tirage des greffes (flux
`[270926, 78, 0]` : 391 (a), 200 (a0), 200 (b)), l'effet de la règle e et
les trois sortes de greffes.

**Graines neuves : 10 à 19** (flux d'initialisation, de mondes et de
tirages du test 32 avec ces graines).

## Prédictions fixées (celles du test 32)

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de
liberté).

| | Prédiction | Critère |
|---|---|---|
| **LOOP1** | L'état transmis le fait survivre mieux que le témoin coupé | moyenne de (survie de A − survie de B, même graine) ≥ **0,08**, borne basse > 0 |
| **LOOP2** | Il dépasse ce que permet l'événement du tour seul | moyenne de (survie de A − 0,723) ≥ **0,05**, borne basse > 0 |
| **LOOP3** | L'état transmis porte les besoins : greffé, il fait choisir selon les besoins de la vie d'origine | m = moyenne sur les greffes (a) de ΔP(R) × e, par graine ; moyenne des m ≥ **0,10**, borne basse > 0, et m > 0,05 pour au moins 8 graines sur 10 |
| **LOOP4** | C'est le besoin, pas l'histoire | moyenne sur les graines de \|ΔP(R)\| des greffes (b) ≤ **la moitié** de celle des greffes (a), et la différence (a) − (b), graine par graine, a une borne basse > 0 |

**Critère global** : LOOP1 à LOOP4, test valide.

**Publié sans seuil** (comme au test 32, et ce que sa relecture a demandé) :
- graine par graine ; courbes ; part des décisions qui suivent la règle
  « besoins » ; \|ΔP\| des greffes (a0) ; résultats selon g ;
- le décodeur linéaire de E et N dans l'état transmis, **et le même
  décodeur sur le réseau non appris de la même graine** ;
- **à événement égal au tour j** : \|ΔP\| des greffes (a) et (b), et
  l'effet aligné des greffes (a) ;
- les greffes (b) sans autre histoire (donneuse et receveuse ont vécu les
  mêmes événements et actions jusqu'au tour j ; 38 des 200 dans ce tirage,
  compté avant ce protocole, sans modèle), et LOOP4 sans elles ;
- l'état passé par la porte : la valeur moyenne de z.

**Validité** (celle du test 32) : B ≥ 0,60 ; coupure ≤ 1e-6 ; greffe de
soi ≤ 1e-6 ; au moins 100 greffes (a) et (b) ; les 10 graines apprises.

## Ce que le résultat dira

**Si LOOP1 à LOOP4 passent.** Seulement en apprenant à survivre, un petit
transformeur dont le passé ne passe que par un état transmis d'un tour au
suivant apprend à y porter ses besoins cachés et à s'en servir. Greffé
dans une autre vie, cet état fait choisir selon les besoins de la vie
d'origine ; venu d'une vie aux mêmes besoins mais à l'histoire différente,
il change peu le choix. C'est une récurrence au sens strict, et ce qu'elle
porte est surtout le niveau de ses besoins, que rien ne lui a nommé.

**Si LOOP1 et LOOP2 passent, mais LOOP3 ou LOOP4 échoue.** L'état sert à
survivre, mais la greffe ne montre pas qu'il porte surtout les besoins.

**Si LOOP1 échoue.** Même avec ce réglage, l'état transmis ne le fait pas
survivre assez mieux que le témoin coupé.

**Si l'apprentissage n'a pas eu lieu** (B sous 0,60), le test n'est pas
valide.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti.
- Le réglage a été choisi en regardant la boucle (au pilote, sur d'autres
  graines et d'autres mondes) : cela peut la favoriser ; le témoin reçoit
  le même réglage.
- La récompense (la survie) vient du monde, qui connaît les besoins ; elle
  ne nomme ni le besoin ni l'action. La tête de valeur du critique prédit
  le retour de survie.
- La greffe lit des vies écrites par une règle, pas des vies vécues par
  l'agent ; elle ne sépare pas le niveau des besoins de ce qui en commande
  le choix.
- Ces modèles sont petits, le monde aussi. Des agents récurrents qui
  apprennent une mémoire par la récompense, et des états à porte (GRU,
  LSTM), sont connus ; l'apport revendiqué se limite à ce montage, à son
  témoin, à la mesure causale par greffe et au pré-enregistrement.

## Exécution

- **Code** : `research/tiny_loop_gate.py` (à écrire), qui reprend le réseau
  et l'apprentissage du pilote (`research/tiny_loop_pilot.py`) et les
  mesures du test 32 (`research/tiny_loop.py`) ; torch sur le processeur
  local, un fil par processus.
- **Verdicts** : numpy seulement, vérifiés en CI.
