# Protocole pré-enregistré — La boucle : un état qui ne passe que d'un tour au suivant (test 32)

Rédigé le 5 octobre 2026, **avant l'écriture du code et toute exécution** ;
l'heure est celle du commit. Seuils fixés, un échec est un résultat.

Au moment d'écrire : le test 31 n'est pas fini (A appris pour 10 graines
sur 10, B pour 8, C pour 8 ; premiers résultats bruts publiés), et rien de
ce test n'a été essayé.

## D'où vient ce test (dit tel quel)

- **Tests 22 à 27** (modèle de langage, publiés). Sous un masque où le
  passé n'est visible qu'à travers ses propres tokens portés, l'agent porte
  son besoin d'un tour à l'autre (tests 22, 23). Mais un tour lit
  directement les états portés de **tout** son passé : ce n'est pas une
  récurrence au sens strict, et il n'y a pas de relais pas à pas (test 27).
- **Tests 29 et 31** (petits transformeurs, deux couches). Une précision
  vérifiée après lecture (publiée le 5 octobre) : avec deux couches, l'état
  porté d'un tour ne contient que l'événement de ce tour et les actions
  passées. Le besoin est **recomposé** à chaque décision ; il n'est pas
  porté comme un état qui dure.
- **Ce qui manque** : un état qui ne passe **que** d'un tour au suivant, et
  qui doit donc porter, à lui seul, tout ce qui sert du passé.

**Ce qui est déjà connu, et que ce test ne revendique pas.** Des agents
récurrents apprennent une mémoire par la seule récompense (par exemple
DRQN, 2015). Des transformeurs à mémoire transmise d'un segment au suivant
existent (Feedback Transformer, 2020 ; Recurrent Memory Transformer,
2022). L'apprentissage par renforcement homéostatique existe aussi
(Keramati et Gutkin, 2014). Ce test ajoute une mesure **causale**,
pré-enregistrée, de ce que porte l'état : on le greffe d'une vie à une
autre.

**Question.** Un petit transformeur dont le passé ne passe que par un état
transmis d'un tour au suivant, et qui n'apprend que de sa survie, apprend-il
à porter ses besoins cachés dans cet état ?

## Le monde

Celui du programme et du test 31 : deux besoins, énergie (E) et nourriture
(N), de 1 à 8, six événements, deux actions (R recharge E de 3, M recharge
N de 3), 30 tours. À chaque tour, l'événement arrive ; si un besoin tombe à
0, la vie s'arrête ; sinon l'agent choisit. **Récompense** : 1 par tour vécu
après la décision, rien d'autre.

## Les agents

- **A, « boucle ».** À chaque tour t, un petit transformeur (2 couches,
  dimension 64, 4 têtes, appris de zéro) lit 4 positions :
  1. **l'état s_t**, un vecteur de dimension 64 ;
  2. **la dernière action** (R, M, ou « aucune » au premier tour) ;
  3. **l'événement** du tour ;
  4. **« Choix »**.

  En sortie de « Choix » : le choix (R ou M), la valeur (pour
  l'apprentissage), et **l'état suivant s_{t+1}** = la sortie de « Choix »
  normalisée. s_1 est un vecteur appris. **Un tour ne voit jamais les tours
  passés** : seul s_t passe d'un tour au suivant. Le même vecteur décide et
  se transmet.
- **B, « coupée »** (le témoin). Le même réseau, mais l'état est coupé :
  à chaque tour, s_t est remplacé par le vecteur initial appris. B ne voit
  donc que la dernière action et l'événement du tour.

## L'apprentissage

Le réglage P2 du pilote (acteur-critique, choisi sur le témoin du test 31,
`docs/TINY_SURVIVAL_PILOT_PLAN.md`) :
- retour escompté (γ = 0,9) des tours vécus après la décision ; avantage =
  retour − valeur, normalisé dans le lot ;
- pertes : politique, + 0,5 × valeur, − 0,01 × entropie ;
- AdamW, taux 3e-4, **4 000 mises à jour de 64 vies** ;
- le gradient traverse l'état d'un tour à l'autre, sur toute la vie.

**Graines** 0 à 9. Flux neufs : initialisation `[270926, 76, graine]`,
mondes `[270926, 74, graine, mise à jour, vie]`, tirages des actions
`[270926, 75, graine, mise à jour]`.

## Les mesures

1. **Survie** sur les 256 mondes de mesure (flux `[270926, 44]`), en
   prenant l'action la plus probable (à égalité exacte, l'autre que la
   dernière). Repères sur ces mondes : règle « événement » 0,723 ; règle
   « besoins » 0,906.
2. **La greffe.**
   - **Les vies** : 128 vies neuves écrites par la règle « événement » avec
     30 % d'actions au hasard (comme aux tests 26 et 28), flux
     `[270926, 77]`. L'agent les lit : ses états sont calculés à partir des
     événements et des actions écrits.
   - Pour une vie receveuse r, un tour j et un tour t = j + g (g = 1 ou 2),
     r vivante en t : on prend une vie donneuse d, vivante au tour j, qui a
     **la même action écrite au tour j**. On remplace l'état s_{j+1} de r
     (celui que « Choix » du tour j transmet) par celui de d, puis on lit
     la suite de r jusqu'au tour t.
   - **ΔP(R)** = probabilité de choisir R au tour t avec la greffe − sans.
   - **Effet de la règle e** : on rejoue r à partir des besoins de d après
     l'événement du tour j, avec l'action du tour j et les événements et
     actions de r ensuite. e = +1 si la règle « besoins » passe alors de M
     à R au tour t, −1 de R à M, 0 sinon. Une donneuse dont le rejeu fait
     mourir r avant t n'est pas admissible.
   - **Trois sortes** (besoins après l'événement du tour j) :
     **(a)** autres besoins, e ≠ 0 ; **(a0)** autres besoins, e = 0
     (publié sans seuil) ; **(b)** mêmes besoins (autre histoire).
   - **Tirage** (flux `[270926, 78, 0]`), sans modèle, le même pour toutes
     les graines et les deux agents. On parcourt les receveuses dans
     l'ordre ; pour chaque r et chaque g, on tire au plus 2 tours t ; pour
     chacun, une donneuse de chaque sorte parmi les admissibles, s'il en
     existe. On s'arrête à **400 (a)**, **200 (a0)**, **200 (b)**.
     Compté avant ce protocole, sans modèle, sur ces 128 vies : des 4 670
     places (r, g, t) possibles, 3 835 ont une donneuse (a), 4 647 une
     donneuse (a0), 3 461 une donneuse (b).

## Prédictions fixées

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de
liberté).

| | Prédiction | Critère |
|---|---|---|
| **LOOP1** | Seulement en apprenant à survivre, l'état transmis le fait survivre mieux que le témoin coupé | moyenne de (survie de A − survie de B, même graine) ≥ **0,08**, borne basse > 0 |
| **LOOP2** | Il dépasse ce que permet l'événement du tour seul | moyenne de (survie de A − 0,723) ≥ **0,05**, borne basse > 0 |
| **LOOP3** | L'état transmis porte les besoins : greffé, il fait choisir selon les besoins de la vie d'origine | pour chaque graine, m = moyenne sur les greffes (a) de ΔP(R) × e ; moyenne des m ≥ **0,10**, borne basse > 0, et m > 0,05 pour au moins 8 graines sur 10 |
| **LOOP4** | C'est le besoin, pas l'histoire | moyenne sur les graines de \|ΔP(R)\| des greffes (b) ≤ **la moitié** de celle des greffes (a), et la différence (a) − (b), graine par graine, a une borne basse > 0 |

**Critère global** : LOOP1 à LOOP4, test valide.

**Publié sans seuil** :
- graine par graine ; les courbes d'apprentissage ;
- la part des décisions où l'agent prend l'action de la règle « besoins » ;
- \|ΔP(R)\| des greffes (a0) ; les résultats selon g ;
- la part de l'effet de la règle que A retrouve (moyenne de ΔP(R) × e,
  rapportée au changement complet) ;
- **ce que dit l'état** : sur les 128 vies, un décodeur linéaire (moindres
  carrés régularisés, validation croisée par vies) lit E et N après
  l'événement du tour dans s_{t+1} ; part de variance expliquée.

**Validité** :
- **l'apprentissage a eu lieu** : survie moyenne de B ≥ **0,60** ;
- **la coupure** : chez B, changer un événement passé ne change pas le
  choix suivant (écart ≤ 1e-6, graine 0, 4 cas) ;
- **greffe de soi** : écart ≤ 1e-6 (graine 0, 4 cas) ;
- au moins 100 greffes (a) et 100 greffes (b) ; les 10 graines apprises.

## Ce que le résultat dira

**Si LOOP1 à LOOP4 passent.** Seulement en apprenant à survivre, un petit
transformeur dont le passé ne passe que par un état transmis d'un tour au
suivant apprend à y porter ses besoins cachés. Greffé dans une autre vie,
cet état fait choisir selon les besoins de la vie d'origine ; venu d'une
vie aux mêmes besoins mais à l'histoire différente, il change peu le choix.
C'est une récurrence au sens strict, et ce qu'elle porte est surtout un
état de soi : le niveau de ses besoins, que rien ne lui a nommé.

**Si LOOP1 et LOOP2 passent, mais LOOP3 ou LOOP4 échoue.** L'état sert à
survivre, mais la greffe ne montre pas qu'il porte surtout les besoins.

**Si LOOP1 échoue.** À cette taille et avec cet apprentissage, l'état
transmis ne le fait pas survivre assez mieux que le témoin coupé.

**Si l'apprentissage n'a pas eu lieu** (B sous 0,60), le test n'est pas
valide.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti.
- La récompense (la survie) vient du monde, qui connaît les besoins ; elle
  ne nomme ni le besoin ni l'action à prendre.
- La greffe lit des vies écrites par une règle, pas des vies vécues par
  l'agent.
- Le test ne sépare pas le niveau des besoins de ce qui en commande le
  choix.
- Ces modèles sont petits, le monde aussi ; le réglage d'apprentissage a
  été choisi pour un autre réseau (test 31).

## Exécution

- **Code** : `research/tiny_loop.py` (à écrire), torch sur le processeur
  local, un fil par processus.
- **Verdicts** : numpy seulement, vérifiés en CI.
