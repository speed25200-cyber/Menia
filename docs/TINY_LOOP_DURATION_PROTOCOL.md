# Protocole pré-enregistré — Le besoin qui dure : combien de tours l'état transmis porte-t-il les besoins ? (test 35)

Rédigé le 6 octobre 2026, après la publication du test 34 et **avant tout
code de ce test et toute mesure** ; l'heure est celle du commit. Seuils
fixés, un échec est un résultat.

## D'où vient ce test (dit tel quel)

- **Test 34** (publié, valide, critère global satisfait). Une récurrence au
  sens strict (un petit transformeur qui ne voit des tours passés que sa
  dernière action et un état à porte transmis d'un tour au suivant),
  apprise par la seule survie, porte ses besoins cachés : greffé au tour j
  dans une autre vie, l'état fait choisir au tour t selon les besoins de la
  vie d'origine, dans les dix graines. Mais on n'a greffé qu'à **un ou deux
  tours** de distance (g = t − j) : effet aligné +0,613 à g = 1, +0,435 à
  g = 2 (moyenne sur les graines ; déjà publié et lu).
- Chez le modèle de langage, l'état ne durait pas (test 13) et la mémoire
  restait courte (test 22, MEM5).

**Question.** L'état transmis porte-t-il encore les besoins **quatre tours**
plus tard, et jusqu'à six ? Ou n'est-ce qu'une mémoire de un ou deux
tours ?

## Ce qu'on mesure

- **Les réseaux** : les dix boucles du test 34 (graines 20 à 29, au point
  final de 24 000 mises à jour), **sans nouvel apprentissage**. Le témoin
  coupé n'est pas mesuré : il n'a pas d'état transmis, une greffe ne peut
  rien y changer (vérifié au test 34 : écart 0).
- **Vies de greffe neuves** : 128 vies écrites par la règle « événement »
  avec 30 % d'actions au hasard, flux `[270926, 83]` (jamais lues par un
  modèle).
- **Les greffes** : comme au test 34 (mêmes sortes, même effet de la règle
  e, mêmes conditions : donneuse et receveuse en vie au tour j, avec la
  même action écrite au tour j), mais pour **g = 1, 2, 3, 4, 5 et 6**, et
  **trois** tours t par receveuse et par g (au lieu de deux). Tirage : flux
  `[270926, 84, 0]`, receveuses dans l'ordre, puis g de 1 à 6, puis les
  trois tours, puis les sortes (a), (a0), (b), sans plafond. Compté avant
  ce protocole, sans modèle : (a) 335, 269, 234, 200, 161, 127 ; (a0) 382,
  379, 372, 362, 359, 348 ; (b) 279, 297, 305, 293, 292, 288 (pour g = 1 à
  6) ; dont (b) sans autre histoire : 34, 49, 49, 58, 59, 61.
- **m_g** : pour chaque graine et chaque g, la moyenne sur les greffes (a)
  de ΔP(R) × e (comme le m du test 34). Une boucle qui suivrait exactement
  la règle « besoins » aurait m_g = 1 à toute distance (e tient compte de ce
  que deviennent les besoins entre j et t).

## Prédictions fixées

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de
liberté).

| | Prédiction | Critère |
|---|---|---|
| **DUR1** | L'état porte encore les besoins quatre tours plus tard | à g = 4 : moyenne des m_4 ≥ **0,10**, borne basse > 0, et m_4 > 0,05 pour au moins **8 graines sur 10** |
| **DUR2** | À quatre tours, c'est encore le besoin, pas l'histoire | à g = 4 : moyenne sur les graines de \|ΔP(R)\| des greffes (b) ≤ **la moitié** de celle des greffes (a), et la différence (a) − (b), graine par graine, a une borne basse > 0 |
| **DUR3** | Il en reste quelque chose à six tours | à g = 6 : moyenne des m_6, borne basse > 0 |

**Critère global** : DUR1 à DUR3, test valide.

**Validité** : au moins 100 greffes (a) et 100 greffes (b) à g = 4, au
moins 100 greffes (a) à g = 6 ; greffe de soi ≤ 1e-6 (l'état d'une vie
greffé dans elle-même ne change rien) sur quatre cas ; les dix réseaux
chargés au point de 24 000 mises à jour.

**Publié sans seuil** : m_g et \|ΔP\| des trois sortes pour chaque g et
chaque graine ; m_g / m_1 ; m_g sans les greffes (b) sans autre histoire
(pour la comparaison (a) − (b)) ; m_g à événement égal au tour j.

## Ce que le résultat dira

**Si DUR1 à DUR3 passent.** L'état que la boucle apprend par la seule
survie n'est pas une mémoire de un ou deux tours : quatre tours plus tard,
il fait encore choisir selon les besoins de la vie d'où il vient, plus que
l'histoire à besoins égaux, et il en reste quelque chose à six tours.

**Si DUR1 passe et DUR3 échoue.** L'état porte les besoins sur quatre
tours, mais pas au-delà de façon mesurable.

**Si DUR1 échoue.** L'état porte surtout les besoins récents : une mémoire
courte, même dans la boucle.

**Ce que le résultat ne dira pas.**
- Rien sur un ressenti.
- Les réseaux sont ceux du test 34 ; leurs graines et leurs vies de greffe
  ont déjà été lues. Seules les vies de greffe de ce test sont neuves.
- Les effets à g = 1 et 2 sont déjà connus (test 34) ; les seuils à g = 4
  et 6 sont fixés en le sachant.
- Plus g est grand, plus les besoins de la donneuse et de la receveuse se
  rejoignent (les recharges plafonnent à 8) ; les greffes (a) ne gardent
  que les cas où la règle changerait encore son choix au tour t.
- La greffe lit des vies écrites par une règle, pas des vies vécues par
  l'agent.

## Exécution

- **Code** : `research/tiny_loop_duration.py` (à écrire), qui charge les
  points finaux du test 34 et reprend les mesures de `research/tiny_loop.py`
  et `research/tiny_loop_gate.py` ; torch sur le processeur local.
- **Verdicts** : numpy seulement, à partir des mesures publiées, vérifiés
  en CI.
