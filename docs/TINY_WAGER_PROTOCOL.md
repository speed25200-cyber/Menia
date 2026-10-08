# Protocole pré-enregistré — Le pari : sait-il quand il ne sait pas ? (test 37)

Rédigé le 8 octobre 2026, **après le pilote** (`docs/TINY_WAGER_PILOT_PLAN.md`,
publié dans `docs/LLM_NEED_RESULTS.md`) et **avant tout apprentissage** sur
les graines de ce test ; l'heure est celle du commit. Seuils fixés, un échec
est un résultat.

## D'où vient ce test (dit tel quel)

- **Tests 34 à 36** (publiés, valides, critères globaux satisfaits). Une
  récurrence au sens strict, apprise par la seule survie, porte ses besoins
  cachés dans l'état transmis, sur plusieurs tours, et peut y porter deux
  contenus distincts, chacun surtout vers sa sortie.
- **Le pilote** (publié). Dans le monde « pari » (les événements forts
  cachés une fois sur deux sous un même « ? » ; un pari après chaque
  décision, +0,5 si l'action sert un besoin le plus bas, −2 sinon), sur les
  graines 110 à 112, la boucle gagne +3,85 par vie sur le témoin coupé. Une
  greffe qui change le pari de la règle idéale sans changer son action fait
  parier dans le même sens (m_pari +0,48 à +0,55) et bouge peu l'action
  (0,12 à 0,17 contre 0,51 à 0,59), aussi à moyennes de croyance proches
  (à 0,5 près : +0,43 à +0,51). Dans l'autre sens (greffes (k)), le pari
  bouge à peu près la moitié de l'action (0,25 à 0,28 contre 0,47 à 0,58) et
  baisse en moyenne : il suit aussi le contenu de la croyance. À probabilité
  exactement égale de la règle, le pari ne prédit pas la réussite mieux que
  le hasard, ce qui est attendu de tout agent. Selon la règle fixée d'avance, le réglage retenu est **P1
  (48 000 mises à jour)**.

**Question.** L'état transmis, appris par la seule survie et par les
paris, porte-t-il, en plus de ce que l'agent croit de ses besoins, une
grandeur qui suit **la fiabilité de cette croyance**, et s'en sert-il pour
parier ?

## Le monde, l'agent, l'apprentissage

Ceux du pilote (`research/tiny_wager.py`), avec le réglage P1 (48 000
mises à jour de 64 vies, taux 3e-4), pour la boucle **et** le témoin coupé.
**Graines neuves : 40 à 49.** Récompense, survie et paris mesurés sur
**256 mondes neufs** (flux `[270926, 136]`), action et pari les plus
probables.

## Les greffes

128 vies de greffe neuves écrites par la règle du pilote (action de la
règle, remplacée au hasard avec une probabilité de 0,3), flux
`[270926, 137]` ; tirage `[270926, 138, 0]`, comme au pilote (t = j + 1 ou
j + 2, tous les tours, même action au tour j, une donneuse de chaque sorte
au plus par tour t et par écart) :

- **(u)** l'action de la règle au tour t ne change pas, son pari change
  (e_pari = ±1) ; « moyennes proches » si les moyennes de E et de N au tour
  t, avec et sans greffe, diffèrent chacune d'au plus 0,5 ;
- **(k)** l'action de la règle change (e_action = ±1), son pari non ;
- **(b)** la croyance de la donneuse au tour j est celle de la receveuse,
  d'une autre vie.

Compté avant ce protocole, sans modèle : (u) 4 220 (dont 357 à moyennes
proches ; e_pari +1 : 1 536, −1 : 2 684), (k) 3 307, (b) 1 460 (dont 408
de même histoire).

## Prédictions fixées

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de
liberté). ΔP : changement de la probabilité au tour t, avec la greffe moins
sans. Moyennes sur les graines.

| | Prédiction | Critère |
|---|---|---|
| **BET1** | L'état transmis sert à gagner | récompense par vie de la boucle − celle du témoin (même graine) : moyenne ≥ **1,0**, borne basse > 0 |
| **BET2** | Le pari suit la fiabilité portée | greffes (u) : m_pari = moyenne de ΔP(pari) × e_pari ; moyenne ≥ **0,10**, borne basse > 0, et m_pari > 0,05 pour au moins **8 graines sur 10** |
| **BET3** | Pas seulement l'écart des besoins | greffes (u) à moyennes proches (à 0,5 près, pas égales) : moyenne de m_pari ≥ **0,10**, borne basse > 0 |
| **BET4** | Pas un déplacement commun | greffes (u) : m_pari sur les seules greffes e_pari = +1 **et** sur les seules e_pari = −1, chacun à borne basse > 0 |
| **BET5** | La fiabilité va vers le pari | greffes (u) : \|ΔP(R)\| ≤ **la moitié** de \|ΔP(pari)\|, différence graine par graine à borne basse > 0 |
| **BET6** | Une croyance égale bouge peu le pari | \|ΔP(pari)\| (b) ≤ la moitié de \|ΔP(pari)\| (u), différence à borne basse > 0 |

**Critère global** : BET1 à BET6, test valide.

**Validité** : coupure ≤ 1e-6 (sous le témoin, changer un événement passé
ne change rien) et greffe de soi ≤ 1e-6, sur la graine 40, quatre cas
chacun ; au moins 100 greffes (u), 100 (k), 100 (b) et 50 (u) à moyennes
proches ; les 10 graines apprises ; empreinte du tirage identique pour
chaque graine.

**Publié sans seuil** : graine par graine ; dans le sens (k), m_action et
\|ΔP(pari)\| contre \|ΔP(R)\| (le pilote montre que le pari y bouge à peu
près la moitié de l'action ; ce sens n'entre pas dans le critère) ;
ΔP(pari) signé sur (b) ; survie, part des tours pariés et des paris
gagnés ; courbes ; aire sous la courbe de P(pari) (réussite contre échec)
au total, par tranches de 0,1 et **à probabilité exactement égale** de la
règle (arrondie à 1e-4) ; résultats selon l'écart g.

## Ce que le résultat dira

**Si BET1 à BET6 passent.** Seulement en apprenant à survivre et à parier,
un petit transformeur dont le passé ne passe que par un état transmis (et
sa dernière action) porte dans cet état, en plus de ce qu'il croit de ses
besoins, une grandeur qui suit la fiabilité de cette croyance, et s'en
sert pour parier. Greffé,
l'état d'une vie où l'agent idéal garderait la même action mais
parierait autrement fait parier autrement, dans les deux sens, aussi à
moyennes de croyance proches, en bougeant peu l'action.

**Si BET2 passe mais BET3 échoue.** Le pari suit surtout l'écart entre les
besoins crus, pas leur fiabilité.

**Si BET2 échoue.** Le pari ne suit pas de façon fiable la confiance
portée.

**Ce que le résultat ne dira pas.**
- Rien sur un ressenti. Le pari est une sortie apprise, payée par le
  monde ; ce n'est ni une parole ni un sentiment de confiance.
- L'incertitude vient du monde (les « ? »), pas d'un doute de l'agent sur
  lui-même. Au pilote, le pari ne repère pas les erreurs propres de
  l'agent au-delà de la règle idéale.
- Le monde a été choisi parmi d'autres pour que la fiabilité compte ; le
  réglage a été choisi au pilote en regardant le pari (sur d'autres graines
  et d'autres vies).
- La séparation dans l'autre sens (k) n'est pas testée : au pilote, elle
  n'était que partielle ; le pari suit aussi le contenu de la croyance.
- Les moyennes proches ne sont pas égales : BET3 réduit la part du contenu
  sans l'écarter.
- La greffe lit des vies écrites par une règle, pas des vies vécues par
  l'agent.
- Petit réseau, petit monde.

## Exécution

- **Code** : `research/tiny_wager_test.py` (à écrire), qui reprend
  `research/tiny_wager.py` ; torch sur le processeur local, quatre
  processus à un fil ; chaque fichier brut commis seul dès son écriture.
- **Verdicts** : numpy seulement, vérifiés en CI.
