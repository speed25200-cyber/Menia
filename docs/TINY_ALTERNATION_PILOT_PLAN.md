# Plan pré-enregistré du pilote — L'alternance : deux causes, un espace de travail, un temps interne (avant le test 42)

Rédigé le 17 octobre 2026, pendant le test 37, **avant tout code de ce
pilote et toute exécution** ; l'heure est celle du commit. Le pilote ne
sera lancé qu'à la fin des apprentissages du test 37, pour ne pas les
ralentir. Rien dans ce plan ne change le test 37.

## Pourquoi

Hesse, Tsao et al. (bioRxiv, 16 octobre 2026 ;
`docs/LITERATURE_CHECK_ALTERNANCE_2026-10-17.md`) enregistrent le cortex
inférotemporal et préfrontal du macaque pendant une rivalité. À entrée et
percept constants, la même population **alterne** environ toutes les 60 ms
entre une représentation fidèle de l'image vue et une représentation fidèle
de l'image supprimée. Les deux états se distinguent par une étiquette faible
(gain ×1,21). Moyennée, l'activité ressemble à un mélange faiblement modulé.
L'alternance peut se lire comme un **échantillonnage** de la croyance
(Gershman, Vul et Tenenbaum, 2012).

Le monde « pari » du test 37 contient une rivalité : un « ? » cache soit
« tu cours », soit « il fait froid ». Sans modèle (1 500 vies du flux de
préparation `[270926, 951]`), sur les 7 474 décisions prises à un « ? »,
la probabilité de « tu cours » selon la règle a pour quantiles 5 %, 50 % et
95 % : 0,44 ; 0,50 ; 0,55. **99,7 %** de ces décisions sont ambiguës
(entre 0,2 et 0,8), et l'action de la règle **diffère** selon la cause dans
**66,8 %** d'entre elles.

La question : quand l'espace de travail ne peut tenir qu'**un contenu à la
fois**, un petit agent appris par la seule survie et par les paris
apprend-il à porter les deux causes **en alternant** dans un temps interne,
de façon **coordonnée** entre ses unités, et sa confiance (le pari) lit-elle
cette alternance ? C'est l'indicateur GWT-2 (espace de travail limité) mis
devant la question de Hesse et al., et HOT-2 (le pari) devant la lecture
d'un état d'accès.

**Repères sans modèle** (1 500 vies du flux `[270926, 952]`, tirages
`[270926, 953, K, vie]`, calculés avant ce plan). Un échantillonneur tire K
couples (E, N) de la croyance exacte à chaque décision. Il choisit l'action
qui sert un besoin le plus bas dans le plus d'échantillons (à égalité,
l'autre que la dernière). Il parie si la part des échantillons où elle sert
dépasse 0,8.

| Croyance | Survie | Récompense par vie |
|---|---|---|
| exacte (la règle) | 0,629 | 32,76 |
| 64 échantillons | 0,622 | 32,60 |
| 16 échantillons | 0,606 | 32,07 |
| **8 échantillons** | 0,601 | **31,84** |
| 4 échantillons | 0,584 | 31,52 |
| 2 échantillons | 0,583 | 29,49 |
| **1 échantillon** (une seule hypothèse) | 0,530 | **26,39** |

Tenir une seule hypothèse par tour coûte 6,4 points. En tenir huit
successives n'en coûte que 0,9.

## Le monde

Celui du test 37, inchangé (`research/tiny_wager.py`) : brouillard 0,5 sur
« tu cours » et « il fait froid », pari +0,5 ou −2, récompense de 1 par
tour vécu plus les paris.

## L'agent

Le réseau à porte du test 37 garde son rôle : à chaque tour, à partir de
l'état transmis, de la dernière action, de l'événement ou « ? » et du token
« Choix », il calcule h (dimension 64) et l'état transmis au tour suivant,
comme au test 37. **Les sorties ne lisent plus h**, elles lisent un espace
de travail :

- **Espace de travail** : 16 unités binaires z ∈ {0, 1}¹⁶.
- **Temps interne** : à chaque tour, K micro-pas. Un micro-état m
  (dimension 64, nul au premier micro-pas de chaque tour) évolue par une
  cellule GRU dont l'entrée est (h, z du micro-pas précédent) ; à chaque
  micro-pas, chaque unité est **tirée** (Bernoulli) avec la probabilité
  sigmoïde(Linéaire(m)), indépendamment des autres unités. La
  **coordination entre unités n'est pas imposée** : elle ne peut venir que
  du micro-état partagé et du retour de z dans la cellule.
- **Lecture** : à chaque micro-pas, logits de l'action et du pari lus sur z
  (couches linéaires) ; la probabilité de l'action et celle du pari au tour
  sont la **moyenne des probabilités** sur les K micro-pas. La valeur est
  lue sur h.
- **Apprentissage** : l'acteur-critique du test 37 (P1 : 48 000 mises à
  jour de 64 vies, AdamW 3e-4, γ = 0,9, valeur × 0,5, entropie 0,01 sur
  chaque sortie). Le gradient traverse les tirages par l'estimateur
  « droit » (straight-through : valeur tirée vers l'avant, gradient de la
  probabilité vers l'arrière).

**Note datée (17 octobre, après le commit de ce plan, avant tout code) : ce
que lit le pari.** Tel qu'écrit plus haut, le pari serait une moyenne de
lectures faites micro-pas par micro-pas. Une telle moyenne ne peut pas lire
l'**accord** entre micro-pas, et l'intervention « unanimité imposée » contre
« partage imposé » ne mesurerait que l'écart entre les paris lus sur A et
sur B. Correction, avant tout code : **l'action** reste lue sur z à chaque
micro-pas, probabilités moyennées (un vote). **Le pari** est lu sur un
**lecteur** séparé : une seconde cellule GRU (dimension 64, nulle au premier
micro-pas de chaque tour) dont la seule entrée est z, micro-pas après
micro-pas ; le pari est lu sur son état au dernier micro-pas. Le lecteur ne
voit ni h ni le micro-état qui produit z : il ne sait que ce qui est passé
par l'espace de travail, et dans quel ordre. Sous intervention, le lecteur
reçoit la suite de z imposée. Rien d'autre ne change.

**Les bras du pilote** :
- **S8** : K = 8, unités tirées ;
- **S1** : K = 1, unités tirées (une seule hypothèse par tour) ;
- **D8** : K = 8, unités **non tirées** (z = la probabilité elle-même,
  continue) : l'espace peut tenir un mélange, sans bruit ;
- **coupé** : S8 dont l'état transmis est remplacé à chaque tour par le
  vecteur initial appris, comme au test 37.

Graines **120 à 122**. Flux : initialisation `[270926, 140, graine]`,
mondes d'apprentissage 141, choix d'apprentissage 142, tirages des unités
143 ; **256 mondes du pilote**, flux `[270926, 144]` (tirages des unités à
l'évaluation : flux 145).

## Ce que le pilote mesure

Sur les mondes du pilote, action et pari les plus probables (la
probabilité moyenne sur les micro-pas), tirages des unités gardés :
- la survie, la récompense moyenne par vie, la part des tours où l'agent
  parie et la part des paris gagnés ;
- à chaque décision prise à un « ? » : les K vecteurs z.

**Gabarits** (comme les images présentées seules chez Hesse et al.). Pour
chaque décision à un « ? », on rejoue la même vie jusqu'à ce tour, avec le
« ? » de ce tour remplacé par « tu cours » (A) puis par « il fait froid »
(B), et 16 tirages indépendants des unités (flux 146). Gabarit de A : la
fréquence de chaque unité sur ces micro-pas ; de même pour B (bornée à
[0,01 ; 0,99]).

**Mesures descriptives**, pour chaque bras à K = 8 (S8 et D8 ; pour D8, z
continu est traité comme une probabilité) :
- **état de chaque micro-pas** : le signe du rapport de vraisemblance de
  Bernoulli entre les gabarits A et B (état 1 : A ; état 2 : B) ;
- **alternance** : la part des décisions à un « ? » où les deux états
  apparaissent parmi les 8 micro-pas, et le nombre moyen de changements ;
- **lecture sur l'autre moitié** (le contrôle de circularité de Hesse et
  al.) : les 16 unités sont coupées en deux moitiés fixes (unités paires,
  impaires) ; l'état est inféré sur une moitié ; sur l'autre moitié, on
  mesure l'**indice de penchant** = (distance au gabarit B − distance au
  gabarit A) de la fréquence moyenne des micro-pas en état 1, moins le même
  indice pour l'état 2 ; positif si l'autre moitié suit l'alternance ;
  moyenne des deux sens ;
- **deux états contre un** : vraisemblance des z sur l'autre moitié sous le
  mélange des deux gabarits, contre le meilleur gabarit unique w·A + (1−w)·B
  (w balayé de 0 à 1 par pas de 0,05) ;
- **modèles témoins**, simulés avec les mêmes gabarits et le même nombre de
  micro-pas, analysés de même : « moyenne » (chaque unité tirée avec
  (A + B)/2) et « non coordonné » (chaque unité bascule seule entre A et B,
  avec la durée moyenne des états mesurée chez l'agent) ;
- **le pari lit-il l'alternance ?** À probabilité égale selon la règle
  (tranches de 0,1), corrélation de rang entre P(pari) et l'**accord** =
  |part des micro-pas en état 1 − 0,5| ; et une **intervention** : au même
  tour, z imposé sur les 8 micro-pas au tirage le plus probable du gabarit A
  (« unanimité imposée ») ou alterné A, B, A, B… (« partage imposé ») ;
  ΔP(pari) = P(pari | unanimité) − P(pari | partage) ;
- **gain** des états : la distance moyenne au point médian des gabarits,
  pour les micro-pas de l'état qui correspond à l'action choisie et pour
  les autres (l'asymétrie « ×1,21 » de Hesse et al.).

**Publié sans seuil** : tout ce qui précède, par graine et par bras, et les
courbes.

**Note datée (17 octobre, après relecture indépendante du code, avant toute
exécution du pilote).** La relecture a trouvé un point bloquant pour les
mesures (pas pour la règle de choix). Une variation **commune** à toutes les
unités, ou **liée au rang du micro-pas** (le premier micro-pas part d'un
micro-état nul), passerait pour une alternance coordonnée. Il suffit pour
cela que A et B diffèrent par leur niveau d'activité moyen, ce que le plan
attend lui-même (le « gain »). Sur un modèle sans aucune alternance, avec
un tel motif au premier micro-pas et B = 1,3 A, la lecture sur l'autre
moitié donne +0,24, contre moins de 0,03 pour les deux témoins. Les données
pour le corriger après coup n'auraient pas été gardées. Corrections, avant
toute exécution :
- **Gabarits par micro-pas.** On garde, pour A et pour B, la fréquence de
  chaque unité **à chaque micro-pas** (8 × 16, sur les 16 tirages). La
  lecture principale étiquette chaque micro-pas avec les gabarits **de ce
  micro-pas**. La lecture avec le gabarit moyen, telle qu'écrite plus haut,
  est publiée aussi.
- **Un témoin de plus, « niveau ».** C'est la moyenne (A + B)/2 de chaque
  micro-pas, mise à l'échelle du niveau d'activité de l'agent à ce
  micro-pas. Ce témoin hérite de toute alternance portée par le seul
  niveau, ce qui en fait une barre plus haute. Le témoin « moyenne » devient
  celui de chaque micro-pas.
- **Témoin « non coordonné ».** La probabilité de bascule par micro-pas est
  celle qui redonne le **nombre moyen de changements** de l'agent. Avec
  1/D, la durée simulée était plus courte que celle mesurée.
- **Intervention.** On publie la quantité du plan (unanimité A moins
  partage A B A B…) et, ajoutée, la forme symétrique (les deux unanimités
  contre les deux partages). La forme du plan mêle le contenu A et l'accord.
  Pour D8, on impose aussi les gabarits continus, puisque des z binaires
  sortent de ce qu'il a appris.
- **Lecture du pari.** La corrélation de rang est calculée aussi à
  l'intérieur de chaque côté de la part d'état 1 (au-dessus, au-dessous de
  0,5). Ainsi, un lecteur qui ne suivrait que le contenu A ne compte pas.
  Les tranches sont celles de la probabilité, selon la règle, que l'action
  de l'agent serve un besoin le plus bas (comme au test 37).
- **Un nul.** Chaque mesure est refaite sur le même réseau **avant tout
  apprentissage** (même graine, mêmes mondes).
- **Témoins pour tous les bras à 8 micro-pas**, D8 compris (probabilités
  continues pour D8) et le coupé ; flux des témoins `[270926, 147, bras,
  graine, …]`.
- Techniques : unités tirées exactement binaires vers l'avant ; tests du
  côté torch (gradient à travers les tirages, lecteur sous intervention,
  coupé, reprise exacte) ; tests des mesures sur des modèles connus, dont le
  scénario de la relecture.

La règle de choix ne change pas.

## Règle de choix, fixée ici

- **Le réglage S8 est fiable** si, pour les trois graines, sa récompense
  moyenne par vie dépasse celle du témoin coupé de la même graine d'au moins
  **1,0** et celle de S1 de la même graine d'au moins **0,5**.
- **Si S8 est fiable**, le test 42 sera écrit avec S8, sur dix graines
  neuves, avec des critères fixés après le pilote et avant tout
  apprentissage sur ces graines : alternance, lecture sur l'autre moitié
  au-dessus des modèles témoins, deux états préférés à un, et un pari qui
  lit l'accord (intervention).
- **Si S8 n'est pas fiable**, le pilote échoue ; il est publié, et le test
  42 n'est pas lancé sous cette forme.

## Ce que l'on attend, dit avant

- D8 peut tenir un mélange sans bruit : il n'a aucune raison d'alterner ; on
  s'attend à un état unique (« moyenne ») à chaque « ? ».
- S8 doit tirer ses unités : s'il apprend à porter les deux causes, il peut
  le faire en alternant. Mais ses unités peuvent aussi rester bruitées sans
  coordination (le modèle « non coordonné »), ou rester sur une seule cause.
- Un résultat nul (pas d'alternance coordonnée, ou un pari qui ne lit pas
  l'accord) est probable et serait publié tel quel.

## Ce que ce pilote ne dira pas

Le tirage des unités est imposé par l'architecture : le bruit n'est pas
appris. Ce qui pourrait l'être, c'est son **organisation** : quelles
hypothèses les tirages portent, s'ils basculent ensemble, et si le pari les
lit. Une alternance dans un petit réseau ne serait pas « la même chose »
que celle d'un cortex. Ce serait une analogie fonctionnelle, mesurée avec
les mêmes contrôles. Le préprint lui-même étudie un corrélat sans rapport
chez le singe. Rien sur un ressenti.

Tout est publié : bras, survies, récompenses, mesures par graine, courbes,
choix.
