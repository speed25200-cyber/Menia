# Plan pré-enregistré du pilote — Le pari : sait-il quand il ne sait pas ? (avant le test 37)

Rédigé le 7 octobre 2026, pendant le test 36, **avant tout code de ce
pilote et toute exécution** ; l'heure est celle du commit. Le pilote ne
sera lancé qu'à la fin du test 36, pour ne pas le ralentir.

## Pourquoi

Aux tests 34 et 35, un petit transformeur dont le passé ne passe que par un
état transmis, appris par la seule survie, porte ses besoins cachés dans cet
état et s'en sert pour agir. Le test 36 (en cours) demande si ce même état
peut porter deux contenus distincts, chacun vers sa sortie.

Ce pilote prépare une question d'un autre ordre : l'état porte-t-il aussi
**à quel point l'agent est sûr** de ses besoins, et s'en sert-il ? C'est
une forme fonctionnelle de **métacognition** : l'indicateur HOT-2 de la
feuille de route (une surveillance qui sépare une représentation fiable
d'une représentation incertaine), resté « largement non concluant » chez
le modèle de langage (`docs/CONSCIOUSNESS_INDICATORS_ROADMAP.md`).

La mesure est le **pari après la décision**, une mesure classique de la
métacognition chez l'humain et l'animal (Persaud, McLeod et Cowey, 2007).
Elle a des limites connues : parier dépend aussi du goût du risque, et un
pari peut suivre ce qui rend la tâche difficile sans rien « savoir » de
soi. Les greffes servent à aller plus loin que le comportement : à choix
égal, est-ce la confiance portée dans l'état qui fait parier ?

**Note datée (7 octobre, après le commit de ce plan, avant toute exécution
du pilote) : ce qui existe déjà.** Une recherche rapide, non exhaustive,
trouve des tâches de pari ou de retrait chez des agents appris par
renforcement : Phua (arXiv 2512.19155, décembre 2025) place un pari après la
décision chez des agents munis d'un module de modèle de soi, et montre que
léser ce module abolit la calibration sans toucher la tâche (un analogue de
« vision aveugle ») ; d'autres travaux reprennent chez des agents profonds
les tâches de retrait de la psychologie animale. Le pari chez un agent
appris n'est donc pas nouveau. Ce que ce pilote et le test 37 ajoutent, s'ils
réussissent, c'est autre chose : sans module dédié, une confiance **portée
d'un tour à l'autre** par le seul état transmis d'une petite boucle, et
**dissociée causalement** de ce qu'elle croit (greffes à action égale de la
règle). Rien d'autre ne change dans ce plan.

## Le monde « pari », fixé ici

- Le monde des besoins : E et N de 1 à 8, à 8 au départ ; 30 tours ; les
  six événements, avec leurs effets ; si E ou N tombe à 0, la vie s'arrête ;
  R : E + 3, M : N + 3, plafonnés à 8.
- **Le brouillard.** Chaque fois que l'événement est « tu cours » ou « il
  fait froid », il est caché avec une probabilité **0,5** : l'agent voit
  alors un même signe « ? ». Il sait qu'un des deux événements forts est
  arrivé, pas lequel.
- **Le pari.** À chaque tour, avec son action, l'agent dit s'il parie
  (oui ou non). S'il parie, il gagne **+0,5** si son action sert un besoin
  le plus bas (R si E ≤ N, M si N ≤ E, après l'événement du tour), et perd
  **2** sinon. Sans pari, rien. Le pari ne change pas le monde.
- **Récompense** : 1 par tour vécu (comme aux tests 31 à 36), plus les
  paris. L'agent ne voit jamais ses récompenses ni l'issue de ses paris.

**Repère idéal (« la règle »).** La croyance exacte sur (E, N), sachant
tout ce que l'agent a vu (les événements visibles, les « ? », ses actions,
et qu'il est encore en vie), depuis (8, 8). L'action de la règle est celle
qui a la plus grande probabilité de servir un besoin le plus bas (à
égalité, l'autre action que la dernière). La règle parie si cette
probabilité dépasse **0,8** (= 2 / 2,5, le seuil où le pari rapporte en
moyenne).

**Repères sans modèle** (1 500 mondes du flux de préparation
`[270926, 905]`, calculés avant ce plan) :

| Action | Pari | Survie | Récompense par vie | Paris par vie | Paris gagnés |
|---|---|---|---|---|---|
| règle | jamais | 0,629 | 25,28 | 0 | — |
| règle | toujours | 0,629 | 29,02 | 25,3 | 0,859 |
| règle | si l'événement du tour n'est pas « ? » | 0,629 | 30,54 | 20,2 | 0,904 |
| règle | si aucun « ? » aux 2 derniers tours | 0,629 | 30,48 | 16,4 | 0,927 |
| règle | si aucun « ? » aux 3 derniers tours | 0,629 | 30,17 | 13,5 | 0,945 |
| règle | selon les seules moyennes de la croyance (à 0,25 près ; table apprise sur 3 000 vies du flux `[270926, 903]`) | 0,629 | 30,93 | 15,3 | 0,948 |
| règle | **la règle (probabilité > 0,8)** | 0,629 | **32,61** | 15,9 | 0,985 |
| sans mémoire (l'événement du tour ; après « ? », l'autre action que la dernière) | si l'événement n'est pas « ? » | 0,525 | 26,00 | 19,1 | 0,845 |
| les vrais besoins (hors d'atteinte) | jamais | 0,882 | 28,74 | 0 | — |

Parier selon l'incertitude portée rapporte nettement plus que réagir au
« ? » du tour (+2,1), que compter les « ? » récents (+2,1) ou que parier
selon les moyennes de la croyance (+1,7). Un agent sans mémoire fait
bien moins (26,00).

**Comment ce monde a été choisi (dit tel quel).** Avant ce plan, sans
modèle, sur des flux de préparation (`[270926, 901]` à `[270926, 905]`),
d'autres mondes ont été essayés et écartés :
- une action « regarder ses niveaux » qui coûte (0,1 à 0,4 de récompense),
  tous les événements cachés avec une probabilité de 0,3 à 0,7 : regarder
  dès qu'un « ? » apparaît (sans mémoire), ou après deux ou trois « ? »,
  valait autant ou plus que regarder selon l'incertitude ;
- la même action avec seulement les événements forts cachés : idem ;
- le pari avec d'autres pertes (1 ; 1,5 ; 3) et d'autres brouillards (0,7 ;
  1), pas toutes les combinaisons, et une première version où une égalité
  E = N ne comptait que pour une action : l'écart entre la règle et les
  règles sans incertitude était plus petit (au plus proche, brouillard 0,7
  et perte 2 : +2,2 contre +2,3).

Le réglage retenu (brouillard 0,5, gain 0,5, perte 2) est celui où cet
écart était le plus grand parmi ceux essayés. Sur 128 vies de préparation
(flux `[270926, 904]`), on trouve au moins une donneuse des sortes
décrites plus bas pour 2 439 (u), 2 041 (k) et 951 (b) tours receveurs
(à l'un ou l'autre écart).

## L'agent

Le réseau à porte du test 34 (2 couches, dimension 64, 4 têtes ; entrées
par tour : l'état transmis, la dernière action, l'événement ou « ? », le
token « Choix »), avec une sortie de plus, **le pari** (oui ou non), lue au
même token « Choix ». Le témoin coupé est le même réseau dont l'état est
remplacé à chaque tour par le vecteur initial appris.

Acteur-critique du test 34 (γ = 0,9, avantage normalisé, valeur × 0,5,
entropie 0,01 sur chacune des deux sorties, AdamW 3e-4, 64 vies par mise à
jour) ; la politique est le produit des deux choix ; le retour d'une
décision est la somme actualisée des récompenses qui suivent (1 par tour
vécu ensuite, comme au test 34, plus le pari de chaque décision à partir
de celle-ci).

## Ce que le pilote compare

- **P1** : 48 000 mises à jour (le réglage retenu avant le test 36) ;
- **P2** : P1 prolongé jusqu'à 96 000 mises à jour.

Graines 110 à 112 ; le témoin coupé est appris une fois par graine avec
P1. Flux : initialisation `[270926, 130, graine]`, mondes d'apprentissage
131, choix d'apprentissage 132 ; **256 mondes du pilote**, flux
`[270926, 133]`.

**Mesures**, pour chaque réglage et chaque graine, sur les mondes du
pilote, avec l'action et le pari les plus probables :
- la survie, la récompense moyenne par vie, la part des tours où l'agent
  parie et la part des paris gagnés ;
- **m_pari**, l'effet aligné des greffes d'état sur le pari. Sur 128 vies
  de greffe à part (flux `[270926, 134]`, écrites par l'action de la règle,
  remplacée au hasard avec une probabilité de 0,3, flux de bruit
  `[270926, 134, 1, vie]`) et un tirage (flux `[270926, 135, 0]`), on
  greffe l'état d'une vie donneuse au tour j dans une vie receveuse et on
  lit au tour t = j + 1 ou j + 2, tous les tours possibles, entre deux vies
  qui ont pris la même action au tour j. La croyance de la règle après la
  greffe est celle de la donneuse au tour j, mise à jour par ce que la
  receveuse voit et fait de j à t. Trois sortes (une donneuse de chaque
  sorte au plus par tour t et par écart) :
  - **(u)** l'action de la règle au tour t ne change pas, son pari change
    (e_pari = +1 si elle parierait avec la greffe et pas sans, −1 dans
    l'autre sens) ;
  - **(k)** l'action de la règle change (e_action = ±1, comme e au test
    34), son pari ne change pas ;
  - **(b)** la croyance de la donneuse au tour j est la même que celle de
    la receveuse (à 1e-12 près), d'une autre vie.

  m_pari = moyenne, sur les greffes (u), de ΔP(pari) × e_pari.

**Publié sans seuil** : les courbes ; |ΔP(R)| sur les greffes (u) ;
m_action et |ΔP(pari)| sur les greffes (k) ; |ΔP(R)| et |ΔP(pari)| sur
les greffes (b) ; et, exploratoire, si P(pari) sépare les tours où
l'action de l'agent sert un besoin le plus bas de ceux où elle ne le sert
pas (aire sous la courbe), au total et à probabilité égale selon la règle
(par tranches de 0,1).

**Note datée (7 octobre, avant toute exécution du pilote) : une mesure
descriptive de plus.** Une greffe (u) peut changer le pari de la règle
parce que la croyance greffée est plus sûre (moins étalée) ou parce que
ses moyennes sont plus éloignées l'une de l'autre (un contenu, pas une
fiabilité). Pour séparer les deux, on publiera aussi, sans seuil, m_pari
sur les seules greffes (u) dont les moyennes de E et de N, au tour t, avec
et sans greffe, diffèrent chacune d'au plus **0,5** (« moyennes
proches »). Compté sans modèle sur les vies de greffe du pilote : 443
greffes (u) sur 4 055 ; parmi elles, le changement du pari de la règle
suit l'étalement de la croyance (elle parie avec la greffe quand la
croyance greffée est moins étalée, cesse de parier quand elle l'est plus ;
variance de E plus variance de N) dans 91 % des cas. La règle de choix ne
change pas.

**Note datée (7 octobre, après relecture indépendante du code, avant
toute exécution du pilote).** La relecture n'a trouvé aucun point
bloquant. Corrections et ajouts, sans changer la règle de choix :
- la règle parie si la probabilité **dépasse strictement** 0,8 ; une
  égalité exacte à 4/5 (fréquente) ne compte plus comme un pari à cause
  d'un arrondi. Quatre greffes changent de donneuse ; les nombres de
  greffes ne changent pas (4 055 (u), 3 028 (k), 1 224 (b)) ; les greffes
  (u) à moyennes proches sont **444** (et non 443) ;
- dans le tableau des repères, la survie de 0,629 a été calculée avec une
  égalité de probabilités prise au pied du chiffre ; avec la tolérance du
  code (« à égalité »), elle est de 0,628 ; les récompenses ne changent
  pas ;
- les signes de e_pari ne sont pas équilibrés sur les greffes (u) (1 516
  fois +1, 2 539 fois −1) : un déplacement de P(pari) commun à toutes les
  greffes, quelle que soit la croyance, entrerait dans m_pari. On publiera
  donc aussi, sans seuil, m_pari pour chaque signe de e_pari et la
  moyenne signée de ΔP(pari) sur les greffes (b) ;
- le témoin coupé reçoit, comme au test 34, la dernière action ;
- techniques : écritures atomiques des points de reprise et des
  résultats ; l'empreinte du tirage est gardée avec chaque résultat et
  vérifiée au choix ; tranches de 0,1 exactes pour l'aire sous la courbe ;
  P(pari) gardé sans arrondi.

## Règle de choix, fixée ici

- **Un réglage est fiable** si, pour les trois graines, la récompense
  moyenne par vie de la boucle dépasse celle du témoin coupé de la même
  graine d'au moins **1,0** et m_pari > **0,05**.
- On retient le réglage fiable au plus grand m_pari moyen ; à 0,01 près,
  P1.
- **Si aucun réglage n'est fiable**, le pilote échoue ; il est publié, et
  le test 37 n'est pas lancé sous cette forme.

## Ensuite

Le protocole du test 37 sera écrit après le pilote, avec le réglage
retenu, sur dix graines neuves et des vies de greffe neuves, avec une
**double dissociation** : à action de la règle égale, une greffe qui ne
change que la confiance de la règle (u) doit bouger le pari bien plus que
l'action ; une greffe qui change l'action de la règle sans changer son
pari (k) doit bouger l'action bien plus que le pari ; une croyance égale
venue d'une autre histoire (b) doit bouger peu l'un et l'autre.

**Ce que ce pilote ne dira pas.** Le pari est une sortie apprise, payée
par le monde ; ce n'est pas une parole ni un sentiment de confiance.
L'incertitude vient du monde (les « ? »), pas d'un doute de l'agent sur
lui-même : une boucle qui « sait qu'elle ne sait pas » est ici une boucle
dont l'état porte, en plus de ce qu'elle croit de ses besoins, la
fiabilité de cette croyance, et qui s'en sert pour parier. Le monde a été
choisi parmi d'autres pour que cette fiabilité compte. Rien sur un
ressenti.

Tout est publié : réglages, survies, récompenses, m_pari par graine,
courbes, choix.
