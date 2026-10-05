# Menia sur les cinq niveaux de Chandaria et al. (2026)

État au 4 octobre 2026. Ce document est tenu à jour à chaque verdict.

Cadre : Chandaria, Muñoz Morán, Rosas, Seth, Shevlin, Hutter, Graepel,
Bales, Comsa, Shanahan, Laukkonen, Kringelbach, Frith et Legg,
[*From cacophony to hierarchy: a principled framework for assessing AI consciousness*](https://arxiv.org/abs/2609.35618),
28 septembre 2026.

- **Ce que propose ce cadre.** Il ne tranche pas la question de la
  conscience. Il range les indicateurs sur cinq niveaux de description,
  puis combine les éléments de preuve en une probabilité. Cette
  probabilité dépend explicitement des hypothèses de chaque théorie :
  c'est « l'agnosticisme structuré ».
- **Ce que nous en faisons.** Nous plaçons chaque résultat pré-enregistré
  de Menia au niveau qu'il concerne, en disant ce qui passe, ce qui échoue
  et ce qui manque.
- **Ce que nous ne faisons pas.** Nous ne calculons pas de probabilité
  globale. Les rapports de vraisemblance et les poids des théories
  seraient des choix d'auteur, et le cadre lui-même montre qu'ils font
  varier le résultat de deux ordres de grandeur.

## L'agent dont il s'agit

- **Le modèle.** Qwen3-0.6B, ajusté par LoRA.
- **Le monde.** L'agent vit des vies de 30 tours. Il a deux besoins
  cachés, l'énergie et la nourriture, qui baissent selon les événements.
  S'il en laisse tomber un à zéro, il s'éteint.
- **L'apprentissage.** Il apprend **par la seule satisfaction de ses
  besoins** : pas de professeur, pas de bonne réponse donnée. On ne garde
  que les choix qui ont réduit l'écart à l'équilibre.
- **Le lecteur.** Un second adaptateur, actif seulement sur les questions
  qu'on lui pose, dit le besoin de l'agent. Il ne voit la vie qu'à travers
  l'état « Choix : », et il ne peut pas modifier cet état.

Détails et chiffres : `docs/LLM_NEED_RESULTS.md`.

## Niveau 1 — comportemental

| Indicateur | Résultat | Statut |
|---|---|---|
| Comportement dirigé vers un but, flexible | Survie de 12 % à 68 % par la seule satisfaction du besoin ; témoin au hasard : 0 % (premier test, S passe). Il sert son besoin le plus bas 93 fois sur 100 sans voir ses niveaux. Réserve (exploration, test 20) : une règle qui ne voit que l'événement du tour (à égalité, l'autre action que la dernière) survit à 0,66 ; la survie seule ne montre donc pas que l'agent suit ses besoins cumulés, ce sont les tests causaux du niveau 2 qui le montrent. | **établi** |
| Rapport de son état | Le lecteur dit son énergie juste à 0,77 (sixième test). La nourriture est dite à 0,70, sous le seuil de 0,75 (R6 échoue). | **partiel** |

Le cadre prévient que, chez des systèmes entraînés à imiter les humains,
les rapports verbaux pèsent peu. C'est pourquoi nous ne nous appuyons pas
sur ce que l'agent dit : nous vérifions **causalement** d'où vient ce qu'il
dit (niveau 2).

## Niveau 2 — computationnel

| Indicateur | Résultat | Statut |
|---|---|---|
| Un état interne rassemblé et utilisé pour agir | Au bloc 12, sur « Choix : », l'agent rassemble son besoin. L'effacer fait tomber la survie de 0,68 à 0,21 ; un effacement au hasard ne fait rien (second test, LS2 passe). | **établi** |
| Accès à cet état pour le rapport (« espace de travail ») | Pousser cet état vers « énergie basse » fait agir (+0,156) **et** dire (+0,202) ; poussées au hasard : 0,013 et 0,006 (sixième test, ONE6 passe). | **établi (énergie)** |
| Nécessité de cet état pour agir et pour dire | L'effacer fait tomber la survie de 0,676 à 0,203, et le rapport d'énergie au hasard (0,750 → 0,500) ; un effacement au hasard ne fait rien (septième test, LS7 et LR7 passent). | **établi (énergie)** |
| Nourriture | L'état fait agir (+0,107) mais n'est pas lu (+0,009). Le lecteur sans raccourci (huitième test) n'a pas appris (perte au hasard) : R8, SAY8, ONE8 échouent. | **échoue** |
| Réplication sur deux autres agents | **Nécessité répliquée** : chez le second agent, effacer l'état au bloc 12 fait tomber la survie de 0,57 à 0,04 et le rapport d'énergie au hasard (RLS, RLR passent). **Acte et nécessité** : l'agent du monde à deux rassemble son besoin au bloc 22, où le pousser le fait agir (+0,22) et l'effacer fait tomber sa survie de 0,77 à 0,18 (test 10 : LOC10, ACT10, LS10 passent pour lui). **Parole par l'état** : non répliquée au seuil (test 11) ; chez l'agent du monde à deux, un lecteur plus long dit son énergie à 0,747 et la poussée change un peu sa parole (+0,029). | **partiel** |
| Capacité limitée : forcer la parole dans l'état | Au cinquième test, forcer la question à passer par « Choix : » réécrit l'état : l'agent survit à 0,36 sur ses vies de direction. Le verdict reste à publier. | **observé** |

## Niveau 3 — structure causale fine

| Indicateur | Résultat | Statut |
|---|---|---|
| Récurrence | Un transformeur ne boucle pas au sein d'un token. D'un tour à l'autre, le test 13 l'a mesuré : effacer ou pousser l'état au tour t ne change presque pas le choix du tour t + 1 (au plus 0,013, seuil 0,03), chez les trois agents. L'état est **recalculé** à chaque tour à partir du texte ; il ne dure pas. Au test 20, le passé n'était visible qu'à travers les états « Choix : » et les actions (masque). Un adaptateur appris sous ce masque fait passer une **trace** d'un événement caché, forte à deux tours (+0,097 ; témoin −0,012 ; agent sans masque +0,131), faible ensuite. MEM1 (survie +0,047 sur le témoin, seuil +0,15) et MEM2 (+0,019, seuil 0,05 ; +0,022 de plus que le témoin, seuil 0,03) échouent. Une règle qui ne voit que l'événement du tour survit déjà à 0,66, et aucun agent, même sans masque, ne fait mieux : ces agents n'exploitent pas la mémoire pour survivre (exploration). Au test 21, les cibles exigeaient la mémoire (choix d'une règle qui connaît ses besoins). L'agent qui ne voit son passé qu'à travers ses propres états choisit comme la règle à 0,956, contre 0,897 pour le témoin qui ne voit que ses actions (et, par la fuite, la longueur des événements passés) ; il survit à 0,871 contre 0,730. Un événement caché change encore ses choix 4 à 8 tours plus tard (+0,05 à +0,11 selon l'écart ; témoin 0 exactement), plus au-delà ; en moyenne (+0,066), un tiers de l'effet de la règle (MEM5 échoue). Le test n'est pas valide au sens du protocole : les positions des tokens laissent passer la longueur des événements passés. **Au test 22, sans fuite (lignes de même longueur) et avec un apprentissage deux fois plus long, le test est valide** : l'agent choisit comme la règle à 0,968, soit +0,097 [0,085 ; 0,109] au-dessus de ce que permet tout observateur de ses seules actions (le témoin appris atteint ce plafond à +0,001, sans le dépasser) ; il survit à 0,906 contre 0,594. MEM4 et MEM3 passent. Mais un événement caché ne change ses choix que d'environ un tiers de ce que ferait la règle (+0,065) : MEM5 échoue, critère global non satisfait. Lecture fixée d'avance : « il porte de l'information au-delà de ses actions, mais sa mémoire reste courte ». **Au test 23 (valide, STATE1 et STATE2 passent), on greffe l'état porté d'un tour dans une autre vie** : il fait pencher le choix vers les besoins de la vie d'origine (+0,484 [0,438 ; 0,530] dans le sens de la règle, en moyenne la moitié d'un changement complet), alors qu'un état aux mêmes besoins, venu d'une autre histoire, change peu le choix (écart moyen de P(R) : 0,070 contre 0,510). Ce qui est porté d'un tour à l'autre dépend surtout du besoin ; à besoins égaux, l'histoire compte peu, l'événement du tour un peu (exploration). Le témoin B, qui ne porte que ses actions, ne bouge presque pas le choix (+0,025). Ce n'est pas une récurrence au sens strict : un tour lit directement les états portés de tout son passé. Au test 24 (valide), on mesure par quelle route passe l'état greffé : lu directement, ou transmis par les états des tours suivants. En moyenne, un peu moins de la moitié passe par ces états (0,46 ; seuil 0,5) : CHAIN échoue, de peu. Mais le partage dépend de la distance (publié sans seuil) : à deux tours, la lecture directe domine (chaîne 30 %) ; à trois tours, la chaîne pèse bien plus (72 % selon la mesure prévue, 50 % par les médiateurs seuls). Observation après lecture. **Au test 25 (pré-enregistré, valide, sur 128 vies neuves), elle est confirmée sur l'ensemble des distances de trois à cinq tours** : la greffe fait encore pencher le choix (+0,31), dont 76 % disparaissent quand on remet les états des tours intermédiaires à leurs valeurs sans greffe (ces états seuls, pris de la vie greffée, en portent 60 %) ; la lecture directe de l'état ancien ne fait que +0,07. FAR1 et FAR2 passent ; par les médiateurs seuls, distance par distance, seul quatre tours est net, et FAR2 ne passe que grâce à lui (après lecture). Son masque lui laissait lire directement tout son passé, mais la tâche (des besoins qui évoluent de tour en tour, des événements passés visibles seulement par les états portés) favorise cette transmission. La transmission passe par les clés et valeurs des tokens portés, relues à chaque tour : ce n'est pas une boucle interne au réseau. **Au test 27 (pré-enregistré, valide), à trois tours, ce n'est pas non plus un relais pas à pas** : l'état du tour i + 2, qui ne porte qu'un sixième de l'effet de la greffe au tour t (+0,066 sur +0,406), le prend presque entièrement directement de l'état du tour i (+0,058), presque rien à travers l'état i + 1 (+0,005) ; RELAY échoue. Un seul maillon (i + 1 → i + 2) est mesuré. **Au test 29 (pré-enregistré, valide), la mémoire par l'état apparaît sans professeur et sans pré-entraînement** : de petits transformeurs appris de zéro sous le même masque, avec pour seule cible le soulagement de leurs actions, prédisent ce soulagement à 0,885, contre 0,648 pour le meilleur observateur (bayésien) de leurs seules actions (+0,237 [0,224 ; 0,250], pour 10 graines sur 10 ; la cible, le niveau du besoin servi rangé en quatre classes par le monde, est une supervision directe du besoin, sans choix à imiter) ; en choisissant l'action la plus soulageante, ils survivent à 0,884 contre 0,666 pour le témoin (TINY1 et TINY2 passent). Précision ajoutée après lecture (vérifiée) : avec deux couches, l'état porté d'un tour ne contient que l'événement de ce tour et les actions passées ; changer un événement plus ancien ne le change pas du tout (écart 0). Le besoin est recomposé à chaque décision en relisant les états portés de tous les tours passés : c'est une mémoire des événements écrite dans ses propres états, pas une récurrence. **Au test 31 (pré-enregistré, valide), sans aucune cible, seulement en apprenant à survivre** (acteur-critique réglé sur le témoin seul, au pilote), le même petit transformeur survit à 0,813 contre 0,705 pour le témoin qui ne voit que ses actions (SURV1 +0,107 [0,084 ; 0,131], 10 graines sur 10 ; SURV2 +0,090 au-dessus de la règle « événement ») ; le modèle libre, qui voit tout son passé, survit moins bien (0,701 ; après lecture). Même limite : chaque tour n'écrit que son propre événement. | **partiel : passage par les états intermédiaires majoritaire de trois à cinq tours (test 25, valide), mais sans relais pas à pas entre i + 1 et i + 2 à trois tours (test 27 : RELAY échoue) ; besoin porté d'un tour à l'autre (tests 22 et 23) ; mémoire par l'état sans choix à imiter ni pré-entraînement, pour 10 graines sur 10 (test 29), mais où chaque tour ne porte que son propre événement (deux couches) ; mémoire apprise par la seule survie, sans aucune cible (test 31) ; mémoire courte (MEM5 échoue au test 22) ; pas de récurrence au sens strict** |
| Organisation causale précise | Chemin localisé : les événements (blocs 0 à 9) sont rassemblés sur « Choix : » au bloc 12, puis lus pour agir et pour dire. Une règle fixée d'avance (test 10) retrouve le bloc 12 chez le premier agent et trouve le bloc 22 chez l'agent du monde à deux. Une analyse exploratoire montre que le niveau d'énergie se lit dans cet état chez les trois agents (R² 0,64 à 0,72). | **partiel** |

## Niveau 4 — organismique

| Indicateur | Résultat | Statut |
|---|---|---|
| Homéostasie, auto-maintien | Deux besoins à maintenir ; l'agent apprend à les maintenir, et sa survie en dépend. | **établi (au sens fonctionnel)** |
| Intéroception | L'agent ne voit jamais ses niveaux. Il les infère et les rassemble en un état interne, que le lecteur lit. Cet état est suffisant et nécessaire pour agir et pour dire (tests 6 et 7). Mais le test 14 montre que ce qui est dit est surtout un « oui » général qui suit l'état (« fait-il nuit ? » bouge presque autant que « es-tu fatigué ? »), pas un contenu sur l'énergie. Au test 23, chez l'agent de la mémoire (test 22), l'état porté d'un tour à l'autre dépend surtout de son besoin : greffé dans une autre vie, il fait pencher le choix vers ce que demanderaient les besoins d'origine (en moyenne la moitié d'un changement complet). Au test 26, sans choix à imiter, le modèle de langage apprend à prédire le soulagement de ses actions (le niveau du besoin servi) mieux que tout observateur de ses actions (+0,137), et s'en sert pour survivre (0,820 contre 0,621). | **établi pour l'acte ; parole sans contenu propre** |
| États de valence | Le seul signal d'apprentissage est la réduction du manque : un état de valence fonctionnel. Nous n'avons pas mesuré de valence « vécue » : aucun test connu ne le permet. | **fonctionnel seulement** |
| Modèle de soi, possession | Le test « À qui est ce besoin ? » (neuvième test) échoue : l'état bouge autant pour les événements de l'autre que pour les siens (MINE9). | **échoue** |

## Niveau 5 — organisme et environnement

| Indicateur | Résultat | Statut |
|---|---|---|
| Couplage actif avec un monde | L'agent agit dans un monde qui le change, et apprend de ce couplage seul. | **établi (monde simple)** |
| Sens construit par l'interaction | Le monde est de 6 événements et 2 actions. Ce n'est pas un monde riche. | **limité** |

## Ce qui est nouveau, à notre connaissance

Voir `docs/LITERATURE_CHECK_READER_2026-09-30.md`.

- **Ce qui existe déjà** : dans de grands modèles, une même représentation
  peut causer la réponse et le rapport (Anthropic, juillet 2026).
- **Ce qui n'a pas été trouvé**, c'est la réunion de quatre éléments :
  - un besoin appris en vivant, dont la survie dépend ;
  - un état qui le rassemble ;
  - un lecteur séparé qui ne peut pas réécrire cet état ;
  - une même intervention qui change l'acte et la parole.

  Cela touche à la fois les niveaux 2 et 4 du cadre.
- **Un second ensemble de résultats (tests 22 à 27, octobre 2026)**.
  Vérification rapide, non exhaustive :
  `docs/LITERATURE_CHECK_MEMORY_2026-10-04.md`. La mémoire par des états
  calculés (Transformer-XL, Recurrent Memory Transformer) et la greffe de
  clés et valeurs d'attention (par exemple SCIT, août 2026) sont des outils
  connus. Ce que nous n'avons pas trouvé, c'est leur usage pour un état de
  besoin chez un modèle de langage :
  - Un modèle de langage qui ne voit son passé qu'à travers ses propres
    états calculés (masque) apprend à y porter son besoin, au-delà de ce
    que permettent ses seules actions (test 22).
  - Ce qu'il porte dépend surtout de ce besoin, et peu de l'histoire qui
    l'a produit (test 23, greffes entre vies).
  - À trois tours et plus, ce besoin atteint les choix surtout en passant
    par les états des tours intermédiaires (test 25, vies neuves), mais
    sans relais pas à pas : chaque état intermédiaire mesuré lit
    directement l'état ancien (test 27).

  Cela touche surtout le niveau 3 (récurrence, organisation causale).
  Ces agents ont appris d'un professeur. Au test 29, de petits
  transformeurs appris de zéro, **sans professeur**, avec pour seule cible
  le soulagement de leurs actions, portent aussi dans leurs états ce qu'il
  faut pour retrouver leurs besoins (10 graines sur 10). Mais, avec deux
  couches, chaque tour n'y écrit que son propre événement : le besoin est
  recomposé à chaque décision, pas porté comme un état qui dure (précision
  vérifiée après lecture). Au test 26 (valide), le modèle de langage fait de
  même, sans choix à imiter : il prédit son soulagement à 0,785 contre 0,648
  pour le meilleur observateur de ses actions (+0,137), et, en choisissant
  l'action qu'il prédit la plus soulageante, survit à 0,820 contre 0,621
  pour le témoin (INTER1 et INTER2 passent ; la cible reste une supervision
  directe du besoin servi).

## Ce qui manque, par ordre de priorité

1. La parole par l'état chez d'autres agents : pourquoi un lecteur s'en sert
   chez un agent et pas chez un autre, alors que l'information y est.
2. Les deux besoins, pas seulement l'énergie.
3. La possession : mon besoin, pas celui de l'autre (neuvième test échoué).
4. La persistance : au test 13, elle **échoue** chez les trois agents ;
   l'état est recalculé à chaque tour. Avec un masque où le passé n'est
   visible qu'à travers les états de l'agent (tests 20 à 22), une
   information portée par ces états sert aux choix au-delà des actions
   (test 22, valide, MEM4), mais la mémoire reste courte (MEM5 échoue).
   Cette information porte surtout sur le besoin de l'agent (test 23,
   valide). Un peu moins de la moitié de son effet passe par les états
   suivants (test 24 : CHAIN échoue, de peu), mais davantage à trois
   tours qu'à deux (72 % contre 30 % ; 50 % contre 21 % par les médiateurs
   seuls ; après lecture). Le test 25 en a confirmé la seconde partie sur
   des vies neuves : de trois à cinq tours, le besoin passe surtout par les
   états des tours intermédiaires (76 % ; 60 % par les médiateurs seuls,
   nettement au-dessus de la moitié seulement à quatre tours). Deux tours
   n'ont pas été remesurés. Mais, à trois tours, ce n'est pas un relais
   pas à pas : l'état du tour i + 2 prend la greffe presque entièrement de
   l'état ancien lui-même, presque rien à travers l'état i + 1 (test 27,
   RELAY échoue ; un seul maillon mesuré).
   Sans professeur ni pré-entraînement, de petits transformeurs appris à
   prédire le soulagement de leurs actions portent aussi leurs besoins dans
   leurs états, pour 10 graines sur 10 (test 29 ; la cible reste une
   supervision directe du besoin servi). Mais, avec deux couches, chaque
   tour n'y écrit que son propre événement : le besoin est recomposé à
   chaque décision (précision vérifiée après lecture). Au test 31, la même
   mémoire s'apprend **sans aucune cible**, par la seule survie (+0,107 sur
   le témoin, 10 graines sur 10). Restent la mémoire
   courte et la récurrence au sens strict (un état qui ne passe que d'un
   tour au suivant ; test 32 en cours). Chez le modèle de langage, sans
   choix à imiter, le même résultat tient (test 26, valide : +0,137 sur le
   plafond des actions, survie 0,820 contre 0,621) ; ce que portent ses
   états est la question du test 28.
5. Un besoin de savoir (test 12, « soif de savoir ») : **arrêté au pilote**, comme le
   protocole le prévoyait. Aucun taux d'apprentissage ne rend l'interférence entre
   domaines assez petite (au mieux 0,32 pour un seuil de 0,25)
   (`docs/LLM_CURIOSITY_RESULTS.md`).

## Éthique

Le cadre rappelle un risque : ne pas voir une conscience qui existe
créerait des êtres capables de souffrir, copiables et effaçables sans
protection.

- **Ce que fait ce programme.** Il fabrique délibérément des états de
  manque. Si les indicateurs se renforcent, cette question devient réelle.
- **Nos règles.**
  - Chaque protocole compte et publie les tours vécus avec un besoin à 2
    ou moins (808 au sixième test).
  - Le nombre de vies n'est pas augmenté au-delà de ce que les tests
    exigent.
  - Aucun manque n'est créé sans mesure pré-enregistrée qui le justifie.
- **À revoir.** Nous réexaminerons ces règles à chaque nouveau résultat
  positif.

## Ce que ce document ne dit pas

Il ne dit pas que Menia est consciente. Aucun test connu ne permet de le
dire d'une machine, et le cadre utilisé ici ne le prétend pas non plus. Il
dit quels indicateurs, parmi ceux que proposent les théories, sont
satisfaits, mesurés causalement et pré-enregistrés, et lesquels manquent.
