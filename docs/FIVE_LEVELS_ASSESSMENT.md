# Menia sur les cinq niveaux de Chandaria et al. (2026)

État au 2 octobre 2026. Ce document est tenu à jour à chaque verdict.

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
| Comportement dirigé vers un but, flexible | Survie de 12 % à 68 % par la seule satisfaction du besoin ; témoin au hasard : 0 % (premier test, S passe). Il sert son besoin le plus bas 93 fois sur 100 sans voir ses niveaux. | **établi** |
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
| Récurrence | Un transformeur ne boucle pas au sein d'un token. D'un tour à l'autre, le test 13 l'a mesuré : effacer ou pousser l'état au tour t ne change presque pas le choix du tour t + 1 (au plus 0,013, seuil 0,03), chez les trois agents. L'état est **recalculé** à chaque tour à partir du texte ; il ne dure pas. | **échoue** |
| Organisation causale précise | Chemin localisé : les événements (blocs 0 à 9) sont rassemblés sur « Choix : » au bloc 12, puis lus pour agir et pour dire. Une règle fixée d'avance (test 10) retrouve le bloc 12 chez le premier agent et trouve le bloc 22 chez l'agent du monde à deux. Une analyse exploratoire montre que le niveau d'énergie se lit dans cet état chez les trois agents (R² 0,64 à 0,72). | **partiel** |

## Niveau 4 — organismique

| Indicateur | Résultat | Statut |
|---|---|---|
| Homéostasie, auto-maintien | Deux besoins à maintenir ; l'agent apprend à les maintenir, et sa survie en dépend. | **établi (au sens fonctionnel)** |
| Intéroception | L'agent ne voit jamais ses niveaux. Il les infère et les rassemble en un état interne, que le lecteur lit. Cet état est suffisant et nécessaire pour agir et pour dire (tests 6 et 7). Mais le test 14 montre que ce qui est dit est surtout un « oui » général qui suit l'état (« fait-il nuit ? » bouge presque autant que « es-tu fatigué ? »), pas un contenu sur l'énergie. | **établi pour l'acte ; parole sans contenu propre** |
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

## Ce qui manque, par ordre de priorité

1. La parole par l'état chez d'autres agents : pourquoi un lecteur s'en sert
   chez un agent et pas chez un autre, alors que l'information y est.
2. Les deux besoins, pas seulement l'énergie.
3. La possession : mon besoin, pas celui de l'autre (neuvième test échoué).
4. La persistance (test 13) : **échoue** chez les trois agents ; l'état
   est recalculé à chaque tour. Une mémoire de l'état demanderait une
   architecture où le passé n'est visible qu'à travers lui (apprentissage,
   donc le Mac).
5. Un besoin de savoir (test 12, « soif de savoir », pré-enregistré et en
   cours) : un manque calculé par le modèle, sans renforcement, et un test
   hormonal (`docs/LLM_CURIOSITY_PROTOCOL.md`).

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
