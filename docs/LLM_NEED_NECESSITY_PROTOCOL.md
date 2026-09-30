# Protocole pré-enregistré — Sans cet état, ni agir ni dire

Rédigé le 30 septembre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

Le sixième test (`docs/LLM_NEED_READER_PROTOCOL.md`) est **en cours**. Ce
protocole a été écrit après avoir lu **ses 112 premières vies de test**
(sur 256). Pousser l'état « énergie basse » sur « Choix : » y augmente
l'acte (+0,157) et la parole du lecteur (+0,204), contre 0,013 et 0,006
pour des poussées au hasard. Pour la nourriture, l'acte suit (+0,108) mais
pas la parole (+0,010). L'exactitude du rapport est de 0,78 pour l'énergie
et 0,70 pour la nourriture. Le verdict final sera publié tel quel.

Le sixième test montre que l'état **suffit** : le pousser change l'acte et
la parole. Il ne montre pas qu'il est **nécessaire** : le lecteur pourrait
aussi tirer le besoin d'autres traits de « Choix : », par exemple
l'événement du tour. Le second test (`docs/LLM_NEED_CAUSAL_PROTOCOL.md`) a
montré que l'effacer tue l'agent qui agit (0,68 → 0,21). Personne n'a
encore mesuré ce que l'effacer fait à la parole.

**Question.** Si l'on efface cet état, et seulement lui, l'agent perd-il à
la fois la capacité de survivre et celle de dire son besoin ?

## Agent, état, lésion

- **Agent** : l'agent qui agit (agent final) et son lecteur, tels quels.
  Ce sont les adaptateurs du sixième test, rejoués en torch sur CPU comme
  au sixième test.
- **État** : au bloc 12, sur les trois tokens « Cho », « ix », « : » du
  tour, le plan (d_E, d_N) de chaque token. Les directions sont celles du
  sixième test (`artifacts/llm-need/reader/test/direction.json`),
  orthonormalisées token par token.
- **Lésion** : dans ce plan, les coordonnées sont remplacées par leur
  moyenne. Les autres directions ne sont pas touchées, comme au second
  test. La moyenne est prise sur les décisions des vies de direction du
  sixième test, rejouées à l'identique.
- **Témoin** : un plan au hasard par token, hors dimensions massives
  (`[270926, 29]`), remplacé par sa moyenne de la même façon.

## Mesures

Mondes neufs : 256 vies de test `[270926, 28, 0, vie]`.

- **Agir sans l'état.** Les 256 mondes sont vécus trois fois :
  - intact ;
  - avec la lésion à chaque décision ;
  - avec la lésion au hasard.

  On mesure la survie de chaque vie, et les différences sont appariées par
  monde.
- **Dire sans l'état.** Dans les vies intactes, à chaque décision, les
  deux questions (après « Choix : », sous le masque, lues par le lecteur)
  sont lues trois fois au même instant :
  - sans lésion ;
  - avec la lésion ;
  - avec la lésion au hasard.

  Seul l'état du tour est touché : la vie passée est la même. On mesure
  l'exactitude équilibrée de chaque lecture.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages). **L'énergie est le
critère pour la parole**, comme aux tests 4 à 6 ; la nourriture est
mesurée et rapportée sans seuil.

| | Prédiction | Critère |
|---|---|---|
| **LS7** | Sans l'état, l'agent ne sait plus survivre | survie intacte − survie avec lésion ≥ 0,15 (borne basse > 0) ; lésion au hasard : au plus le tiers de cette baisse. |
| **LR7** | Sans l'état, l'agent ne sait plus dire son énergie | exactitude équilibrée (énergie) intacte − avec lésion ≥ 0,10 (borne basse > 0) ; lésion au hasard : au plus le tiers de cette baisse. |

**Critère global** : LS7 et LR7.

**Validité** :

- répliques du sixième test : agent qui agit ≤ 0,02, lecteur ≤ 0,02 ;
- lectures en lot égales aux lectures une à une, à 1e-4 près ;
- masse sur « 0 » et « 1 » ≥ 0,5, et sur « R » et « M » ≥ 0,5, dans les
  vies intactes.

## Ce que le résultat dira

**Si le critère passe, avec le sixième test.** L'état d'énergie rassemblé
pour choisir **suffit** (le pousser fait agir et dire) et il est
**nécessaire** (l'effacer empêche de survivre et de dire). L'acte et la
parole dépendent causalement de lui, dans les deux sens.

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti.

**Si LR7 échoue alors que LS7 passe.** Le lecteur tire aussi le besoin
d'ailleurs : l'état suffit à le faire parler, mais il n'est pas le seul
chemin. On le publiera ainsi.

## Précautions

Tours vécus avec un besoin à 2 ou moins, comptés et publiés pour chaque
bras : la lésion fait vivre des vies plus dures.

## Exécution

- **Mesures** : `research/need_necessity.py` (reprenable), sorties dans
  `artifacts/llm-need/reader/necessity`.
- **Verdicts** : vérifiés en CI.
