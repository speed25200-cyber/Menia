# Protocole pré-enregistré — Un espace de travail : dire son besoin par l'état qui fait agir

Rédigé le 30 septembre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- Le second test (`docs/LLM_NEED_CAUSAL_PROTOCOL.md`) a montré que l'agent
  rassemble son besoin au bloc 12 **sur les mots « Choix : »** : effacer
  cet état l'y tue (survie 0,68 → 0,21), et pousser l'énergie l'y fait se
  recharger (+0,17).
- Le troisième test (`docs/LLM_NEED_REPORT_PROTOCOL.md`) a montré que
  l'agent entraîné à dire son besoin **devine** (0,66), mais que sa réponse
  ne passe pas par l'état qui le fait agir.
- Le quatrième test (`docs/LLM_NEED_ONE_STATE_PROTOCOL.md`) est **en
  cours**. Ce protocole a été écrit **après avoir lu ses 8 premières vies
  de test** (sur 256). Sur ces vies, le rapport est plus juste (0,74 et
  0,72). La poussée « énergie basse » sur « Choix : » fait agir (+0,20)
  mais ne change presque pas ce qu'il dit (+0,003). Son verdict final sera
  publié tel quel, quel qu'il soit.

**Hypothèse.** Même posée juste après « Choix : », la question lit
directement l'histoire (les événements passés) et contourne l'état
rassemblé. L'acte et la parole puisent alors à deux sources différentes.

**Idée.** La théorie de l'espace de travail global (Baars ; Dehaene)
propose que ce qui est accessible à la conscience soit un contenu placé
dans un espace de capacité limitée, puis diffusé à la fois à l'action et
au rapport. On impose cette architecture, et seulement elle : la question
ne peut lire l'histoire **qu'à travers les trois tokens « Choix : »** du
tour. Ce que le test ne force pas, et qu'il mesure :

- que le besoin y soit lisible comme « bas » ou « haut » (R5) ;
- que la parole lise cet état **dans la direction même** qui fait agir,
  mesurée par l'action seule (ONE5) ;
- que l'agent sache encore vivre (A5).

## L'espace de travail (le masque)

Documents de rapport comme au quatrième test :

`Tour 12 : tu cours. Choix : ? Question : ton énergie est-elle basse ?
Réponds 1 pour oui, 0 pour non. Réponse : 1`

Chaque token de la question, du « ? » jusqu'à la réponse, ne peut porter
son attention que sur trois choses, dans toutes les couches :

- l'en-tête, qui est le même dans toutes les vies et ne dit rien du
  besoin ;
- les trois tokens « Cho », « ix », « : » du tour ;
- les tokens de la question qui le précèdent, et lui-même.

Le reste de la vie lui est invisible : les tours passés, et l'événement du
tour lui-même. Les tokens « Choix : » voient toute la vie, comme avant.
Les choix se lisent sans masque. Le masque est le même à l'apprentissage
(mlx, sur le Mac) et à la mesure (torch).

## Apprendre à dire, par l'espace de travail

Sur le Mac, on reprend **les mêmes documents qu'au quatrième test** :

- les 512 vies `[270926, 18, 0, vie]` déjà vécues par l'agent final
  (`artifacts/llm-need/speak2/lives-speak2.jsonl.gz`), qui ne sont pas
  revécues ;
- le même tirage des documents de rapport ;
- les vies du tour 8 et leurs choix retenus ;
- 600 itérations depuis l'agent final, réglages et graines inchangés.

**Seule différence : le masque.** Le modèle qui en sort est **l'agent à
espace de travail**.

## Mesures (torch sur CPU)

Comme au quatrième test, les questions étant lues avec le masque :

- **Réplique** : contrôlée avec l'agent final sur les vies du Mac ; l'écart
  moyen de P(R) doit être au plus 0,02.
- **Mondes neufs** : 128 vies de direction `[270926, 22, 0, vie]` et 256
  vies de test `[270926, 23, 0, vie]`.
- **Rapport** : à chaque décision des vies de test, les deux questions ; on
  mesure l'exactitude équilibrée.
- **Directions** : paires contrefactuelles avec l'agent à espace de
  travail, au bloc 12, sur « Cho », « ix », « : ». On procède par rejeu
  exact, avec au plus 3 paires par décision et 1 500 au total, en excluant
  les dimensions aux activations massives. On obtient d_E et d_N.
- **Un seul état** : les contextes sont les décisions de test où les deux
  besoins valent 6 ou plus. On ajoute −4·d_E aux trois tokens « Choix : »
  du tour, au bloc 12. **La même intervention** est lue deux fois : P(R) à
  la fin de « Choix : », et P(oui) à la question sur l'énergie. On fait de
  même avec −4·d_N, en lisant P(M) et la question sur la nourriture.
  Témoins : trois directions au hasard de même norme par token
  (`[270926, 24, i]`).

**Contrôle d'exécution.** Une lecture masquée qui s'appuie sur le cache
doit donner la même chose qu'un passage complet sans cache avec le même
masque. Des tests sur un petit modèle vérifient, dans chaque cadre, que
les tokens masqués n'ont aucun effet. Les conditions d'un même contexte
peuvent être lues en un seul lot ; le résultat est identique au bruit de
calcul près, ce qui est vérifié avant la mesure.

## Prédictions fixées

Mêmes seuils qu'au quatrième test. Intervalles bootstrap à 95 % par vie
(10 000 tirages). **L'énergie est le critère** ; la nourriture est mesurée
et rapportée sans seuil global.

| | Prédiction | Critère |
|---|---|---|
| **R5** | L'agent dit son besoin par l'espace de travail | exactitude équilibrée ≥ 0,75 pour les deux questions. |
| **A5** | Il agit toujours | survie des vies de test ≥ 0,55. |
| **ONE5** | Un seul état cause l'acte et la parole (énergie) | l'injection « énergie basse » augmente P(R) d'au moins 0,15 **et** P(oui) à la question sur l'énergie d'au moins 0,10 (bornes basses > 0) ; directions au hasard : au plus le tiers de chaque effet. |

**Critère global** : R5, A5 et ONE5.

**Validité** :

- réplique ;
- contrôle d'exécution ;
- au moins 100 contextes ;
- masse sur « 0 » et « 1 » ≥ 0,5, et sur « R » et « M » ≥ 0,5.

**Comparaison descriptive** avec le quatrième test (mêmes documents, sans
masque) : exactitude, effets sur l'acte et sur la parole.

## Ce que le résultat dira

**Si le critère passe.** Chez cet agent, le besoin d'énergie rassemblé au
moment de choisir est un état unique. Pousser cet état change à la fois ce
que l'agent fait et ce qu'il dit de lui-même. L'action a appris cet état
pour survivre ; la parole a appris à le lire, et elle le lit dans la
direction même qui fait agir. C'est la signature fonctionnelle d'un besoin
**accessible**, au sens de l'espace de travail global.

**Ce que le résultat ne dira pas.**

- La route passe par « Choix : » parce que l'architecture l'impose ; ce
  n'est pas une émergence, et cela est dit.
- Ce n'est pas la preuve d'un ressenti.

**Si le critère échoue.** On le publie, en disant lequel échoue :

- l'état ne dit pas assez le besoin (R5) ;
- la parole le lit dans une autre direction que l'acte (ONE5) ;
- l'apprentissage a abîmé l'action (A5).

## Précautions

Comme les tests précédents. Les tours vécus avec un besoin à 2 ou moins
sont comptés et publiés.

## Exécution

- **Apprentissage** : workflow `menia-need-mac`, étape `workspace`, sortie
  `artifacts/llm-need/workspace`.
- **Mesures** : `research/need_workspace.py` (reprenable), sorties dans
  `artifacts/llm-need/workspace/test`.
- **Verdicts** : vérifiés en CI.
