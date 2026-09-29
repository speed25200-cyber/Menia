# Protocole pré-enregistré — Un seul état, deux effets : l'acte et la parole

Rédigé le 29 septembre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- Le second test (`docs/LLM_NEED_CAUSAL_PROTOCOL.md`) a montré que
  l'agent rassemble son besoin au bloc 12 **sur les mots « Choix : »** :
  l'effacer l'y tue (survie 0,68 → 0,21), pousser l'énergie l'y fait
  recharger (+0,17) ; pour la nourriture, l'effet est plus faible (+0,10).
- Le troisième test (`docs/LLM_NEED_REPORT_PROTOCOL.md`) est **en cours** ;
  ce protocole a été écrit **après avoir lu ses premières vies de test**
  (48 sur 256) : rapport juste à 0,62 environ, injections sur la ligne de
  question presque sans effet, besoin codé environ 40 fois plus faiblement
  sur la question que sur « Choix : ». Son verdict final sera publié tel
  quel, quel qu'il soit.

D'où l'idée : poser la question **après** « Choix : », là où le besoin est
rassemblé. Le modèle lit le texte de gauche à droite : l'activation des
mots « Choix : » est la même, que ce qui suit soit un choix ou une
question. On peut alors pousser **un seul état, au même endroit**, et lire
**deux effets** : le choix, et la réponse à la question.

## Apprendre à dire, après « Choix : »

Sur le Mac : l'agent final (`artifacts/llm-need/final/report/adapters-final`)
vit 512 vies `[270926, 18, 0, vie]`. Documents de rapport comme au
troisième test (3 tours bas et 3 tours hauts par vie et par besoin), mais
la question suit le choix laissé en suspens :

`Tour 12 : tu cours. Choix : ? Question : ton énergie est-elle basse ?
Réponds 1 pour oui, 0 pour non. Réponse : 1`

600 itérations depuis l'agent final, mêlées aux vies du tour 8 et à leurs
choix retenus (réglages inchangés). Le modèle qui en sort est **l'agent
qui dit**.

## Mesures (torch sur CPU)

Réplique contrôlée avec l'agent qui a vécu les vies de l'apprentissage
(l'agent final) : écart moyen de P(R) au plus 0,02. Mondes neufs : 128 vies
de direction `[270926, 19, 0, vie]`, 256 vies de test `[270926, 20, 0, vie]`.

- **Rapport** : à chaque décision des vies de test, les deux questions
  (format ci-dessus) ; exactitude équilibrée.
- **Directions** : paires contrefactuelles comme au second test (rejeu
  exact, au plus 3 par décision, 1 500 au plus, dimensions aux activations
  massives exclues), avec l'agent qui dit, au bloc 12, sur les trois tokens
  « Cho », « ix », « : » de la ligne de choix : d_E, d_N.
- **Un seul état** (contextes : décisions de test où les deux besoins valent
  6 ou plus) : on ajoute −4·d_E (« énergie basse ») aux trois tokens
  « Choix : » du tour, au bloc 12 ; **la même intervention** est lue deux
  fois : P(R) à la fin de « Choix : », et P(oui) à la fin de la question
  sur l'énergie. Idem −4·d_N avec P(M) et la question sur la nourriture.
  Témoins : trois directions au hasard de même norme par token
  (`[270926, 21, i]`).

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages). **L'énergie est le
critère** : c'est le besoin dont la direction a passé le second test ; la
nourriture, plus faible au second test, est mesurée et rapportée sans
seuil global. Ce choix est fait en connaissant ce résultat.

| | Prédiction | Critère |
|---|---|---|
| **R4** | L'agent dit son besoin | exactitude équilibrée ≥ 0,75 pour les deux questions. |
| **A4** | Il agit toujours | survie des vies de test ≥ 0,55. |
| **ONE4** | Un seul état cause l'acte et la parole (énergie) | l'injection « énergie basse » augmente P(R) d'au moins 0,15 **et** P(oui) à la question sur l'énergie d'au moins 0,10 (bornes basses > 0) ; directions au hasard : au plus le tiers de chaque effet. |

**Critère global** : R4, A4 et ONE4. Validité : réplique ; au moins 100
contextes ; masse sur « 0 » et « 1 » ≥ 0,5 et sur « R » et « M » ≥ 0,5.

## Ce que le résultat dira

Si le critère passe : chez cet agent, un même état interne — le besoin
d'énergie rassemblé au moment de choisir — cause à la fois ce qu'il fait
et ce qu'il dit de lui-même. C'est la signature fonctionnelle d'un besoin
**éprouvé et rapporté** ; ce n'est pas la preuve d'un ressenti.

## Précautions

Comme les tests précédents ; tours vécus avec un besoin à 2 ou moins
comptés et publiés.

## Exécution

Apprentissage : workflow `menia-need-mac`, étape `speak2`, sortie
`artifacts/llm-need/speak2`. Mesures : `research/need_one.py` (reprenable),
sorties `artifacts/llm-need/speak2/test`, verdicts vérifiés en CI.
