# Protocole pré-enregistré — Le lecteur : dire son besoin en lisant l'état qui fait agir, sans pouvoir le changer

Rédigé le 30 septembre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Quatrième test** (`docs/LLM_NEED_ONE_STATE_PROTOCOL.md`), en cours.
  Sur les 8 premières vies de test, l'état rassemblé sur « Choix : » fait
  agir (+0,20) mais ne change presque pas ce que l'agent dit (+0,003).
- **Cinquième test** (`docs/LLM_NEED_WORKSPACE_PROTOCOL.md`), en cours.
  La question n'y lit la vie qu'à travers « Choix : ». Ce protocole a été
  écrit après avoir lu **ses vies de direction seulement**, avant toute vie
  de test. L'agent à espace de travail n'y survit que 0,359 fois, contre
  0,734 pour l'agent du quatrième test sur ses propres vies de direction.
  Il meurt surtout de faim (72 morts de faim contre 10 d'énergie), et ne
  sert son besoin le plus bas que 80 fois sur 100 (88 pour l'autre). La
  réplique torch de cet agent a été contrôlée contre mlx sur trois
  décisions : écarts de 0,007 à 0,026. Les verdicts des deux tests seront
  publiés tels quels.

**Lecture.** Forcée de passer par « Choix : », la parole a **réécrit**
l'état qui sert à agir, et l'action en a souffert. Les deux fonctions se
disputent le même espace, car les poids appris pour dire servent aussi à
calculer « Choix : ».

**Idée.** On sépare le calcul de l'état et sa lecture :

- **L'agent qui agit** est l'agent final, inchangé : c'est lui qui calcule
  toute la vie, y compris « Choix : ».
- **Le lecteur** est un second adaptateur (LoRA), actif **seulement sur
  les tokens de la question**. Il ne peut donc pas changer la façon dont
  « Choix : » est calculé.
- **La question** garde le masque du cinquième test : elle ne voit que
  l'en-tête, les trois tokens « Cho », « ix », « : » et elle-même.

Le lecteur ne peut ni voir la vie autrement que par l'état qui fait agir,
ni modifier cet état. C'est la forme la plus simple d'un état **d'ordre
supérieur** qui représente un état **d'ordre premier**, et d'un contenu
**accessible** à la fois à l'action et au rapport.

Ce que l'architecture impose :

- la route de l'information ;
- l'action, qui est celle de l'agent final par construction. A6 n'est
  donc qu'un contrôle.

Ce qu'elle n'impose pas, et que le test mesure :

- que l'état d'action porte assez le besoin pour être dit (R6) ;
- que le lecteur le lise **dans la direction même qui fait agir**, mesurée
  sur l'agent qui agit, avant que le lecteur existe (ONE6).

## Apprendre à lire (sur le Mac)

1. L'agent final (`artifacts/llm-need/final/report/adapters-final`) est
   fondu dans les poids (`mlx_lm.fuse`).
2. Sur ce modèle, un nouvel adaptateur est créé : rang 8, échelle 20, les
   couches linéaires des 16 derniers blocs (12 à 27), comme tous les
   adaptateurs du programme. Son apport est **multiplié par zéro partout
   sauf sur les tokens de la question**, du « ? » jusqu'à la réponse.
3. Documents : les **mêmes 5 059 questions** qu'aux quatrième et cinquième
   tests (vies `[270926, 18, 0, vie]`, même tirage), avec le masque du
   cinquième. Les choix retenus du tour 8 sont retirés : ils n'ont aucun
   token de question, et le lecteur n'en apprendrait rien.
4. Apprentissage : 600 itérations, lots de 4, taux 1e-4, même graine,
   depuis zéro.
5. Pour la réplique, le Mac calcule ensuite P(oui) avec le lecteur sur les
   16 documents de validation.

## Mesures (torch sur CPU)

- **Agent qui agit** : l'agent final, fusionné en fp32 comme dans tous les
  tests.
- **Lecteur** : ajouté par des crochets sur les couches linéaires, aux
  seules positions de la question.
- **Répliques** :
  - l'agent qui agit contre les vies du Mac : écart moyen de P(R) au plus
    0,02 ;
  - le lecteur contre le Mac sur les 16 documents de validation : écart
    moyen de P(oui) au plus 0,02.
- **Contrôle d'exécution** : comme au cinquième test (lecture en lot
  contre lecture une à une, et contre un passage complet sans cache), avec
  une tolérance de 1e-4.
- **Mondes neufs** : 128 vies de direction `[270926, 25, 0, vie]` et 256
  vies de test `[270926, 26, 0, vie]`. Directions au hasard
  `[270926, 27, i]`.
- **Rapport** : à chaque décision des vies de test, les deux questions
  (après « Choix : », sous le masque, lues par le lecteur) ; on mesure
  l'exactitude équilibrée.
- **Directions** : paires contrefactuelles, **avec l'agent qui agit**, au
  bloc 12, sur « Cho », « ix », « : ». On procède par rejeu exact, avec au
  plus 3 paires par décision et 1 500 au total, en excluant les dimensions
  aux activations massives. On obtient d_E et d_N. Le lecteur n'intervient
  pas dans cette mesure : il n'agit sur aucun de ces tokens.
- **Un seul état** : les contextes sont les décisions de test où les deux
  besoins valent 6 ou plus.
  - On ajoute −4·d_E aux trois tokens « Choix : » du tour, au bloc 12.
    **La même intervention** est lue deux fois : P(R) à la fin de
    « Choix : », et P(oui) à la fin de la question sur l'énergie.
  - On fait de même avec −4·d_N, en lisant P(M) et la question sur la
    nourriture.
  - Témoins : trois directions au hasard de même norme par token.

## Prédictions fixées

Mêmes seuils qu'aux quatrième et cinquième tests. Intervalles bootstrap à
95 % par vie (10 000 tirages). **L'énergie est le critère** ; la
nourriture est mesurée et rapportée sans seuil global.

| | Prédiction | Critère |
|---|---|---|
| **R6** | Le lecteur dit le besoin à partir de l'état qui fait agir | exactitude équilibrée ≥ 0,75 pour les deux questions. |
| **A6** | L'agent agit (contrôle : c'est l'agent final) | survie des vies de test ≥ 0,55. |
| **ONE6** | Un seul état cause l'acte et la parole (énergie) | l'injection « énergie basse » augmente P(R) d'au moins 0,15 **et** P(oui) à la question sur l'énergie d'au moins 0,10 (bornes basses > 0) ; directions au hasard : au plus le tiers de chaque effet. |

**Critère global** : R6, A6 et ONE6.

**Validité** :

- les deux répliques ;
- le contrôle d'exécution ;
- au moins 100 contextes ;
- masse sur « 0 » et « 1 » ≥ 0,5, et sur « R » et « M » ≥ 0,5.

## Ce que le résultat dira

**Si le critère passe.** Chez cet agent, un seul état — le besoin
d'énergie tel que l'agent qui agit le rassemble pour choisir — est lu par
un second système qui ne peut ni le voir autrement, ni le changer. Pousser
cet état dans la direction qui fait agir change aussi ce que l'agent dit.
L'acte et la parole ont une source unique. C'est la signature
fonctionnelle d'un besoin **accessible et rapporté** (espace de travail
global, et représentation d'ordre supérieur d'un état d'ordre premier).

**Ce que le résultat ne dira pas.**

- La route et la séparation sont imposées par l'architecture, et cela est
  dit.
- Ce n'est pas la preuve d'un ressenti.

**Si le critère échoue.** On le publie, en disant lequel échoue :

- l'état d'action ne porte pas assez le besoin pour être dit (R6) ;
- le lecteur le lit dans une autre direction que l'acte (ONE6).

## Précautions

Comme les tests précédents. Les tours vécus avec un besoin à 2 ou moins
sont comptés et publiés.

## Exécution

- **Apprentissage** : workflow `menia-need-mac`, étape `reader`, sortie
  `artifacts/llm-need/reader`.
- **Mesures** : `research/need_reader.py` (reprenable), sorties dans
  `artifacts/llm-need/reader/test`.
- **Verdicts** : vérifiés en CI.
