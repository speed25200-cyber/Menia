# Protocole pré-enregistré — Prédire ce que ses actes font à son corps : la mémoire par l'état sans professeur (test 26)

Rédigé le 4 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Tests 22 à 24** (publiés, valides). L'agent A, qui ne voit son passé
  qu'à travers ses propres états portés, a appris à porter son besoin d'un
  tour à l'autre : il choisit comme la règle « besoins » mieux que tout
  observateur de ses seules actions, et ce qu'il porte dépend surtout de son
  besoin. **Mais il a appris d'un professeur** : ses cibles étaient les
  choix d'une règle qui connaît les besoins.
- **L'apprentissage par la seule récompense, tel que le programme le
  faisait** (garder les choix dont la récompense dépasse la moyenne), ne
  suffit pas ici. Calculé sans modèle avant ce protocole, sur des vies aux
  actions tirées au hasard : même avec une mémoire parfaite, un agent qui
  imite ses choix gardés ne retrouverait le choix de la règle « besoins »
  qu'à 0,76 environ. Un observateur de ses seules actions et de l'événement
  du tour fait déjà 0,88. Ce test ne l'utilise donc pas.
- **L'idée.** Après chaque choix, le monde donne un signal sans
  professeur : le **soulagement** que l'action apporte. Il dépend du niveau
  du besoin servi (plus le besoin est bas, plus le soulagement est fort).
  Apprendre à le **prédire** avant de le sentir demande de savoir où en
  sont ses besoins. Sous le masque, il faut donc les porter d'un tour à
  l'autre. C'est un apprentissage par prédiction de ce que ses actes font à
  son corps, sans cible de choix.

**Question.** Un modèle de langage qui apprend seulement à prédire le
soulagement de ses actions, et qui ne voit son passé qu'à travers ses
propres états, apprend-il à porter ses besoins ? Et s'il choisit l'action
dont il prédit le plus grand soulagement, survit-il mieux qu'un témoin qui
ne porte que ses actions ?

## Le monde et les textes

- **Le monde** est celui du programme : deux besoins, énergie (E) et
  nourriture (N), de 1 à 8, six événements, deux actions (R recharge E de
  3, M recharge N de 3), 30 tours.
- **Les lignes** sont celles du test 22 : chaque ligne d'événement est
  remplie de « - » jusqu'à la même longueur. On y ajoute le soulagement de
  l'action prise :
  `Tour 3 : calme. - - - - Choix : R Soulagement : moyen.`
- **Le soulagement** vaut la baisse du manque que l'action apporte. Il ne
  dépend que du niveau v du besoin servi, avant la recharge. Il est écrit
  en quatre mots, chacun d'un seul token dans le tokenizer de Qwen3-0.6B
  (vérifié) :
  - « aucun » si v ≥ 7 ;
  - « petit » si v = 5 ou 6 ;
  - « moyen » si v = 3 ou 4 ;
  - « fort » si v = 1 ou 2.

  Toutes les lignes ont donc la même longueur en tokens.
- **Qui écrit les actions des vies d'apprentissage.** Une règle qui ne
  connaît pas les besoins : la règle « événement » (servir le besoin que
  l'événement du tour frappe le plus ; à égalité, l'autre action que la
  dernière), avec 30 % d'actions au hasard. Calculé avant ce protocole,
  sans modèle : ces vies survivent à 0,33, durent 20 tours en moyenne, et
  donnent 166 923 décisions sur 8 192 vies.

## Les agents et l'apprentissage

- **Les cibles** : seulement le mot du soulagement, à chaque décision.
  **Aucune cible de choix.** L'agent n'apprend jamais quelle action
  prendre.
- **Deux agents**, appris de la même façon, avec les masques du test 22 :
  - **A (« porte »)** : un tour voit l'en-tête, sa propre ligne et, pour
    chaque tour passé, ses tokens portés (« Cho », « ix », « : » et
    l'action). Le soulagement des tours passés **n'est pas porté** ;
  - **B (« actions »)**, le témoin : les tours suivants ne voient que les
    actions passées.
- **Apprentissage** : à partir de l'agent final fondu, un nouvel
  adaptateur (rang 8, échelle 20, blocs 12 à 27, lots de 4, taux 1e-4,
  graine du programme), **4 000 itérations**, comme au test 22 (morceaux de
  250 itérations repris exactement, ordre des lots fixé par époque).
- **Les vies** :
  - apprentissage : 8 192 vies, flux `[270926, 58]` ;
  - validation tenue à l'écart : 32 vies, flux `[270926, 59]` ;
  - mesure de la prédiction : 128 vies tenues à l'écart, flux
    `[270926, 60]` ;
  - survie : les 256 mondes du test 22, flux `[270926, 44]`.

  Les vies sans cible (mortes avant leur première décision) sont retirées.

## Les mesures

1. **Prédiction du soulagement.** Sur les 128 vies tenues à l'écart, à
   chaque décision, on lit les probabilités des quatre mots après
   « Soulagement : ». Le niveau prédit est le plus probable. **Précision** :
   part des décisions où il est juste.
2. **Le plafond des actions.** Un observateur bayésien exact connaît le
   monde et la règle qui écrit les actions. Il voit l'événement du tour et
   toutes les actions passées, mais pas les événements passés. Il prédit le
   niveau le plus probable. Calculé avant ce protocole sur 256 autres vies
   de la même règle : 0,646 ; sur les 128 vies de la mesure : 0,648. Une
   mémoire parfaite des besoins donnerait 1.
3. **Choisir par le soulagement prédit.** Sur les 256 mondes de survie,
   l'agent vit ses vies ainsi : à chaque décision, il prédit le soulagement
   des deux actions (deux lectures, « Choix : R Soulagement : » et « Choix :
   M Soulagement : ») et prend celle dont le niveau attendu (0 à 3, pondéré
   par ses probabilités) est le plus grand ; à égalité, l'autre action que
   la dernière. Le vrai soulagement est écrit, puis la vie continue. Cette
   règle de lecture est fixée par nous : elle ne s'apprend pas.

   Repères calculés avant ce protocole, sur ces 256 mondes :
   - la même lecture avec les niveaux exacts : 0,852 ;
   - la règle « événement » : 0,723 ;
   - la règle « besoins » : 0,906.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages). Jugées après les
4 000 itérations.

| | Prédiction | Critère |
|---|---|---|
| **INTER1** | En apprenant à prédire son soulagement, A porte ses besoins au-delà de ses actions | précision de A − plafond des actions ≥ **0,10** (borne basse > 0) |
| **INTER2** | Choisir par le soulagement prédit le fait survivre | survie de A − survie de B, même lecture, mêmes mondes ≥ **0,08** (paires de vies, borne basse > 0) |

**Critère global** : INTER1 et INTER2, test valide.

**Publié sans seuil** :
- précision de A et de B selon le niveau ;
- part des décisions où A, en choisissant par le soulagement prédit, prend
  l'action de la règle « besoins » (sur les 128 vies de mesure, lecture
  des deux actions sans changer la vie) ;
- les pertes de validation ;
- la précision après 2 000 itérations (A et B).

**Validité** :
- **réplique** : le Mac calcule la probabilité du vrai mot aux tours 3, 10
  et 20 des 32 vies de validation ; torch la redonne à 0,02 près en
  moyenne ;
- **cache** : lecture avec le cache égale lecture du texte entier (écart ≤
  1e-4, 4 décisions) ;
- **masque de A** : changer un événement passé ne change pas la sortie du
  bloc 0 aux tokens des tours suivants (≤ 1e-5) ;
- **masque de B, sans fuite** : changer un événement passé pour un
  événement de longueur naturelle différente ne change pas la prédiction
  aux décisions suivantes (≤ 1e-5, 4 décisions) ;
- **masse** sur les quatre mots ≥ 0,5 en moyenne chez A ;
- **précision de B ≤ plafond des actions + 0,02** (sinon une fuite reste).

## Ce que le résultat dira

**Si INTER1 et INTER2 passent.** Sans professeur, en apprenant seulement à
prédire ce que ses actes font à son corps, un modèle de langage qui ne voit
son passé qu'à travers ses propres états apprend à y porter ses besoins.
Il prédit son soulagement mieux que tout observateur de ses seules
actions. Et en choisissant l'action dont il attend le plus grand
soulagement, il survit mieux qu'un témoin qui ne porte que ses actions.

**Si INTER1 passe et INTER2 échoue.** Il porte ses besoins pour prédire,
mais choisir par cette prédiction ne le fait pas assez survivre de plus.

**Si INTER1 échoue.** Prédire son soulagement ne suffit pas, ici, à lui
faire porter ses besoins au-delà de ses actions.

**Si le test n'est pas valide.** Le verdict n'est pas revendiqué, et la
cause est publiée.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti. « Soulagement » est le nom d'un
  signal du monde, pas un vécu.
- La règle qui choisit l'action la plus soulageante est fixée par nous.
- Le signal de soulagement vient du monde, qui connaît les besoins : c'est
  un retour de ses actes, pas une consigne sur quoi faire.
- Ce n'est pas un état unique mis à jour à chaque tour : un tour lit
  directement les tokens portés de tout son passé.

## Précautions

256 × 2 vies neuves font vivre des manques à deux agents, comme au test
22. Les tours vécus avec un besoin à 2 ou moins sont comptés et publiés.

## Exécution

- **Mac** : étape `memory` du workflow `menia-need-mac`, avec les textes du
  soulagement (à écrire) ; sortie `artifacts/llm-need/relief`.
- **Mesures** : torch sur le processeur local (à écrire).
- **Verdicts** : numpy seulement, vérifiés en CI.
