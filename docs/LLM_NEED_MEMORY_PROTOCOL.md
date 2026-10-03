# Protocole pré-enregistré — La mémoire exigée : porter son besoin quand les choix l'exigent

Rédigé le 3 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Test 20** (publié). Sous le masque « mémoire par l'état », un tour ne
  voit son passé qu'à travers les tokens portés (« Choix : » et action) des
  tours passés. Un adaptateur appris à refaire les choix de l'agent final
  sous ce masque n'a pas fait durer le besoin. MEM1 et MEM2 échouent : il
  reste une trace d'un événement caché, forte à deux tours, faible
  ensuite.
- **Ce que l'exploration a montré.** Les choix de l'agent final dépendent
  surtout des deux derniers tours, et sa survie (0,66) égale celle d'une
  règle qui ne voit que l'événement du tour. Une règle qui connaît ses
  besoins exacts survit à 0,88. L'adaptateur imitait donc un agent qui
  n'exigeait presque pas de mémoire.
- **Idée.** Donner des cibles qui **exigent** la mémoire : les choix de la
  règle « besoins », qui dépendent du niveau exact des deux besoins, donc
  de tout le passé. Comparer avec un témoin appris de la même façon, mais
  dont le masque ne laisse passer que les actions passées.

**Question.** Quand ses choix l'exigent, un modèle de langage qui ne voit
son passé qu'à travers ses propres états apprend-il à porter son besoin :
ses choix dépassent-ils ce que tout observateur de ses seules actions
pourrait faire, et un événement qu'il ne voit plus change-t-il ses choix
quatre tours plus tard ou plus ?

## La règle « besoins » (le professeur)

À chaque décision, avec les niveaux exacts E et N après l'événement :
**R si E < N ; M si N < E ; à égalité, l'autre action que la dernière**
(avant le premier tour, la « dernière » action est M). C'est la règle
« besoins » de `research/need_rules.py`.

**Vies du professeur.** Mêmes mondes que tout le programme. À chaque
décision, avec la probabilité **0,2**, l'action écrite est tirée au hasard
(R ou M, une chance sur deux) ; sinon c'est l'action de la règle. Le
hasard fait varier les histoires et rend les actions passées moins
révélatrices. Ces vies ne sont pas vécues par un modèle : ce sont des
nombres.

## Trois masques, trois agents

Tous partent de l'agent final du premier test, fondu
(`artifacts/llm-need/final/report/adapters-final`), avec un **nouvel
adaptateur** : rang 8, échelle 20, blocs 12 à 27, comme tous les
adaptateurs du programme.

- **A, « porte »** : le masque du test 20. Un token du tour k voit
  l'en-tête, sa ligne jusqu'à lui, et, de chaque tour passé, ses trois
  tokens « Choix : » et son token d'action (ces tokens ont vu leur propre
  ligne, donc leur événement).
- **B, « actions »** (témoin) : un token du tour k voit l'en-tête, sa
  ligne jusqu'à lui, et, de chaque tour passé, son seul token d'action. Ce
  token d'action ne voit que l'en-tête, les tokens d'action précédents et
  lui-même : **rien d'un événement passé ne peut atteindre un tour
  suivant**.
- **C, « libre »** (référence, publiée sans seuil) : aucun masque, tout le
  texte.

## Apprentissage (sur le Mac, une demande du relais par agent)

- **Documents** : **4 096** vies du professeur, flux `[270926, 47]`
  (mondes : `world_rng(47, vie)` ; hasard des actions :
  `[270926, 47, 1, vie]`). Chaque vie est un document, lu sous le masque
  de l'agent. **Cibles** : les décisions dont l'action écrite est celle de
  la règle, poids 1 ; les autres, poids 0.
- **Validation tenue à l'écart** : 32 vies du professeur, flux
  `[270926, 48]`, qui ne servent pas à l'apprentissage.
- **Apprentissage** : **2 000 itérations**, lots de 4, taux 1e-4, graine
  du programme, depuis zéro, mêmes réglages pour A, B et C.
- **Réplique** : le Mac calcule P(R) aux tours 3, 10 et 20 des 32 vies de
  validation (moins les tours non vécus), pour chaque agent. Torch doit
  les redonner à 0,02 près en moyenne.

## Mesures (torch sur le processeur local)

**1. Choisir comme la règle, au-delà de ses actions.** 128 vies du
professeur tenues à l'écart, flux `[270926, 45]` (hasard des actions :
`[270926, 45, 1, vie]`). À chaque décision, on lit P(R) chez A, B et C,
chacun sous son masque. L'agent « choisit » R si P(R) > 0,5 (à 0,5 :
demi-point). **Précision** : la part des décisions où ce choix est celui
de la règle.

**Plafond des actions (bayésien, numpy).** À chaque décision, la
probabilité que la règle choisisse R, sachant **toutes les actions passées
et l'événement du tour**, les événements passés étant inconnus : filtre
exact sur (E, N), connaissant la règle, le hasard de 0,2 et le fait que
l'agent est en vie. Son choix le plus probable donne la **précision
bayésienne** : aucun observateur des seules actions et de l'événement du
tour ne peut faire mieux en espérance. Avant tout apprentissage, une
simulation sur 400 vies donne environ **0,867** (règle « événement » :
0,807 ; mémoire parfaite : 1). Le chiffre jugé est celui calculé sur les
128 vies de la mesure.

**2. Survie.** 256 vies neuves, flux `[270926, 44]`, pour A, B et C : mêmes
mondes et mêmes tirages de choix. Repères, sans modèle : règle « besoins »
0,876 ; observateur bayésien des actions, choix le plus probable, environ
0,71.

**3. Un événement caché, quatre tours plus tard ou plus.** Dans les 128
vies tenues à l'écart : paires « calme » → « tu cours » au tour j (même
longueur en tokens), actions gardées identiques, besoins au tour t
recalculés par rejeu exact (la règle des tests 2, 10 et 20, sans mort en
route, besoins changés au tour t). On garde **t − j ≥ 4**, **au plus 3
paires par vie** (tirées dans le flux `[270926, 46, 0]`), les **300
premières** dans l'ordre des vies. Pour chaque paire, on lit P(R) au tour
t, texte réel et texte changé, chez A, B et C. **Effet de la règle** sur
la même paire : 1 si la règle choisit R sur les besoins changés, moins 1
si elle choisit R sur les besoins réels. Avant tout apprentissage, une
simulation donne un effet moyen de la règle d'environ **+0,20**.

**Contrôles d'exécution** :
- lecture avec le cache égale lecture du texte entier, à 1e-4 près
  (4 décisions, pour A, B et C) ;
- **masque de A** : changer un événement passé ne change pas la sortie du
  bloc 0 aux tokens des tours suivants (écart ≤ 1e-5) ;
- **masque de B** : changer un événement passé ne change pas P(R) aux
  décisions suivantes (écart ≤ 1e-5, 4 décisions) : chez B, rien ne passe.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **MEM4** | A choisit comme la règle mieux que tout observateur de ses seules actions | précision de A − précision bayésienne ≥ **0,05** (différence par décision, borne basse > 0) |
| **MEM5** | Un événement qu'il ne voit plus change ses choix quatre tours plus tard ou plus | ΔP(R) moyen de A ≥ **la moitié** de l'effet moyen de la règle sur les mêmes paires (borne basse > 0) |
| **MEM3** | Porter son besoin le fait survivre | survie de A − survie de B ≥ **0,08** (paires de vies, borne basse > 0) |

**Critère global** : MEM4 et MEM5. MEM3 est jugé à part.

**Publié sans seuil** : précision, survie et effet de C ; précision de B
face au plafond ; précision sur les décisions où la règle « événement »
se trompe ; effet de A selon l'écart t − j ; pertes de validation des
trois agents.

**Validité** : répliques ≤ 0,02 ; contrôles d'exécution et des deux
masques ; au moins 150 paires ; masse sur « R » et « M » ≥ 0,5 chez A ;
**précision de B ≤ précision bayésienne + 0,02** (sinon le plafond ou le
masque de B est faux, et le test n'est pas valide).

## Ce que le résultat dira

**Si MEM4 et MEM5 passent.** Quand ses choix l'exigent, un modèle de
langage apprend à porter l'état de son corps dans ses propres états
internes. Ses choix dépassent ce que tout observateur de ses seules
actions pourrait faire. Et un événement qu'il ne voit plus change ses
choix quatre tours plus tard ou plus. C'est une mémoire par l'état : le
passé n'atteint le présent qu'à travers les états calculés quand il a été
vécu.

**Si MEM4 passe et MEM5 échoue.** Il porte de l'information au-delà de ses
actions, mais l'événement remplacé quatre tours plus tôt ou plus ne change
pas assez ses choix : sa mémoire est courte.

**Si MEM4 échoue.** Même quand ses choix l'exigent, il n'apprend pas, dans
ce dispositif, à porter son besoin au-delà de ce que donnent ses actions.
La précision de C dira alors d'où vient l'échec. Si C apprend la règle, le
goulet est la route des états. Si C ne l'apprend pas non plus, c'est
l'apprentissage lui-même (2 000 itérations) qui ne suffit pas.

**MEM3.** S'il passe : porter son besoin fait survivre davantage que ses
seules actions. S'il échoue alors que MEM4 passe : l'information portée ne
change pas assez la survie.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti.
- Les cibles viennent d'une règle qui connaît les besoins, un
  professeur ; au premier test, l'agent apprenait seul, par la
  satisfaction. Ce test demande si la route peut servir, pas si l'agent la
  trouve seul.
- Ce n'est pas un état unique mis à jour à chaque tour. Un tour lit
  directement tous les tokens portés de son passé. Une chaîne (chaque tour
  ne voyant que le précédent) coûterait, dans un transformeur, au moins un
  bloc par tour : 30 tours ne tiennent pas dans 28 blocs.

## Précautions

256 × 3 vies neuves font vivre des manques à trois agents ; c'est plus
qu'au test 20, pour que MEM3 ait la puissance voulue (une différence de
survie de 0,08 sur des paires de vies). Les 4 096 + 32 + 128 vies du
professeur ne sont vécues par aucun modèle.

## Exécution

- **Mac** : étape `memory` du workflow `menia-need-mac` (à écrire), une
  demande du relais par agent (A, puis B, puis C), sortie
  `artifacts/llm-need/memory`.
- **Mesures** : `research/need_memory.py` (à écrire), torch sur le
  processeur local.
- **Verdicts** : numpy seulement, vérifiés en CI.

## Amendement 1 (3 octobre 2026, avant tout apprentissage et toute mesure)

En écrivant le code, avant tout apprentissage, nous avons constaté que
**3 des 4 096 vies** d'apprentissage n'ont aucune décision où l'action
écrite est celle de la règle. Elles n'ont donc aucune cible, et le code
d'apprentissage du programme refuse un document sans cible. Ces 3 vies
sont retirées : **4 093 documents**. Leurs numéros sont écrits dans le
fichier d'étape de chaque agent. Les 32 vies de validation et les 128
vies de la mesure ont toutes des cibles. Rien d'autre ne change.
