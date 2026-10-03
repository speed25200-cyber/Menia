# Protocole pré-enregistré — La mémoire sans fuite (test 22)

Rédigé le 3 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Test 21** (publié, non valide). Cibles : les choix de la règle
  « besoins ». L'agent A, qui ne voit son passé qu'à travers ses propres
  états, choisit comme la règle à 0,956 et survit à 0,871. Le témoin B, qui
  ne voit que ses actions, est à 0,897 et 0,730. Un événement caché change
  les choix de A 4 à 8 tours plus tard (+0,066 ; B : 0 exactement), mais
  moins que la moitié exigée de l'effet de la règle : MEM5 échoue.
- **Pourquoi le test 21 n'est pas valide.** B dépasse le plafond bayésien
  de 0,026 (le maximum permis était 0,02). Les lignes d'événement n'ont pas
  la même longueur (12, 14 ou 16 tokens). La distance entre deux actions
  passées dit donc la longueur de l'événement qui les sépare, et le masque
  ne cache pas les positions.
- **Ce que le test 21 a montré d'autre.** L'agent qui voit tout le texte
  ne faisait pas mieux que A. La perte de A baissait encore à la fin de ses
  2 000 itérations.

**Question.** Sans fuite des positions, et avec un apprentissage deux fois
plus long, l'agent qui ne voit son passé qu'à travers ses propres états
choisit-il mieux que tout observateur de ses seules actions ? Et un
événement qu'il ne voit plus change-t-il ses choix quatre tours plus tard
ou plus, d'au moins la moitié de ce que ferait la règle ?

## Ce qui change par rapport au test 21

1. **Toutes les lignes d'événement ont la même longueur.** Après le point
   de l'événement, un remplissage « - » complète la ligne jusqu'à la
   longueur de la plus longue (« tu trouves des baies ») :
   - 4 « - » pour « calme », « tu cours », « orage » ;
   - 2 pour « il fait froid », « tu te reposes » ;
   - aucun pour « tu trouves des baies ».

   Exemple : « Tour 3 : calme. - - - - Choix : R ». Vérifié avec le
   tokenizer de Qwen3-0.6B : à chaque tour de 1 à 30, toutes les lignes
   ont le même nombre de tokens, et les actions tombent aux mêmes positions
   dans toutes les vies. Le remplissage vaut pour tous les textes du test
   (apprentissage, vies, paires).
2. **Apprentissage deux fois plus long.** 8 192 vies du professeur, flux
   `[270926, 49]` (même règle, même hasard de 0,2), et **4 000
   itérations**. Elles sont faites en deux builds de 2 000 sur le Mac : le
   second reprend exactement le premier (mêmes lots dans le même ordre,
   moments d'Adam repris).
3. **Deux agents seulement : A (« porte ») et B (« actions »)**, mêmes
   masques qu'au test 21. Les deux partent de l'agent final fondu, avec un
   nouvel adaptateur (rang 8, échelle 20, blocs 12 à 27, lots de 4, taux
   1e-4, graine du programme). L'agent libre C n'est pas refait : au test
   21, il ne faisait pas mieux que A.

Tout le reste est celui du test 21 (`docs/LLM_NEED_MEMORY_PROTOCOL.md`) :
- la règle ;
- la validation tenue à l'écart (32 vies `[270926, 48]`) ;
- les 128 vies tenues à l'écart de la mesure (`[270926, 45]`, mêmes vies,
  textes remplis) ;
- le plafond bayésien des actions ;
- les 256 vies de survie (`[270926, 44]`) ;
- les 300 paires « calme » → « tu cours » (t − j ≥ 4, au plus 3 par vie,
  `[270926, 46, 0]` : les mêmes paires qu'au test 21) ;
- les contrôles du cache et de la réplique.

Les vies sans cible sont retirées, comme à l'amendement 1 du test 21.

## Contrôles d'exécution

- Lecture avec le cache égale lecture du texte entier, à 1e-4 près
  (4 décisions, A et B).
- **Masque de A** : changer un événement passé ne change pas la sortie du
  bloc 0 aux tokens des tours suivants (écart ≤ 1e-5).
- **Masque de B, sans fuite** : changer un événement passé pour un
  événement **de longueur naturelle différente** (« calme » → « tu trouves
  des baies », rendus égaux par le remplissage) ne change pas P(R) aux
  décisions suivantes (écart ≤ 1e-5, 4 décisions).
- Réplique : le Mac calcule P(R) aux tours 3, 10 et 20 des 32 vies de
  validation ; torch les redonne à 0,02 près en moyenne.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages). Jugées après les
4 000 itérations.

| | Prédiction | Critère |
|---|---|---|
| **MEM4** | A choisit comme la règle mieux que tout observateur de ses seules actions | précision de A − précision bayésienne ≥ **0,05** (borne basse > 0) |
| **MEM5** | Un événement qu'il ne voit plus change ses choix quatre tours plus tard ou plus | ΔP(R) moyen de A ≥ **la moitié** de l'effet moyen de la règle sur les mêmes paires (borne basse > 0) |
| **MEM3** | Porter son besoin le fait survivre | survie de A − survie de B ≥ **0,08** (paires de vies, borne basse > 0) |

**Critère global** : MEM4 et MEM5, test valide. MEM3 est jugé à part.

**Validité** : répliques ≤ 0,02 ; contrôles du cache et des deux masques ;
au moins 150 paires ; masse sur « R » et « M » ≥ 0,5 chez A ; **précision
de B ≤ précision bayésienne + 0,02**.

**Publié sans seuil** :
- A et B après les 2 000 premières itérations : précision et effet des
  paires (pas de vies de survie), pour voir si un apprentissage plus long
  rend la mémoire plus longue ;
- l'effet selon l'écart t − j ;
- la précision là où la règle « événement » se trompe ;
- les pertes de validation.

## Ce que le résultat dira

**Si le critère global passe.** Sans fuite, un modèle de langage dont les
choix l'exigent apprend à porter l'état de son corps dans ses propres
états. Ses choix dépassent ce que tout observateur de ses seules actions
pourrait faire. Un événement qu'il ne voit plus change ses choix quatre
tours plus tard ou plus, d'au moins la moitié de ce que ferait la règle.
C'est une mémoire par l'état : le passé n'atteint le présent qu'à travers
les états calculés quand il a été vécu.

**Si MEM4 passe et MEM5 échoue.** Il porte de l'information au-delà de ses
actions, mais sa mémoire reste courte, même avec un apprentissage deux fois
plus long.

**Si MEM4 échoue.** Sans la fuite, il ne choisit pas assez mieux que ses
seules actions ne le permettent.

**MEM3.** S'il passe : porter son besoin fait survivre davantage que ses
seules actions.

**Si le test n'est pas valide.** Le verdict n'est pas revendiqué, et la
cause est publiée.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti.
- Les cibles viennent d'une règle qui connaît les besoins, un professeur.
  Ce test demande si la route peut servir, pas si l'agent la trouve seul.
- Ce n'est pas un état unique mis à jour à chaque tour. Un tour lit
  directement les tokens portés de tout son passé.

## Précautions

256 × 2 vies neuves font vivre des manques à deux agents (une de moins
qu'au test 21). Les vies du professeur ne sont vécues par aucun modèle.

## Exécution

- **Mac** : étape `memory` du workflow `menia-need-mac`, avec le remplissage
  et la reprise (à écrire) ; quatre builds (A puis B, deux chacun), sortie
  `artifacts/llm-need/memory-no-leak`.
- **Mesures** : torch sur le processeur local (à écrire).
- **Verdicts** : numpy seulement, vérifiés en CI.
