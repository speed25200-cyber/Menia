# Protocole pré-enregistré — Sans pré-entraînement : de petits transformeurs portent-ils leurs besoins ? (test 29)

Rédigé le 4 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Aucun résultat du test 26 n'est
connu au-delà des journaux d'apprentissage du Mac (pertes de validation,
répliques après 500 et 750 itérations). Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Tests 22 à 27.** Un modèle de langage pré-entraîné (Qwen3-0.6B, ajusté
  par LoRA), qui ne voit son passé qu'à travers ses propres états portés,
  apprend à y porter son besoin. Mais il a appris d'un professeur, et il
  part d'un modèle qui sait déjà beaucoup.
- **Test 26** (en cours sur le Mac) : le même modèle apprend **sans
  professeur**, seulement à prédire le soulagement de ses actions.
- **Ce qu'on ne sait pas.** Ce résultat dépend-il du pré-entraînement, ou
  vient-il de l'architecture (un transformeur sous ce masque) ? Et est-il
  robuste d'une graine à l'autre ? Le Mac ne permet qu'un agent par bras.

**Question.** De petits transformeurs appris **de zéro**, sous les mêmes
masques, avec la seule cible du soulagement, portent-ils leurs besoins
au-delà de leurs actions ? Sur dix graines ?

## Le monde, les vies, la cible

Ceux du test 26 (`docs/LLM_NEED_RELIEF_PROTOCOL.md`) :
- vies écrites par la règle « événement » avec 30 % d'actions au hasard ;
- apprentissage : 8 192 vies, flux `[270926, 58]` ;
- mesure : 128 vies tenues à l'écart, flux `[270926, 60]` ;
- survie : les 256 mondes du test 22, flux `[270926, 44]` ;
- cible : le niveau du soulagement (« aucun », « petit », « moyen »,
  « fort »), seule cible ; **aucune cible de choix**.

## Les textes, en tokens simples

Chaque tour est écrit en **5 tokens**, toujours aux mêmes positions :
1. l'événement (6 tokens possibles) ;
2. « Choix » ;
3. l'action (R ou M) ;
4. « Soulagement » ;
5. le niveau (4 tokens possibles) — la cible, prédite au token 4.

Un tour où la vie s'éteint s'écrit : l'événement, puis « Fin ». Un token
de début ouvre la vie, et chaque token reçoit un plongement de position
(appris).

## Les masques

Comme aux tests 22 et 26 :
- **A (« porte »)** : un token voit le début, les tokens de son propre tour
  jusqu'à lui, et, pour chaque tour passé, seulement ses tokens portés
  (« Choix » et l'action) ;
- **B (« actions »)** : un token voit le début, les tokens de son tour
  jusqu'à lui, et seulement les actions des tours passés. Comme au test 22,
  le token de l'action ne voit, lui, que le début, les actions passées et
  lui-même : il ne peut pas porter l'événement de son tour ;
- **C (« libre »)**, publié sans seuil : le masque causal ordinaire, tout
  le passé visible.

## Les modèles et l'apprentissage

- Transformeur décodeur : 2 couches, dimension 64, 4 têtes, sans
  pré-entraînement.
- Apprentissage : AdamW, taux 1e-3, lots de 32 vies, **3 000 pas**, perte
  sur le token du niveau seulement.
- **10 graines** par masque (flux `[270926, 70, graine]` pour
  l'initialisation et l'ordre des lots). Les graines de A, B et C sont les
  mêmes.

## Les mesures

Pour chaque graine et chaque masque :
1. **Précision** : sur les 128 vies tenues à l'écart, à chaque décision, le
   niveau le plus probable est-il le bon ?
2. **Plafond des actions** : l'observateur bayésien exact du test 26
   (0,648 sur ces vies).
3. **Survie** : sur les 256 mondes, à chaque décision, l'agent prédit le
   soulagement des deux actions et prend celle dont le niveau attendu est
   le plus grand ; à égalité, l'autre action que la dernière (la lecture du
   test 26).

## Prédictions fixées

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de liberté).

| | Prédiction | Critère |
|---|---|---|
| **TINY1** | Sans pré-entraînement, A porte ses besoins au-delà de ses actions | moyenne sur les graines de (précision de A − plafond) ≥ **0,10**, borne basse > 0, et au moins 8 graines sur 10 au-dessus de **0,05** |
| **TINY2** | Choisir par le soulagement prédit le fait survivre | moyenne sur les graines de (survie de A − survie de B, même graine) ≥ **0,08**, borne basse > 0 |

**Critère global** : TINY1 et TINY2, test valide.

**Publié sans seuil** :
- C (« libre ») : précision et survie ;
- les résultats graine par graine ;
- les pertes d'apprentissage.

**Validité** :
- **masque de B** : changer un événement passé ne change aucune
  prédiction des tours suivants (écart ≤ 1e-6, vérifié sur 4 vies, à la
  graine 0) ;
- **masque de A** : changer un événement passé ne change pas la sortie de
  la première couche aux tokens des tours suivants (écart ≤ 1e-6, même
  vérification) ;
- **précision moyenne de B ≤ plafond + 0,02** (sinon une fuite reste).

## Ce que le résultat dira

**Si TINY1 et TINY2 passent.** La mémoire par l'état ne dépend pas du
pré-entraînement. Un petit transformeur appris de zéro, sous ce masque, en
apprenant seulement à prédire ce que ses actes font à son corps, porte ses
besoins dans ses propres états, mieux que tout observateur de ses seules
actions, et s'en sert pour survivre. Et c'est vrai sur dix graines.

**Si TINY1 passe et TINY2 échoue.** Il porte ses besoins pour prédire, mais
choisir par cette prédiction ne le fait pas assez survivre de plus.

**Si TINY1 échoue.** Sans pré-entraînement, à cette taille et avec cet
apprentissage, il n'apprend pas à porter ses besoins au-delà de ses
actions. Le résultat du test 26, s'il passe, tiendrait alors au modèle
pré-entraîné ou à sa taille.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti.
- Le signal de soulagement vient du monde, qui connaît les besoins.
- Un autre tokenage, d'autres tailles ou durées d'apprentissage pourraient
  donner autre chose.

## Exécution

- **Code** : `research/tiny_relief.py` (à écrire), torch sur le processeur
  local.
- **Verdicts** : numpy seulement, vérifiés en CI.
