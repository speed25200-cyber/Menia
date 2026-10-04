# Protocole pré-enregistré — Sans professeur, que porte l'état ? (test 28)

Rédigé le 4 octobre 2026, **avant l'écriture du code, avant la fin de
l'apprentissage du test 26 et avant toute mesure de ses agents** ; l'heure
est celle du commit. Seuls des journaux d'apprentissage du Mac ont été vus
(pertes de validation et répliques des 32 vies de validation après 500 et
750 itérations). Seuils fixés, un échec est un résultat.

## D'où vient ce test (dit tel quel)

- **Test 23** (publié, valide). Chez l'agent appris d'un professeur (test
  22), l'état porté d'un tour, greffé dans une autre vie, fait choisir
  selon les besoins de la vie d'origine. Un état aux mêmes besoins, venu
  d'une autre histoire, change peu le choix. Ce qui est porté dépend
  surtout du besoin.
- **Test 26** (en cours). Un agent A apprend **sans professeur** : sa seule
  cible est le mot du soulagement que son action apporte (« aucun »,
  « petit », « moyen », « fort »), qui dépend du niveau du besoin servi. Il
  ne voit son passé qu'à travers ses états portés.
- **La question.** Si cet agent porte quelque chose d'un tour à l'autre,
  est-ce son besoin ?

## L'agent et les vies

- **L'agent** : A du test 26 (« porte »), après 4 000 itérations, sous son
  masque, textes du soulagement.
- **Témoin publié sans seuil** : B du test 26 (« actions »).
- **Les vies** : **128 vies neuves**, écrites comme au test 26 (règle
  « événement », 30 % d'actions au hasard), flux `[270926, 63]`.

## La greffe

Celle du test 23, sur les textes du soulagement. Pour une vie receveuse r,
un tour j et un tour t = j + g (g = 1 ou 2), r vivante en t :
1. on lit r jusqu'à la fin du tour j, sous le masque ;
2. de même pour une vie donneuse d, vivante au tour j, qui a **la même
   action écrite au tour j** ;
3. on remplace, dans toutes les couches, les clés et valeurs des 4 tokens
   portés du tour j (« Cho », « ix », « : » et l'action) par celles de d ;
4. on continue la lecture de r jusqu'à « Soulagement : » au tour t, pour
   l'action écrite de r au tour t.

On lit les probabilités des quatre mots (normalisées). **Niveau attendu** :
0 × P(aucun) + 1 × P(petit) + 2 × P(moyen) + 3 × P(fort). **ΔL** = niveau
attendu avec la greffe − sans.

**Effet des besoins.** On rejoue r à partir des besoins de d au tour j, avec
les actions et les événements de r. On calcule le vrai niveau du soulagement
au tour t pour l'action écrite de r. **e** = ce niveau − le vrai niveau de r
(de −3 à +3). Une donneuse dont le rejeu fait mourir r avant t n'est pas
admissible.

**Trois sortes de greffes** :
- **(a)** autres besoins au tour j, e ≠ 0 ;
- **(a0)** autres besoins, e = 0 (publié sans seuil) ;
- **(b)** mêmes besoins au tour j (autre histoire).

**Tirage** (flux `[270926, 64, 0]`). On parcourt les vies receveuses dans
l'ordre. Pour chaque r et chaque g ∈ {1, 2}, on tire au plus 2 tours t.
Pour chacun, on tire une donneuse de chaque sorte parmi les admissibles,
s'il en existe. On s'arrête à **400 (a)**, **200 (a0)** et **200 (b)**.
Compté avant ce protocole, sans modèle, sur toutes les vies : 493 (a), 506
(a0), 388 (b) possibles.

**Calcul** : un seul fil par processus ; les greffes peuvent être réparties
entre processus.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie receveuse (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **SELF1** | L'état greffé fait prédire le soulagement selon les besoins de la vie donneuse | sur les greffes (a) : moyenne de ΔL × signe(e) ≥ **0,15** niveau (borne basse > 0) |
| **SELF2** | C'est le besoin, pas l'histoire | moyenne de \|ΔL\| des greffes (b) ≤ **la moitié** de celle des greffes (a), et la différence a une borne basse > 0 |

**Critère global** : SELF1 et SELF2, test valide.

**Publié sans seuil** :
- \|ΔL\| des greffes (a0) ;
- les résultats selon g ;
- la pente de ΔL sur e (part de l'effet des besoins que A retrouve) ;
- les mêmes mesures chez le témoin B.

**Validité** :
- sans greffe, la lecture continuée égale la lecture du texte entier
  (écart ≤ 1e-4 sur les quatre probabilités, 4 cas) ;
- greffe de soi : écart ≤ 1e-6 (4 cas) ;
- mêmes positions des tokens portés chez d et r (vérifié à chaque
  greffe) ;
- au moins 100 greffes (a) et 100 greffes (b).

## Ce que le résultat dira

**Si SELF1 et SELF2 passent.** Sans professeur, ce que l'agent porte d'un
tour à l'autre dépend surtout de son besoin. Greffé dans une autre vie, il
fait prédire le soulagement selon les besoins de la vie d'origine ; venu
d'une vie aux mêmes besoins mais à l'histoire différente, il change peu la
prédiction.

**Si SELF1 passe et SELF2 échoue.** L'état porté fait prédire selon les
besoins d'origine, mais il porte aussi autre chose de l'histoire.

**Si SELF1 échoue.** Greffé ailleurs, l'état porté ne fait pas prédire le
soulagement selon les besoins d'origine.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti.
- La greffe porte sur les 4 tokens d'un seul tour.
- Le signal de soulagement vient du monde, qui connaît les besoins : c'est
  un retour de ses actes, pas une consigne sur quoi faire.
- Il ne sépare pas le niveau des besoins de ce qui commande le soulagement
  de l'action écrite.

## Précautions

Aucune vie n'est vécue par un modèle : on relit des vies écrites par une
règle.

## Exécution

- **Mesures** : `research/need_relief_graft.py` (à écrire), torch sur le
  processeur local, après l'apprentissage du test 26.
- **Verdicts** : numpy seulement, vérifiés en CI.
