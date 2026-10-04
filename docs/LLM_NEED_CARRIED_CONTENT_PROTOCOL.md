# Protocole pré-enregistré — Que porte l'état ? (test 23)

Rédigé le 4 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Test 22** (publié, valide). L'agent A ne voit son passé qu'à travers
  ses propres états : les tokens portés « Choix : » et l'action de chaque
  tour passé. Il choisit comme la règle « besoins » à 0,968, bien au-dessus
  de tout observateur de ses seules actions (0,871). Ses états portés
  contiennent donc une information sur son passé, qui sert à ses choix
  (MEM4 passe).
- **Ce qu'on ne sait pas.** Quelle est cette information ? Deux
  possibilités :
  - **le besoin** : un état de soi, le niveau de l'énergie et de la
    nourriture à ce tour ;
  - **une trace de l'histoire** : les événements, que les tours suivants
    additionneraient.

  Les tests 2 à 7 avaient montré que l'état « Choix : » de l'agent final
  rassemble le besoin du tour. Ici, la question est ce qui est **porté**
  d'un tour à l'autre.
- **Idée : la greffe.** Avec les lignes remplies du test 22, chaque tour
  occupe les mêmes positions dans toutes les vies. On peut donc prendre les
  clés et valeurs d'attention des tokens portés d'un tour j (« Cho », « ix »,
  « : » et l'action), dans toutes les couches, chez une vie « donneuse ». On
  les met à la place de celles d'une vie « receveuse », puis on lit le choix
  de la receveuse au tour t, un ou deux tours plus tard. Sous le masque, les
  tours suivants ne voient le tour j qu'à travers ces tokens.

**Question.** Greffé dans une autre vie, l'état porté du tour j fait-il
choisir selon **les besoins de la vie donneuse** ? Et un état porté venant
d'une vie qui avait **les mêmes besoins** au tour j, mais une autre
histoire, laisse-t-il le choix presque inchangé ?

## L'agent et les vies

- **L'agent** : A, le « porte » du test 22, après 4 000 itérations
  (`artifacts/llm-need/memory-no-leak/route-3`), sous son masque, textes
  remplis.
- **Témoin publié sans seuil** : B, le « actions » du test 22
  (`actions-4`). Ses tokens portés ne sont que ses actions.
- **Les vies** : les 128 vies du professeur tenues à l'écart du test 22
  (flux `[270926, 45]`).

## La greffe

Pour une vie receveuse r, un tour j et un tour t = j + g (g = 1 ou 2), r
vivante en t :
1. Lire le texte de r jusqu'à la fin du tour j, sous le masque, et garder
   le cache (clés et valeurs de toutes les couches).
2. Faire de même pour une vie donneuse d, vivante au tour j, qui a **la même
   action écrite au tour j** que r.
3. Remplacer, dans le cache de r et dans toutes les couches, les clés et
   valeurs des 4 tokens portés du tour j par celles de d. Les positions sont
   les mêmes (vérifié).
4. Continuer la lecture de r, du tour j + 1 jusqu'à « Choix : » du tour t,
   et lire P(R).

**ΔP(R)** = P(R) avec la greffe − P(R) sans greffe.

**Effet de la règle.** On rejoue la vie r à partir des besoins de d au tour
j (après l'événement, avant l'action). On garde les actions et les
événements de r de j à t. On regarde si la règle choisirait autrement au
tour t :
- +1 si elle passe de M à R ;
- −1 si elle passe de R à M ;
- 0 sinon.

Une greffe dont le rejeu fait mourir r avant t est écartée.

**Deux sortes de greffes** :
- **(a) besoins différents** : (E, N) de d au tour j ≠ (E, N) de r au tour j ;
- **(b) mêmes besoins** : (E, N) de d au tour j = (E, N) de r au tour j.
  L'histoire diffère (ce sont deux vies), souvent l'événement du tour j
  aussi.

**Tirage** (flux `[270926, 50, 0]`). On parcourt les vies receveuses dans
l'ordre. Pour chaque r et chaque g ∈ {1, 2}, on tire au plus 2 tours t
(avec j = t − g ≥ 1). Pour chacun, on tire une donneuse de chaque sorte
parmi les vies admissibles, s'il en existe. On s'arrête à **400 greffes
(a)** et **200 greffes (b)**.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie receveuse (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **STATE1** | L'état greffé fait choisir selon les besoins de la vie donneuse | sur les greffes (a) dont l'effet de la règle n'est pas nul : moyenne de ΔP(R) × (effet de la règle) ≥ **0,10** (borne basse > 0) |
| **STATE2** | C'est le besoin, pas l'histoire | moyenne de \|ΔP(R)\| sur les greffes (b) ≤ **la moitié** de la moyenne de \|ΔP(R)\| sur les greffes (a) à effet non nul, et la différence des deux a une borne basse > 0 |

**Critère global** : STATE1 et STATE2.

**Publié sans seuil** :
- \|ΔP(R)\| des greffes (a) dont l'effet de la règle est nul ;
- les résultats selon g (1 ou 2) ;
- les mêmes mesures chez le témoin B ;
- la part de l'effet de la règle que A retrouve.

**Validité** :
- **sans greffe**, la lecture continuée égale la lecture du texte entier
  (écart ≤ 1e-4, 4 cas) ;
- **une greffe de soi** (d = r) donne ΔP(R) = 0 (écart ≤ 1e-6, 4 cas) ;
- les positions des tokens portés du tour j sont les mêmes chez d et r ;
- au moins 100 greffes (a) à effet non nul et 100 greffes (b).

## Ce que le résultat dira

**Si STATE1 et STATE2 passent.** L'état que l'agent porte d'un tour à
l'autre contient son besoin. Greffé dans une autre vie, il fait choisir
selon les besoins de la vie d'origine. Un état aux mêmes besoins, venu
d'une autre histoire, ne change presque rien. Ce qui est porté est un état
de soi (le niveau de ses besoins), pas une trace des événements.

**Si STATE1 passe et STATE2 échoue.** L'état porté fait choisir selon les
besoins d'origine, mais il porte aussi autre chose de l'histoire, qui
change les choix.

**Si STATE1 échoue.** Greffé ailleurs, l'état porté ne fait pas choisir
selon les besoins d'origine. Ce qu'il porte n'est pas, ou pas seulement, le
besoin, ou la vie qui le reçoit le recalcule.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti.
- L'agent a appris d'un professeur qui connaît les besoins.
- La greffe porte sur les 4 tokens d'un seul tour : les tours précédents
  de la receveuse restent les siens.

## Précautions

Aucune vie neuve n'est vécue : on relit des vies du professeur, qui ne
sont vécues par aucun modèle.

## Exécution

- **Mesures** : `research/need_carried_content.py` (à écrire), torch sur le
  processeur local.
- **Verdicts** : numpy seulement, vérifiés en CI.
