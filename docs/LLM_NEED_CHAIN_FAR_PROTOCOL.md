# Protocole pré-enregistré — À distance, le besoin passe-t-il d'état en état ? (test 25)

Rédigé le 4 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Test 24** (publié, valide). On greffe dans une autre vie l'état porté
  d'un tour i (les tokens « Cho », « ix », « : » et l'action, dans toutes
  les couches). Puis on mesure quelle part de son effet sur le choix du
  tour t passe par les états portés des tours intermédiaires (les
  médiateurs).
  - En moyenne, cette part vaut 0,46 (seuil 0,5) : CHAIN échoue, de peu.
  - **Après lecture** (publié sans seuil), le partage dépend de la
    distance : 30 % à deux tours (un tour intermédiaire), **72 %** à trois
    tours (deux tours intermédiaires). L'effet direct baisse nettement
    (+0,335 puis +0,097).
  - **La relecture indépendante** a montré que cette part (effet total
    moins effet direct) contient un terme non additif. Comptée par la
    lecture « par les médiateurs seuls », la part de la chaîne vaut 21 % à
    deux tours et **50 %** à trois.
- **Ce qu'on ne sait pas.** Cette observation est faite après lecture, sur
  un seul découpage et sur les mêmes 128 vies. Est-ce vrai sur des vies
  neuves, et plus loin ?

**Question.** Sur des vies neuves, à trois, quatre et cinq tours, le
besoin porté par l'état d'un tour atteint-il le choix **surtout** en
passant d'état en état ?

## L'agent et les vies

- **L'agent** : A, le « porte » du test 22, après 4 000 itérations
  (`artifacts/llm-need/memory-no-leak/route-3`), sous son masque, textes
  remplis. Le même qu'aux tests 23 et 24.
- **Les vies** : **128 vies neuves** du professeur, flux `[270926, 52]`
  (même règle, même hasard de 0,2). Elles n'ont jamais servi, ni à
  l'apprentissage, ni à une mesure.

## La greffe et les lectures

Celles du test 24 (`docs/LLM_NEED_CHAIN_PROTOCOL.md`), sans changement :
- même donneuse : vivante au tour i, même action écrite au tour i,
  d'autres besoins, et la règle changerait de choix au tour t ;
- mêmes lectures de P(R) au tour t : propre, totale, directe (médiateurs
  remis à leurs valeurs propres), par les médiateurs seuls ;
- mêmes effets alignés sur la règle : total, direct, en chaîne (total −
  direct).

Seul change **l'écart h = t − i : 3, 4 ou 5** (deux à quatre médiateurs).

**Tirage** (flux `[270926, 53, 0]`). On parcourt les 128 vies dans
l'ordre. Pour chaque r et chaque h ∈ {3, 4, 5}, on tire au plus 3 tours t
(avec i = t − h ≥ 1, et r vivante de i à t). Pour chacun, on tire une
donneuse parmi les admissibles, s'il en existe.

Le nombre de greffes de ce tirage a été compté sur les vies du professeur,
sans modèle, avant d'écrire ce protocole : 219 (h = 3), 217 (h = 4) et 173
(h = 5).

**Calcul.** Un seul fil de calcul par processus (amendement 1 du test 24).
Les greffes peuvent être réparties entre plusieurs processus : chaque
greffe est calculée seule, de la même façon.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie receveuse (10 000 tirages). Sur
toutes les greffes (h = 3, 4, 5) :

| | Prédiction | Critère |
|---|---|---|
| **FAR1** | Le besoin passe surtout d'état en état, compté comme au test 24 | la borne basse de (effet en chaîne − la moitié de l'effet total) est **> 0** |
| **FAR2** | Même chose, compté sans le terme non additif | la borne basse de (effet par les médiateurs seuls − la moitié de l'effet total) est **> 0** |

C'est plus exigeant qu'au test 24 :
- la part de la chaîne doit être **clairement** au-dessus de la moitié,
  pas seulement en moyenne ;
- elle doit l'être des deux façons de compter.

**Condition préalable** : moyenne de l'effet total ≥ **0,10**, borne basse
> 0. Sans elle, FAR1 et FAR2 ne sont pas jugés.

**Critère global** : FAR1 et FAR2, avec la condition préalable, test
valide.

**Publié sans seuil** :
- les résultats selon h (3, 4, 5) : total, direct, en chaîne, par les
  médiateurs seuls, terme non additif, et les parts ;
- les parts sur l'ensemble et leurs intervalles.

**Validité** (celle du test 24) :
- lecture prolongée sans greffe égale à la lecture du texte entier (écart
  ≤ 1e-4, 4 cas) ;
- greffe de soi : les trois lectures greffées égalent la lecture propre
  (écart ≤ 1e-6, 4 cas) ;
- mêmes positions des tokens portés chez la donneuse et la receveuse
  (vérifié à chaque greffe) ;
- au moins 150 greffes pour chaque h.

## Ce que le résultat dira

**Si FAR1 et FAR2 passent.** Au-delà de deux tours, le besoin porté par l'état d'un
tour atteint les choix surtout en passant par les états des tours
suivants. Chaque état reprend celui d'avant et le transmet : c'est une
transmission d'état en état, une récurrence au sens fonctionnel. L'agent
l'a apprise alors que son masque lui permettait de lire directement tout
son passé. L'observation du test 24 est confirmée sur des vies neuves.

**Si FAR1 passe et FAR2 échoue.** Plus de la moitié de l'effet disparaît
quand on coupe les états intermédiaires, mais ces états seuls n'en portent
pas la moitié : ils transmettent surtout **avec** l'état d'origine encore
lisible. La transmission d'état en état n'est que partielle.

**Si FAR1 échoue (condition préalable remplie).** Sur des vies neuves, à
trois tours et plus, la part transmise d'état en état n'est pas clairement
au-dessus de la moitié. L'observation du test 24 n'est pas confirmée.

**Si la condition préalable échoue.** À ces distances, la greffe ne change
pas assez le choix pour mesurer la route. FAR1 et FAR2 ne sont pas jugés.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti.
- L'agent a appris d'un professeur qui connaît les besoins.
- Remettre les médiateurs à leurs valeurs sans greffe crée un état que
  l'agent ne rencontre jamais : c'est la limite habituelle de ces
  décompositions. Plus h est grand, plus il y a de tours remis, ce qui
  peut à lui seul baisser l'effet direct. La lecture par les médiateurs
  seuls crée elle aussi un mélange jamais rencontré (tour i sans greffe,
  tours suivants issus de la greffe), mais elle ne dépend pas de la
  lecture directe. C'est pourquoi les deux façons de compter sont exigées.
- Il ne dira pas comment l'état est mis à jour d'un tour à l'autre.
- La transmission passe par les clés et valeurs des tokens portés,
  relues à chaque tour : ce n'est pas une boucle à l'intérieur du réseau.

## Précautions

Aucune vie neuve n'est vécue par un modèle : on relit des vies du
professeur.

## Exécution

- **Mesures** : `research/need_chain_far.py` (à écrire), torch sur le
  processeur local.
- **Verdicts** : numpy seulement, vérifiés en CI.
