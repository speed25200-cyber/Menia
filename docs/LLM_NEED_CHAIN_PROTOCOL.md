# Protocole pré-enregistré — Le besoin passe-t-il d'un état porté au suivant ? (test 24)

Rédigé le 4 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Test 22** (publié, valide). L'agent A ne voit son passé qu'à travers
  ses propres états portés : les tokens « Choix : » et l'action de chaque
  tour passé. Il choisit comme la règle « besoins » bien mieux que tout
  observateur de ses seules actions (MEM4 passe). Mais ce n'est pas une
  récurrence au sens strict : sous le masque, un tour peut lire
  **directement** les états portés de tout son passé.
- **Test 23** (en cours de publication). Au moment d'écrire ce protocole,
  nous connaissons les résultats de A, calculés avant la fin du témoin B :
  greffé dans une autre vie, l'état porté d'un tour fait choisir selon les
  besoins de la vie donneuse (environ 0,48 de l'effet de la règle, un ou
  deux tours plus tard). Un état aux mêmes besoins, venu d'une autre
  histoire, change peu le choix. Rien n'a été mesuré sur la question
  ci-dessous.
- **Ce qu'on ne sait pas.** Le besoin porté au tour i atteint le choix du
  tour t par deux routes possibles :
  - **directe** : le tour t lit lui-même l'état porté du tour i ;
  - **en chaîne** : l'état porté du tour i + 1 lit celui du tour i et le
    transmet, et ainsi de suite jusqu'au tour t.

  Si la chaîne domine, chaque état porté reprend le précédent et le met à
  jour : c'est une récurrence au sens fonctionnel, apprise alors que le
  masque permettait de tout lire directement.

**Question.** Quand on greffe l'état porté du tour i, quelle part de son
effet sur le choix du tour t passe par les états portés des tours
intermédiaires ?

## L'agent et les vies

- **L'agent** : A, le « porte » du test 22, après 4 000 itérations
  (`artifacts/llm-need/memory-no-leak/route-3`), sous son masque, textes
  remplis (lignes de même longueur).
- **Les vies** : les 128 vies du professeur tenues à l'écart du test 22
  (flux `[270926, 45]`).
- Le témoin B n'est pas mesuré : la question porte sur la façon dont A fait
  passer son besoin d'un tour à l'autre.

## La greffe et les trois lectures

Une receveuse r, un tour i, un tour t = i + h (h = 2 ou 3), r vivante de i
à t. Une donneuse d, vivante au tour i, avec **la même action écrite au
tour i** que r, **d'autres besoins** (E, N) que r au tour i, et un **effet
de la règle non nul** au tour t. L'effet de la règle se calcule comme au
test 23 : on rejoue r à partir des besoins de d au tour i, avec les
actions et les événements de r.

Les **médiateurs** sont les tokens portés (« Cho », « ix », « : » et
l'action) des tours i + 1 à t − 1.

On lit P(R) au « Choix : » du tour t dans quatre cas :
1. **propre** : la vie de r, sans greffe ;
2. **totale** : les clés et valeurs des tokens portés du tour i sont
   celles de d, dans toutes les couches ; les tours suivants sont calculés
   avec cette greffe ;
3. **directe** : comme la totale, puis on remet aux médiateurs les clés et
   valeurs qu'ils ont dans la lecture propre. Seule la route directe reste
   ouverte ;
4. **par les médiateurs seuls** (publiée sans seuil) : la lecture propre,
   où seuls les médiateurs reçoivent les clés et valeurs qu'ils ont dans
   la lecture totale ; le tour i reste celui de r.

Pour chaque greffe, avec e l'effet de la règle (+1 ou −1) :
- **effet total** = (P totale − P propre) × e ;
- **effet direct** = (P directe − P propre) × e ;
- **effet en chaîne** = (P totale − P directe) × e.

**Tirage** (flux `[270926, 51, 0]`). On parcourt les 128 vies receveuses
dans l'ordre. Pour chaque r et chaque h ∈ {2, 3}, on tire au plus 3 tours
t (avec i = t − h ≥ 1, et r vivante de i à t). Pour chacun, on tire une
donneuse parmi les admissibles, s'il en existe. Une donneuse dont le rejeu
fait mourir r avant t n'est pas admissible.

Le nombre de greffes de ce tirage a été compté sur les vies du professeur,
sans modèle, avant d'écrire ce protocole : 266 (h = 2) et 225 (h = 3).
Avec 2 tours par vie, h = 3 n'en donnait que 142, sous le minimum de 150 :
d'où 3 tours.

## Prédiction fixée

Intervalles bootstrap à 95 % par vie receveuse (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **CHAIN** | Le besoin passe surtout d'un état porté au suivant | sur toutes les greffes (h = 2 et 3) : moyenne de l'effet en chaîne ≥ **la moitié** de la moyenne de l'effet total, et la borne basse de l'effet en chaîne > 0 |

**Condition préalable** : moyenne de l'effet total ≥ **0,10**, borne basse
> 0. Sans elle, la part de la chaîne n'a pas de sens, et CHAIN n'est pas
jugé.

**Critère global** : CHAIN, avec la condition préalable, test valide.

**Publié sans seuil** :
- l'effet direct ;
- l'effet par les médiateurs seuls ;
- les résultats selon h (2 ou 3) ;
- la part de la chaîne et son intervalle.

**Validité** :
- **lecture prolongée** : sans greffe, lire jusqu'au tour i, puis
  prolonger jusqu'au tour t − 1, puis finir, égale la lecture du texte
  entier (écart ≤ 1e-4, 4 cas) ;
- **greffe de soi** (d = r) : P totale, P directe et P par les médiateurs
  seuls égalent P propre (écart ≤ 1e-6, 4 cas) ;
- les positions des tokens portés des tours i à t − 1 sont les mêmes chez
  d et r ;
- au moins 150 greffes pour chaque h.

## Ce que le résultat dira

**Si CHAIN passe.** Le besoin porté par l'état d'un tour atteint les choix
suivants surtout en passant par les états des tours qui suivent. Chaque
état porté reprend le précédent et le transmet. C'est une mise à jour
d'état en chaîne, une récurrence au sens fonctionnel. L'agent l'a apprise
alors que son masque lui permettait de lire directement tout son passé.

**Si CHAIN échoue (condition préalable remplie).** Les choix lisent surtout
directement l'état ancien ; les états suivants en transmettent moins de la
moitié. Ce n'est pas une chaîne, mais surtout une lecture directe du passé.

**Si la condition préalable échoue.** La greffe à deux ou trois tours ne
change pas assez le choix pour mesurer la part de la chaîne. CHAIN n'est
pas jugé.

**Ce que le résultat ne dira pas.**
- Ce n'est pas la preuve d'un ressenti.
- L'agent a appris d'un professeur qui connaît les besoins.
- Remettre les médiateurs à leurs valeurs propres crée un état que l'agent
  ne rencontre jamais (un tour i greffé, des tours suivants non). C'est la
  limite habituelle de ces décompositions.
- Une chaîne ne dit pas comment l'état est mis à jour (par exemple,
  s'il ajoute l'effet de l'événement) : ce test ne le mesure pas.

## Précautions

Aucune vie neuve n'est vécue : on relit des vies du professeur, qui ne
sont vécues par aucun modèle.

## Exécution

- **Mesures** : `research/need_chain.py` (à écrire), torch sur le
  processeur local.
- **Verdicts** : numpy seulement, vérifiés en CI.
