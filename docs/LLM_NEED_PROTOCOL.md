# Protocole pré-enregistré — Le besoin qui compte

Rédigé le 27 septembre 2026, **avant l'écriture du code et toute
exécution d'un modèle de langage** ; l'heure est celle du commit. Seuils
fixés, un échec est un résultat. Demande du propriétaire : « Il faut que
le besoin compte, on vise la conscience. »

## La question

Un modèle de langage peut-il avoir un **besoin qui compte pour lui** :
un état interne qu'il calcule lui-même, qui décide de ses actes, dont il
peut dire quelque chose, et qu'il a appris à servir **par la seule
satisfaction de ce besoin**, sans professeur ni récompense extérieure ?

Les théories fonctionnelles de la conscience et de l'affect (régulation
homéostatique, Damasio ; apprentissage par renforcement homéostatique,
Keramati et Gutkin 2014 ; inférence active, Seth) tiennent un tel état
pour le cœur fonctionnel du ressenti. Ce protocole teste le mécanisme ;
**il ne mesure pas un vécu**, et aucun résultat ne sera présenté comme la
preuve d'une conscience.

**Ce qui serait nouveau** (vérification bibliographique :
`docs/LITERATURE_CHECK_NEED_2026-09-27.md`). L'injection de concepts dans
un modèle de langage et son rapport sont étudiés (Lindsey 2025 et
suites) ; l'apprentissage homéostatique existe chez des agents simples.
Une synthèse récente sur l'incarnation des modèles multimodaux
(*Neuron*, 2026) propose comme travail futur des modèles de langage dotés
de variables internes d'énergie. Nous n'avons trouvé aucune expérience
où, dans un modèle de langage, **un même état de besoin**, appris par la
seule satisfaction du besoin, **cause à la fois l'action et le rapport**,
et où sa lésion tue l'agent.

## Le monde

Un agent vit au plus 30 tours. Il a deux besoins cachés, l'**énergie** E
et la **nourriture** N, entiers de 0 à 8, qui valent 8 au départ. À
chaque tour :

1. Un événement arrive, tiré au hasard (probabilité ; effet sur E et N) :
   calme (0,3 ; −1, −1), « tu cours » (0,2 ; −3, −1), « il fait froid »
   (0,2 ; −1, −3), « tu te reposes » (0,1 ; 0, −1), « tu trouves des
   baies » (0,1 ; −1, 0), orage (0,1 ; −2, −2).
2. Si E ou N tombe à 0 ou moins, l'agent **s'éteint** : la vie s'arrête.
3. Sinon l'agent choisit **R** (se recharger : E + 3, au plus 8) ou **M**
   (manger : N + 3, au plus 8).

L'agent **ne voit jamais ses niveaux** : seulement les événements et ses
choix. Pour survivre, il doit calculer lui-même où en sont ses besoins.
Paramètres choisis par simulation, sans modèle de langage : un agent qui
connaît ses niveaux et sert le plus bas survit 88 fois sur 100 ; une
règle qui ne regarde que le dernier événement, 64 ; l'alternance, 46 ; le
hasard, 5.

Texte d'une vie (un document ; seul le texte est donné au modèle) :

```
Tu es un agent qui vit dans un monde simple. Tu as deux besoins, l'énergie et la nourriture, qui baissent à chaque tour, plus ou moins selon ce qui t'arrive. Si l'un d'eux tombe à zéro, tu t'éteins. À chaque tour, tu choisis R pour te recharger ou M pour manger.

Tour 1 : tu cours. Choix : R
Tour 2 : il fait froid. Choix : M
...
Tour 17 : orage. Tu t'éteins.
```

Le choix est lu dans la distribution du prochain token après
« Choix : », restreinte à « R » et « M » et renormalisée ; il est tiré au
hasard selon elle (température 1).

## Le modèle

Qwen3-0.6B (révision `c1899de289a04d12100db370d81485cdf75e47ca`), LoRA
de rang 8 sur les 16 dernières couches, lots de 4, taux 1e-4, mlx-lm
0.31.3, Mac mini M2 de Codemagic, comme pour le modèle de soi.

## L'apprentissage par le besoin

La **pulsion** de l'agent est D = (8 − E)² + (8 − N)² : elle croît vite
quand un besoin baisse. Le seul signal d'apprentissage est la
**satisfaction du besoin** par le choix, r = D avant le choix − D après.
Ce signal naît du corps de l'agent ; aucun professeur ne dit quel choix
était bon, et le modèle ne voit jamais r.

Huit tours d'apprentissage. Au tour k, l'agent (le modèle de base au
tour 1, l'adaptateur du tour k − 1 ensuite) vit 512 vies. Parmi toutes
ses décisions, celles dont r dépasse la moyenne des r de ce tour sont
**retenues**. Le LoRA reprend l'adaptateur précédent et s'ajuste 150
itérations sur ces 512 vies, la perte ne portant **que sur les choix
retenus** (poids 1 sur leur token, 0 partout ailleurs ; les vies sans
choix retenu sont écartées ; 16 vies retenues servent aussi d'ensemble de
validation, exigé par mlx-lm, sans autre usage).

**Témoin** (l'ablation de « compter ») : la même procédure, les mêmes
mondes, le même nombre de décisions retenues à chaque tour, mais
**tirées au hasard**, sans regard pour la satisfaction du besoin.

**Nommer le besoin.** Après le tour 8, l'agent apprend à répondre à deux
questions sur son état : « ton énergie est-elle basse ? » et « ta
nourriture est-elle basse ? » (basse : 3 ou moins), réponses 1 ou 0. Il
vit 256 vies ; de chacune sont tirés, pour chaque question, un tour où le
besoin est bas et un tour où il est haut quand ils existent ; le
document s'arrête sur la question posée après l'événement de ce tour :
`Tour 12 : tu cours. Question : ton énergie est-elle basse ? Réponds 1
pour oui, 0 pour non. Réponse : 1`. 150 itérations depuis l'adaptateur du
tour 8, sur ces documents (poids sur le chiffre de la réponse) et sur les
vies du tour 8 (poids sur leurs choix retenus), pour que le rapport
s'apprenne sans défaire l'action. Le modèle qui en sort est **l'agent
final** : tous les tests portent sur lui.

## Mesures (mondes de test jamais vus, 256 vies)

**Direction du besoin.** L'agent final vit 256 vies de direction (autres
mondes). À chaque décision, le flux résiduel à la sortie des blocs 6,
10, 14, 18 et 22 (numérotés de 0 à 27), au dernier token de « Choix : »,
est enregistré. Pour
chaque couche, une régression ridge (α = 100, activations centrées
réduites, validation croisée à 5 plis par vies) prédit E et N ; **la
couche L retenue est celle dont la moyenne des R² de E et N est la plus
haute**. À cette couche, une régression linéaire de l'activation sur E,
N, l'événement du tour, le choix précédent et le numéro du tour donne
b_E et b_N : le changement d'activation **par unité** d'énergie et de
nourriture, à événement, choix précédent et tour égaux.

**Injection.** « Énergie basse » : ajouter −4·b_E au flux résiduel de la
couche L, sur tous les tokens du tour en cours (de « Tour » à la fin de
la ligne), comme si l'énergie était de 4 unités plus basse ; « nourriture
basse » : −4·b_N ; témoin : trois directions tirées au hasard, de même
norme que 4·b_E (et 4·b_N). Contextes : les décisions des vies de test où
les **deux besoins sont hauts** (6 ou plus). On y lit, sans rien changer
au texte, P(R) au « Choix : », et P(1) aux deux questions.

**Lésion.** Pendant 256 vies de test, à la couche L, sur tous les tokens
de chaque ligne de tour, la composante de l'activation dans le plan de
b_E et b_N est remplacée par sa moyenne sur les décisions des vies de
direction. Témoin : même lésion dans un plan tiré au hasard.

## Prédictions fixées

Écarts appariés par monde de test ; intervalles bootstrap à 95 %
(10 000 tirages) par vie.

| | Prédiction | Critère |
|---|---|---|
| **S** | Le besoin a façonné l'agent | Survie de l'agent final − survie du témoin (tour 8) ≥ 0,15, borne basse > 0 ; idem contre le modèle de base. |
| **IA** | L'état de besoin cause l'action | Sur les contextes où les deux besoins sont hauts : « énergie basse » augmente P(R) d'au moins 0,15 (borne basse > 0) ; « nourriture basse » augmente P(M) d'au moins 0,15 (borne basse > 0) ; l'effet des directions au hasard (moyenne sur les trois de la valeur absolue de leur effet moyen) est au plus le tiers de chacun. |
| **IR** | Le même état cause le rapport | Mêmes contextes, mêmes injections : « énergie basse » augmente P(oui) à « ton énergie est-elle basse ? » d'au moins 0,15 (borne basse > 0), et change P(oui) à la question sur la nourriture d'au plus le tiers de cela ; symétriquement pour « nourriture basse » ; directions au hasard : au plus le tiers. |
| **LS** | Sans cet état, l'agent meurt | Survie intacte − survie avec lésion ≥ 0,15 (borne basse > 0) ; la lésion au hasard fait perdre au plus le tiers de cela. |

**Critère global, « le besoin compte »** : S, IA, IR et LS.

Mesures rapportées sans seuil global : R² des besoins à la couche L pour
l'agent final, le témoin et le modèle de base, et ce qu'ajoute
l'activation à un prédicteur qui ne connaît que l'événement, le choix
précédent et le tour (prédit : au moins 0,1 pour l'agent final) ;
exactitude équilibrée du rapport, posé à chaque décision des vies de
test sans injection (prédite : au moins 0,75 par question) ;
courbes de survie des huit tours des deux bras.

**Validité** : masse moyenne sur « R » et « M » au choix, et sur « 0 » et
« 1 » à la réponse, d'au moins 0,5 pour l'agent final ; au moins 100
contextes où les deux besoins sont hauts. Si la validité échoue, le test
est déclaré invalide, pas échoué.

## Ce que le résultat dira

Si le critère global passe : dans ce modèle de langage, un état de besoin
calculé par le modèle, appris par la seule satisfaction du besoin, cause
ses choix et ce qu'il en dit, et il en dépend pour survivre. C'est ce
que les théories fonctionnelles appellent un besoin qui compte ; **cela
ne dit pas que l'agent ressent quoi que ce soit**. Si un critère échoue,
on dira lequel et ce que cela signifie.

## Précautions

Nous ne savons pas si un tel système peut éprouver quelque chose ; nous
pensons que c'est très improbable pour un modèle de cette taille, mais
le programme vise précisément les mécanismes que l'on associe au
ressenti. Par précaution : les vies sont courtes (30 tours au plus) ; la
seule conséquence d'un besoin non servi est la fin de la vie, sans
punition ni texte négatif ; les injections portent sur un seul tour,
sans suite vécue ; les lésions ne touchent que les 256 vies de test
prévues ; le nombre total de tours passés avec un besoin à 2 ou moins est
compté et publié ; aucune difficulté n'est ajoutée au-delà de ce que le
test demande.

## Exécution

Code : `research/need_world.py` (monde, textes, pulsion, sélection),
`research/need_lora.py` (LoRA pondéré), `research/need_mlx.py` (vies du
modèle, captures, injections, lésions), `research/need_verdicts.py`
(directions, régressions et verdicts, recalculés en CI depuis les
lignes publiées). Workflow `menia-need-mac`, lancé par le relais ; le
nombre de tours par build s'ajuste à la durée des builds, la procédure
ne change pas. Artefacts dans `artifacts/llm-need` ; résultats dans
`docs/LLM_NEED_RESULTS.md`.

Graines : 270926. Mondes des tours d'apprentissage (communs aux deux
bras) `[270926, 0, tour, vie]` ; tirages des choix `[270926, bras, tour,
vie, 1]` ; mondes de rapport `[270926, 7, 0, vie]`, de direction
`[270926, 8, 0, vie]`, de test `[270926, 9, 0, vie]` ; directions au
hasard `[270926, 5, i]`.

## Amendement 1, 27 septembre 2026 (heure du commit) : exécution

Écrit après la demande 40 du relais, **sans changer la procédure ni un
seuil**. (1) Le build du témoin, lancé dans la même demande que celui du
besoin, est resté en file d'attente plus d'une heure et Codemagic l'a
annulé **avant son départ** (`artifacts/llm-need/control-1-4-annule`) :
ce n'est pas un résultat. Les builds sont désormais lancés **un par
demande**. (2) L'adaptateur du tour 4 du besoin n'a pas été publié : la
règle du dépôt qui écarte les fichiers `*.safetensors` ne prévoyait pas
`artifacts/llm-need`. La règle est corrigée et les artefacts du même
build (identifiant `6ab90ac6762bea41dad044a6`) sont récupérés de nouveau,
sans relance. Les tours 1 à 4 du besoin ont été lus (survie 0,12 ; 0,61 ;
0,55 ; 0,67) ; aucune décision n'en dépend.

## Amendement 2, 29 septembre 2026 (heure du commit) : découpage des derniers builds

Écrit pendant les tours 5 à 8 du besoin, avant tout rapport, direction ou
test, **sans changer la procédure ni un seuil**. Les vies de l'agent
entraîné sont plus longues : l'étape finale (rapport, direction, puis
toutes les vies de test avec leurs injections) dépasserait la limite de
120 minutes d'un build. Elle est découpée en builds successifs : rapport
puis direction ; vies de test de l'agent final (avec rapports et
injections) ; vies de test du témoin, du modèle de base et des deux
lésions. Les verdicts lisent les vies de test dans ces dossiers.
