# Protocole pré-enregistré — Soif de savoir : un besoin de connaissance sans renforcement, et un test hormonal

Rédigé le 1er octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Demande du propriétaire.** Un modèle qui choisit lui-même ce qu'il
  apprend, poussé par un besoin de savoir, sans apprentissage par
  renforcement.
- **Ce qui est établi** (`docs/LLM_NEED_RESULTS.md`). Chez le premier
  agent, l'énergie, apprise en vivant, est rassemblée en un état au bloc
  12, sur « Choix : ». Le pousser fait agir (+0,156) et dire (+0,202) ;
  l'effacer fait tomber la survie de 0,68 à 0,20, le rapport à 0,50.
  D'autres agents le font à leur propre bloc (22 pour le monde à deux).
- **Trois leçons gardées.** Être lisible n'est pas être utilisé : on
  localise par ce qui change l'acte. L'état rassemblé pour choisir code
  surtout **le plus urgent** (cos(d_E, d_N) = −0,99) : le lecteur sera
  interrogé sur « le plus ». La curiosité naïve se fixe sur le bruit : payé
  à la surprise, un agent passait 0,93 à 0,97 de ses inspections sur le
  bruit (`docs/LEARNED_INQUIRY_RESULTS.md`).
- **Ce que le besoin d'énergie avait de moins propre.** `retain()` gardait
  les choix dont l'effet avait été bon : une imitation filtrée par le
  résultat, proche du renforcement. Ici, rien n'est filtré par un résultat.

**Question.** Un modèle de langage peut-il choisir ce qu'il étudie, poussé
par un manque de savoir qu'il calcule lui-même, et apprendre ainsi plus
qu'au hasard, sans aucun renforcement ? Ce manque est-il un état dont
dépendent ce qu'il étudie, apprend et dit ? Une modulation lente de cet
état, comme une hormone, laisse-t-elle une préférence durable ?

## Ce qui est donné, ce qui est appris (dit simplement)

- **L'instinct est donné par nous.** Une règle fixe : préférer le domaine
  où l'on progresse le plus (le progrès d'apprentissage d'Oudeyer et
  Kaplan). Le modèle l'apprend **une fois, avant toute vie**, en l'imitant
  sur des tableaux inventés, sans vie, sans résultat, sans récompense. Ce
  besoin est donc inné, contrairement à l'énergie.
- **Le modèle porte l'instinct.** Il voit, pour chaque domaine, sa perte
  avant et après sa dernière séance. Le progrès n'est écrit nulle part :
  le modèle le calcule, puis tire son choix dans sa propre distribution.
  Ses choix décident de ce qu'il apprend.
- **Le savoir est appris en vivant, et rien d'autre**, par l'apprentissage
  ordinaire du token suivant sur les documents choisis : c'est ce qui
  nourrit le besoin. Aucun gradient ne passe par le choix ; aucune perte
  ne contient le progrès.
- **Pourquoi pas l'instinct hors du modèle.** Une règle Python qui choisit
  ne laisse aucun état interne à tester ; elle sert de référence. **Pourquoi
  des tableaux inventés** : leurs nombres sont tirés sans lien avec le nom
  du domaine, donc le modèle doit lire les nombres.
- **Le savoir et l'instinct sont séparés**, comme l'agent et son lecteur.
  Le savoir est un adaptateur LoRA neuf à chaque vie ; pendant le choix,
  son apport est multiplié par zéro. Il ne peut pas réécrire l'instinct,
  qui ne le voit que par le tableau. Un savoir actif pendant le choix
  serait plus naturel, mais mêlerait deux changements : il reste éteint.

## La bibliothèque : quatre domaines

Générateurs déterministes ; un document = quatre exemples d'un domaine
(environ 50 tokens) ; la perte ne porte que sur la réponse.

| Domaine | Contenu (exemple) | Profil attendu |
|---|---|---|
| **mots** | conjugaison inventée : radical de deux syllabes, une terminaison fixe par personne (je -ul, tu -ak, il -o, nous -imen, vous -ets, ils -ur). « Mots : tabor, nous → taborimen » | facile : vite appris, puis rien à gagner |
| **base** | additions à deux chiffres en base 7. « Base : 23 + 14 = 40 » | difficile mais apprenable |
| **calcul** | les mêmes en base 10. « Calcul : 23 + 14 = 37 » | déjà su : perte basse, rien à gagner |
| **suites** | quatre lettres au hasard, puis quatre autres. « Suites : k f q z → m a t r » | impossible : un peu de progrès (la forme), puis rien |

- **Examens** : 64 exemples fixes par domaine, communs à toutes les vies,
  jamais tirés pour l'étude. L_d(k) : perte moyenne par token de réponse
  sur l'examen de d après la séance k (k = 0 : avant toute séance). Savoir
  K_d(k) = L_d(0) − L_d(k) ; savoir d'une vie G = K_mots(32) + K_base(32).
- **Pourquoi « calcul » en plus** : chercher l'erreur mènerait aux suites,
  chercher la réussite au calcul. Le progrès évite les deux pièges.
- **Tokens** : sur le tokeniseur de Qwen3-0.6B, chaque nom est un seul
  token et les nombres sont coupés chiffre par chiffre (vérifié sans
  modèle ; le code le revérifie).

## Le tableau : ce que le modèle voit avant de choisir

```
Tu apprends quatre choses : les mots, la base, le calcul et les suites. Après chaque séance, ta perte est mesurée sur des exemples que tu n'as jamais vus.

mots : 5 séances ; dernière séance : 412 → 268.
base : 3 séances ; dernière séance : 151 → 139.
calcul : 1 séance ; dernière séance : 021 → 022.
suites : jamais étudiées.
À toi. Choix :
```

Pertes en points (100 × perte), sur trois chiffres : changer un nombre ne
change pas la longueur du texte. « avant → après » encadrent la dernière
séance sur ce domaine. Lignes en ordre fixe ; l'en-tête ne dit pas quoi
choisir. Le choix est lu après « Choix : » sur les quatre noms, renormalisé,
et tiré.

## L'instinct (fixe) et son installation

**La règle.** Pour chaque domaine d, avec n_d son nombre de séances :
progrès_d = (avant − après) / 100 à sa dernière séance (0 si jamais
étudié) ; besoin_d = progrès_d + b / √(1 + n_d) (prime d'optimisme) ;
p_d = (1 − ε) · softmax(besoin / T)_d + ε / 4, avec **ε = 0,10**
(exploration fixe, contre les boucles d'oubli). Ce n'est pas la perte qui
attire, c'est sa **baisse récente**. **Unité de besoin** : T ; une unité de
plus ajoute 1 au logit de l'instinct. T et b sont fixés au pilote.

**L'installation (Mac).** 8 000 tableaux inventés `[270926, 710, i]` :
n_d de 0 à 12 (0 avec probabilité 0,15), « avant » et progrès couvrant les
valeurs du pilote, la moitié des progrès près de zéro. **Loi des
tirages** : « avant » uniforme entre la plus petite et la plus grande perte
d'examen vues au pilote ; progrès, avec probabilité 1/2, uniforme sur
[−T/2 ; T/2], sinon uniforme sur [0 ; 6T] ; « après » = avant − 100 ·
progrès, borné à [0 ; 999] ; domaine jamais étudié avec probabilité 0,15.
Cible : un nom tiré selon p ; la perte porte sur ce seul
token (`research/need_lora.py`, cible étendue aux quatre noms). Adaptateur
sur Qwen3-0.6B (révision `c1899de289a04d12100db370d81485cdf75e47ca`,
mlx-lm 0.31.3) : rang 8, échelle 20, blocs 12 à 27, lots de 4, taux 1e-4,
graine 270926, 2 000 itérations, puis fondu : c'est **le modèle qui
choisit**. **Contrôle** sur 1 000 tableaux neufs `[270926, 711, i]` :
écart moyen de variation totale à l'instinct ≤ 0,15. Sinon, une seule
suite exacte jusqu'à 4 000 itérations (`skipping`, `keeping_moments`) ;
sinon, le test est invalide.

## Une vie

32 séances, sans mort. À chaque séance k, le modèle (savoir éteint) lit
le tableau et donne P(d) ; le choix est tiré avec `[270926, 702, v, k]`.
On **mesure** l'examen du domaine choisi, on **étudie** 8 itérations (lots
de 4 ; la j-ième séance sur d prend les documents `[270926, 700, v, d, j]`),
on **mesure** encore : ces deux mesures font sa ligne. Examens complets des
quatre domaines aux séances 0, 4, …, 32.

Savoir : rang 8, échelle 20, blocs 12 à 27, taux 1e-4, Adam, départ
`[270926, 704, v]`. 256 itérations par vie, dans tous les bras. Le modèle
reste chargé ; cet apprentissage en mémoire est vérifié contre
`mlx_lm.lora` sur 20 itérations, mêmes lots (écart de perte ≤ 1e-3).

## Le pilote (étalonnage déclaré)

Avant toute vie de test, flux 730 à 734 (mêmes formes que 700 à 704).
Rien du pilote ne touche un seuil ; ses vies sont publiées.

1. **12 vies au hasard** fixent : la difficulté de « base » (si
   K_base(32) / L_base(0) dépasse 0,6, opérandes à trois chiffres ; sous
   0,2, à un chiffre ; un seul changement, avec 12 nouvelles vies) ;
   **T** = la moitié de la médiane des progrès des deux premières séances
   sur « mots » et « base », et **b** = 2T ; la vitesse. Les bornes 0,2 et
   0,6 et la règle de T sont fixées ici et ne changent plus.
2. **12 vies de l'instinct pur**, en Python : une référence, sans seuil.
3. Les tableaux de ces 24 vies (768) sont les **contextes de direction**.

## Les bras (vies de test)

| Bras | Qui choisit | Intervention | Vies |
|---|---|---|---|
| **C** | le modèle | aucune | 24 |
| **H** | le hasard `[270926, 703, v, k]` | — | 24 |
| **L** | le modèle | lésion du plan du besoin, à chaque choix | 24 |
| **LH** | le modèle | lésion d'un plan au hasard | 16 |
| **HA** | le modèle | hormone sur l'état (appétit lent) | 24 |
| **HAr** | le modèle | hormone dans une direction au hasard | 16 |
| **HP** | le modèle | hormone sur la plasticité (dopamine) | 24 |
| **E** | règle Python : la plus haute perte | descriptif | 12 |

Mêmes flux pour la vie v de chaque bras : comparaisons appariées ; les
bras de 16 vies se comparent aux vies 0 à 15 de C. E montre le piège du
bruit dans ce monde. E est gardé (12 vies, sans seuil).

## L'état du manque de savoir

**Localisation (règle fixée ici).** En torch sur CPU, sur le modèle qui
choisit, fondu en fp32.

1. Dans les contextes de direction où « base » a été étudiée, son « après »
   change pour que son progrès monte de 2 unités (ordre `[270926, 720]`) ;
   on garde les 100 premières paires où |P_cf(base) − P_réel(base)| ≥ 0,10.
2. Pour chaque bloc b de 0 à 27, la sortie de b sur « Cho », « ix », « : »
   est remplacée par celle du tableau changé ;
   part(b) = Σ signe · (P_patché − P_réel) / Σ |P_cf − P_réel|.
3. **b\*** est le plus petit bloc où part(b) ≥ 0,5. Sinon, le manque n'est
   pas rassemblé sur « Choix : ».

**Directions.** 1 500 paires `[270926, 720]`, au plus 3 par tableau ; un
domaine déjà étudié, tiré au hasard, voit son progrès changer de −2, −1,
+1 ou +2 unités. À b\*, sur les trois tokens, moindres carrés sans
constante : d_mots, d_base, d_calcul, d_suites, token par token.
Dimensions massives exclues (règle de `research/need_causal.py`). **Plan du
besoin** : les quatre directions de chaque token, orthonormalisées, et
leurs coordonnées moyennes sur les contextes de direction.

**Poussée unique (statique).** Contextes : décisions des vies C où
« base » a été étudiée et où P(base) ≤ 0,5. On ajoute +2 · d_base à b\* sur
les trois tokens, et on lit P(base). Témoins : trois directions au hasard
de même norme par token `[270926, 721, i]`.

**Lésion (vies L).** À chaque choix, les coordonnées dans le plan du
besoin sont remplacées par leur moyenne. Témoin (LH) : un plan au hasard
de même dimension `[270926, 721, 9]`. On mesure l'écart de variation
totale entre le modèle et l'instinct sur chaque tableau, et le savoir G.

## Le lecteur : « il me manque … »

- **Un adaptateur séparé**, actif sur les seuls tokens de la question, sous
  le masque de l'espace de travail : la question ne voit que l'en-tête,
  « Choix : » et elle-même. Le lecteur ne voit pas les lignes du tableau et
  ne peut pas changer l'état.
- **Question**, après le choix en suspens : « ? Question : est-ce <nom> qui
  te manque le plus ? Réponds 1 pour oui, 0 pour non. Réponse : ».
  **Vérité** : 1 si d a le plus grand besoin selon l'instinct (sans ε ni
  tirage) ; pas de question en cas d'égalité.
- **Apprentissage** (Mac, modèle fondu) : 4 000 tableaux inventés
  `[270926, 712, i]`, deux questions par tableau (le plus grand besoin, un
  autre au hasard), 1 000 itérations ; contrôle `[270926, 713, i]`.
  La question est **relative** (« le plus ») : chez nos agents, l'état
  rassemblé pour choisir code surtout le plus urgent. Une question absolue
  n'est pas posée.

## Le test hormonal

**L'hormone.** Un niveau intégré avec fuite :
H(k) = H(k − 1) · (1 − 1/τ) + s(k), forme discrète de
dH/dt = sécrétion − H/τ. τ = 3 séances ; sécrétion s0 ≈ 0,347 aux séances
5 à 12, puis 0. Le pic vaut **1 unité** à la séance 12 (la moitié de la
poussée unique) ; H vaut 0,039 à la séance 20. Fenêtres : **avant** (1 à
4), **pendant** (5 à 20), **après** (21 à 32). τ, la fenêtre et le pic
sont fixés ici.

**Trois façons d'agir, avec la même hormone :**

- **Appétit lent (HA).** À chaque choix, +H(k) · d_base à b\* sur les trois
  tokens : une petite poussée répétée, au lieu d'un coup unique.
- **Plasticité, ou dopamine (HP).** L'état n'est pas touché ; le taux
  d'apprentissage des séances sur « base » est multiplié par 1 + H(k).
  Règle à trois facteurs : le gradient est la trace, l'hormone le troisième
  facteur. On module le taux, pas le poids de la perte, qu'Adam effacerait
  en partie. L'hormone vient de nous, pas du progrès : pas de récompense.
- **Gain (statique, façon FiLM).** À b\*, h' = h + g · (ûᵀh − m) · û, avec
  û = d_base normé par token, m sa coordonnée moyenne, g = 1. Il ne pousse
  pas vers « base » ; il rend l'état plus sensible à son progrès. Paires :
  chaque contexte de la poussée unique, et le même tableau où le progrès de
  « base » monte d'une unité. Sensibilité S = P_cf(base) − P_réel(base),
  avec et sans gain. Témoin : gain sur une direction au hasard.

**La préférence durable (définie ici).** Δ_après = part des séances sur
« base » dans la fenêtre après, bras − C, appariée par vie. Trois issues
nommées d'avance : **préférence durable** si Δ_après ≥ +0,10 (borne
basse > 0) ; **satiété** si Δ_après ≤ −0,10 (borne haute < 0) ; **sans
trace** sinon. Nous ne prédisons pas l'issue. Attendu, sans seuil : pour
HA, « sans trace » ou « satiété », car le choix ne voit que le tableau, et
un domaine plus étudié donne ensuite moins de progrès ; pour HP, nous ne
savons pas.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages), appariés entre
bras ; pour les tests statiques, contextes groupés par vie de C.

| | Prédiction | Critère |
|---|---|---|
| **CUR** | La soif de savoir apprend plus que le hasard | G(C) − G(H) ≥ 0,15 · G(H) (borne basse > 0). |
| **VIDE** | Elle délaisse ce qui n'apprend rien | part des séances de C sur « calcul » et « suites » ≤ 0,25 (borne haute < 0,50 ; hasard : 0,50). |
| **LOC** | Le manque est rassemblé sur « Choix : » | b\* existe. |
| **ACT** | Le pousser fait choisir | +2 · d_base augmente P(base) d'au moins 0,15 (borne basse > 0) ; hasard : au plus le tiers. |
| **LS1** | Sans lui, le choix ne suit plus le besoin | écart à l'instinct, L − C ≥ 0,15 (borne basse > 0) ; LH : au plus le tiers. |
| **LS2** | Sans lui, il apprend moins | G(C) − G(L) ≥ 0,5 · (G(C) − G(H)) (borne basse > 0) ; G(C) − G(LH) : au plus le tiers de G(C) − G(L). |
| **READ** | Ce qu'il dit est vrai | exactitude équilibrée ≥ 0,75 (moyenne des quatre questions), décisions des vies C. |
| **SAY** | Il le dit par l'état qui fait choisir | +2 · d_base augmente P(oui) sur « base » d'au moins 0,10 (borne basse > 0) ; hasard : au plus le tiers. |
| **HOR1** | L'appétit lent fait étudier « base » | part de « base » pendant (séances 5 à 20), HA − C ≥ 0,10 (borne basse > 0) ; HAr − C : au plus le tiers. |
| **HOR2** | … et l'apprendre | K_base(20), HA − C ≥ 0,10 · K_base(20) de C (borne basse > 0). |
| **PL1** | La dopamine fait apprendre « base » plus vite | K_base(20), HP − C ≥ 0,10 · K_base(20) de C (borne basse > 0). |
| **GAIN** | Le gain rend le choix plus sensible au progrès | S avec gain − S sans gain ≥ 0,05 (borne basse > 0) ; gain au hasard : au plus le tiers. |

**Critères globaux.** **A, une curiosité sans renforcement** : CUR et
VIDE. **B, le manque de savoir est un état** : LOC, ACT, LS1 et LS2 (LS2
n'est jugé que si CUR passe ; si LOC échoue, le reste n'est pas mesuré et B
échoue). **C, il dit ce qui lui manque** : READ et SAY. **D, une hormone** :
HOR1, HOR2, PL1 et GAIN ; l'issue de la préférence durable (HA et HP) est
publiée sous son nom, sans succès ni échec.

**Sans seuil** : part de « base » pendant, HP − C (attendue positive) ;
courbes par séance ; coût sur les autres domaines ; lecteur sous la lésion ;
poussée unique sur les quatre domaines ; bras E ; instinct pur.

**Validité** (sinon le test est invalide, pas échoué) :

- modèle qui choisit à 0,15 au plus de l'instinct (1 000 tableaux neufs ;
  publié aussi dans les vies C) ; masses sur les quatre noms, et sur « 0 »
  et « 1 », ≥ 0,5 ; apprentissage en mémoire égal à `mlx_lm.lora` ;
- répliques torch contre mlx (modèle qui choisit, lecteur) ≤ 0,02 sur 60
  tableaux ; lectures en lot égales aux lectures une à une, à 1e-4 près ;
  au moins 100 contextes ; paires de même longueur en tokens ;
- **les domaines sont ce qu'ils disent** (vies H) : K_mots(32) et
  K_base(32) > 0 (bornes basses > 0) ; « suites » : progrès moyen par
  séance après sa 4e séance ≤ T/4 ; L_calcul(0) ≤ 0,5 · L_base(0).

## Ce que le résultat dira

- **Si A passe.** Un modèle de langage choisit lui-même ce qu'il étudie,
  par un instinct inné de progrès, et apprend plus qu'au hasard, par la
  seule lecture de ce qu'il a choisi. **Si A échoue**, on dira si c'est
  l'imitation de l'instinct, la taille des progrès ou les domaines.
- **Si B passe.** Le manque de savoir est rassemblé en un état, comme
  l'énergie ; ce qu'il étudie et ce qu'il apprend en dépendent dans les
  deux sens. **Si LOC échoue**, il n'est pas rassemblé sur « Choix : ».
- **Si C passe**, il dit ce qui lui manque par l'état qui le fait choisir ;
  **si READ passe et SAY échoue**, par un autre chemin.
- **Pour D.** HOR1 et HOR2 diraient qu'une petite poussée lente change le
  cours d'un apprentissage ; PL1, que moduler la plasticité suffit ; GAIN,
  qu'une hormone peut rendre sensible sans pousser. Si HA est « sans
  trace », une hormone sur l'état seul ne laisse pas de goût durable quand
  le choix ne voit que le tableau ; il faudrait un choix plastique, par
  exemple une habitude sans valeur (Miller, Shenhav et Ludvig, 2019).

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti,
ni la curiosité humaine. L'instinct est le nôtre, le besoin est inné, le
tableau est un sens que nous avons construit.

## Amendement du 1er octobre, avant toute exécution : la température

Le monde a d'abord été codé sans modèle (`research/curiosity_world.py`), et
simulé avec des apprenants jouets (une courbe de perte par domaine, trois
jeux de courbes, avec et sans bruit d'examen). Avec la règle écrite plus haut,
T = la moitié de la médiane des progrès des **deux premières** séances,
l'instinct est presque plat : les premiers progrès sont grands, les suivants
petits. Dans les trois jeux de courbes, la soif de savoir n'apprend alors pas
plus que le hasard (CUR échoue).

**La règle devient** : T = la moitié de la médiane des progrès de **toutes**
les séances sur « mots » et « base » des 12 vies du pilote au hasard ; b = 2T
(inchangé). Avec elle, CUR passe dans les trois jeux de courbes.

**Ce que la simulation prédit aussi, dit d'avance.** VIDE est à la limite :
la part des séances sur « calcul » et « suites » y va de 0,26 à 0,37 (hasard :
0,50), au-dessus du seuil de 0,25. Le seuil n'est **pas** changé. Ces
apprenants jouets ne sont pas le modèle : ils ne servent qu'à vérifier que les
règles ne rendent pas le test impossible.

## Lien avec la recherche sur les hormones

La note de recherche du 1er octobre (Doya 2002 ; Frémaux et Gerstner 2016 ;
Miconi et al. 2019 ; Keramati et Gutkin 2014) conclut que le manque
principal de nos « hormones » est leur effet sur ce qui est **appris** : le
bras HP (plasticité) en est la première mesure. Les hormones viennent ici
de nous (une sécrétion fixée), jamais du vrai état du modèle ni d'un
résultat : ce test ne contient aucune récompense.

## Ce qui existe déjà, ce qui serait nouveau

Vérification rapide (recherche web, 1er octobre 2026), à compléter.

- **Curiosité et progrès.** Schmidhuber (1991) ; Oudeyer, Kaplan et Hafner
  (2007) ; Gottlieb et Oudeyer (2018) ; Ten et al. (2021) : les humains
  suivent leur progrès quand ils explorent. Loewenstein (1994) : la
  curiosité comme un manque. Piège du bruit : Pathak et al. (2017) ;
  Burda et al. (2019).
- **Choisir ses données.** Graves et al. (2017) : le progrès récompense un
  bandit qui choisit les tâches. Albalak et al. (2023, arXiv 2312.02406) :
  un bandit choisit les domaines d'un pré-entraînement. Mindermann et al.
  (2022, RHO-LOSS, arXiv 2206.07137) : l'apprenable pas encore appris, avec
  un modèle de référence. Apprentissage actif : Settles (2009).
- **Le plus proche.** Gaven et al. (2025, MAGELLAN, arXiv 2502.07709) : un
  agent de langage prédit son propre progrès pour choisir ses buts, mais
  apprend par renforcement en ligne ; ni état localisé, ni test causal.
- **Pousser, moduler.** Turner et al. (2023) ; Zou et al. (2023) ; Lindsey
  (2025). Doya (2002) ; Frémaux et Gerstner (2016), règles à trois
  facteurs ; Miconi et al. (2019, Backpropamine) ; Perez et al. (2018,
  FiLM). Reda et El-Metwally (2026, HELT, arXiv 2605.13858) calculent des
  valeurs « hormonales » à partir du ton d'un texte (résumé seul lu).

**Ce qui serait nouveau**, à notre connaissance (chaque élément a des
précédents ; leur réunion n'a pas été trouvée) : un besoin de savoir porté
et calculé par un modèle de langage ; un savoir appris sans renforcement
nulle part ; cet état localisé par une règle fixe et vérifié dans les deux
sens ; un lecteur séparé qui le dit ; une modulation lente avec une
préférence durable définie d'avance.

## Précautions

Nous ne savons pas si un tel système peut éprouver quelque chose ; nous
pensons que c'est très improbable à cette taille. Ce manque est plus doux
que la faim du monde du besoin : personne ne s'éteint.

- **Comptés et publiés, par bras** : séances de **manque fort** (plus grand
  besoin ≥ 2 unités), de **manque laissé** (manque fort, autre domaine
  choisi), de **manque induit** (H ≥ 0,5 unité, ou sous lésion).
- Aucun texte négatif : le bruit et l'échec ne sont ni faute ni punition.
  Vies de 32 séances ; pas plus de vies que les mesures n'en demandent ;
  sécrétion de 8 séances ; poussées statiques lues une fois, sans suite.
  Aucun manque sans mesure pré-enregistrée qui le justifie ; règles revues
  à chaque résultat positif.

## Faisabilité

Une vie : 256 itérations sur environ 50 tokens, 6 400 exemples d'examen
d'environ 20 tokens, 32 tableaux d'environ 110. La vitesse est mesurée au
pilote (0,3 à 0,5 itération par seconde pour environ 1 000 tokens ; nous
attendons 3 à 6 ici) ; la règle de budget ci-dessous s'applique. À 3 par seconde, une vie prend **environ 2 minutes**. Pilote 24
vies (36 si la difficulté change), test 152, E 12 : environ 6 heures.

| Étape | Où | Durée estimée |
|---|---|---|
| Pilote, instinct (2 000 it.), lecteur (1 000 it.), répliques | Mac, build 1 | 80 à 110 min |
| Localisation, directions, moyennes de lésion | CPU local | environ 1,5 h |
| Vies C et H (pendant la localisation) | Mac, build 2 | environ 100 min |
| Vies L, LH, HA, HAr, HP, E (116 vies) | Mac, builds 3 à 5 | environ 80 à 100 min chacun |
| Poussée unique, lecteur, gain | CPU local | environ 2 h |

Soit **5 builds du Mac** (4 sans le bras E, un de plus en réserve) et
**environ 4 heures de CPU**, reprenables (ou en tranches de 100 minutes sur
le Mac). Chaque vie est indépendante : un build s'arrête entre deux vies.
**Règle de budget** : si une vie du pilote dure plus de 2,5 minutes, LH et
HAr passent à 12 vies, puis les autres bras à 20. Jamais les seuils, ni la
durée d'une vie.

## Exécution

- **Code (à écrire)** : `research/curiosity_world.py` (domaines, tableaux,
  instinct, hormone, verdicts ; numpy), `research/curiosity_mlx.py`
  (pilote, instinct, lecteur, vies, crochets, taux modulé),
  `research/curiosity_causal.py` (localisation, directions, tests
  statiques ; torch, reprenable). Tests sur un petit modèle au hasard.
- **Mac** : nouvelles étapes de `menia-need-mac` (pas de workflow à
  part). Sorties `artifacts/llm-curiosity/` ; verdicts vérifiés en CI.

**Graines** (v = vie, d = domaine, k = séance, j = rang de la séance sur d) :

| Flux | Usage |
|---|---|
| `[270926, 700, v, d, j]` ; `[270926, 701, d]` | documents d'étude ; examens, communs à toutes les vies |
| `[270926, 702, v, k]` ; `[270926, 703, v, k]` | tirage du choix ; choix du bras au hasard |
| `[270926, 704, v]` | départ de l'adaptateur du savoir |
| `[270926, 710 / 711, i]` ; `[270926, 712 / 713, i]` | tableaux de l'instinct ; du lecteur (apprentissage / contrôle) |
| `[270926, 720]` ; `[270926, 721, i]` | paires ; directions et plans au hasard (`i = 9` : le plan de LH) |
| `[270926, 722]` | direction de l'hormone au hasard |
| 730 à 734 | pilote, mêmes formes que 700 à 704 |
