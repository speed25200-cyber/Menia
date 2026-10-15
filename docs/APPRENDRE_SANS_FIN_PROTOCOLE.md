# Programme V — Apprendre sans fin : la boucle jour–nuit, le choix de ce qu'on apprend, la croissance

Rédigé le 15 octobre 2026, **avant tout code et toute exécution** de ce programme. Ce document fixe les
questions, les bras, les mesures et les critères des tests 38 à 41. Comme pour les tests précédents,
chaque test a ensuite **un pilote, puis son propre protocole pré-enregistré** (seuils définitifs, graines,
flux), écrit après le pilote et avant tout apprentissage sur ses graines. Un seuil peut être resserré à
ce moment, jamais relâché. Un échec est un résultat, publié tel quel.

## 0. Le but, et ce qu'on peut en mesurer

Le but du propriétaire : **une IA qui veut apprendre seule, sans fin, sans nouveau pré-entraînement à
chaque fois, en commençant petite et en grandissant au fil de l'apprentissage, comme un humain.**

Mis en conditions mesurables, sur des familles de tâches vérifiables par programme :

| | Condition | Mesure |
|---|---|---|
| **A** | Elle apprend de sa propre activité | Gain sur un banc jamais entraîné, nuit après nuit |
| **B** | Elle va au-delà de ce qu'on lui donne | Gain sur un banc **hors de la distribution du générateur** |
| **C** | Elle n'oublie pas | Perte de bits/octet sur un texte tenu à l'écart, et précision sur les familles anciennes |
| **D** | Elle choisit ce qu'elle apprend | Part de l'apprentissage nocturne consacrée à ce qu'elle rate, contre le hasard |
| **E** | Sans fin | La pente de B reste positive sur les dernières nuits, générateur retiré |
| **F** | Elle grandit utilement | À capacité finale égale, grandir en apprenant fait mieux qu'être grand dès le départ |

Ce que le programme ne mesurera pas : un « vouloir » au sens vécu, une conscience, un apprentissage
illimité dans des poids fixes (`prophet_agi/docs/35` §0 : 229 M paramètres tiennent ≈ 57 Mo de faits).
« Sans fin » veut dire ici : une mémoire externe qui grandit, des poids qui gardent les savoir-faire,
une capacité qui s'ajoute quand elle manque.

## 1. Ce qui existe déjà, et ce que ce programme réutilise

Voir `docs/PROPHET_AGI_DANS_MENIA.md` pour le détail. En bref :

- **La boucle fermée** : `prophet_agi/scripts/closed_loop.py` (tâches tirées d'un générateur, vérifiées
  par programme, promues, ajustement avec rejeu, mesure par tour, reprise tour par tour). Bras `closed`,
  `oracle`, `frozen`.
- **Le proposeur** : `prophet_agi/prophet/agent/propose.py`, et la règle des six runs (`prophet_agi/docs/39`).
- **Les familles vérifiables** : `calc`, `lookup`, `files`, `count`, `replace`
  (`prophet_agi/prophet/agent/tasks.py`), et leurs bancs (`prophet_agi/prophet/eval/agent_bench.py`).
- **La comparaison appariée** : `prophet_agi/prophet/eval/paired.py` (bootstrap apparié).
- **L'ajustement par étapes en MLX** : `research/need_mlx.py` (LoRA rang 8, 16 couches, lots de 4).
- **Le pari et la greffe** : `research/tiny_wager.py`, tests 34 à 37.
- **La croissance neutre** : la récurrence sur donneur retenu, logits identiques au bit près
  (`prophet_agi/docs/28`).

## 2. Préalables (avant le test 38)

1. **Finir le test 37** (graines 44 à 49) et publier son verdict. Le test 39 en dépend : si le pari ne
   suit pas la fiabilité, le choix appris du test 39 part sans base mécanique, et on le dit.
2. **Vérification bibliographique**, comme `docs/LITERATURE_CHECK_*.md`, sur : « Language Models Need
   Sleep » (arXiv 2606.03979), « Gate-Zero Growth » (arXiv 2607.14571), ELLE et LOIRE (croissance
   neutre en apprentissage continu), SEAL (arXiv 2506.10943), TT-SI (ACL Findings 2026), Absolute Zero
   (arXiv 2505.03335), « Self-Monitoring Benefits from Structural Integration » (arXiv 2604.11914),
   « Operational Proto-Introspection in Looped Language Models » (arXiv 2607.18553), et le papier
   COLM 2026 sur les critères d'introspection (arXiv 2605.26242). Ce qui est déjà montré est retiré
   des revendications.
3. **Porter la boucle fermée sur un modèle de langue existant** (travail d'ingénierie, pas un test) :
   rendre les tâches des cinq familles en texte, lire les actions du modèle, garder les vérificateurs
   de Prophet. Contrôle de validité : avec le modèle de base figé, le banc de chaque famille donne la
   même précision sur deux exécutions (déterminisme, écart ≤ 1 tâche sur 60).

**Modèle de départ** : Qwen3-0.6B, celui des tests 26 à 28 et de `research/need_mlx.py`. Le passage à
plus grand est le test 41.

## 3. Le monde

- **Le jour** : 5 familles, 40 tâches neuves par famille et par jour, tirées d'un flux que ni la nuit,
  ni les bancs, ni les autres graines n'utilisent. Jusqu'à 3 tentatives par tâche ; le vérificateur
  décide.
- **Une famille réservée**, jamais montrée pendant les nuits 1 à 10, introduite à la nuit 11 : elle
  mesure si la boucle apprend du nouveau sans écraser l'ancien.
- **Le banc hors distribution** : les tâches que le générateur ne produit jamais (définition de
  `prophet_agi/docs/33` : opérandes plus longs, compositions de deux opérations, clés plus profondes).
- **La rétention** : bits/octet sur le texte tenu à l'écart du pilote réel de Prophet
  (`prophet_agi/docs/13`, 376 documents), et précision sur ARC-Easy (protocole de `prophet_agi/docs/20`).

## 4. Test 38 — La boucle jour–nuit, sans croissance

**Question.** Un petit modèle de langue, qui agit le jour et apprend la nuit de ses propres épisodes
vérifiés, avec un ajustement par étapes et une révision de l'ancien, progresse-t-il nuit après nuit,
au-delà de son générateur, sans oublier ?

**La nuit** (bras complet) :
1. Garder les épisodes réussis et vérifiés, dans une mémoire externe datée (rien de non vérifié n'entre).
2. Trier ce qui a manqué : familles et formes où le taux de réussite du jour est le plus bas.
3. Proposer des exercices proches des échecs (le proposeur de Prophet), vérifiés par programme ; ne
   promouvoir que ceux que le modèle résout dans la nuit.
4. Ajuster par étapes (recette des tests `LLM_MARK_*`) : appui sur la forme la plus facile de ce qui
   manque, ajouts répétés, consolidation à parts égales ; une part fixe de rejeu de l'ancien et du
   texte de base ; taux ÷ √r (`prophet_agi/docs/39`, SI-8c).

**Bras** (même modèle de départ, même budget de tâches et de pas par nuit) :

| Bras | Ce qui change |
|---|---|
| **Complet** | La nuit ci-dessus |
| **Sans choix** | Mêmes étapes, mais les épisodes et propositions de la nuit sont tirés au hasard parmi ceux du jour |
| **Sans révision** | Pas de rejeu de l'ancien ni du texte de base |
| **En vrac** | Pas d'étapes : tout l'ajustement de la nuit mélangé |
| **Figé** | Aucun apprentissage (le plancher, et la variance des bancs) |

**Durée** : 20 nuits. **Graines** : 3 au pilote, 5 au test, neuves.

**Prédictions** (seuils provisoires ; définitifs dans le protocole du test) :

| | Prédiction | Critère provisoire |
|---|---|---|
| **N1** | Elle apprend | Banc des familles connues, nuit 20, Complet − Figé ≥ 0,10, borne basse > 0 |
| **N2** | Au-delà du générateur | Banc hors distribution, nuit 20, Complet − Figé ≥ 0,05, borne basse > 0 |
| **N3** | Sans oublier | Bits/octet : hausse ≤ 0,05 ; ARC-Easy : baisse ≤ 2 points ; famille réservée apprise sans que les anciennes perdent plus de 0,03 |
| **N4** | La révision compte | Sans révision oublie plus que Complet (différence de bits/octet, borne basse > 0) |
| **N5** | Les étapes comptent | Complet − En vrac sur le banc hors distribution, borne basse > 0 |
| **N6** | Sans fin | Pente du banc hors distribution sur les nuits 11 à 20 > 0, borne basse > 0 |

**Critère global** : N1, N2, N3 et N6. N4 et N5 disent ce qui porte le résultat.

**Arrêt** : si au pilote le bras Complet ne dépasse pas le bras Figé à la nuit 10, le test n'est pas
lancé ; on publie le pilote et on cherche le mur, comme `prophet_agi/docs/39` l'a fait pour SI-1.

## 5. Test 39 — Le choix appris de ce qu'on apprend

**D'où il vient.** Au test 38, le tri de la nuit suit une règle écrite par nous (le taux de réussite du
jour). Le but demande que le modèle choisisse lui-même. Menia a les deux outils : le pari (test 37)
et la greffe (tests 34 à 36).

**Question.** Un lecteur appris de l'état du modèle, qui prédit ses propres échecs avant de répondre,
choisit-il ce qu'il faut apprendre aussi bien que la règle, et mieux que le hasard ?

**Le signal.** Un petit lecteur, sur l'état du modèle avant la réponse, entraîné à parier « je vais
réussir » sur les tâches du jour. Branché **sur la décision** (il choisit les épisodes et les
propositions de la nuit), pas à côté : arXiv 2604.11914 montre qu'un module de confiance posé à côté
s'effondre et ne sert pas.

**Bras** : tri par le lecteur appris ; tri par la règle (test 38) ; tri au hasard ; tri par l'oracle
(le vrai taux d'échec de la nuit suivante, la borne haute).

**Prédictions provisoires** :
- **CH1** : sur le banc hors distribution à la nuit 20, lecteur ≥ règle − 0,03, et lecteur − hasard,
  borne basse > 0.
- **CH2** : le pari du lecteur bat le témoin aux étiquettes mélangées (la leçon du Colab 23), borne
  basse > 0.
- **CH3** : greffer l'état d'une tâche ratée dans une tâche réussie fait baisser le pari du lecteur
  (même méthode qu'au test 37) : le signal vient de l'état, pas de la surface du texte.

**Critère global** : CH1 et CH2. CH3 dit si le lecteur lit l'état ou devine.

## 6. Test 40 — Grandir quand il le faut

**Question.** Un modèle qui commence petit et grandit quand il sature apprend-il mieux, ou oublie-t-il
moins, qu'un modèle de la taille finale entraîné dès le départ sur les mêmes nuits ?

**La croissance.** Neutre au moment de grandir : on insère des blocs dont la sortie est l'identité (et
des projections initialisées à zéro), ou on répète le cœur comme au donneur retenu (`prophet_agi/docs/28`).
**Contrôle de validité à chaque croissance** : écart maximal des logits avant et après ≤ 1e-5 sur
256 entrées fixes, avant tout apprentissage.

**Quand grandir** (règle fixée avant le test) : sur trois nuits de suite, le gain sur le banc hors
distribution reste sous ε **et** le coût de rétention monte au-dessus de δ (ε et δ tirés du pilote du
test 38). Un cran à la fois ; au plus quatre crans en 30 nuits.

**Bras** :

| Bras | Ce qui change |
|---|---|
| **Grandit** | Qwen3-0.6B, puis un cran à chaque saturation |
| **Petit fixe** | Qwen3-0.6B, jamais de croissance |
| **Grand fixe** | La taille finale du bras Grandit dès la nuit 1 (même croissance neutre faite d'avance), mêmes nuits |
| **Grandit au hasard** | Même nombre de crans que Grandit, à des nuits tirées au hasard |

Budget égal en tokens d'entraînement entre Grandit et Grand fixe.

**Prédictions provisoires** :
- **G1** : Grandit ≥ Grand fixe − 0,02 sur le banc hors distribution à la nuit 30, avec moins de
  calcul total (le petit modèle coûte moins pendant ses premières nuits).
- **G2** : Grandit oublie moins que Petit fixe (bits/octet), borne basse > 0.
- **G3** : grandir à la saturation fait mieux que grandir au hasard, borne basse > 0.

**Critère global** : G1 et G2. G3 dit si le moment compte. Si G1 échoue, la croissance n'apporte rien
de plus qu'un modèle grand dès le départ ; c'est le résultat à publier.

**Ce qui est déjà connu** : la croissance neutre existe (Net2Net, Masked Structural Growth, ELLE, LOIRE).
Ce que ce test ajoute, s'il passe : le déclenchement par la saturation mesurée, et la comparaison à
budget égal contre le modèle grand dès le départ, dans une boucle qui choisit ce qu'elle apprend.

## 7. Test 41 — Passage à l'échelle

Seulement si les tests 38 et 40 passent. Même protocole, modèle de départ Qwen3-4B ou Gemma 4 E4B,
3 graines. Question unique : les critères N1, N2, N3, N6 et G1 tiennent-ils à 4 milliards ? Un résultat
qui ne tient qu'à 0,6 milliard est publié comme tel.

## 8. Matériel et budget

- **Tests 38 à 40** : Qwen3-0.6B et ses crans tiennent sur un Mac de 32 Go (MLX, `research/need_mlx.py`)
  ou sur une carte de 16 Go. Plafond fixé : une nuit ≤ 3 h sur la machine choisie ; sinon on réduit
  les tâches par jour, jamais le nombre de graines.
- **Test 41** : un modèle de 4 milliards en LoRA tient sur 32 Go en 4 bits ; la pleine précision demande
  une A100 80 Go louée.
- Toutes les durées sont à mesurer au pilote du test 38 ; aucune n'est estimée ici.
- **Une seule machine par exécution**, notée dans le protocole : la reprise exacte en dépend
  (`prophet_agi/docs/34` §9).

## 9. Ce que les résultats diront, et ne diront pas

**S'ils passent** : un petit modèle de langue progresse nuit après nuit sur des tâches vérifiables, au-delà
de son générateur, sans oubli mesurable, en choisissant ce qu'il apprend d'après un signal lu dans son
propre état, et en grandissant quand il sature, mieux qu'un modèle grand dès le départ.

**Ce qu'ils ne diront pas** :
- Rien sur un vécu, un désir, une conscience.
- Rien au-delà des familles vérifiables par programme : le vérificateur borne ce qui s'apprend seul
  (`prophet_agi/docs/35` §0, borne 1).
- Rien d'« infini » : 20 à 30 nuits, et une pente.

## 10. Ordre d'exécution

1. Test 37 jusqu'au verdict.
2. Vérification bibliographique (§2.2) ; retrait de ce qui est déjà montré.
3. Portage de la boucle (§2.3) et son contrôle de validité.
4. Pilote du test 38 (3 graines, 10 nuits), puis protocole pré-enregistré du test 38, puis le test.
5. Pilote, protocole et test 39.
6. Pilote, protocole et test 40.
7. Test 41 si 38 et 40 passent.
