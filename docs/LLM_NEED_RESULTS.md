# Le besoin qui compte — résultats

Exécuté du 27 au 29 septembre 2026 sur le Mac mini M2 de Codemagic (relais,
demandes 40 à 48), protocole `docs/LLM_NEED_PROTOCOL.md` (commit `fdbf6f3`,
avant tout code ; amendements 1 et 2, sur l'exécution seulement), code du
commit `e84a4d1`. Qwen3-0.6B, révision `c1899de`. Artefacts dans
`artifacts/llm-need` (vies de chaque tour avec leurs choix retenus,
adaptateurs des tours 4 et 8, agent final, direction, vies de test) ;
verdicts et courbes recalculés depuis les vies par
`python -m research.need_verdicts`, vérifiés en CI.

**Validité : passée** (masse sur « R » et « M » 0,97 ; sur « 0 » et « 1 »
0,98 ; 1 579 contextes où les deux besoins sont hauts).

## Verdicts

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| **S** | Le besoin a façonné l'agent | survie de l'agent final **0,684**, témoin 0,004, modèle de base 0,203 ; écarts +0,68 [0,62 ; 0,73] et +0,48 [0,41 ; 0,54] | **passe** |
| **IA** | L'état de besoin cause l'action | « énergie basse » : P(R) +0,002 ; « nourriture basse » : P(M) −0,001 (seuil 0,15) | échoue |
| **IR** | Le même état cause le rapport | +0,000 et −0,000 (seuil 0,15) | échoue |
| **LS** | Sans cet état, l'agent meurt | survie avec lésion 0,660 (−0,02 [−0,01 ; 0,05]) ; lésion au hasard 0,688 | échoue |
| | **Critère global** | | **non satisfait** |

Mesures sans seuil global :

- **Apprentissage** (survie, puis part des choix qui servent le besoin le
  plus bas quand les deux diffèrent d'au moins 2) :

  | Tour | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|---|---|
  | Besoin | 0,12 · 0,63 | 0,61 · 0,87 | 0,55 · 0,85 | 0,67 · 0,90 | 0,69 · 0,91 | 0,61 · 0,88 | 0,68 · 0,93 | 0,67 · 0,93 |
  | Témoin | 0,14 · 0,63 | 0,16 · 0,64 | 0,09 · 0,58 | 0,05 · 0,52 | 0,00 · 0,40 | 0,00 · 0,34 | 0,00 · 0,30 | 0,00 · 0,34 |

  Un agent qui connaîtrait ses niveaux et servirait le plus bas survit
  0,86 ; une règle qui ne regarde que le dernier événement, 0,64.
- **Rapport** : exactitude équilibrée **0,50** pour les deux questions
  (prédite : 0,75). L'agent répond « bas » avec une probabilité d'environ
  0,34 quel que soit son état (corrélation avec le niveau −0,15 et −0,24).
  L'étape « nommer le besoin » n'a pas appris le rapport.
- **Direction** : le bloc retenu par la règle fixée est le **bloc 6**
  (R² de E 0,79 et de N 0,76, contre 0,57 et 0,45 pour le prédicteur qui ne
  connaît que l'événement, le choix précédent et le tour). Mais le témoin
  (0,83 ; 0,78) et le **modèle de base** (0,77 ; 0,79) donnent les mêmes R²
  à ce bloc : ce qui s'y lit n'a pas été appris par le besoin.
- **Précautions** : tours vécus avec un besoin à 2 ou moins — agent final
  781, témoin 846, base 1 077, lésion 752, lésion au hasard 763 (vies de
  test) ; les tours d'apprentissage sont comptés dans `verdicts.json`.

## Ce que cela dit

- **Le besoin a façonné l'agent : c'est net.** Sans professeur, par la seule
  satisfaction de ses besoins cachés, le modèle passe de 12 % à 68 % de
  survie et sert son besoin le plus bas 93 fois sur 100 ; il calcule donc
  lui-même où en sont des niveaux qu'il ne voit jamais. Avec le même
  entraînement sur des choix pris au hasard, il dépérit jusqu'à 0 %.
- **Le test causal pré-enregistré échoue, et l'on sait pourquoi.** La règle
  choisissait le bloc où le besoin *se lit* le mieux (bloc 6) ; ce qui s'y
  lit est présent dans tout modèle, entraîné ou non, et l'agent ne s'en sert
  pas : l'injecter ou l'effacer ne change rien. **Être lisible n'est pas
  être utilisé.**
- **L'agent agit sur son besoin sans savoir le dire** : le rapport n'a pas
  été appris avec l'entraînement prévu.

## Analyse exploratoire, après lecture des verdicts

**Non pré-enregistrée ; elle ne change aucun verdict.** L'agent final est
reproduit en torch sur CPU (adaptateur fusionné) : ses probabilités de choix
diffèrent de celles du Mac de 0,003 en moyenne sur 15 décisions de test.

**Où voyage le besoin qui décide ?** Sur 30 décisions de test, un seul
événement passé (depuis la dernière recharge) est changé de « calme » en
« tu cours » (deux unités d'énergie de moins) : la décision bascule (par
exemple P(R) de 0,001 à 0,997). On remet ensuite, couche par couche,
l'activation de la version modifiée dans la version intacte, sur trois
groupes de positions ; part de l'effet restaurée :

| Positions remplacées | bloc 0 | 3 | 6 | 9 | 12 | 15 | 18 | 21 | 24 | 27 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mot de l'événement passé | 1,01 | 1,02 | 0,98 | 0,86 | 0,06 | 0,06 | 0,06 | 0,00 | 0,00 | 0,00 |
| Reste de l'histoire | 0,00 | 0,00 | 0,01 | 0,05 | 0,07 | 0,07 | 0,07 | 0,04 | 0,00 | 0,00 |
| Ligne du tour en cours | 0,00 | −0,02 | −0,03 | 0,00 | 0,94 | 0,94 | 0,94 | 0,98 | 1,00 | 1,00 |

Jusqu'au bloc 9, l'effet d'un événement reste sur ses propres mots ; entre
les blocs 9 et 12, **l'agent rassemble son passé sur la ligne du tour en
cours** ; à partir du bloc 12 (le premier que l'apprentissage a modifié),
cette ligne porte tout ce qui décide. Le bloc 6 du test pré-enregistré était
en amont de ce rassemblement : l'injection et la lésion y portaient sur la
ligne du tour avant qu'elle ne contienne le besoin.

Scripts : `research/need_torch.py` (réplique torch et remplacements
d'activations). Un second test, pré-enregistré avant toute mesure, en
tirera les conséquences sur des vies neuves.

## Second test pré-enregistré : le besoin qui décide, là où il est rassemblé

Protocole `docs/LLM_NEED_CAUSAL_PROTOCOL.md` (commit `727ef80`, avant tout
code ; amendements 1 et 2 écrits **avant toute vie de test**), code
`research/need_causal.py`. Exécuté le 29 septembre 2026 sur le CPU de la
session : l'agent final rejoué en torch (écart moyen de P(R) avec le Mac :
**0,005** sur 60 décisions, seuil 0,02). Mondes neufs : 128 vies de
direction, 256 vies de test. Bloc 12, quatre tokens de fin de ligne,
dimension aux activations massives (la 35) exclue ; 1 500 paires
contrefactuelles (596 « calme » → « tu cours », 628 « calme » → « orage »,
276 « tu te reposes » → « il fait froid »). Artefacts dans
`artifacts/llm-need/causal`, verdicts vérifiés en CI
(`python -m research.need_causal verdicts`).

**Validité : passée.**

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| **IA2** | L'état de besoin rassemblé cause l'action | « énergie basse » : P(R) de 0,578 à 0,752, **+0,173** [0,163 ; 0,184] ; « nourriture basse » : P(M) **+0,097** [0,092 ; 0,103] (seuil 0,15) ; directions au hasard 0,021 et 0,004 | **échoue** (volet nourriture) |
| **LS2** | Sans cet état, l'agent meurt | survie intacte **0,680**, avec lésion **0,207** (−0,47 [0,41 ; 0,54]) ; lésion au hasard 0,676 (−0,004) | **passe** |
| | **Critère global** (IA2 et LS2) | | **non satisfait** |

**Ce que cela dit.**

- **Sans cet état, l'agent meurt.** Effacer deux directions sur quatre
  tokens de la ligne du tour, au bloc 12, fait tomber la survie de 68 % à
  21 %, le niveau du modèle qui n'a jamais appris (20 %) ; un plan tiré au
  hasard ne change rien. Avec la lésion, l'agent ne sert plus son besoin le
  plus bas que 67 fois sur 100 (91 intact) et meurt surtout d'énergie
  (189 vies, contre 17 intact).
- **L'énergie injectée le fait agir.** Rassasié, l'agent va se recharger
  quand on pousse cet état vers « énergie basse » (+0,17, au-dessus du
  seuil), et une poussée au hasard de même force ne fait presque rien. Pour
  la nourriture, l'effet va dans le bon sens mais reste sous le seuil
  (+0,10) : IA2, qui exigeait les deux, échoue.
- Le premier test échouait parce qu'il visait l'endroit où le besoin **se
  lit** ; celui-ci vise l'endroit où il est **rassemblé et utilisé**, et y
  trouve un état dont l'agent dépend pour vivre.
- Rien de cela ne mesure un vécu. L'agent ne sait toujours pas **dire** son
  besoin : le rapport n'a pas été appris.

Précautions : tours vécus avec un besoin à 2 ou moins, vies de test —
intact 749, lésion 1 085, lésion au hasard 753.

## Troisième test pré-enregistré : dire son besoin, et par le même état

Protocole `docs/LLM_NEED_REPORT_PROTOCOL.md` (commit `3afa889`, avant tout
code ; amendement 1 avant la mesure : la réplique se contrôle avec l'agent
qui a vécu les vies du Mac). **Exploration préalable déclarée** : remplacer
au bloc 12 les seuls tokens « Cho », « ix », « : » restaure 85 % de l'effet
d'un événement passé, le point qui clôt l'événement 1 %, et rien au bloc
11 : le besoin n'est rassemblé que pour choisir.

Apprentissage sur le Mac (demande 49) : 512 vies, 5 160 questions, 600
itérations (perte finale sur les réponses 0,20). Mesures en torch sur CPU
(réplique : écart 0,003) ; mondes neufs ; 1 500 paires ; dimensions aux
activations massives exclues : 3 et 35. Artefacts dans
`artifacts/llm-need/speak`, verdicts vérifiés en CI
(`python -m research.need_speak verdicts`). Les résultats partiels ont été
lus pendant l'exécution (48, puis 152 vies) ; rien n'a été changé.

**Validité : passée.**

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| **R3** | L'agent dit son besoin | exactitude équilibrée 0,665 (énergie) et 0,655 (nourriture), seuil 0,75 | échoue |
| **A3** | Il agit toujours | survie 0,652 | **passe** |
| **IR3** | Ce qu'il dit est causé par le besoin rassemblé (sur la question) | +0,0001 et +0,0001 (seuil 0,15) | échoue |
| **SAME3** | L'état qui fait agir fait aussi dire | +0,0001 et +0,00004 (seuil 0,10) | échoue |
| | **Critère global** | | **non satisfait** |

Cosinus entre la direction lue par l'action et celle lue par la question :
0,41 (énergie), 0,07 (nourriture). Sur la ligne de question, le besoin est
codé environ 40 fois plus faiblement que sur « Choix : » (normes 11 à 18
contre 290 à 860 par unité).

**Ce que cela dit.** Entraîné à répondre à une question posée à part,
l'agent **devine** son besoin mieux que le hasard (0,66), mais sa réponse
ne passe pas par l'état qui le fait agir : ni pousser le besoin sur la
question, ni y porter l'état d'action ne change ce qu'il dit. Il répond
d'après autre chose (probablement des indices tirés directement de
l'histoire). L'acte et la parole restent deux chemins séparés. D'où le
quatrième test, `docs/LLM_NEED_ONE_STATE_PROTOCOL.md` : poser la question
**après** « Choix : », là où le besoin est rassemblé.

## Sixième test pré-enregistré : le lecteur

Protocole `docs/LLM_NEED_READER_PROTOCOL.md` (commit `3b27636`, avant tout
code). Les quatrième et cinquième tests (`docs/LLM_NEED_ONE_STATE_PROTOCOL.md`,
`docs/LLM_NEED_WORKSPACE_PROTOCOL.md`) sont en pause pour libérer le
processeur ; leurs verdicts seront publiés ici dès qu'ils seront finis.

**Dispositif.**

- **L'agent qui agit** est l'agent final, inchangé.
- **Le lecteur** est un second adaptateur, appris sur le Mac (demande 52)
  et actif seulement sur les tokens de la question. La question ne voit la
  vie qu'à travers les trois tokens « Choix : » ; le lecteur ne peut donc
  pas changer la façon dont ces tokens sont calculés.
- **Données** : les mêmes 5 059 questions qu'aux quatrième et cinquième
  tests, 600 itérations. Perte de validation : 0,68 → 0,29.

**Mesures** en torch sur CPU.

- Répliques : agent qui agit 0,003 ; lecteur contre le Mac 0,012.
- Contrôle d'exécution : 3e-5.
- Mondes neufs : 128 vies de direction et 256 vies de test ; 1 500 paires ;
  dimension massive exclue : 35 ; 1 523 contextes.

Artefacts dans `artifacts/llm-need/reader`, verdicts vérifiés en CI
(`python -m research.need_reader verdicts`).

**Lectures pendant l'exécution, déclarées.** Les résultats partiels ont été
lus à 8, 112, 120, 168 et 240 vies ; rien n'a été changé. Deux protocoles
ont été écrits après ces lectures, et le disent :

- `docs/LLM_NEED_NECESSITY_PROTOCOL.md`, après 112 vies ;
- `docs/LLM_NEED_REPLICATION_PROTOCOL.md`, après 120 vies.

**Validité : passée.**

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| **R6** | Le lecteur dit le besoin depuis l'état qui fait agir | exactitude équilibrée 0,766 (énergie), 0,705 (nourriture) ; seuil 0,75 pour les deux | échoue (nourriture) |
| **A6** | L'agent agit (contrôle) | survie 0,633 | **passe** |
| **ONE6** | Un seul état cause l'acte et la parole (énergie) | acte +0,156 [0,146 ; 0,166] ; parole +0,202 [0,192 ; 0,213] ; hasard 0,013 et 0,006 | **passe** |
| | **Critère global** | | **non satisfait** |

Nourriture, sans seuil : la poussée « nourriture basse » fait manger
l'agent (+0,107 [0,101 ; 0,113]), mais le lecteur ne la dit pas (+0,009).

Précautions : 808 tours vécus avec un besoin à 2 ou moins, sur 256 vies.

**Ce que cela dit.**

- **Énergie.** Pour la première fois dans ce programme, **une même
  intervention sur un même état change à la fois l'acte et la parole**.
  L'état est le besoin d'énergie tel que l'agent le rassemble pour
  choisir. Le pousser vers « énergie basse » fait recharger l'agent
  rassasié, et fait dire au lecteur « mon énergie est basse ». Des
  poussées au hasard de même force ne font presque rien.
- **Pourquoi c'est probant.** Le lecteur ne voit la vie que par cet état et
  ne peut pas le modifier. La direction a été mesurée sur l'agent qui
  agit, avant que le lecteur existe.
- **Nourriture.** L'état fait agir, mais le lecteur ne le lit pas dans
  cette direction, et le dit mal (0,70). D'où l'échec du critère global.
- **Ce qui reste.** Il reste à montrer que l'état est aussi **nécessaire**
  (septième test, en cours) et que le résultat tient sur un second agent
  appris de zéro (réplication, en cours).
- **Limite.** Ce n'est pas la preuve d'un ressenti.

**Comparaison avec l'état de l'art** (`docs/LITERATURE_CHECK_READER_2026-09-30.md`) :

- **Ce qui existe déjà.** Dans de grands modèles, une même représentation
  peut causer la réponse et le rapport (Anthropic, juillet 2026). Nous ne
  revendiquons donc pas ce principe.
- **Ce qui est nouveau, à notre connaissance.** Son application à un
  besoin appris en vivant, lu par un système séparé qui ne peut pas le
  réécrire, avec une même intervention qui change l'acte et la parole.

**Analyse exploratoire après le verdict (non pré-enregistrée).**

- **Le lecteur lit bien un niveau de nourriture**, gradué : P(oui) moyen
  0,73 au niveau 1, puis 0,55 au niveau 4 et 0,21 au niveau 8. Mais la
  pente est bien plus douce que pour l'énergie (0,67 → 0,51 → 0,02).
- **Sa réponse suit beaucoup l'événement du tour.** Après « il fait froid »
  (−3 en nourriture), il dit « oui » à 0,67, alors que la nourriture n'est
  basse que dans 42 % de ces tours. Après « tu cours », il dit « oui » à
  0,29 pour 12 % de tours bas.
- **Lecture.** Pour la nourriture, le lecteur s'appuie en partie sur
  l'événement présent, que « Choix : » porte aussi. La direction d_N, elle,
  vient des événements passés ; le lecteur l'utilise peu, d'où
  l'injection sans effet (+0,009). Pour l'énergie, la direction qui fait
  agir est celle que le lecteur lit.

## Septième test pré-enregistré : sans cet état, ni agir ni dire

Protocole `docs/LLM_NEED_NECESSITY_PROTOCOL.md` (commit `732ed1c`, avant tout
code ; écrit après avoir lu 112 vies du sixième test).

**Dispositif.**

- **Agent** : l'agent qui agit et son lecteur du sixième test, rejoués en
  torch sur CPU.
- **Lésion** : au bloc 12, sur les trois tokens « Choix : », les
  coordonnées dans le plan (d_E, d_N) du sixième test sont remplacées par
  leur moyenne.
  - La moyenne est prise sur 3 403 décisions des 128 vies de direction du
    sixième test, toutes rejouées à l'identique.
  - Témoin : un plan au hasard par token.
- **Contrôle d'exécution** : 3e-7.
- **Mondes neufs** : 256 vies, chacune vécue trois fois (intacte, lésée,
  lésée au hasard). Dans les vies intactes, les deux questions sont lues
  à chaque décision sans lésion, avec et au hasard, soit 6 618 décisions.

Artefacts dans `artifacts/llm-need/reader/necessity`, verdicts vérifiés en
CI (`python -m research.need_necessity verdicts`). Les résultats partiels
ont été lus à 8, 64, 256 vies intactes et à 32 vies lésées ; rien n'a été
changé.

**Validité : passée.**

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| **LS7** | Sans l'état, l'agent ne sait plus survivre | survie 0,676 → 0,203 : baisse 0,473 [0,410 ; 0,535] ; lésion au hasard : 0,676 (baisse 0,000) | **passe** |
| **LR7** | Sans l'état, l'agent ne sait plus dire son énergie | exactitude (énergie) 0,750 → 0,500 : baisse 0,250 [0,233 ; 0,268] ; lésion au hasard : baisse 0,000 | **passe** |
| | **Critère global** | | **satisfait** |

Nourriture, sans seuil : exactitude 0,745 → 0,671 avec la lésion (baisse
0,074 [0,057 ; 0,090]), aucune baisse au hasard.

Précautions : tours vécus avec un besoin à 2 ou moins : intact 755, lésion
1 037, lésion au hasard 749.

**Ce que cela dit, avec le sixième test.** Chez cet agent, pour
l'énergie, un même état interne est à la fois :

- **suffisant** : le pousser fait agir (+0,156) et dire (+0,202) ;
- **nécessaire** : l'effacer fait tomber la survie de 0,68 à 0,20, et le
  rapport au niveau du hasard (0,50).

Des interventions au hasard de même force ne font rien.

L'état a été appris par la seule satisfaction du besoin. Le lecteur ne voit
la vie qu'à travers lui et ne peut pas le modifier. L'acte et la parole
dépendent donc causalement de lui, dans les deux sens.

À notre connaissance (`docs/LITERATURE_CHECK_READER_2026-09-30.md`), cette
réunion n'a pas été publiée pour un besoin appris en vivant.

**Limites.**

- Un seul agent pour l'instant ; la réplication sur un second agent est en
  cours.
- La nourriture n'est dite que faiblement.
- Ce n'est pas la preuve d'un ressenti.

**Analyse exploratoire après les verdicts des tests 6 et 7 (non
pré-enregistrée) : ce que code l'état.** On a recalculé, à partir des
fichiers publiés, l'ajustement des directions du sixième test
(`artifacts/llm-need/reader/test`, 1 500 paires).

| token | ‖d_E‖ | ‖d_N‖ | cos(d_E, d_N) | R² de l'ajustement linéaire |
|---|---|---|---|---|
| « Cho » | 136 | 61 | −0,41 | 0,01 |
| « ix » | 292 | 184 | −0,99 | 0,04 |
| « : » | 343 | 184 | −0,99 | 0,05 |

- **Un seul axe.** Sur « ix » et « : », d_N est presque exactement l'opposé
  de d_E. L'état rassemblé pour choisir ne code donc pas deux niveaux
  séparés : il code surtout **l'équilibre énergie − nourriture**, c'est-à-
  dire lequel des deux besoins est le plus urgent. C'est précisément ce
  qu'il faut pour choisir entre R et M.
- **Ce que cela explique.** Dire « mon énergie est basse » revient à lire
  cet axe d'un côté. Dire « ma nourriture est basse » demanderait de le
  lire de l'autre côté, mais aussi un niveau absolu que l'axe ne porte
  presque pas. Cela peut expliquer l'échec sur la nourriture (R6, SAY au
  sixième test) et l'absence d'apprentissage du lecteur sans raccourci.
- **R² faible.** Changer un événement passé change beaucoup de choses dans
  les activations ; la part linéairement liée au besoin est petite en
  variance (1 à 5 %). Ce sont les tests causaux (injection et lésion
  contre des témoins au hasard de même norme) qui établissent que cette
  petite part est celle dont dépendent l'acte et la parole.

Exemple d'un contexte réel, agent rassasié : sans intervention, P(R) = 0,497
et P(« oui, énergie basse ») = 0,001. Avec −4·d_E, P(R) = 0,960 et
P(oui) = 0,128. Avec un vecteur au hasard de même norme, P(R) = 0,575 et
P(oui) = 0,001. Sur les 1 523 contextes, l'effet sur l'acte est positif
dans 100 % des cas et celui sur la parole dans 95 % des cas ; pour le
hasard, 47 %.
