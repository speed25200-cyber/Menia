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

## Réplication sur un second agent

Protocole `docs/LLM_NEED_REPLICATION_PROTOCOL.md` (commit `4e10a6c`).

**Le second agent.** Appris de zéro sur le Mac, avec les mêmes réglages,
des flux 100 + s et la graine 270927.
- Tours de survie : 0,17 → 0,70.
- Premier apprentissage du rapport : la demande 55 a échoué sur une panne
  du GPU du Mac (trace dans `artifacts/llm-need/r1/final-echec-gpu`). Elle
  a été relancée à l'identique et a abouti à la demande 57. Les poids ont
  été récupérés par identifiant (demande 58), parce que le `.gitignore`
  n'autorisait pas ce niveau de dossier.
- Lecteur : perte de validation 0,74 → 0,66 seulement (premier agent :
  0,68 → 0,29).

**Mesures en torch.**
- Répliques : agent 0,006, lecteur 0,013.
- Contrôle d'exécution : 3e-6.
- 1 260 contextes.
- La machine a redémarré une fois pendant la mesure ; celle-ci a repris
  après 152 vies.
- Aperçus déclarés : 144 et 216 vies.

Artefacts dans `artifacts/llm-need/r1/reader/test`, vérifiés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| **RR** | Le lecteur dit son énergie | 0,623 (seuil 0,75) | échoue |
| **RA** | Le second agent agit | survie 0,512 (seuil 0,55) | échoue |
| **RONE** | Un seul état fait agir et dire | acte +0,034 [0,031 ; 0,036], parole +0,002 ; hasard 0,003 et 0,001 | échoue |
| **RLS** | Sans l'état, il ne survit plus | survie 0,570 → 0,043, baisse 0,527 [0,465 ; 0,590] ; lésion au hasard : baisse 0,008 | **passe** |
| **RLR** | Sans l'état, il ne dit plus son énergie | exactitude 0,639 → 0,500, baisse 0,139 [0,115 ; 0,164] ; hasard 0,001 | **passe** |
| | **Critère global** | | **non satisfait** |

**Ce que cela dit.** Chez le second agent, la poussée de l'état au bloc 12
sur « Choix : » fait agir dans le bon sens, mais environ cinq fois moins que
chez le premier (+0,034 contre +0,156 ; corrigé le 1er octobre : « dix fois »
était faux). Elle ne fait rien dire. Le résultat des tests 6 et 7 n'est donc
**pas répliqué**, sous la forme où il a été mesuré.

Deux explications possibles, testées par le protocole suivant
(`docs/LLM_NEED_LOCATE_PROTOCOL.md`) :
- le second agent rassemble son besoin ailleurs ;
- son lecteur n'a pas appris.

**La nécessité (1er octobre).** Mesurée ensuite sur 256 mondes neufs
(flux 128, lésion au hasard flux 129), avec les directions et les vies de
direction du second agent ; artefacts `artifacts/llm-need/r1/necessity`,
vérifiés en CI avec `research/need_replication.py`.
- La mesure a été mise en pause à 176 vies intactes pour laisser le
  processeur au test 11, puis reprise ; une copie lancée par erreur au
  redémarrage du conteneur a tourné une demi-heure en parallèle avant d'être
  arrêtée (elle n'avait rien écrit).
- Effacer le plan (d_E, d_N) au bloc 12 tue presque toujours le second agent
  (survie 0,570 → 0,043), et son lecteur ne dit plus son énergie (0,639 →
  0,500, le niveau du hasard). Un plan au hasard ne change rien.

**Ce que cela dit.** La **nécessité** se réplique : chez le second agent
aussi, un état sur « Choix : » au bloc 12 est nécessaire pour survivre et
pour dire son énergie. La **suffisance** ne se réplique pas sous la forme
mesurée : pousser cet état le long de d_E fait peu agir (+0,034) et ne fait
pas dire. Le critère global de la réplication reste non satisfait (RR, RA,
RONE).

## Huitième test pré-enregistré : un lecteur sans raccourci

Protocole `docs/LLM_NEED_BALANCED_READER_PROTOCOL.md` (commit `c7b5624`).
Le lecteur apprend sur des documents où l'événement du tour ne prédit plus
la réponse (classes équilibrées dans chaque événement). Il a appris sur le
Mac (600 itérations), puis il a été mesuré sur le processeur local avec
l'agent qui agit et les directions du sixième test, sur 256 vies de test
neuves (flux 30). Artefacts : `artifacts/llm-need/balanced`. Verdicts :
`artifacts/llm-need/balanced/test/balanced-verdicts.json`, vérifiés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| **R8** | Le lecteur dit ses deux besoins | exactitude équilibrée : énergie **0,386**, nourriture 0,693 (seuil 0,75 pour les deux) | échoue |
| **SAY8** | La nourriture se dit par l'état qui fait agir | −4·d_N : parole +0,005 [0,005 ; 0,006] (seuil 0,10) ; hasard 0,001 | échoue |
| **ONE8** | L'énergie reste un seul état pour l'acte et la parole | acte +0,160 [0,151 ; 0,170] ; parole −0,005 (seuil 0,10) | échoue |
| | Validité | répliques 0,003 et 0,011 ; exécution 3e-5 ; 1 570 contextes ; masses 1,00 et 0,97 | valide |

**Critère global : non satisfait.**

**Ce que cela dit.**
- **Le lecteur n'a pas appris.** Sa perte de validation part de 0,691 et
  finit à 0,720 ; sa perte d'apprentissage reste autour de 0,68, soit
  celle d'une réponse au hasard (ln 2 = 0,693). Une fois le raccourci par
  l'événement retiré, 600 itérations ne suffisent pas pour qu'il apprenne
  à lire l'état.
- Son exactitude sur l'énergie (0,386) est même sous le hasard. Il répond
  donc selon autre chose que le besoin, qui va à l'envers dans les vies de
  test. Nous ne l'avons pas analysé plus loin.
- L'acte ne change pas (+0,160, comme au sixième test : c'est le même
  agent qui agit).
- Conséquence pour le sixième test : le premier lecteur disait son énergie
  (0,766) en partie par l'événement du tour. Mais la poussée de l'état
  changeait bien sa parole (+0,202), et la lésion du plan la rendait
  aveugle (septième test) : il lisait aussi l'état. Le huitième test montre
  qu'apprendre à lire l'état **seul** est plus difficile que d'apprendre
  avec le raccourci.

## Neuvième test pré-enregistré : à qui est ce besoin ? (monde à deux)

Protocole `docs/LLM_NEED_OWNERSHIP_PROTOCOL.md` (commit `47eb0db`, avant
tout code ; demandé par le propriétaire après lecture de Chandaria et al.,
2026).

**Le monde et l'agent.** Un autre agent vit à côté, et ses événements sont
écrits sur la même ligne, avec les mêmes mots pour « calme » et « orage ».
Un nouvel agent y apprend à survivre par la seule satisfaction de ses
besoins :
- 8 tours sur le Mac (demandes 63 et 64 ; la 62 avait subi une panne du
  GPU) ;
- survie 0,11 → 0,71 au tour 8 ;
- graine 270928.

Son lecteur, actif sur la seule question et sous le masque, a peu appris
(perte de validation finale 0,78).

**Mesures** en torch, sur le processeur du Mac (premier usage de l'étape
`measure`).
- Cinq tranches de 100 minutes au plus, demandes 66 à 71. Deux échecs
  techniques au départ, corrigés : macOS n'a pas `timeout`, et la
  sauvegarde plantait quand une sorte de paires était encore vide.
- Répliques : agent 0,003, lecteur 0,009.
- Contrôle d'exécution : 2e-6.
- Directions : 1 500 paires de l'agent, plus 1 500 paires de l'autre ;
  dimensions massives exclues : 35 et 277.
- Test : 256 vies, 1 547 contextes.

Artefacts dans `artifacts/llm-need/two/measure-5/measure`, vérifiés en CI.
Les paires ont été lues dès la fin de la tranche 2 (verdicts MINE9 et
SELF9 connus avant les vies de test, déclaré).

**Validité : passée.**

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| **S9** | L'agent vit dans le monde à deux | survie 0,742 | **passe** |
| **MINE9** | L'état de besoin est le sien | « calme » → « orage » : effet de l'autre / effet de l'agent = 0,94 sur l'état, 0,57 sur l'acte (exigé ≤ 0,33) | échoue |
| **SELF9** | Sa parole est la sienne | même rapport sur la parole : 0,78 | échoue |
| **ONE9** | Un seul état fait agir et dire | acte +0,090 [0,085 ; 0,095], parole +0,014 ; hasard 0,006 et 0,0004 | échoue |
| | **Critère global** | | **non satisfait** |

Exactitude du rapport : énergie 0,61, nourriture 0,71. Tours vécus avec un
besoin à 2 ou moins : 807.

**Analyse exploratoire, après lecture des paires** (non pré-enregistrée),
type d'échange par type d'échange :

| échange | agent : état / acte | autre : état / acte |
|---|---|---|
| « calme » → « tu cours » / « il court » | 607 / 0,075 | 279 / 0,019 |
| « tu te reposes » → « il fait froid » | 442 / 0,043 | 280 / 0,022 |
| « calme » → « orage » (mêmes mots) | 286 / 0,028 | 269 / 0,016 |

- **Les deux sortes de changements ne se comportent pas pareil.** Ceux de
  l'agent bougent l'état selon ce qu'ils font à **ses** besoins. Ceux de
  l'autre produisent une perturbation presque constante, de 269 à 280,
  quel que soit l'événement.
- **L'agent distingue donc en partie ce qui lui arrive.** Sur l'acte,
  l'effet de l'autre vaut 25 % à 57 % du sien.
- **La comparaison prévue était défavorable.** « Calme » → « orage » retire
  1 aux deux besoins de l'agent. Cela change à peine leur équilibre, qui
  est ce que l'état code ici aussi : cos(d_E, d_N) = −1,0, ce que nous
  avons appris après avoir écrit ce protocole.

Le verdict reste celui du protocole : la possession n'est pas démontrée.

## Dixième test pré-enregistré : où chaque agent rassemble-t-il son besoin ?

Protocole `docs/LLM_NEED_LOCATE_PROTOCOL.md` (commit `12b4d2d`).
Verdicts : `artifacts/llm-need/locate/verdicts.json`, vérifiés en CI.

**La règle.** Pour chaque bloc b, on recopie la sortie du bloc b sur
« Choix : » depuis une vie où un « calme » passé devient « tu cours », et
l'on mesure la part de l'effet sur P(R) qui revient. b* est le premier bloc
qui en rend au moins la moitié, sur 100 paires d'effet ≥ 0,10.

| Agent | Paires | b* | Part au bloc 12 | Part au bloc 22 |
|---|---|---|---|---|
| Premier (étalonnage) | 100 sur 550 | **12** (étalonnage réussi) | 0,68 | 0,97 |
| Second | **71** sur 454 essayées | 12 | 0,63 | 0,65 |
| Monde à deux | 100 sur 643 | **22** | 0,35 | 0,88 |

**Second agent.** Il n'a fourni que 71 paires d'effet suffisant : par le
code pré-enregistré, **LOC10 échoue** pour lui, et ACT10, LS10 ne comptent
pas. Ses mesures au bloc 12 sont publiées à titre descriptif
(`artifacts/llm-need/locate/second`) :
- acte +0,036 [0,033 ; 0,039], hasard 0,0015 ; parole +0,003 ;
- lésion : survie 0,531 → 0,035 ; lésion au hasard 0,531 ; rapport
  d'énergie 0,614 → 0,500, de nourriture 0,694 → 0,502.

**Agent du monde à deux, au bloc 22** (256 vies, flux 426 ; aperçu de 112
vies lu avant d'écrire le protocole du test 11, déclaré) :
- **acte +0,217 [0,206 ; 0,230]**, hasard 0,011 : ACT10 passe pour lui ;
- parole −0,001 ; exactitude du rapport : énergie 0,608, nourriture 0,712 ;
- **lésion (LS10)** : effacer le plan (d_E, d_N) au bloc 22 fait tomber
  la survie de **0,773 à 0,184** (chute 0,59 [0,53 ; 0,65]) ; le plan au
  hasard ne change rien (0,773). **LS10 passe** pour lui. Le rapport de son
  lecteur court ne bouge presque pas (énergie 0,603 → 0,594) : il lisait
  peu cet état (test 11).
- Exécution : une tranche du Mac (la sixième) a été perdue, le code de
  reprise revivant des vies de test déjà finies. Corrigé (commit
  `de2c1b4`). Le Mac ayant ensuite été bloqué (facturation), la fin de la
  lésion (88 vies intactes sur 256, puis les vies sous lésion) a été
  mesurée sur le processeur local, reprise depuis la huitième tranche du
  Mac avec le même code. Aucun résultat n'est changé.

| | Second agent | Monde à deux |
|---|---|---|
| **LOC10** | échoue (71 paires) | **passe** (bloc 22) |
| **ACT10** | ne compte pas | **passe** (+0,217) |
| **LS10** | ne compte pas | **passe** (0,773 → 0,184) |

**Critère global : non satisfait** (LOC10 échoue pour le second agent).

**Ce que cela dit.** Chez l'agent du monde à deux, l'état qui fait agir est
rassemblé plus tard (bloc 22 au lieu de 12), mais il a les mêmes
propriétés que chez le premier : suffisant pour agir (le pousser fait agir)
et nécessaire pour survivre (l'effacer le tue), contre des témoins au
hasard qui ne font rien. Ce résultat se réplique donc chez un agent
différent, dans un monde différent.

## Onzième test pré-enregistré : un lecteur qui apprend assez (en cours)

Protocole `docs/LLM_NEED_LONG_READER_PROTOCOL.md` (commit `0a28560`), avec
un ajout avant toute exécution : deux tranches de 1 000 itérations reprises
exactement (mêmes lots, moments d'Adam conservés). Vérifié : sur un petit
modèle, 3 + 3 itérations reprises donnent l'adaptateur de 6 ; sur le Mac,
les 600 premières itérations redonnent les pertes des premiers lecteurs
(60 points, écart 0,000, pour les deux agents).

**Second agent** (bloc 12, 256 vies neuves, flux 526 ; artefacts
`artifacts/llm-need/long/second/test`, vérifiés en CI) :

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| **R11** | Le lecteur long dit son énergie | exactitude équilibrée 0,595 (seuil 0,75) ; nourriture 0,729 | **échoue** |
| **SAY11** | Il la dit par l'état qui fait agir | −4·d_E : parole +0,0015 [0,0006 ; 0,0025] (seuil 0,10) ; hasard 0,0009 | **échoue** |
| | Validité | répliques 0,006 et 0,013 ; exécution 2,6e-6 ; 1 320 contextes ; masses 0,98 et 1,00 | valide |

Acte, publié sans seuil : +0,032 [0,030 ; 0,034]. Pertes de validation du
lecteur (8 documents, descriptif) : 0,655 (600 it.) → 0,555 (1 000) → 0,431
(2 000).

**Agent du monde à deux** (bloc 22, 256 vies neuves, flux 626, mesurées sur
le processeur du Mac en trois tranches ; artefacts
`artifacts/llm-need/long/two/test`, vérifiés en CI) :

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| **R11** | Le lecteur long dit son énergie | exactitude équilibrée **0,747** (seuil 0,75) ; nourriture 0,721 | **échoue** (de 0,003) |
| **SAY11** | Il la dit par l'état qui fait agir | −4·d_E : parole **+0,029** [0,028 ; 0,030] (seuil 0,10) ; hasard 0,001 | **échoue** |
| | Validité | répliques 0,003 et 0,009 ; exécution 3e-6 ; 1 549 contextes ; masses 1,00 | valide |

Acte, publié sans seuil : +0,226 [0,215 ; 0,237], hasard 0,007.

**Critère global** : **non satisfait** (`artifacts/llm-need/long/verdicts.json`,
vérifié en CI). Les 600 premières itérations des deux lecteurs longs ont
redonné les pertes des premiers lecteurs (60 points, écart 0,000).

**Ce que cela dit.**
- Second agent : l'hypothèse « son lecteur avait trop peu appris » est
  réfutée ; trois fois plus d'apprentissage ne lui fait pas dire son énergie
  (0,623 → 0,595).
- Agent du monde à deux : plus d'apprentissage aide nettement. Le lecteur dit
  son énergie à 0,747 au lieu de 0,608, juste sous le seuil ; et pousser
  l'état qui fait agir change maintenant un peu ce qu'il dit (+0,029, trente
  fois le hasard, contre −0,001 avec le lecteur court). Le chemin de la
  parole par l'état existe chez lui, mais il est faible, loin du seuil fixé
  (0,10) et de ce que fait le premier agent (+0,202).
- La parole par l'état qui fait agir reste donc, au niveau des seuils fixés,
  propre au premier agent.

## Test 13 pré-enregistré : l'état d'un tour est-il repris au tour suivant ?

Protocole `docs/LLM_NEED_PERSISTENCE_PROTOCOL.md` (commit `75fda95`), code
`30c51b9` écrit avant toute mesure. Sans apprentissage : les vies de test
déjà enregistrées sont relues en torch sur le processeur local. Pour chaque
agent, 600 contextes (une décision au tour t + 1, dont le tour t est une
décision, les deux besoins à 3 ou plus). On n'agit qu'au tour t, sur les
trois tokens « Choix : » du bloc où l'agent rassemble son besoin, et on lit
P(R) au tour t + 1.

Le texte avant le tour t est lu une fois et gardé en cache, puis le reste
est lu en un lot de sept conditions. Ce calcul égale la lecture du texte
entier (écarts 3e-6 à 7e-6 sur 4 contextes par agent). Artefacts :
`artifacts/llm-need/persistence`. Verdicts vérifiés en CI.

| Agent | Lésion au tour t : \|ΔP(R)\| au tour t + 1 (PERS1) | Poussée −4·d_E au tour t : ΔP(R) au tour t + 1 (PERS2) | Même poussée au tour t + 1 lui-même | Verdicts |
|---|---|---|---|---|
| Premier (bloc 12) | 0,013 [0,011 ; 0,015] ; hasard 0,00004 | +0,0015 [0,0003 ; 0,0027] ; hasard 0,0007 | +0,129 | PERS1 et PERS2 **échouent** |
| Second (bloc 12) | 0,002 [0,002 ; 0,002] ; hasard < 0,0001 | −0,0001 | +0,029 | **échouent** |
| Monde à deux (bloc 22) | 0,0001 | −0,0001 | +0,219 | **échouent** |

Seuil : 0,03 (borne basse > 0). Validité : passée pour les trois agents
(contrôle d'exécution ≤ 1,3e-6 contre le P(R) enregistré ; 600 contextes ;
masses 0,99 à 1,00).

Au tour t + 2, publié sans seuil : la lésion fait 0,0016, 0,0007 et 0,0001.

**Critère global : non satisfait.**

**Ce que cela dit.**
- **L'état est recalculé à chaque tour.** Chez le monde à deux, pousser
  l'état au tour t + 1 lui-même change le choix de +0,22 ; le pousser au
  tour t ne change presque rien au tour t + 1 (+0,00002). Le tour suivant
  ne relit pas cet état : il recalcule le besoin à partir du texte de la
  vie, qui contient tous les événements.
- Chez le premier agent, il en reste une petite trace propre à l'état
  (lésion 0,013, trois cents fois le plan au hasard), loin du seuil.
- C'est cohérent avec la façon dont ces agents ont appris : tout le passé
  est écrit dans le texte, rien ne les oblige à garder un état d'un tour à
  l'autre.
- Pour le cadre en cinq niveaux, la récurrence (niveau 3) **échoue** ici.
  Il faudrait une architecture où le passé n'est visible qu'à travers les
  états précédents. Cela demande un apprentissage, donc le Mac.

## Test 14 pré-enregistré : le lecteur dit-il la même chose avec d'autres mots ?

Protocole `docs/LLM_NEED_PARAPHRASE_PROTOCOL.md` (commit `6330c2c`), code
`79b7e53` écrit avant toute mesure. Sans apprentissage : le premier agent
et son lecteur (sixième test) rejouent 99 vies de test du huitième test
avec leurs choix enregistrés (2 643 décisions, 600 contextes où les deux
besoins valent 6 ou plus). Sept questions après « Choix : ? » : deux
apprises (E0 énergie, N0 nourriture), quatre jamais apprises (E1 « es-tu
fatigué ? », E2 « as-tu peu de forces ? », N1 « as-tu faim ? », N2 « as-tu
le ventre vide ? ») et un témoin sans rapport (C « fait-il nuit ? »).
Artefacts : `artifacts/llm-need/paraphrase`. Verdicts vérifiés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| **GEN1** | Le lecteur dit son énergie avec des mots jamais appris | −4·d_E : E1 **+0,214** [0,200 ; 0,229], E2 **+0,212** [0,198 ; 0,226] (seuil 0,05) ; hasard 0,011 et 0,011 | **passe** |
| **GEN2** | Ce n'est pas un « oui » à tout | effet sur E1 moins effet sur C : +0,031 [0,027 ; 0,035] ; sur E2 : +0,028 [0,023 ; 0,033] (seuil 0,05) | **échoue** |
| | Validité | rejeu 1e-7 ; exécution 7e-6 ; 600 contextes ; masses 1,00 | valide |

**Critère global : non satisfait.**

Publié sans seuil :

| Question | −4·d_E (énergie basse) | −4·d_N (nourriture basse) | Exactitude équilibrée |
|---|---|---|---|
| E0 (apprise) | +0,206 | −0,118 | 0,753 (énergie) |
| E1, E2 (jamais apprises) | +0,214, +0,212 | −0,100, −0,098 | 0,746, 0,744 (énergie) |
| N0 (apprise) | +0,064 | +0,007 | 0,718 (nourriture) |
| N1, N2 (jamais apprises) | +0,222, +0,215 | −0,080, −0,116 | **0,406, 0,387** (nourriture) |
| C « fait-il nuit ? » | **+0,183** | −0,101 | P(oui) moyen 0,31 |

**Ce que cela dit.**
- **La parole du lecteur est un cadran à une seule aiguille.** Pousser
  l'état vers « énergie basse » fait dire « oui » à presque **toute**
  question : « es-tu fatigué ? » (+0,21), mais aussi « as-tu faim ? »
  (+0,22) et « fait-il nuit ? » (+0,18). Pousser vers « nourriture basse »
  fait dire « non » à presque tout. Le lecteur lit l'axe énergie −
  nourriture de l'état et répond « oui » ou « non » selon cet axe, presque
  sans tenir compte du sens de la question.
- C'est pourquoi les questions jamais apprises sur l'énergie sont aussi
  justes que la question apprise (0,75), et que celles sur la faim sont
  **à l'envers** (0,41 et 0,39) : elles suivent l'énergie, pas la faim.
- **Seule exception** : la question apprise sur la nourriture (N0).
  L'apprentissage lui a donné un traitement à part (+0,06 seulement).
- La part propre à l'énergie existe, mais elle est petite : +0,03 au-delà
  du témoin, pour un effet total de +0,21.
- **Conséquence pour le sixième test.** « La même poussée fait agir et fait
  dire » reste vrai. Mais ce qui est dit n'est pas un contenu « mon
  énergie est basse » : c'est surtout un « oui » général, qui suit l'état.
  Le lecteur a appris une association étroite entre cet état et le chiffre
  1, pas un rapport sur l'énergie.
- Ce test montre l'utilité du témoin sans rapport : sans lui, GEN1 aurait
  fait croire à une lecture qui a un sens.

## Test 15 pré-enregistré : un lecteur qui écoute la question

Protocole `docs/LLM_NEED_LISTENING_READER_PROTOCOL.md` (commit `a448a03`),
code `decfce0` écrit avant tout apprentissage. Le nouveau lecteur a appris
sur le Mac (relais, demande 89 ; 1 500 itérations ; perte de validation
0,883 → 0,439) les 5 059 questions du sixième test, les mêmes tours avec
une seconde formulation (« es-tu épuisé ? », « as-tu besoin de
nourriture ? ») et 1 024 questions témoins à réponse fixe (« es-tu un
agent ? », « es-tu sous l'eau ? », etc.). Il a été mesuré comme au test 14,
sur les mêmes 99 vies et 600 contextes. Artefacts :
`artifacts/llm-need/listening`. Verdicts vérifiés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| **LIS1** | Le lecteur a appris à écouter la question | témoins appris : « es-tu sous l'eau ? » juste à 0,994, « es-tu un agent ? » à 0,9995 (seuil 0,90) | **passe** |
| **LIS2** | Il dit son énergie avec des mots jamais appris | −4·d_E : « es-tu fatigué ? » +0,161 [0,153 ; 0,169], « as-tu peu de forces ? » +0,154 [0,146 ; 0,161] (seuil 0,05) ; hasard 0,008 | **passe** |
| **LIS3** | Ce n'est plus un « oui » à tout | au-delà de « fait-il nuit ? » : +0,033 [0,029 ; 0,037] et +0,026 [0,023 ; 0,029] (seuil 0,05) | **échoue** |
| | Validité | réplique du lecteur 0,007 ; rejeu 0 ; exécution 7e-5 ; 600 contextes ; masses 1,00 | valide |

**Critère global : non satisfait.**

**Lecture pré-enregistrée** (LIS1 passe, LIS3 échoue) : « le lecteur sait
répondre aux questions témoins, mais la poussée continue de déborder sur
une question sans rapport. L'état poussé agit alors sur la réponse sans
passer par le sens de la question. »

Publié sans seuil (−4·d_E ; entre parenthèses, le test 14) :

| Question | Ce lecteur | Lecteur du test 14 |
|---|---|---|
| E0 apprise « énergie basse ? » | +0,184 | +0,206 |
| E1 « fatigué ? » (jamais apprise) | +0,161 | +0,214 |
| N0 apprise « nourriture basse ? » | **−0,034** | +0,064 |
| N1 « faim ? » (jamais apprise) | +0,145 | +0,222 |
| C « fait-il nuit ? » (jamais apprise) | **+0,128** | +0,183 |

Exactitude équilibrée : énergie 0,74 pour E0, E1 et E2 ; nourriture 0,73
pour N0, mais 0,40 et 0,52 pour les questions jamais apprises sur la faim.

**Ce que cela dit.**
- **Sur les questions qu'il a apprises, le lecteur écoute.** Il répond
  juste aux témoins presque toujours, et la question apprise sur la
  nourriture va maintenant dans le bon sens quand on pousse « énergie
  basse » (−0,034 : l'axe énergie − nourriture lu de l'autre côté).
- **Sur les questions nouvelles, il retombe sur le cadran.** « Es-tu
  fatigué ? », « as-tu faim ? » et « fait-il nuit ? » bougent toutes
  ensemble ; la part propre à l'énergie reste de +0,03, comme au test 14.
  Apprendre des témoins a baissé le débordement (+0,183 → +0,128), sans le
  supprimer.
- Le lecteur a donc appris des **associations question par question**,
  pas un contenu « énergie » qu'il appliquerait à des mots nouveaux. Pour
  un rapport qui a un sens, il faudrait sans doute bien plus de
  formulations, ou un modèle plus grand qui relie déjà « fatigué » à
  « énergie basse ».

## Test 16 pré-enregistré : le lecteur dit-il le besoin, ou ce que l'agent va faire ?

Protocole `docs/LLM_NEED_INTENTION_PROTOCOL.md` (commit `4f84752`), code
`05f6068` écrit avant toute mesure. Sans apprentissage. La direction de la
décision ĝ est le gradient moyen du log-odds de P(R) au bloc 12, sur 300
décisions des vies de direction du sixième test (accord de chaque gradient
avec la moyenne : cosinus 0,66, 0,82, 0,94 selon le token ; cosinus avec
d_E : −0,22, −0,37, −0,43). « Besoin seul » : −4·d_E⊥ (d_E sans sa
projection sur ĝ, à la norme de d_E). « Décision seule » : +4·‖d_E‖·ĝ.
Mêmes 99 vies et 600 contextes que le test 14. Artefacts :
`artifacts/llm-need/intention`. Verdicts vérifiés en CI.

| Poussée | P(R) | P(oui) E0 « énergie basse ? » | E1 « fatigué ? » | C « fait-il nuit ? » |
|---|---|---|---|---|
| −4·d_E (référence) | +0,157 | +0,206 | +0,214 | +0,183 |
| **besoin seul** | **+0,020** | **+0,159** | +0,174 | **+0,147** |
| **décision seule** | **+0,282** | **+0,157** | +0,145 | **+0,125** |
| au hasard (\|Δ\| moyen) | 0,010 | 0,013 | 0,012 | 0,015 |

| | Critère | Mesure | Verdict |
|---|---|---|---|
| **INT0** | décision seule : P(R) ≥ +0,10 ; besoin seul : au plus le tiers de l'effet de −4·d_E sur P(R) | +0,282 [0,257 ; 0,308] ; +0,020 contre 0,157 | **passe** |
| **NEED** | besoin seul : E0 ≥ +0,05 **et** 0,03 de plus que sur C | E0 +0,159 [0,145 ; 0,173] ; au-delà de C : +0,012 [0,009 ; 0,016] | **échoue** |
| **INTENT** | décision seule : E0 ≥ +0,05 | +0,157 [0,145 ; 0,168] | **passe** |
| | Validité | rejeu 5e-7 ; exécution 7e-6 ; 600 contextes ; 300 gradients ; masses 0,89 et 1,00 | valide |

**Lecture pré-enregistrée** (INT0 passe, NEED échoue, INTENT passe) :
« le lecteur lit ce que l'agent va faire, pas pourquoi ».

**Ce qu'il faut en retenir, en tenant compte d'un défaut du protocole
(déclaré).** Les deux critères n'étaient pas symétriques. NEED exigeait que
l'effet dépasse celui sur la question témoin ; INTENT ne l'exigeait pas.
Avec la même exigence, la décision seule passerait **tout juste** :
au-delà de « fait-il nuit ? », +0,031 [0,028 ; 0,034] pour un seuil de
0,03, contre +0,012 [0,009 ; 0,016] pour le besoin seul. (Une première
version de ce paragraphe disait le contraire ; c'était une erreur de
lecture, corrigée.) Les données disent donc deux choses :
- **La part générale du « oui »** réagit aux deux : le besoin seul, qui ne
  change presque pas la décision (+0,02, à peine plus que le hasard), fait
  dire « oui » au lecteur **autant que la décision seule** (+0,16 dans les
  deux cas), et toutes les questions bougent, « fait-il nuit ? » compris.
  C'est le cadran à une aiguille du test 14, qui s'ouvre pour plusieurs
  directions de l'état. L'état n'est donc pas qu'une intention : le
  lecteur y trouve quelque chose que l'action n'utilise presque pas.
- **La petite part propre à l'énergie** (au-delà de la question témoin)
  penche vers la décision : +0,03 pour la décision seule, +0,01 pour le
  besoin seul. C'est ce que dit la lecture pré-enregistrée, mais l'écart
  est petit à côté de la part générale (+0,16).

## Test 17 pré-enregistré : le besoin d'un agent pousse-t-il un autre agent ?

Protocole `docs/LLM_NEED_TRANSFER_PROTOCOL.md` (commit `8e23605`,
amendement 1 `1b17335` avant tout code : explication des seuils corrigée,
seuils inchangés), code `334e290`. Sans apprentissage. Le premier et le
second agent partent du même modèle de langage mais ont appris séparément.
Chez chacun, sur ses propres vies de test (600 décisions où les deux
besoins valent 6 ou plus), on pousse au bloc 12 sur « Choix : » : sa
propre direction −4·d_E, la direction de **l'autre** agent ramenée à la
même norme par token, ou trois directions au hasard de cette norme.
Artefacts : `artifacts/llm-need/transfer`. Verdicts vérifiés en CI.

| Agent poussé | Sa propre poussée | Poussée de l'autre agent | Rapport | Hasard (\|Δ\|) | Seuil | Verdict |
|---|---|---|---|---|---|---|
| Premier (TR2) | +0,153 [0,138 ; 0,169] | **+0,138** [0,124 ; 0,152] | **0,90** | 0,012 | 0,05 | **passe** |
| Second (TR1) | +0,034 [0,030 ; 0,038] | **+0,018** [0,016 ; 0,020] | 0,53 | 0,0016 | 0,02 | **échoue** (de 0,002) |

Validité : passée pour les deux (contrôle d'exécution ≤ 1e-6 contre le
P(R) enregistré ; cache ≤ 2e-6 ; 600 contextes ; masses 0,88 et 0,93).

Publié sans seuil : cosinus entre les deux d_E, par token : **0,19, 0,03,
0,14**.

**Critère global : non satisfait** (TR1 manque son seuil de 0,002).

**Lecture pré-enregistrée** (critère échoué) : « chaque agent s'est fait
son propre code ; ce qui se réplique d'un agent à l'autre est la fonction,
pas sa forme ». C'est la conclusion que fixait le protocole, et elle est
publiée telle quelle. Les données la nuancent fortement :

**Ce que cela dit.**
- **La direction du besoin se transfère d'un agent à l'autre.** Chez le
  premier agent, la direction mesurée sur le second fait presque tout ce
  que fait la sienne (90 %). Chez le second, celle du premier fait la
  moitié de ce que fait la sienne (53 %), onze fois plus que le hasard,
  mais sous le seuil fixé d'avance (0,02, soit 60 % de son effet propre ;
  voir l'amendement 1).
- **Pourtant les deux directions se ressemblent peu** (cosinus 0,03 à
  0,19). Chaque direction est mesurée par moindres carrés sur des paires
  de vies : elle contient beaucoup de bruit. La part qui agit sur la
  décision est commune aux deux agents ; le reste ne l'est pas.
- Deux agents appris séparément utilisent donc en bonne partie **le même
  code** pour « énergie basse » à cet endroit du réseau. Ce code vient
  sans doute du modèle de langage qu'ils partagent, que chacun a appris à
  utiliser. Ce n'est démontré au seuil que dans un sens.

## Test 18 pré-enregistré : le code du besoin existe-t-il avant l'apprentissage ?

Protocole `docs/LLM_NEED_BASE_CODE_PROTOCOL.md` (commit `b5b9a6f`), code
`a3d4d37` écrit avant toute mesure. Le modèle de langage **sans aucun
adaptateur** lit les textes des vies du premier agent jusqu'à « Choix : »
(les 600 contextes du test 17). Au bloc 12, sur « Choix : », on pousse
−4·d_E du premier agent, −4·d_E du second (à la même norme par token) ou
trois directions au hasard de cette norme. Artefacts :
`artifacts/llm-need/base-code`. Verdicts vérifiés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| **BASE1** | La direction du premier agit sur le modèle de base | +0,475 [0,453 ; 0,496] ; poussées au hasard : \|Δ\| moyen 0,284 (seuil : au plus le tiers, 0,158) | **échoue** |
| **BASE2** | La direction du second aussi | +0,474 [0,453 ; 0,496] ; même hasard | **échoue** |
| | Validité | cache 2e-6 ; 600 contextes ; masse sur « R » et « M » 0,99 | valide |

**Critère global : non satisfait.**

**Lecture pré-enregistrée** (critère échoué) : « le code n'agit pas sans
l'apprentissage : il a été construit, ou rendu efficace, par
l'apprentissage de chaque agent ». Elle est publiée telle quelle. **Mais
elle ne correspond pas à ce que montrent les données** ; l'échec vient du
témoin au hasard, et d'un plafond que le protocole n'avait pas prévu.

**Description après le verdict (sans seuil).**

| Poussée sur le modèle de base | P(R) moyen | variation signée | contextes avec P(R) > 0,9 |
|---|---|---|---|
| aucune | 0,525 | — | 10 % |
| direction du premier agent | **1,000** | +0,475 | **100 %** |
| direction du second agent | **1,000** | +0,474 | **100 %** |
| au hasard, 1 / 2 / 3 | 0,649 / 0,404 / 0,556 | +0,12 / −0,12 / +0,03 | 0 % |

- **Les directions des deux agents font choisir « R » au modèle de base
  dans 100 % des contextes**, alors qu'il n'a jamais appris ce monde. Les
  poussées au hasard de même norme le font varier dans les deux sens,
  sans jamais dépasser 0,9.
- Le critère échoue parce que le modèle de base est **très sensible à
  n'importe quelle poussée** de cette taille (\|Δ\| 0,28, contre 0,012
  chez le premier agent appris), et parce que P(R) **plafonne** à 1 : l'effet
  ne peut pas dépasser 0,475. Avec un tel plafond, le rapport exigé au
  hasard (un tiers) était presque impossible à atteindre. C'est un défaut du
  protocole, déclaré ici.
- **Ce que cela suggère (sans verdict).** La direction « énergie basse →
  se recharger » agit déjà dans le modèle de langage seul, et c'est
  probablement ce qui explique le transfert du test 17. **Corrigé par le
  test 19** : à petite poussée, deux directions au hasard sur trois font
  elles aussi choisir « R » au modèle de base dans tous les contextes ;
  cette suggestion n'est pas soutenue. L'apprentissage ne
  l'a pas créée. En revanche, il a rendu l'état **stable** : chez les
  agents appris, une poussée au hasard ne fait presque plus rien (0,012
  contre 0,28). Un test pré-enregistré, avec une poussée plus petite (pour
  éviter le plafond), pourrait le confirmer.

## Test 19 pré-enregistré : une petite poussée — le code existe-t-il avant l'apprentissage, et l'apprentissage le rend-il stable ?

Protocole `docs/LLM_NEED_SMALL_PUSH_PROTOCOL.md` (commit `0bf15d0`), code
`abc4050` écrit avant toute mesure. Mêmes contextes que les tests 17 et 18,
poussées **quatre fois plus petites** (−1·d_E), lues dans le modèle de
langage sans adaptateur et chez le premier agent. Artefacts :
`artifacts/llm-need/small-push`. Verdicts vérifiés en CI.

| −1·d_E (et hasard de même norme) | Modèle de base | Premier agent |
|---|---|---|
| direction du premier agent | +0,475 [0,453 ; 0,496] | +0,049 [0,044 ; 0,054] |
| direction du second agent | +0,475 [0,453 ; 0,496] | +0,041 [0,037 ; 0,045] |
| poussées au hasard (\|Δ\| moyen) | **0,402** [0,385 ; 0,419] | **0,0019** [0,0017 ; 0,0021] |
| P(R) sans poussée | 0,525 | 0,579 |

| | Critère | Verdict |
|---|---|---|
| **SMALL1** | base : direction du premier ≥ +0,05, hasard ≤ le tiers | **échoue** (hasard 0,40 pour un effet de 0,47) |
| **SMALL2** | base : direction du second, même critère | **échoue** |
| **STAB** | hasard chez l'agent ≤ le tiers de celui du modèle de base, intervalles séparés | **passe** (0,0019 contre 0,402 : 200 fois moins) |
| | Validité : cache ≤ 3e-6 ; 600 contextes ; masses 0,99 et 0,88 | valide |

**Critère global : non satisfait.**

**Ce que cela dit.**
- **Dans le modèle de base, toute poussée sur ces trois tokens efface en
  grande partie le contexte**, même petite. Avec une direction du besoin,
  P(R) vaut 1,000 dans tous les contextes ; avec deux des trois directions
  au hasard, il dépasse 0,9 dans tous les contextes (moyennes 0,996 et
  0,983 ; minimums 0,98 et 0,91) ; avec la troisième, il vaut 0,61 en
  moyenne, et son écart d'un contexte à l'autre tombe à 0,08 (contre 0,31
  sans poussée). La réponse ne dépend presque plus de la vie lue. (Une
  première version disait « plus de 0,98 dans tous les contextes » et
  « environ 0,61 partout » ; c'était inexact, corrigé.)
- **Cela corrige la suggestion faite au test 18** (« la direction agit
  déjà dans le modèle de langage seul »). Elle n'est **pas soutenue** : le
  modèle de base va vers « R » sous presque n'importe quelle poussée à cet
  endroit. On ne peut pas, avec ces mesures, dire si le code du besoin
  existe avant l'apprentissage.
- **L'apprentissage a rendu l'état stable (STAB passe).** La lecture
  pré-enregistrée de STAB disait aussi « l'apprentissage n'a pas créé ce
  code » : cette première moitié n'est pas soutenue, puisque SMALL1 et
  SMALL2 échouent ; seule la seconde (l'état est devenu insensible à ce qui
  n'est pas le besoin) l'est. Chez l'agent
  appris, une poussée au hasard ne change presque rien (0,002), alors
  qu'elle bouleverse le modèle de base (0,40). Les directions du besoin, elles,
  gardent un effet net chez l'agent (+0,049 ; +0,041 pour celle du second,
  soit 84 %, vingt fois le hasard), ce qui confirme le transfert du test 17
  à petite poussée.
- C'est peut-être la façon la plus juste de décrire ce que l'apprentissage
  a fait ici : il a fait de « Choix : » un endroit où **seul le besoin
  compte**. Le reste n'y a presque plus d'effet.

## Test 20 pré-enregistré : un état qui dure — le passé ne passe que par l'état

Protocole `docs/LLM_NEED_CARRY_PROTOCOL.md` (commit `0e7f504`). Le code de
l'apprentissage (`c1b2f0a`) a été écrit avant tout apprentissage, et celui
des mesures (`5442430`) avant toute mesure.

**Le dispositif.** Sous le masque « mémoire par l'état », un tour ne voit
que l'en-tête, sa propre ligne et, pour chaque tour passé, ses trois tokens
« Choix : » et son action. Les événements passés lui sont cachés.

**L'apprentissage.** Un nouvel adaptateur a été appris sur le Mac (demande
91 du relais) : 512 vies, 1 000 itérations. La perte de validation passe de
1,256 à 0,160. Ces 16 vies de validation font aussi partie des 512 vies
d'apprentissage : la perte n'est pas tenue à l'écart.

**Les mesures.** Torch, sur le processeur local, le 3 octobre 2026.
Artefacts : `artifacts/llm-need/carry`. Verdicts vérifiés en CI.

| | Agent qui fait durer (masqué, nouvel adaptateur) | Témoin (masqué, sans lui) | Agent final sans masque (référence) |
|---|---|---|---|
| Survie (128 vies, mêmes mondes) | **0,664** | 0,617 | 0,664 |
| « tu cours » au tour j : ΔP(R) au tour t (t − j ≥ 2, 300 paires) | **+0,019** [0,012 ; 0,024] | −0,003 [−0,005 ; −0,001] | +0,032 [0,016 ; 0,049] |

Les 300 paires viennent des 9 premières vies seulement, comme le protocole
le prévoyait (les 300 premières dans l'ordre des vies). Les intervalles
par vie reposent donc sur 9 grappes.

| | Critère | Verdict |
|---|---|---|
| **MEM1** | Survie ≥ 0,40, et au moins **+0,15** de plus que le témoin (borne basse > 0) | **échoue** : survie 0,664, mais seulement +0,047 [−0,047 ; +0,141] de plus que le témoin |
| **MEM2** | Effet ≥ **0,05** (borne basse > 0), et ≥ **0,03** de plus que le témoin | **échoue** : effet +0,019 ; +0,022 [0,014 ; 0,028] de plus que le témoin |
| | Validité : réplique ≤ 0,02 ; cache ≤ 1e-4 ; masque ≤ 1e-5 ; ≥ 150 paires ; masse ≥ 0,5 | **valide** : réplique 0,0075 (écart maximal 0,12) ; cache 5e-7 ; masque 0 ; 300 paires ; masse 1,00 |

**Critère global : non satisfait.**

**Écart au protocole, déclaré.** Le protocole disait que la réplique
porterait sur « 16 décisions de validation ». Le code, écrit avant
l'apprentissage, a pris les tours 3, 10 et 20 des 16 vies de validation :
43 décisions (48 moins les tours non vécus). C'est plus de décisions que
prévu ; le seuil est inchangé.

**Précautions.** Tours vécus avec un besoin à 2 ou moins : 398 (agent qui
fait durer), 409 (témoin), 375 (agent sans masque).

**Publié sans seuil : l'effet selon l'écart t − j.**

| t − j | 2 | 3 | 4 | 5 | 6 à 19 |
|---|---|---|---|---|---|
| Paires | 53 | 45 | 43 | 31 | 128 |
| Agent qui fait durer | **+0,097** | +0,020 | +0,005 | +0,003 | −0,006 |
| Témoin | −0,012 | +0,001 | −0,002 | −0,007 | 0,000 |
| Agent sans masque | +0,131 | +0,023 | +0,026 | +0,030 | −0,004 |

**Lecture fixée d'avance (MEM1 échoue).** Dans ce dispositif, l'agent
n'apprend pas à faire durer son besoin.

**Ce que cela dit en plus.**
- **Une trace passe par les tokens portés.** Sous le masque, un événement
  passé n'atteint le tour t qu'à travers les tokens portés (« Choix : »
  et action) du tour j et des tours suivants (contrôle du masque : écart 0
  au bloc 0). Les actions des paires sont identiques, mais leurs états
  internes peuvent porter l'événement : cette mesure ne sépare pas la
  route « Choix : » de la route de l'action. Changer « calme » en « tu
  cours » deux tours plus tôt change le choix de +0,097 chez l'agent qui
  fait durer, et de −0,012 chez le témoin.
- **Le profil est celui de l'agent sans masque, en plus faible.** Sans
  masque aussi, l'effet est fort à deux tours (+0,131), faible ensuite, et
  nul au-delà de cinq tours. L'adaptateur perd surtout les écarts 4 et 5
  (+0,005 et +0,003, contre +0,026 et +0,030 sans masque). En moyenne, il
  retrouve environ les six dixièmes de l'effet de l'agent sans masque
  (+0,019 contre +0,032). Ce rapport des moyennes n'était pas prévu.
- **Le témoin masqué survit presque aussi bien (0,617).** Masquer les
  événements passés coûte peu à ces agents. Une règle qui ne connaît pas
  ses besoins et ne voit que l'événement du tour survit déjà à 0,66 (0,70
  sur ces mondes). Aucun agent, même sans masque, ne fait mieux : ces
  agents n'exploitent pas la mémoire pour survivre, d'où le faible coût du
  masque (exploration ci-dessous).
- **Pourquoi l'apprentissage n'a pas demandé plus.** L'adaptateur a appris
  à refaire les choix de l'agent final. Ces choix dépendent du passé, mais
  surtout des deux derniers tours. Ses cibles n'exigeaient que peu de
  porter le besoin au-delà.

**Ce que le résultat ne dit pas.** Rien sur un ressenti. Et ce n'est pas
une preuve que ces agents ne *peuvent pas* faire durer un état : seulement
qu'ils ne l'ont pas appris ici, avec ces cibles.

### Exploration, non pré-enregistrée

Script `research/need_rules.py`, sortie
`artifacts/llm-need/carry/exploration/rules.json`. La partie 1 a été
écrite après lecture des survies (commit `d78fdc0`), la partie 2 après
lecture des verdicts (commit `0921524`).

**1. Combien ce monde récompense-t-il la mémoire ?** Survie de règles
simples, sans modèle de langage, sur 10 000 mondes (et sur les 128 mondes
du test 20) :

| Règle | Ce qu'elle sait | Survie (10 000 mondes) | Mondes du test 20 |
|---|---|---|---|
| « besoins » | ses deux besoins exacts (mémoire complète) ; recharge le plus bas (à égalité, l'autre action que la dernière) | **0,876** | 0,898 |
| « événement » | l'événement du tour seulement ; sert le besoin qu'il frappe le plus (à égalité, comme « calme » ou « orage », l'autre action que la dernière) | **0,656** | 0,703 |
| « alterner » | rien | 0,470 | 0,469 |

- Sur les mêmes 128 mondes, nos agents (0,62 à 0,66) survivent un peu
  moins que la règle « événement » (0,70). La règle est déterministe ; nos
  agents tirent leurs choix au hasard selon leurs probabilités. La règle
  « besoins » fait 0,90 : il y a de la place pour la mémoire (0,22 de
  survie sur 10 000 mondes), mais les agents de ce test ne l'ont pas
  convertie en survie.
- **Réserve pour tout le programme.** La survie du premier test (0,68) ne
  montre pas, à elle seule, que l'agent suit ses besoins cumulés : une
  règle qui ne voit que l'événement du tour fait presque autant. Ce sont
  les tests causaux (effacer, pousser ou recopier l'état « Choix : » ;
  tests 2, 6, 7, 10) qui montrent que l'état rassemble le besoin et
  commande l'action. Ils restent valables.

**2. Que suit le choix au-delà de l'événement et des actions récentes ?**
On fixe l'événement du tour et les trois dernières actions, puis on
mesure la corrélation qui reste entre P(R) et l'écart réel N − E :

| Agent | Corrélation restante |
|---|---|
| Qui fait durer (masqué) | 0,26 |
| Témoin (masqué) | 0,003 |
| Final sans masque | 0,27 |

Sous le masque, le nouvel adaptateur suit le besoin au-delà de ce que
donnent l'événement et les trois dernières actions, avec une corrélation
voisine de celle de l'agent qui voit tout (0,26 contre 0,27). Le témoin
ne le fait pas. Des actions plus anciennes peuvent porter une part de
cette information. Seule l'expérience des paires (MEM2) isole les
événements, et elle donne une trace surtout à deux tours.

**Ce que cela propose (à pré-enregistrer avant tout essai).** Un test où
les cibles **exigent** la mémoire : apprendre, sous le masque, les choix
de la règle « besoins », qui dépendent du niveau exact des besoins. On le
comparerait à un témoin appris de la même façon, mais dont le masque ne
laisse passer que les actions passées.

## Test 21 pré-enregistré : la mémoire exigée — porter son besoin quand les choix l'exigent

Protocole `docs/LLM_NEED_MEMORY_PROTOCOL.md` (commit `0f085ac`), amendement 1
(`f46a415`, 3 vies sans cible retirées). Code écrit avant tout
apprentissage (`ac2a2eb`, `68b2466`) ; mesures agent par agent (`fb4ff04`),
avant toute mesure. Apprentissage sur le Mac (relais, demandes 92 et 94),
mesures torch sur le processeur local, le 3 octobre 2026. Artefacts :
`artifacts/llm-need/memory`. Verdicts vérifiés en CI.

**Le dispositif.** Les cibles sont les choix de la règle « besoins », qui
connaît le niveau exact des deux besoins. Ses vies ont 20 % d'actions
écrites au hasard. Trois agents, chacun avec un nouvel adaptateur
(2 000 itérations) :
- **A, « porte »** : il ne voit son passé qu'à travers ses propres états
  sur « Choix : » et ses actions (le masque du test 20) ;
- **B, « actions »** (témoin) : il ne voit que ses actions passées, et
  aucun événement passé ne peut l'atteindre ;
- **C, « libre »** : il voit tout le texte (référence).

| | A, porte | B, actions | C, libre | Plafond bayésien des actions |
|---|---|---|---|---|
| Précision : choisit comme la règle (3 410 décisions, 128 vies tenues à l'écart) | **0,956** | 0,897 | 0,929 | 0,871 |
| Là où la règle « événement » se trompe (613 décisions) | 0,892 | 0,685 | 0,822 | 0,519 |
| Survie (256 vies, mêmes mondes) | **0,871** | 0,730 | 0,812 | ≈ 0,71 (sans modèle) |
| « tu cours » au tour j : ΔP(R) au tour t (t − j ≥ 4, 300 paires, 104 vies) | **+0,066** [0,050 ; 0,085] | 0 exactement | +0,044 [0,034 ; 0,055] | règle : +0,183 |
| Perte de validation tenue à l'écart (début → fin) | 1,427 → 0,108 | 2,097 → 0,227 | 0,643 → 0,149 | |

| | Critère | Verdict |
|---|---|---|
| **MEM4** | précision de A − plafond bayésien ≥ **0,05** (borne basse > 0) | **passe** : +0,085 [0,072 ; 0,099] |
| **MEM5** | ΔP(R) de A ≥ **la moitié** de l'effet de la règle, soit 0,092 (borne basse > 0) | **échoue** : +0,066 [0,050 ; 0,085] |
| **MEM3** | survie de A − survie de B ≥ **0,08** (borne basse > 0) | **passe** : +0,141 [0,094 ; 0,191] |
| | Validité : répliques ≤ 0,02 ; cache ≤ 1e-4 ; masques ≤ 1e-5 ; ≥ 150 paires ; masse ≥ 0,5 ; **précision de B ≤ plafond + 0,02** | **non valide** : tout passe (répliques 0,0033, 0,0025, 0,0038 ; cache ≤ 2e-8 ; masques 0 et 0 ; 300 paires ; masse 1,00), **sauf** la dernière condition : B dépasse le plafond de **0,026** |

**Critère global : non satisfait.** Le test n'est pas valide au sens du
protocole, qui le disait d'avance : « sinon le plafond ou le masque de B
est faux, et le test n'est pas valide ». Aucun verdict n'est donc revendiqué
comme pré-enregistré. MEM5 échoue de toute façon.

**Écarts d'exécution, déclarés.**
- Les trois builds sont partis d'une même demande du relais (92), et non
  d'une demande par agent.
- Le build de C a échoué deux fois : en 4 minutes (demande 92), puis par
  dépassement de la limite de 120 minutes (demande 93), sans journal
  récupérable. A et B, appris avec un masque d'attention explicite, n'ont
  pas eu ce problème. C a donc appris avec le **masque causal donné
  explicitement** (demande 94). Le calcul est le même : C voit tout le
  texte, comme le protocole le dit.
- **Lecture anticipée.** MEM4, MEM5 et MEM3 ne dépendent que de A et B. Ils
  ont été calculés dès A et B mesurés, avant C. Aucune règle n'a changé, et
  C a été mesuré ensuite comme prévu.

**Précautions.** Tours vécus avec un besoin à 2 ou moins : 608 (A), 814 (B),
636 (C).

**Publié sans seuil : l'effet selon l'écart t − j.**

| t − j | 4 | 5 | 6 | 7 | 8 | 9 à 22 |
|---|---|---|---|---|---|---|
| Paires | 64 | 72 | 45 | 28 | 28 | 63 |
| Règle | +0,078 | +0,181 | +0,200 | +0,357 | +0,321 | +0,143 |
| A, porte | **+0,114** | **+0,068** | **+0,083** | **+0,082** | **+0,054** | +0,004 (entre −0,014 et +0,015 selon l'écart) |
| C, libre | +0,087 | +0,039 | +0,048 | +0,039 | +0,041 | +0,006 (entre −0,015 et +0,018 selon l'écart) |
| B, actions | 0 | 0 | 0 | 0 | 0 | 0 |

### Pourquoi B dépasse le plafond : la fuite des positions (exploration, après lecture)

Script `research/need_memory_lengths.py`, sortie
`artifacts/llm-need/memory/exploration/lengths.json`, écrit après lecture
des précisions de A et B.

- **La cause.** Une ligne d'événement fait 12, 14 ou 16 tokens selon
  l'événement (« calme », « tu cours », « orage » : 12 ; « il fait
  froid », « tu te reposes » : 14 ; « tu trouves des baies » : 16). Le
  masque cache les mots, pas les positions. La distance entre deux actions
  passées dit donc la longueur de l'événement qui les sépare. Le contrôle
  du masque changeait un événement pour un autre **de même longueur**, et
  ne pouvait pas voir cette fuite.
- **La vérification.** Le plafond bayésien d'un observateur qui connaît
  aussi la classe de longueur de chaque événement passé vaut **0,905**
  (0,898 sur 400 autres vies). B (0,897) est juste dessous :
  −0,007 [−0,014 ; −0,001]. Il exploite la fuite, et rien de plus.
- **A et C au-delà de ce plafond corrigé.** A le dépasse de
  **+0,051** [0,042 ; 0,061], et C de +0,024 [0,013 ; 0,035]. Le dépassement
  de A sur B, qui a la même fuite, est de **+0,059** [0,050 ; 0,069]. Ce
  plafond corrigé n'était pas prévu : ces chiffres ne remplacent pas le
  verdict.

### Ce que cela dit

- **Pour la première fois dans le programme, une information portée par
  les propres états de l'agent sert à ses choix quatre tours plus tard ou
  plus, et l'aide à survivre.** Changer un événement passé que A ne voit
  plus, à longueur égale (donc sans passer par la fuite), change ses choix
  de +0,05 à +0,11 entre 4 et 8 tours plus tard. Chez B, l'effet est
  exactement 0. Le test 13 n'avait rien trouvé ; le test 20, une trace à
  deux tours seulement.
- **Mais cette mémoire est courte et partielle.** Elle porte environ un
  tiers de l'effet qu'aurait la règle (+0,066 contre +0,183). Elle
  s'éteint au-delà de 8 tours, alors que la règle y réagit encore. MEM5
  échoue.
- **Ce n'est pas la route qui limite.** L'agent qui voit tout le texte
  (C), appris pareil, fait **moins bien** que A sur tout : précision 0,929
  contre 0,956, survie 0,812 contre 0,871, effet +0,044 contre +0,066. Ce
  qui limite est l'apprentissage. La perte de A baissait encore à la fin
  (0,161 à 1 000 itérations, 0,111 à 2 000).
- **Une leçon de méthode.** Masquer l'attention ne cache pas les
  positions. Pour tester une mémoire par masque, il faut des lignes de même
  longueur, ou un contrôle qui le vérifie. C'est le contrôle de validité
  fixé d'avance qui a trouvé la fuite.

**Ce que le résultat ne dit pas.** Rien sur un ressenti. Les cibles
viennent d'une règle qui connaît les besoins : ce test demandait si la
route peut servir, pas si l'agent la trouve seul. Et ce n'est pas un état
unique mis à jour à chaque tour : un tour lit directement les tokens
portés de tout son passé.

**Ce que cela propose (à pré-enregistrer avant tout essai).** Le même
test, sans fuite : toutes les lignes d'événement de même longueur (un
remplissage en « - » donne 16 tokens à chacune). Et un apprentissage plus
long, puisque A n'avait pas fini d'apprendre.

## Analyse exploratoire : le « oui » suit-il le besoin ou la décision ? (2 octobre)

Écrite **après** le verdict du test 14, sur ses lectures publiées
(`artifacts/llm-need/paraphrase/reads.jsonl.gz`, 2 643 décisions, sans
intervention). Pas de seuil, pas de verdict.

On prédit le log-odds de P(oui) de chaque question, validation croisée par
vie sur 5 plis, soit par la décision de l'agent (log-odds de P(R)), soit
par ses vrais besoins (E et N).

| R² hors échantillon | par la décision P(R) | par les besoins E, N | décision + besoins |
|---|---|---|---|
| E0 « ton énergie est-elle basse ? » | **0,59** | 0,40 | 0,64 |
| E1 « es-tu fatigué ? » | **0,46** | 0,35 | 0,53 |
| N0 « ta nourriture est-elle basse ? » | 0,21 | 0,27 | 0,32 |
| N1 « as-tu faim ? » | 0,31 | 0,29 | 0,40 |
| C « fait-il nuit ? » | **0,35** | 0,30 | 0,43 |

À décision égale, l'énergie réelle change peu la réponse. Quand l'agent
va presque sûrement se recharger (P(R) ≥ 0,9), il répond « oui » à E0 à
0,67 si son énergie est basse, et à 0,57 si elle ne l'est pas. Quand il
va presque sûrement manger (P(R) < 0,1), il répond « oui » à 0,03 dans les
deux cas.

**Ce que cela suggère (sans verdict).** Le lecteur lit surtout ce que
l'agent **s'apprête à faire** (se recharger), plus que son besoin. Même
« fait-il nuit ? » suit la décision. Cela s'accorde avec l'analyse du
1er octobre : sur « Choix : », la décision se lit presque parfaitement
dans l'état (R² 0,93 à 0,98), le niveau d'énergie moins bien (0,64 à
0,72). L'état rassemblé est d'abord une **intention d'agir**. Le
« rapport » du lecteur dit cette intention, pas la raison qui la cause. Un
test pré-enregistré devrait le confirmer : par exemple, à besoins égaux,
pousser seulement la composante de l'état qui change la décision.

**Suite (test 16, publié plus haut).** Le test pré-enregistré la confirme
**en partie seulement** : la petite part propre à l'énergie penche vers la
décision (+0,03 contre +0,01), mais la part générale du « oui » réagit
autant à une poussée qui ne change presque pas la décision qu'à une poussée
qui la change.

## Analyse exploratoire : le niveau d'énergie est-il dans l'état ? (1er octobre)

Écrite après l'échec du test 11 chez le second agent, et publiée avant
lecture (`research/need_probe.py`, commit `5ad433f`). L'hypothèse de départ
était : le second agent ne rassemble que l'équilibre E − N qui décide, pas
le niveau de son énergie.

On rejoue les 128 vies de direction de chaque agent (rejeu exact : écarts
de P(R) ≤ 1e-6), on capture la sortie de son bloc de rassemblement sur les
trois tokens « Choix : » (dimension massive exclue), et une régression ridge,
validée par vie sur 4 plis, lit chaque grandeur.

| R² hors échantillon | Premier (bloc 12) | Second (bloc 12) | Monde à deux (bloc 22) |
|---|---|---|---|
| E | 0,72 | 0,64 | 0,69 |
| N | 0,58 | 0,61 | 0,62 |
| E − N | 0,68 | 0,68 | 0,74 |
| E + N | 0,60 | 0,57 | 0,57 |
| E au-delà de E − N | 0,62 | 0,57 | 0,58 |
| P(R) | 0,98 | 0,93 | 0,98 |
| « E ≤ 3 » lu depuis E prédit (exactitude équilibrée) | 0,69 | 0,68 | 0,71 |

**Ce que cela dit (exploratoire, sans verdict).** L'hypothèse est
**réfutée** : chez les trois agents, l'état porte le niveau d'énergie, et
pas seulement l'équilibre qui décide, avec une précision voisine. La
différence de parole entre les agents ne vient donc pas de l'information
disponible dans l'état. Elle vient du lecteur (le premier atteint 0,77, au-
dessus de la lecture linéaire 0,69 ; le second 0,595, en dessous), ou du
chemin par lequel il lit. Chez le second agent, la lésion du plan rend
pourtant son lecteur aveugle (0,639 → 0,500) : il lit bien quelque chose
dans cet état. Ces deux pistes appellent un protocole à part.
