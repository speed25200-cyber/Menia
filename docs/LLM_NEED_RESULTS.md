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
apprentissage (`ac2a2eb`, `68b2466`), sauf `93af6a1` pour C (voir les
écarts) ; mesures agent par agent (`fb4ff04`), avant toute mesure. Apprentissage sur le Mac (relais, demandes 92 et 94),
mesures torch sur le processeur local, le 3 octobre 2026. Artefacts :
`artifacts/llm-need/memory`. Verdicts vérifiés en CI.

**Le dispositif.** Les cibles sont les choix de la règle « besoins », qui
connaît le niveau exact des deux besoins. Ses vies ont 20 % d'actions
écrites au hasard. Trois agents, chacun avec un nouvel adaptateur
(2 000 itérations) :
- **A, « porte »** : il ne voit son passé qu'à travers ses propres états
  sur « Choix : » et ses actions (le masque du test 20) ;
- **B, « actions »** (témoin) : il ne voit que ses actions passées.
  Aucun mot d'un événement passé ne peut l'atteindre ; sa longueur, si
  (voir plus bas) ;
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
| **MEM4** | précision de A − plafond bayésien ≥ **0,05** (borne basse > 0) | critère atteint, **non revendiqué** (test non valide) : +0,085 [0,072 ; 0,099], chiffre gonflé par la fuite, que A voit aussi (+0,051 au-dessus du plafond corrigé, exploration) |
| **MEM5** | ΔP(R) de A ≥ **la moitié** de l'effet de la règle, soit 0,092 (borne basse > 0) | **échoue** : +0,066 [0,050 ; 0,085] |
| **MEM3** | survie de A − survie de B ≥ **0,08** (borne basse > 0) | critère atteint, **non revendiqué** (test non valide) : +0,141 [0,094 ; 0,191] |
| | Validité : répliques ≤ 0,02 ; cache ≤ 1e-4 ; masques ≤ 1e-5 ; ≥ 150 paires ; masse ≥ 0,5 ; **précision de B ≤ plafond + 0,02** | **non valide** : tout passe (répliques 0,0033, 0,0025, 0,0038 ; cache ≤ 3e-8 (2e-10, 2e-8, 9e-10) ; masques 0 et 0 ; 300 paires ; masse 1,00), **sauf** la dernière condition : B dépasse le plafond de **0,026** |

**Critère global : non satisfait.** Le test n'est pas valide au sens du
protocole, qui le disait d'avance : « sinon le plafond ou le masque de B
est faux, et le test n'est pas valide ». Aucun verdict n'est donc revendiqué
comme pré-enregistré. MEM5 échoue de toute façon.

**Écarts d'exécution, déclarés.**
- Les trois builds sont partis d'une même demande du relais (92), et non
  d'une demande par agent.
- Le build de C a échoué deux fois : en 4 minutes (demande 92), puis par
  dépassement de la limite de 120 minutes (demande 93), sans journal
  récupérable. **La cause de ces échecs n'est pas connue.** A et B, appris
  avec un masque d'attention explicite, n'ont pas eu ce problème. C a donc
  appris avec le **masque causal donné explicitement** (demande 94), et un
  arrêt à 110 minutes qui ne s'est pas déclenché. Le calcul est le même :
  C voit tout le texte, comme le protocole le dit. Ce changement
  (`93af6a1`) a été fait après les mesures de A et après les choix, les
  vies et les contrôles de B.
- **L'apprentissage de C a connu un pic de perte** vers 1 000 itérations
  (de 0,15 à 0,37 entre les itérations 1 000 et 1 020, puis une redescente
  lente), absent chez A et B.
- **Lecture anticipée.** MEM4, MEM5 et MEM3 ne dépendent que de A et B. Ils
  ont été calculés dès A et B mesurés, avant C. Aucune règle n'a changé. C a
  été mesuré ensuite, appris autrement que prévu (masque explicite).

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

- **La cause.** Une ligne de tour fait 12, 14 ou 16 tokens aux tours 1 à 9
  (13, 15 ou 17 ensuite) selon l'événement (« calme », « tu cours »,
  « orage » : 12 ; « il fait froid », « tu te reposes » : 14 ; « tu
  trouves des baies » : 16). Le
  masque cache les mots, pas les positions. La distance entre deux actions
  passées dit donc la longueur de l'événement qui les sépare. Le contrôle
  du masque changeait un événement pour un autre **de même longueur**, et
  ne pouvait pas voir cette fuite.
- **La vérification.** Le plafond bayésien d'un observateur qui connaît
  aussi la classe de longueur de chaque événement passé vaut **0,905**
  (0,898 sur 400 autres vies). B (0,897) est juste dessous :
  −0,007 [−0,014 ; −0,001]. La fuite suffit à expliquer son excès ; il ne
  dépasse pas le plafond corrigé.
- **A et C au-delà de ce plafond corrigé.** A le dépasse de
  **+0,051** [0,042 ; 0,061], et C de +0,024 [0,013 ; 0,035]. Le dépassement
  de A sur B, qui a la même fuite, est de **+0,059** [0,050 ; 0,069]. Ce
  plafond corrigé n'était pas prévu : ces chiffres ne remplacent pas le
  verdict.

### Ce que cela dit

- **Hors verdict (le test n'est pas valide), deux mesures que la fuite ne
  touche pas.** La paire garde la longueur de l'événement, et A et B ont la
  même fuite. Elles montrent qu'une information portée par les propres
  états de l'agent change ses choix 4 à 8 tours plus tard : de +0,05 à
  +0,11 selon l'écart, contre exactement 0 chez B. Cette information passe
  par « Choix : » ou par l'action ; la mesure ne les sépare pas, comme au
  test 20. Et A survit mieux que B (+0,141). Le test 13 n'avait rien
  trouvé de tel ; le test 20, une trace à deux tours seulement.
- **Mais cette mémoire est courte et partielle.** Son effet moyen vaut
  environ un tiers de celui de la règle (+0,066 contre +0,183 ; rapport non
  prévu). C'est plus que la règle à 4 tours (+0,114 contre +0,078), mais un
  sixième à 8 tours. Au-delà de 8 tours, l'effet n'est plus visible
  (+0,004 sur 63 paires, sans intervalle), alors que la règle y réagit
  encore. MEM5 échoue.
- **La route ne semble pas limiter, mais la comparaison est fragile.**
  L'agent qui voit tout le texte (C) fait moins bien que A sur tout :
  précision 0,929 contre 0,956, survie 0,812 contre 0,871, effet +0,044
  contre +0,066. Mais C n'a pas appris tout à fait comme A (masque
  explicite, pic de perte vers 1 000 itérations). Hypothèse non
  pré-enregistrée : c'est l'apprentissage qui limite. La perte
  d'apprentissage de A baissait encore : 0,145 en moyenne sur les
  itérations 751 à 1 000, 0,116 sur 1 751 à 2 000.
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

## Test 22 pré-enregistré : la mémoire sans fuite

Protocole `docs/LLM_NEED_MEMORY_NO_LEAK_PROTOCOL.md` (commit `c35d9ee`).
Amendements, écrits avant les mesures concernées :
- **amendement 1** (`a0d6b1a`) : builds plus courts, ordre des lots fixé
  par époque, arrêt qui arrête l'apprentissage ;
- **amendement 2** (`371ea11`) : morceaux de 250 itérations repris
  exactement.

Code écrit avant tout apprentissage (`88ff796`), et avant les mesures
concernées pour les amendements (`55de952`, `cc1541d`).

Apprentissage sur le Mac (relais, demandes 95 à 102), les 3 et 4 octobre
2026 ; mesures torch sur le processeur local, le 4 octobre 2026. Artefacts :
`artifacts/llm-need/memory-no-leak`. Verdicts vérifiés en CI.

**Ce qui change par rapport au test 21.**
- **Plus de fuite par les positions.** Toutes les lignes d'événement ont la
  même longueur, grâce à un remplissage « - ». Les actions tombent aux
  mêmes positions dans toutes les vies.
- **Un apprentissage deux fois plus long** : 4 000 itérations, sur 8 189
  documents (8 192 vies du professeur, moins 3 sans cible). 2 000 itérations
  font un peu moins d'une époque (2 047 lots), 4 000 environ 1,95.
- **Deux agents** : A (« porte ») et B (« actions »).

| | A, porte | B, actions | Plafond bayésien des actions |
|---|---|---|---|
| Précision : choisit comme la règle (3 410 décisions, 128 vies tenues à l'écart) | **0,968** | 0,872 | 0,871 |
| Là où la règle « événement » se trompe (613 décisions) | 0,887 | 0,514 | 0,519 |
| Survie (256 vies, mêmes mondes) | **0,906** | 0,594 | |
| « tu cours » au tour j : ΔP(R) au tour t (t − j ≥ 4, 300 paires, 104 vies) | **+0,065** [0,049 ; 0,082] | 0 exactement | règle : +0,183 |
| Perte de validation tenue à l'écart (début → après 2 000 → après 4 000 itérations) | 1,916 → 0,112 → 0,069 | 3,318 → 0,291 → 0,294 | |

| | Critère | Verdict |
|---|---|---|
| **MEM4** | précision de A − plafond bayésien ≥ **0,05** (borne basse > 0) | **passe** : +0,097 [0,085 ; 0,109] |
| **MEM5** | ΔP(R) de A ≥ **la moitié** de l'effet de la règle, soit 0,092 (borne basse > 0) | **échoue** : +0,065 [0,049 ; 0,082] |
| **MEM3** | survie de A − survie de B ≥ **0,08** (borne basse > 0) | **passe** : +0,313 [0,254 ; 0,371] |
| | Validité : répliques ≤ 0,02 ; cache ≤ 1e-4 ; masques ≤ 1e-5 ; ≥ 150 paires ; masse ≥ 0,5 ; précision de B ≤ plafond + 0,02 | **valide** : répliques 0,0028 et 0,0059 ; cache 3e-14 et 9e-8 ; masques 0 et 0 (pour B, avec un événement de longueur naturelle différente) ; 300 paires ; masse 1,00 ; **B à +0,001 [−0,002 ; +0,004] du plafond** |

**Critère global (MEM4 et MEM5) : non satisfait**, à cause de MEM5.

**Lectures fixées d'avance.**
- **MEM4 passe et MEM5 échoue** : « Il porte de l'information au-delà de
  ses actions, mais sa mémoire reste courte, même avec un apprentissage
  deux fois plus long. »
- **MEM3 passe** : « porter son besoin fait survivre davantage que ses
  seules actions. »

**Publié sans seuil : après 2 000 itérations.**

| | A | B |
|---|---|---|
| Précision | 0,946 (+0,074 [0,062 ; 0,087] sur le plafond) | 0,874 (+0,003 [−0,001 ; +0,006]) |
| ΔP(R) des paires | +0,057 [0,044 ; 0,072] | 0 exactement |

Doubler l'apprentissage a amélioré la précision de A (+0,074 → +0,097).
L'effet d'un événement caché, lui, n'a que peu bougé (+0,057 → +0,065,
intervalles qui se recouvrent) : un peu plus de 6 à 8 tours, toujours
presque rien au-delà de 9. Au test 21 (lignes non remplies, avec la fuite,
4 093 documents), cet effet valait +0,066 : chiffre non directement
comparable.

**Publié sans seuil : l'effet selon l'écart t − j (4 000 itérations).**

| t − j | 4 | 5 | 6 | 7 | 8 | 9 | 10 à 22 |
|---|---|---|---|---|---|---|---|
| Paires | 64 | 72 | 45 | 28 | 28 | 15 | 48 |
| Règle | +0,078 | +0,181 | +0,200 | +0,357 | +0,321 | +0,333 | +0,083 |
| A, porte | +0,074 | +0,056 | +0,102 | +0,100 | +0,085 | +0,029 | +0,013 |
| B, actions | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Écarts d'exécution, déclarés.**
- **Plusieurs builds ont échoué**, sans rien garder :
  - B, demande 95 : limite de 120 minutes ;
  - A, demande 96 : Mac deux fois plus lent, arrêté vers l'itération 850 ;
  - B, demande 96 : arrêté à la limite de temps, vers l'itération 690.
- **D'où les amendements 1 et 2.** Le total est resté exactement de 4 000
  itérations par agent, chaque morceau reprenant le précédent (mêmes lots,
  moments d'Adam).
- **Les apprentissages réels :**
  - A : 2 000 itérations en un build, puis 1 750 en 7 morceaux, puis 250 ;
  - B : 1 250, 750, 1 750 et 250, par morceaux.
- **Calendrier de l'amendement 2.** Il a été écrit après la publication des
  mesures de A à 2 000 itérations (publiées sans seuil), pendant que le
  build de B de la demande 96 tournait encore. Il ne touche que la façon de
  découper l'apprentissage.
- **Lecture anticipée.** MEM4 et MEM5 de A ont été calculés dès A mesuré,
  avant presque tout l'apprentissage de B : B venait de repartir de zéro.
  Aucune règle n'a changé, et B a été appris et mesuré ensuite comme prévu.
- **Les pertes de validation** sont celles notées par le Mac avant la
  mise à jour de rang N : la valeur « après 2 000 » est celle notée au
  début du build suivant. Celle de B reste plate vers 0,29 dès
  l'itération 250.

**Repères de survie, sans modèle, sur ces 256 mondes** (recalculés à la
relecture) :
- règle « besoins » : 0,906, autant que A ;
- observateur bayésien des actions : 0,727 en prenant le choix le plus
  probable, 0,629 en tirant ses choix comme le font les agents.

B (0,594), qui tire ses choix d'une probabilité souvent incertaine, reste
proche de ce second repère. Une part de l'écart de MEM3 vient donc de ce
tirage, et non de l'information portée.

**Précautions.** Tours vécus avec un besoin à 2 ou moins : 590 (A), 892 (B).

### Exploration, non pré-enregistrée (après lecture des paires de A)

Script `research/need_memory_consistency.py`, sortie
`artifacts/llm-need/memory-no-leak/exploration/consistency.json`.

**L'hypothèse testée.** La paire garde les actions écrites après le tour j.
Avec l'événement changé, ces actions peuvent contredire ce que la règle
aurait fait. Si A se sert de ses actions récentes comme indice de ses
besoins, il réagirait moins quand elles contredisent l'événement.

| Paires | Nombre | Effet sur A | Règle | Part |
|---|---|---|---|---|
| Cohérentes (la règle aurait fait les mêmes actions) | 36 | +0,094 [0,049 ; 0,151] | +0,222 | 42 % |
| Contredites | 264 | +0,061 [0,045 ; 0,078] | +0,178 | 34 % |

L'écart, petit, tient surtout à la répartition des écarts t − j (16 des 36
paires cohérentes sont à t − j = 4). Il ne se retrouve pas à écart égal :
+0,027 contre +0,059 à 5 tours, +0,097 contre +0,102 à 6 tours. Pour les
paires cohérentes, la moyenne reste sous la moitié de la règle (42 %), mais
l'intervalle (de 22 à 68 % de la règle) ne permet pas de conclure.
L'hypothèse n'est donc ni établie ni écartée.

### Ce que cela dit

- **De l'information portée par ses états au-delà de ses actions, établie
  sans fuite (MEM4).** L'agent qui ne voit son passé qu'à travers ses
  propres états internes (« Choix : » et actions) choisit comme la règle
  « besoins » à 0,968. C'est près de 10 points au-dessus de ce que permet
  **tout** observateur de ses seules actions et de l'événement du tour : un
  optimum calculé (en espérance), pas un témoin appris.
- **Le témoin appris atteint ce plafond sans le dépasser** (+0,001
  [−0,002 ; +0,004]). C'est cohérent avec l'absence de fuite. Ce qui le
  montre, c'est le contrôle du masque : 0, même avec un événement de
  longueur naturelle différente.
- **Porter son besoin le fait survivre** : 0,906 contre 0,594 (MEM3). Une
  part de cet écart vient de ce que B tire ses choix d'une probabilité
  incertaine (voir les repères).
- **Mais cette mémoire est courte et partielle.** Un événement qu'il ne voit
  plus change encore ses choix 4 à 8 tours plus tard (de +0,06 à +0,10
  selon l'écart ; témoin : 0 exactement). En moyenne, c'est environ un tiers
  de ce que ferait la règle, et presque rien au-delà de 9 tours. Doubler
  l'apprentissage ne l'a guère changée.
- **Pour le cadre en cinq niveaux** : c'est la première fois, dans un test
  valide, qu'une information portée seulement par des états calculés aux
  tours passés sert aux choix au-delà des actions (MEM4). Le critère global
  n'est pas satisfait (MEM5). Et ce n'est pas une récurrence au sens
  strict : un tour lit directement les états portés de tout son passé.

**Ce que le résultat ne dit pas.**
- Rien sur un ressenti.
- Les cibles viennent d'une règle qui connaît les besoins (un
  professeur) : ce test dit que la route **peut** servir, pas que l'agent
  la trouve seul.
- Ce n'est pas un état unique mis à jour à chaque tour : un tour lit
  directement les tokens portés de tout son passé.
- La mesure ne sépare pas la route « Choix : » de la route de l'action :
  les deux sont portées.

## Test 23 pré-enregistré : que porte l'état ?

Protocole `docs/LLM_NEED_CARRIED_CONTENT_PROTOCOL.md` (commit `50e490e`).
**Amendement 1** (`057f7c7`), écrit avant toute mesure. Le tirage prévu ne
donnait que 80 greffes (a) sur 400 où la règle changerait de choix, sous
le minimum de 100. Les greffes (a) sont donc tirées parmi les donneuses
qui changent le choix de la règle. Les autres forment un groupe à part,
(a0), publié sans seuil.

Code écrit avant toute mesure (`23f07fa`). Mesures torch sur le processeur
local, le 4 octobre 2026. Artefacts : `artifacts/llm-need/carried-content`.
Verdicts recalculés en local (identiques) ; la CI les vérifie à chaque
envoi.

**La greffe.** On prend l'agent A du test 22 (« porte », 4 000
itérations). Sous son masque, les tours suivants ne voient le tour j qu'à
travers ses 4 tokens portés (« Cho », « ix », « : » et l'action). On
remplace, dans toutes les couches, leurs clés et valeurs (ce que les tours
suivants lisent de ces tokens) par celles d'une autre vie, la donneuse,
qui a écrit la même action au tour j. Puis on lit le choix de la vie
receveuse un ou deux tours plus tard (g = 1 ou 2).
- **ΔP(R)** : la probabilité que la receveuse choisisse R avec la greffe,
  moins sans la greffe.
- **Effet de la règle** : on rejoue la vie receveuse avec les besoins de la
  donneuse au tour j, en gardant ses actions et ses événements. +1 si la
  règle passerait alors de M à R, −1 de R à M, 0 sinon.

| Greffes (receveuses tirées parmi les 128 vies tenues à l'écart du test 22) | Nombre (vies receveuses) | A (« porte ») : écart moyen \|ΔP(R)\| | B (« actions ») : écart moyen \|ΔP(R)\| |
|---|---|---|---|
| (a) autres besoins, la règle changerait de choix | 374 (127) | **0,510** | 0,041 |
| (a0) autres besoins, la règle ne changerait pas | 200 (50) | 0,073 | 0,026 |
| (b) mêmes besoins, autre histoire | 200 (60) | **0,070** | 0,022 |

| | Critère | Verdict |
|---|---|---|
| **STATE1** | sur les greffes (a) : moyenne de ΔP(R) × (effet de la règle) ≥ **0,10** (borne basse > 0) | **passe** : +0,484 [0,438 ; 0,530] |
| **STATE2** | \|ΔP(R)\| des greffes (b) ≤ **la moitié** de celui des greffes (a), et différence à borne basse > 0 | **passe** : 0,070 contre 0,510 (seuil 0,255) ; différence +0,440 [0,392 ; 0,489] |
| | Validité : sans greffe ≤ 1e-4 ; greffe de soi ≤ 1e-6 ; mêmes positions (vérifié à chaque greffe) ; ≥ 100 greffes (a) et (b) | **valide** : 5e-9 ; 0 ; oui ; 374 et 200 |

**Critère global (STATE1 et STATE2) : satisfait.**

**Lecture fixée d'avance** (STATE1 et STATE2 passent) : « L'état que
l'agent porte d'un tour à l'autre contient son besoin. Greffé dans une
autre vie, il fait choisir selon les besoins de la vie d'origine. Un état
aux mêmes besoins, venu d'une autre histoire, ne change presque rien. Ce
qui est porté est un état de soi (le niveau de ses besoins), pas une trace
des événements. »

**Nuance (après lecture).** Le test ne sépare pas le niveau des besoins de
ce qui en commande le choix. Et l'événement du tour j compte un peu (voir
l'exploration plus bas).

**Publié sans seuil.**
- **Selon g** : effet aligné +0,470 à un tour (199 greffes (a)), +0,499 à
  deux tours (175) ; \|ΔP(R)\| des greffes (b) : 0,073 et 0,067. L'effet
  ne baisse pas entre un et deux tours.
- **Greffes (a0)** : autres besoins, mais la règle ne changerait pas de
  choix. \|ΔP(R)\| = 0,073, comme les greffes (b) (0,070).
- **La part de l'effet de la règle que A retrouve.** Dans les greffes (a),
  la règle change entièrement de choix (de M à R, ou de R à M). A déplace
  en moyenne sa probabilité de choisir R de 0,484 dans ce sens, soit
  environ la moitié. Par sens : quand la règle passerait à R (178
  greffes), P(R) va en moyenne de 0,062 à 0,556 ; quand elle passerait à M
  (196), de 0,896 à 0,422.
- **Le témoin B** (« actions » du test 22, 4 000 itérations ; sans greffe
  4e-7, greffe de soi 0). Greffés, ses tokens portés changent à peine le
  choix : effet aligné +0,025 [0,018 ; 0,031]. Sous le masque de B, seul le
  token de l'action compte, et il ne voit que les actions passées. La
  seule chose que la greffe change est donc la suite des actions passées
  de la donneuse. Quand ces actions sont les mêmes (92 greffes), l'effet
  est exactement nul. Ces actions disent un peu les besoins de la
  donneuse : d'où le petit effet aligné.

**Écarts d'exécution, déclarés.**
- **Amendement 1**, avant toute mesure (voir plus haut).
- **374 greffes (a) au lieu de 400** : le tirage a parcouru les 128 vies
  receveuses sans atteindre 400. Le protocole fixait un maximum ; le
  minimum de validité (100) est largement atteint.
- **Peu de vies receveuses pour (b) et (a0)** : 60 pour (b), 50 pour (a0).
  Le tirage s'arrête à 200, atteint avant la fin des vies. L'intervalle
  bootstrap tient compte de ce regroupement (tirage par vie).
- **20 des 200 greffes (b) n'ont pas d'autre histoire** (relevé à la
  relecture). La donneuse a vécu les mêmes tours que la receveuse jusqu'au
  tour j (12 fois au tour 1, où mêmes besoins veut dire même événement).
  Leur greffe ne change rien, par construction. Le protocole ne l'avait
  pas prévu. Sans elles, \|ΔP(R)\| des greffes (b) vaut 0,077 ; STATE2
  passe toujours (différence +0,432 [0,383 ; 0,482]).
- **Une donneuse dont le rejeu fait mourir la receveuse** est écartée avant
  le tirage, pas après.
- **Lecture anticipée.** Les verdicts de A ont été calculés dès ses greffes
  finies, avant les greffes du témoin B. Aucune règle n'a changé. B ne
  compte pas dans les verdicts.
- **Le protocole du test 24** (`docs/LLM_NEED_CHAIN_PROTOCOL.md`) a été
  écrit après cette lecture anticipée, et avant la fin de B.

### Exploration, non pré-enregistrée (après lecture, relevée par la relecture indépendante)

Script `research/need_carried_exploration.py`, sortie
`artifacts/llm-need/carried-content/exploration.json`.

- **L'événement du tour j compte un peu.**
  - (a), même événement au tour j chez la donneuse (79 greffes) : +0,31
    [0,22 ; 0,41]. Événement différent (295) : +0,53 [0,48 ; 0,58].
  - (b), événement différent au tour j (106 greffes) : \|ΔP(R)\| 0,098
    [0,056 ; 0,147]. Même événement, autre histoire (74) : 0,047.
  - Une simple trace de l'événement du tour j est exclue : à événement égal,
    la greffe (a) change encore le choix de +0,31. Mais l'état porte aussi
    un peu de l'événement récent, ce qui va avec la mémoire courte du
    test 22.
- **Ce n'est pas une hésitation autour de 0,5.** Dans les greffes (a), la
  probabilité du nouveau choix de la règle dépasse 0,9 dans 144 greffes
  sur 374, reste sous 0,1 dans 102, et se trouve entre les deux dans 128.
  La greffe fait souvent basculer le choix presque entièrement, et souvent
  presque pas.

### Ce que cela dit

- **Ce que l'agent porte d'un tour à l'autre dépend surtout de son
  besoin.** Greffé dans une autre vie, l'état porté d'un seul tour fait
  pencher le choix vers les besoins de la vie d'origine. C'est vrai un
  tour plus tard comme deux. En moyenne, c'est la moitié d'un changement
  complet.
- **Surtout le besoin, pas seulement une trace des événements.** À besoins
  égaux, une autre histoire change peu le choix (0,070 contre 0,510 ;
  0,098 quand l'événement du tour j diffère). L'événement du tour j compte
  un peu (voir l'exploration). L'agent ne voit jamais ses niveaux :
  l'état porte au mieux son estimation de ses besoins.
- **Des besoins différents qui ne changeraient pas le choix de la règle
  (a0) changent aussi peu le choix que des besoins égaux (b)** (publié sans
  seuil). C'est attendu si l'état code les niveaux, puisque la règle ne
  changerait pas de choix. Mais c'est aussi ce qu'on verrait s'il ne
  codait que ce qui commande le choix (par exemple, quel besoin est le plus
  bas). Ce test ne sépare pas les deux.
- **Pour le cadre en cinq niveaux** : au test 22, nous savions que ces
  états portent une information au-delà des actions. Nous savons
  maintenant qu'elle porte surtout sur le besoin de l'agent (ses niveaux,
  ou au moins ce qui en commande le choix).

**Ce que le résultat ne dit pas.**
- Rien sur un ressenti.
- L'agent a appris d'un professeur qui connaît les besoins.
- Aucune vie n'a été vécue avec la greffe : on a lu P(R) au tour t.
- La greffe porte sur les 4 tokens d'un seul tour : les tours précédents
  de la receveuse restent les siens.
- La mesure ne sépare pas « Choix : » de l'action : les 4 tokens sont
  greffés ensemble.

## Test 24 pré-enregistré : le besoin passe-t-il d'un état porté au suivant ?

Protocole `docs/LLM_NEED_CHAIN_PROTOCOL.md` (commit `5a2a908`), écrit après
la lecture anticipée des résultats de A au test 23, avant tout code et
toute mesure. Code écrit avant toute mesure (`0d9a80a`). **Amendement 1**
(`e4f97c0`), avant toute greffe mesurée : avec 4 fils de calcul, la greffe
de soi donnait 1,3e-6, au-dessus de la tolérance de 1e-6 (erreurs
d'arrondi qui dépendent de l'ordre des additions). Les contrôles ont été
refaits, et les greffes calculées, avec un seul fil, tolérances
inchangées ; les contrôles à 4 fils sont publiés. Commande :
`python -m research.need_chain run --threads 1` ; le nombre de fils n'est
pas écrit dans les artefacts.

Mesures torch sur le processeur local, le 4 octobre 2026. Artefacts :
`artifacts/llm-need/chain`. Verdicts recalculés en local (identiques) ; la
CI les vérifie à chaque envoi.

**La question.** Au test 23, l'état porté d'un tour fait pencher le choix
un ou deux tours plus tard. Mais par quelle route ? Le tour t peut lire
**directement** l'état porté du tour i. Ou bien l'état porté du tour i + 1
le lit et le transmet, **en chaîne**.

**La mesure.** Même greffe qu'au test 23. On remplace les tokens portés du
tour i par ceux d'une autre vie. Cette vie a la même action au tour i,
mais d'autres besoins. Avec ses besoins, la règle changerait de choix au
tour t = i + h (h = 2 ou 3). On lit P(R) au tour t de quatre façons :
- **propre** : sans greffe ;
- **totale** : avec la greffe, les tours suivants calculés avec elle ;
- **directe** : la totale, mais les tokens portés des tours i + 1 à t − 1
  (les **médiateurs**) reprennent leurs valeurs sans greffe ;
- **par les médiateurs seuls** : sans greffe au tour i, mais les
  médiateurs reçoivent leurs valeurs de la lecture totale.

Chaque effet est compté dans le sens où la règle changerait le choix
(multiplié par +1 ou −1). L'**effet en chaîne** est l'effet total moins
l'effet direct.

| 491 greffes (128 vies receveuses) | Effet aligné moyen |
|---|---|
| Total | **+0,419** [0,386 ; 0,451] |
| Direct (médiateurs remis à leurs valeurs sans greffe) | +0,226 [0,200 ; 0,253] |
| En chaîne (total − direct) | **+0,193** [0,170 ; 0,215] |
| Par les médiateurs seuls | +0,133 [0,110 ; 0,157] |

| | Critère | Verdict |
|---|---|---|
| Condition préalable | effet total ≥ **0,10** (borne basse > 0) | **remplie** : +0,419 [0,386 ; 0,451] |
| **CHAIN** | effet en chaîne ≥ **la moitié** de l'effet total, et borne basse de l'effet en chaîne > 0 | **échoue** : la chaîne fait 0,461 de l'effet total [0,417 ; 0,504] (+0,193 pour un seuil de 0,209) ; la borne basse (0,170 > 0) est remplie, l'échec vient seulement de la moitié |
| | Validité : lecture prolongée ≤ 1e-4 ; greffe de soi ≤ 1e-6 ; mêmes positions ; ≥ 150 greffes par h | **valide** : 8,7e-7 ; 0 ; oui ; 266 (h = 2) et 225 (h = 3) |

**Critère global : non satisfait.**

**Lecture fixée d'avance** (CHAIN échoue, condition préalable remplie) :
« Les choix lisent surtout directement l'état ancien ; les états suivants
en transmettent moins de la moitié. Ce n'est pas une chaîne, mais surtout
une lecture directe du passé. »

Cette lecture va un peu au-delà des données. La part directe (0,54)
n'est pas nettement au-dessus de la moitié : son intervalle va de 0,50 à
0,58. Et elle ne vaut qu'à deux tours (voir selon h).

**Publié sans seuil : selon h.** Ce partage change fortement avec la
distance.

| | h = 2 (266 greffes, 126 vies) | h = 3 (225 greffes, 114 vies) |
|---|---|---|
| Total | +0,481 [0,434 ; 0,527] | +0,346 [0,299 ; 0,391] |
| Direct | **+0,335** [0,293 ; 0,380] | +0,097 [0,075 ; 0,120] |
| En chaîne | +0,146 [0,117 ; 0,174] | **+0,249** [0,211 ; 0,287] |
| Part de la chaîne | 0,30 [0,25 ; 0,36] | **0,72** [0,67 ; 0,77] |
| Par les médiateurs seuls | +0,101 [0,074 ; 0,130] | +0,172 [0,137 ; 0,209] |

- **À deux tours** (un tour intermédiaire), le choix lit surtout
  directement l'état ancien : la chaîne n'en porte que 30 %.
- **À trois tours** (deux tours intermédiaires), la chaîne pèse bien
  plus : 72 % de l'effet selon la mesure prévue. La lecture directe ne
  fait plus que +0,097.
- **Les effets ne s'additionnent pas** : direct + médiateurs seuls
  (+0,359) reste sous le total (+0,419). L'écart (+0,060 [0,036 ; 0,083])
  n'appartient ni à la seule route directe, ni aux seuls médiateurs ; la
  mesure prévue le compte dans la chaîne. Comptée par les médiateurs
  seuls, la part de la chaîne est plus basse : 0,32 en tout, 0,21 à deux
  tours, 0,50 à trois. L'écart entre deux et trois tours reste net
  (relevé à la relecture).

**Écarts d'exécution, déclarés.**
- **Amendement 1** (un seul fil), avant toute greffe mesurée (voir plus
  haut).
- Le test 23 avait été lu (pour A) avant l'écriture de ce protocole : le
  protocole le dit.

### Ce que cela dit

- **Pas de chaîne dominante en moyenne (CHAIN échoue, de peu).** Sur
  l'ensemble, un peu moins de la moitié de l'effet passe par les états
  intermédiaires (0,46 ; l'intervalle va jusqu'à 0,50).
- **Mais le partage dépend de la distance** (résultats selon h prévus sans
  seuil ; leur lecture ne l'était pas). L'effet direct baisse nettement :
  +0,335 à deux tours, +0,097 à trois. À trois tours, la moitié de l'effet
  ou plus passe par les états intermédiaires (50 à 72 % selon la façon de
  compter). Le besoin semble donc surtout lu directement à deux tours, et
  passer davantage par les états intermédiaires à trois. La relecture a
  vérifié que cet écart tient à t, à i et à l'écart des besoins égaux, et
  au sein des mêmes vies.
- **Ce n'est qu'une observation après lecture.** Elle porte sur un seul
  découpage (h = 2 contre h = 3) et sur les mêmes 128 vies. Elle doit être
  confirmée par un test pré-enregistré, sur des vies neuves et à des
  distances plus grandes, avant d'être revendiquée.
- **Deux biais restent possibles** : l'écart non additif, compté dans la
  chaîne ; et la lecture directe remet à leurs valeurs sans greffe deux
  tours à trois tours de distance, contre un seul à deux tours, ce qui
  crée un conflit plus fort avec le tour greffé. La moyenne générale mêle
  ces deux distances : son verdict dépend de leur proportion (266 et 225
  greffes).
- **Pour le cadre en cinq niveaux** : la récurrence au sens strict (un état
  qui reprend le précédent) n'est pas établie. Une part plus grande passe
  par les états intermédiaires à trois tours qu'à deux, à confirmer.

**Ce que le résultat ne dit pas.**
- Rien sur un ressenti.
- L'agent a appris d'un professeur qui connaît les besoins.
- Remettre les médiateurs à leurs valeurs sans greffe crée un état que
  l'agent ne rencontre jamais : c'est la limite habituelle de ces
  décompositions.
- Il ne dit pas comment l'état est mis à jour d'un tour à l'autre.

## Test 25 pré-enregistré : à distance, le besoin passe-t-il d'état en état ?

Protocole `docs/LLM_NEED_CHAIN_FAR_PROTOCOL.md` (commit `e2920e7`), écrit
après la publication et la relecture du test 24, avant le code du test et
toute mesure (seul le nombre de greffes du tirage avait été compté, sur les
vies du professeur, sans modèle). Code écrit avant toute mesure
(`29e9800`). Mesures torch sur le processeur local, le 4 octobre 2026, en
trois processus à un seul fil (chaque greffe calculée seule). Artefacts :
`artifacts/llm-need/chain-far`. Verdicts recalculés en local (identiques) ;
la CI les vérifie à chaque envoi.

**Pourquoi ce test.** Au test 24, après lecture, la part de l'effet qui
passe par les états des tours intermédiaires semblait grandir avec la
distance : 30 % à deux tours, 72 % à trois (21 % et 50 % en comptant par
les médiateurs seuls). Ce test met à l'épreuve la seconde partie de cette
observation, sur **128 vies neuves**, jamais utilisées, à trois, quatre et
cinq tours : la part qui passe par les états intermédiaires y est-elle
clairement au-dessus de la moitié ? Deux tours ne sont pas remesurés.

**La mesure** est celle du test 24. On greffe les tokens portés du tour i
d'une autre vie (même action au tour i, autres besoins ; avec ses besoins,
la règle changerait de choix au tour t = i + h). On lit P(R) au tour t :
- sans greffe ;
- avec la greffe (**total**) ;
- avec la greffe, mais les tokens portés des tours intermédiaires remis à
  leurs valeurs sans greffe (**direct**) ;
- sans greffe au tour i, mais les tours intermédiaires pris de la lecture
  avec greffe (**par les médiateurs seuls**).

Chaque effet est compté dans le sens où la règle changerait le choix. La
**chaîne** est le total moins le direct.

| 609 greffes (128 vies neuves) | Effet moyen |
|---|---|
| Total | **+0,307** [0,281 ; 0,334] |
| Direct | +0,074 [0,062 ; 0,087] |
| Chaîne (total − direct) | **+0,233** [0,210 ; 0,256] |
| Par les médiateurs seuls | **+0,182** [0,158 ; 0,208] |
| Terme non additif (la part qui n'apparaît que lorsque l'état greffé et les états suivants agissent ensemble : total − direct − médiateurs seuls) | +0,050 [0,036 ; 0,065] |

| | Critère | Verdict |
|---|---|---|
| Condition préalable | effet total ≥ **0,10** (borne basse > 0) | **remplie** : +0,307 [0,281 ; 0,334] |
| **FAR1** | borne basse de (chaîne − la moitié du total) > 0 | **passe** : +0,079 [0,067 ; 0,092] ; la chaîne fait 0,76 du total [0,72 ; 0,79] |
| **FAR2** | borne basse de (médiateurs seuls − la moitié du total) > 0 | **passe** : +0,029 [0,015 ; 0,044] ; les médiateurs seuls font 0,60 du total [0,55 ; 0,64] |
| | Validité : lecture prolongée (le calcul par morceaux redonne la lecture du texte entier) ≤ 1e-4 ; greffe de soi (greffer sa propre vie ne change rien) ≤ 1e-6 ; mêmes positions (vérifié à chaque greffe) ; ≥ 150 greffes par h | **valide** : 5,2e-7 ; 0 ; oui ; 219, 217 et 173 |

**Critère global (FAR1 et FAR2) : satisfait.**

**Lecture fixée d'avance** (FAR1 et FAR2 passent) : « Au-delà de deux
tours, le besoin porté par l'état d'un tour atteint les choix surtout en
passant par les états des tours suivants. Chaque état reprend celui d'avant
et le transmet : c'est une transmission d'état en état, une récurrence au
sens fonctionnel. L'agent l'a apprise alors que son masque lui permettait
de lire directement tout son passé. L'observation du test 24 est confirmée
sur des vies neuves. »

Cette lecture va un peu au-delà des données. La mesure remet ensemble tous
les tours intermédiaires : elle ne montre pas que chaque état reprend celui
d'avant (le test 27 l'a mesuré à trois tours : l'état i + 2 ne reprend
presque rien de l'état i + 1). Et, comptée par les médiateurs seuls, la part n'est nettement
au-dessus de la moitié qu'à quatre tours (voir selon h).

**Publié sans seuil : selon h.**

| | h = 3 (219 greffes, 119 vies) | h = 4 (217, 119) | h = 5 (173, 109) |
|---|---|---|---|
| Total | +0,358 | +0,305 | +0,243 |
| Direct | +0,114 | +0,045 | +0,059 |
| Chaîne (part) | +0,244 (0,68 [0,62 ; 0,74]) | +0,260 (0,85 [0,81 ; 0,89]) | +0,184 (0,76 [0,69 ; 0,82]) |
| Médiateurs seuls (part) | +0,174 (0,49 [0,43 ; 0,54]) | +0,227 (0,74 [0,68 ; 0,80]) | +0,137 (0,56 [0,49 ; 0,63]) |
| Médiateurs seuls − moitié du total | −0,005 [−0,025 ; 0,016] | +0,075 [0,051 ; 0,099] | +0,016 [−0,002 ; 0,034] |
| Terme non additif | +0,070 | +0,033 | +0,047 |

- **Comptée comme au test 24**, la chaîne porte plus de la moitié à chaque
  distance (de 68 à 85 %).
- **Comptée par les médiateurs seuls**, elle porte la moitié à trois tours
  (0,49, comme au test 24 : 0,50), les trois quarts à quatre tours, un peu
  plus de la moitié à cinq tours. Séparément, seul h = 4 est nettement
  au-dessus de la moitié. Le critère porte sur l'ensemble, et **FAR2 ne
  passe que grâce à h = 4** : sans h = 4, +0,004 [−0,012 ; 0,020] (analyse
  après lecture, relevée par la relecture indépendante). FAR1, lui, passe
  sans h = 4 (+0,064 [0,049 ; 0,079]).
- **La part ne grandit pas régulièrement avec la distance** : elle est la
  plus haute à quatre tours et redescend à cinq.
- **La lecture directe de l'état ancien reste faible** à toutes ces
  distances (+0,045 à +0,114).

**Écarts d'exécution, déclarés.**
- **La session de l'agent a redémarré pendant la mesure** (vers 17 h 23 ;
  la machine, elle, n'a pas redémarré). Les trois processus n'ont pas été
  arrêtés : leurs journaux montrent un seul chargement du modèle et une
  progression continue ; ils ont fini normalement (17 h 58, 17 h 59,
  18 h 01). Seule une attente en arrière-plan a été relancée.
- Aucun autre écart.

### Ce que cela dit

- **De trois à cinq tours, prises ensemble, le besoin passe surtout par les
  états des tours intermédiaires.** Une greffe de l'état porté d'un tour
  fait pencher le choix trois à cinq tours plus tard : la probabilité de R
  bouge en moyenne de 0,31 dans le sens où la règle changerait le choix.
  76 % de cet effet disparaissent quand on remet les états intermédiaires
  à leurs valeurs sans greffe. Ces états seuls, pris de la vie greffée, en
  portent 60 %. La lecture directe de l'état ancien ne fait que +0,07.
- **C'est confirmé sur des vies neuves, des deux façons de compter, sur
  l'ensemble des distances.** Par les médiateurs seuls, distance par
  distance, seul quatre tours est net.
- **C'est une forme de récurrence fonctionnelle, apparue avec
  l'apprentissage.** L'architecture ne l'imposait pas : son masque lui
  laisse lire directement les états portés de tout son passé. Mais la tâche
  s'y prête. L'agent décide sur les tokens portés eux-mêmes, et la décision
  dépend des besoins du moment. Ces besoins se déduisent de ceux du tour
  précédent, de l'événement et de l'action (avec un plafond à 8). Et les
  événements passés ne sont visibles qu'à travers les états portés.
  Reprendre l'état précédent est donc une solution naturelle ; l'agent l'a
  trouvée, surtout au-delà de deux tours.
- **Pour le cadre en cinq niveaux** : c'est le premier résultat
  **positif**, pré-enregistré et valide, de Menia sur la récurrence
  elle-même (les tests 13 et 24, valides, avaient échoué), et non
  seulement sur une information portée.

**Ce que le résultat ne dit pas.**
- Rien sur un ressenti.
- L'agent a appris d'un professeur qui connaît les besoins.
- Il ne dit pas si l'état passe d'un tour au suivant à chaque tour, ou en
  sautant des tours : la mesure remet tous les tours intermédiaires
  ensemble.
- Ce n'est pas une boucle à l'intérieur du réseau : la transmission passe
  par les clés et valeurs des tokens portés, relues à chaque tour.
- Remettre les états intermédiaires à leurs valeurs sans greffe, ou les
  prendre seuls de la vie greffée, crée des mélanges que l'agent ne
  rencontre jamais : c'est la limite de ces décompositions. C'est pourquoi
  les deux façons de compter étaient exigées.
- Plus h est grand, plus il y a de tours remis ensemble. Cela peut à lui
  seul baisser l'effet direct et grossir la chaîne comptée comme au
  test 24 (FAR1). La lecture par les médiateurs seuls (FAR2) n'en dépend
  pas ; elle passe, de peu.

## Test 26 pré-enregistré : prédire ce que ses actes font à son corps (le modèle de langage)

Protocole `docs/LLM_NEED_RELIEF_PROTOCOL.md` (commit `eb43887`), écrit avant
le code et toute exécution. Code commis avant tout apprentissage et toute
mesure (`36d4f2b`). Apprentissage sur le Mac (Codemagic, étape `memory`
avec les textes du soulagement), par morceaux de 250 itérations repris
exactement, en cinq demandes du relais (103 à 107 : `29d442f`, `ba3e2c4`,
`a83b3c6`, `bae320b`, `3d3216d`), du 4 au 5 octobre 2026. Mesures torch sur
le processeur local, le 5 octobre (lancé avec `--threads 1`, un fil par
processus ; le nombre de fils n'est pas noté dans les fichiers). Artefacts :
`artifacts/llm-need/relief`. Verdicts recalculés en local (identiques) ; la
CI les vérifie à chaque envoi.

**Ce qu'on teste.** Le modèle de langage (Qwen3-0.6B, à partir de l'agent
final fondu) apprend un nouvel adaptateur avec une seule cible : **le mot
du soulagement** que son action apporte (« aucun », « petit », « moyen »,
« fort »), qui dépend du niveau du besoin servi. **Aucune cible de
choix.** Les vies d'apprentissage sont écrites par une règle qui ignore
les besoins (la règle « événement », 30 % au hasard). Deux masques, ceux du
test 22 : A (« porte », le passé n'est visible qu'à travers ses tokens
portés ; le soulagement passé n'est pas porté) et B (« actions », le
témoin). Puis, sur 256 mondes, l'agent choisit l'action dont il prédit le
plus grand soulagement (règle de lecture fixée par nous).

| | A, porte | B, actions | Plafond des actions (bayésien exact) |
|---|---|---|---|
| Précision du soulagement prédit (128 vies tenues à l'écart, 2 647 décisions) | **0,785** | 0,634 | 0,648 |
| Survie en choisissant par le soulagement prédit (256 mondes du test 22) | **0,820** | 0,621 | — |

Repères sans modèle, sur les mêmes 256 mondes : la même lecture avec les
niveaux exacts 0,852 ; règle « événement » 0,723 ; règle « besoins »
0,906.

| | Critère | Verdict |
|---|---|---|
| **INTER1** | précision de A − plafond des actions ≥ **0,10** (borne basse > 0) | **passe** : +0,137 [0,113 ; 0,161] |
| **INTER2** | survie de A − survie de B, mêmes mondes ≥ **0,08** (paires de vies, borne basse > 0) | **passe** : +0,199 [0,148 ; 0,250] |
| | Validité : réplique (≤ 0,02 en moyenne) ; cache (≤ 1e-4) ; masques (≤ 1e-5) ; masse ≥ 0,5 ; précision de B ≤ plafond + 0,02 | **valide** : réplique 0,006 (A) et 0,007 (B) ; cache 6e-8 et 3e-7 ; masques 0 et 0 ; masse 1,000 ; B à 0,634 (plafond 0,648) |

Intervalles bootstrap à 95 % par vie (10 000 tirages).

**Critère global (INTER1 et INTER2) : satisfait.**

**Lecture fixée d'avance** (INTER1 et INTER2 passent) : « Sans professeur,
en apprenant seulement à prédire ce que ses actes font à son corps, un
modèle de langage qui ne voit son passé qu'à travers ses propres états
apprend à y porter ses besoins. Il prédit son soulagement mieux que tout
observateur de ses seules actions. Et en choisissant l'action dont il
attend le plus grand soulagement, il survit mieux qu'un témoin qui ne porte
que ses actions. »

**Publié sans seuil.**
- **Précision selon le niveau** (A / B / plafond) : « aucun » 0,919 /
  0,754 / 0,795 ; « petit » 0,956 / 0,836 / 0,793 ; « moyen » 0,590 /
  0,407 / 0,503 ; « fort » 0,469 / 0,311 / 0,282 (443, 1 112, 783 et 309
  décisions). A dépasse le plafond à chaque niveau ; les besoins bas
  restent les plus durs à prédire.
- **Part des décisions où la lecture prend l'action de la règle
  « besoins »** (128 vies de mesure, sans changer la vie) : A 0,872 ;
  B 0,840.
- **Après 2 000 itérations** (lectures publiées dès leur fin, le 5 octobre
  à 6 h 43 et 6 h 46 UTC, alors que les itérations 2 001 à 3 500 étaient
  déjà lancées — finies pour A, en cours pour B — ; seules 3 501 à 4 000
  ont été lancées après ; le nombre d'itérations était fixé d'avance) :
  A 0,747, soit +0,099 [0,076 ; 0,122] au-dessus du plafond ; B 0,635
  (−0,013). L'écart de A a continué de croître de 2 000 à 4 000
  itérations ; la hausse vient des niveaux hauts : chez A, « moyen » et
  « fort » ont baissé de 2 000 à 4 000 itérations (0,68 → 0,59 ;
  0,57 → 0,47), « aucun » et « petit » ont monté (0,87 → 0,92 ;
  0,80 → 0,96).
- **Pertes de validation** (32 vies tenues à l'écart, sur les mots du
  soulagement ; notées par le Mac à la fin de chaque morceau, comme au
  test 22) : A 1,18 après 250 itérations, 0,84 à 500, 0,79 à 750, 0,52 à
  2 000 et 0,44 à 4 000 ; B 1,00 après 250, 0,84 à 500, puis entre 0,80 et
  0,86 sans baisser nettement (0,80 à 2 000, 0,81 à 4 000). (Première
  version publiée : « A 0,79 après 250 itérations », erreur de lecture des
  journaux par morceau ; corrigée après relecture.)
- **Précautions** : tours vécus avec un besoin à 2 ou moins, sur les 256
  vies : A 653, B 829 (tour de la mort compris, comme aux tests
  précédents ; 607 et 732 sans lui).

**Écarts d'exécution, déclarés.**
- **Lectures anticipées.** Les lectures finales de A (`fe95bca`) puis de B
  (`b9ec1e8`) ont été publiées dès leur fin, et la valeur d'INTER1 a été
  calculée et annoncée avant la fin des vies (A +0,137 ; B sous le
  plafond). Aucun seuil n'a changé ; INTER2 n'a été lu qu'à la fin.
- **La machine a redémarré** le matin du 5 octobre, pendant les premières
  lectures finales ; rien n'avait été écrit. Le code a été modifié pour
  garder les lectures toutes les 8 vies et reprendre exactement (`227a76f`,
  exécution seulement) ; les lectures ont été relancées de zéro.
- **Le conteneur a redémarré** à 14 h 07 UTC, pendant les vies ; elles ont
  repris de leur sauvegarde (toutes les 8 vies).
- Vers 18 h UTC, un calcul du test 28 a presque rempli la mémoire et ralenti
  la machine ; il a été arrêté puis repris. Sans effet sur ces mesures
  (calcul déterministe à un fil).
- D'autres calculs (tests 28, 31, 32) tournaient en même temps.

### Ce que cela dit

- **Le modèle de langage fait comme les petits transformeurs du test 29**,
  en moins fort (+0,137 contre +0,237 ; survie 0,820 contre 0,884), et loin
  d'une mémoire parfaite (1).
  Sans choix à imiter, en apprenant seulement à prédire le soulagement de
  ses actions, l'agent qui ne voit son passé qu'à travers ses propres
  tokens portés prédit ce soulagement bien mieux que tout observateur de
  ses seules actions (+0,137). Il porte donc, dans ces tokens, une
  information sur ses besoins que ses actions ne donnent pas.
- **Lu par une règle fixée par nous, cela le fait survivre** : 0,820
  contre 0,621 pour le témoin, près de ce que donnerait la même lecture avec
  les niveaux exacts (0,852).

**Ce que le résultat ne dit pas.**
- Rien sur un ressenti. « Soulagement » est le nom d'un signal du monde.
- « Sans professeur » veut dire : sans choix à imiter. **La cible reste une
  supervision directe du besoin** : le niveau du soulagement est le besoin
  servi par l'action, rangé par le monde en quatre classes. Le résultat
  montre que ce besoin, ainsi supervisé, passe par les états portés ; pas
  qu'il émerge sans signal qui le nomme (c'est la question du test 31,
  pour les petits transformeurs).
- La règle qui choisit l'action la plus soulageante est fixée par nous.
- Ce n'est pas un état unique mis à jour à chaque tour : un tour lit
  directement les tokens portés de tout son passé. Avec 28 couches, ces
  tokens peuvent porter plus que l'événement de leur tour (contrairement
  aux petits modèles à deux couches), mais ce qu'ils portent n'est pas
  mesuré ici : c'est la question du test 28 (publié : surtout l'événement
  du tour ; à événement égal, pas plus que ce que disent les actions).
- Un seul apprentissage (une graine), sur un seul modèle.

## Test 27 pré-enregistré : le relais pas à pas

Protocole `docs/LLM_NEED_RELAY_PROTOCOL.md` (commit `4a4527a`), écrit après
la publication du test 25 et sa relecture, avant le code et toute mesure
(seul le nombre de greffes du tirage avait été compté, sans modèle). Code
commis avant toute mesure (`b8dfba9`). Mesures torch sur le processeur
local, le 4 octobre 2026, en trois processus à un seul fil. Artefacts :
`artifacts/llm-need/relay`. Verdicts recalculés en local (identiques) ; la
CI les vérifie à chaque envoi.

**La question.** Au test 25, l'effet d'un état greffé passe surtout par les
états des tours intermédiaires. Mais chaque état reprend-il **celui
d'avant** (i → i + 1 → i + 2), ou chacun prend-il la greffe directement de
l'état ancien (i → i + 2) ? On regarde l'état du tour i + 2, le dernier
avant le choix du tour t = i + 3 (la greffe est faite trois tours avant le
choix), sur **128 vies neuves**.

**La mesure.** Trois versions de l'état porté du tour i + 2 :
- **complet** : tour i greffé, tour i + 1 calculé avec la greffe ;
- **par i seul** : tour i greffé, mais les 4 tokens portés du tour i + 1
  remis à leurs valeurs sans greffe (le reste du tour i + 1 n'est pas
  visible du tour i + 2) avant de calculer le tour i + 2 ;
- **par i + 1 seul** : tour i sans greffe, tour i + 1 pris de la lecture
  avec greffe.

Chaque version est mise seule dans la lecture sans greffe (tours i et i + 1
sans greffe), et on lit P(R) au tour t. L'effet est compté dans le sens où
la règle changerait le choix.

| 209 greffes (116 vies receveuses) | Effet moyen |
|---|---|
| État i + 2 complet | +0,066 [0,044 ; 0,091] |
| État i + 2 par i seul | **+0,058** [0,038 ; 0,082] |
| État i + 2 par i + 1 seul | **+0,005** [−0,002 ; 0,013] |
| Terme non additif (complet − par i seul − par i + 1 seul) | +0,003 [−0,005 ; 0,010] |
| Pour comparaison : effet total au tour t (comme au test 25) | +0,406 [0,355 ; 0,458] |

Intervalles à 95 %, bootstrap par vie receveuse.

| | Critère | Verdict |
|---|---|---|
| Condition préalable | effet de l'état i + 2 complet ≥ **0,05** (borne basse > 0) | **remplie** : +0,066 [0,044 ; 0,091] |
| **RELAY** | borne basse de (par i + 1 seul − la moitié du complet) > 0, et borne basse de (par i + 1 seul − par i seul) > 0 | **échoue** : −0,028 [−0,040 ; −0,017] et −0,053 [−0,077 ; −0,032] |
| | Validité : lecture prolongée (le calcul par morceaux redonne la lecture du texte entier) ≤ 1e-4 ; greffe de soi (greffer sa propre vie ne change aucune lecture) ≤ 1e-6 ; mêmes positions (vérifié à chaque greffe) ; ≥ 150 greffes | **valide** : 4,4e-7 ; 0 ; oui ; 209 |

**Critère global : non satisfait.**

**Lecture fixée d'avance** (RELAY échoue, condition préalable remplie) :
« L'état du tour i + 2 ne reçoit pas le besoin surtout à travers l'état du
tour i + 1. La transmission du test 25 n'est pas, ici, un relais pas à
pas. »

**Écarts d'exécution, déclarés.**
- **Le contrôle de validité s'est d'abord arrêté** : il prenait les quatre
  premières vies, et la deuxième meurt avant le tour 7. Le code a été
  corrigé pour prendre les quatre premières vies vivantes au tour 7
  (`22ac58c`), sans rien changer aux greffes ni aux seuils. Les deux autres
  processus avaient déjà commencé leurs greffes ; le premier a été relancé.
  Aucune greffe n'avait été lue. Le message du commit dit « la troisième
  vie » : c'est la deuxième.

### Ce que cela dit

- **Pas de relais pas à pas.** L'état du tour i + 2 porte un peu de la
  greffe (+0,066), mais il la prend presque entièrement **directement** de
  l'état greffé du tour i (+0,058), et presque rien à travers l'état du
  tour i + 1 (+0,005) (compté par ce qui change le choix du tour t).
- **Ce que cela dit du test 25.** Là-bas, l'effet passait surtout par les
  états intermédiaires. Ici, à trois tours, l'état du tour i + 2 prend ce
  qu'il porte de la greffe directement de l'état ancien, presque rien à
  travers l'état i + 1. Seul ce maillon est mesuré ; on ne sait rien des
  distances de quatre et cinq tours. Et l'état i + 2 seul porte peu de
  l'effet au tour t (+0,066, un sixième de l'effet total, +0,406). Ce test
  ne mesure pas l'état i + 1 seul ; rapproché du test 25 (autres vies : à
  trois tours, les deux états intermédiaires ensemble y portaient +0,17 sur
  +0,36), cela suggère, sans le montrer, que l'état i + 1 compte davantage.
- **Pour le cadre en cinq niveaux** : à trois tours, entre i + 1 et i + 2,
  la transmission par les états intermédiaires (test 25) n'est pas une mise
  à jour d'état pas à pas. Sur ce maillon, ce n'est pas une récurrence au
  sens strict d'un état qui reprend le précédent.

**Ce que le résultat ne dit pas.**
- Rien sur un ressenti.
- L'agent a appris d'un professeur qui connaît les besoins ; et ce n'est
  pas une boucle interne au réseau : tout passe par les clés et valeurs des
  tokens portés, relues à chaque tour.
- Un seul écart (trois tours) et un seul maillon (i + 1 → i + 2) sont
  mesurés.
- L'effet de chaque version n'est compté que par ce qu'il change au choix
  du tour t (états i et i + 1 sans greffe) : ce que l'état i + 2 recevrait
  de l'état i + 1 sans que ce choix s'en serve n'est pas compté.
- Les lectures mélangent des états greffés et non greffés : c'est la
  limite de ces décompositions.

## Test 28 pré-enregistré : sans professeur, que porte l'état ?

Protocole `docs/LLM_NEED_RELIEF_GRAFT_PROTOCOL.md` (commit `39d6f4d`),
commis deux minutes avant le code (`74d0c3d`) ; les greffes possibles qu'il
cite avaient été comptées, sans modèle, avec le tirage de ce code. Les deux
ont été commis avant la fin de l'apprentissage du test 26 et avant toute
mesure de ses agents. Calcul
torch sur le processeur local, le 5 octobre 2026, en 4 parts à un fil.
Artefacts : `artifacts/llm-need/relief-graft`. Verdicts recalculés en local
(identiques) ; la CI les vérifie à chaque envoi.

**Ce qu'on teste.** La greffe du test 23, sur l'agent A du test 26 (le
modèle de langage appris **sans choix à imiter**, seulement à prédire le
soulagement de ses actions). Sur 128 vies neuves écrites par la règle
« événement », on remplace, dans toutes les couches, les clés et valeurs
des 4 tokens portés du tour j d'une vie receveuse par ceux d'une vie
donneuse qui a écrit la même action au tour j. On lit ensuite le
soulagement prédit un ou deux tours plus tard. **ΔL** = niveau attendu (0 à
3) avec la greffe − sans. **e** = ce que changerait, sur le vrai niveau du
soulagement, de donner à la receveuse les besoins de la donneuse au tour j.

| Greffes (400 (a), 200 (a0), 200 (b)) | A (« porte ») : \|ΔL\| moyen | B (« actions », témoin) : \|ΔL\| moyen |
|---|---|---|
| (a) autres besoins, e ≠ 0 | **0,402** | 0,158 |
| (a0) autres besoins, e = 0 | 0,182 | 0,132 |
| (b) mêmes besoins, autre histoire (dont 32 sans autre histoire, voir les écarts) | **0,136** | 0,098 |

| | Critère | Verdict |
|---|---|---|
| **SELF1** | sur les greffes (a) : moyenne de ΔL × signe(e) ≥ **0,15** niveau (borne basse > 0) | **passe** : +0,348 [0,306 ; 0,390] |
| **SELF2** | \|ΔL\| des greffes (b) ≤ **la moitié** de celui des greffes (a), et différence à borne basse > 0 | **passe** : 0,136 contre 0,402 (seuil 0,201) ; différence +0,266 [0,213 ; 0,317] |
| | Validité : sans greffe ≤ 1e-4 ; greffe de soi ≤ 1e-6 ; mêmes positions ; ≥ 100 greffes (a) et (b) | **valide** : 0 ; 0 ; oui ; 400 et 200 |

Intervalles bootstrap à 95 % par vie receveuse (10 000 tirages).

**Critère global (SELF1 et SELF2) : satisfait.**

**Lecture fixée d'avance** (SELF1 et SELF2 passent) : « Sans professeur, ce
que l'agent porte d'un tour à l'autre dépend surtout de son besoin. Greffé
dans une autre vie, il fait prédire le soulagement selon les besoins de la
vie d'origine ; venu d'une vie aux mêmes besoins mais à l'histoire
différente, il change peu la prédiction. »

**Publié sans seuil.**
- **Selon g** (A) : effet aligné +0,428 à un tour, +0,267 à deux tours ;
  \|ΔL\| des greffes (b) : 0,159 et 0,111.
- **Pente de ΔL sur e** (A) : 0,28. A retrouve environ un quart de l'effet
  qu'aurait, sur le vrai niveau, le changement de besoins.
- **Le témoin B** bouge aussi, moins : effet aligné +0,106 [0,088 ; 0,124]
  (pente 0,09). Sous son masque, les tours suivants ne lisent de la
  donneuse que son token d'action, qui résume ses actions passées ; ces
  actions renseignent un peu sur ses besoins (le plafond des actions du
  test 26). L'effet de A est plus de trois fois celui de B.

**Nuance (après lecture, exploration non pré-enregistrée, étendue après la
relecture indépendante ; `research/need_relief_graft_event.py`,
`artifacts/llm-need/relief-graft/exploration-event.json`).** L'événement
du tour j compte beaucoup. On sépare les greffes selon que donneuse et
receveuse avaient le même événement au tour j :

| \|ΔL\| moyen (A / B) | même événement au tour j | autre événement |
|---|---|---|
| (a) autres besoins, e ≠ 0 | 0,185 / 0,163 (74) | 0,451 / 0,157 (326) |
| (a0) autres besoins, e = 0 | 0,117 / 0,132 (45) | 0,201 / 0,131 (155) |
| (b) mêmes besoins | **0,034** / 0,051 (112, dont 32 sans autre histoire) | **0,266** / 0,157 (88) |

| Greffes (a) : effet aligné ΔL × signe(e) | A | B | A − B (apparié) |
|---|---|---|---|
| même événement au tour j (74) | +0,162 [0,107 ; 0,225] | +0,139 [0,100 ; 0,183] | +0,023 [−0,036 ; 0,082] |
| autre événement (326) | +0,390 [0,342 ; 0,437] | +0,098 [0,078 ; 0,119] | +0,292 [0,242 ; 0,343] |

- **L'avantage de A sur le témoin passe par l'événement du tour j.** À
  événement égal, l'effet aligné de A n'est pas plus grand que celui de B,
  qui ne porte que ses actions (différence appariée +0,02 [−0,04 ; 0,08]).
  Presque tout l'écart entre A et B vient des greffes à événement
  différent.
- **À besoins égaux**, un autre événement au tour j bouge déjà la
  prédiction de 0,266 : l'état porté contient fortement **l'événement du
  tour**.
- À événement égal, des besoins différents bougent la prédiction près de
  quatre fois plus que des besoins égaux venus d'une autre histoire
  (0,185 contre 0,048 ; 0,034 avec les 32 greffes sans autre histoire).
  Mais B, qui ne voit pas l'événement, montre le même contraste (0,163
  contre 0,051) : il vient au moins en partie de ce que disent les
  actions, pas d'un besoin porté au-delà.
- **SELF2 passe grâce au partage d'événement.** Les donneuses (b)
  partagent plus souvent l'événement de la receveuse (56 % ; 48 % sans les
  32 greffes sans autre histoire) que les donneuses (a) (18,5 %). À
  événement différent, \|ΔL\| (b) vaut 0,59 fois \|ΔL\| (a) ; à la
  composition d'événements des greffes (a), il vaudrait 0,22, soit 0,55
  fois \|ΔL\| (a), et SELF2 ne passerait pas.
- Pour (b), \|ΔL\| de B dépend aussi du partage d'événement (0,051 contre
  0,157), alors que B ne voit pas l'événement : à besoins égaux, les
  donneuses qui partagent l'événement ont aussi une histoire plus proche.
  Le contraste de A (0,034 contre 0,266) ne mesure donc pas l'événement
  seul.

**Écarts d'exécution, déclarés.**
- **32 des 200 greffes (b) n'ont pas d'autre histoire** (relevé à la
  relecture, comme au test 23) : la donneuse a vécu les mêmes événements et
  actions que la receveuse jusqu'au tour j (j ≤ 3 ; toutes celles de
  j = 1). Leur greffe ne change rien par construction (ΔL = 0). Le
  protocole disait « autre histoire » sans l'imposer. Sans elles,
  \|ΔL\| (b) vaut 0,162 ; SELF2 passe toujours (différence +0,240
  [0,182 ; 0,295]).
- Les greffes ont commencé (17 h 17 UTC) avant la fin des vies du test 26 ;
  les lectures finales de A et B du test 26 avaient été lues. Parts
  lancées : 1 à 17 h 17, 2 vers 17 h 20, 3 à 19 h 38, 4 à 20 h 48.
- Vers 18 h UTC, la deuxième part, en passant de A à B, a chargé deux
  modèles et presque rempli la mémoire ; elle a été arrêtée, puis reprise
  de sa sauvegarde (toutes les 25 greffes) à 18 h 54. Une garde devait
  ensuite arrêter automatiquement la quatrième part en cas de manque de
  mémoire (elle n'a pas eu à le faire). Calcul déterministe à un fil : sans
  effet sur les mesures.
- D'autres calculs (tests 26, 31, 32) tournaient en même temps.

### Ce que cela dit

- **Sans choix à imiter, l'état porté fait prédire le soulagement dans le
  sens des besoins d'origine** (+0,35 niveau), plus de trois fois plus que
  chez le témoin B (+0,11).
- **Mais cet avantage passe par l'événement du tour j.** À événement égal
  (74 greffes (a)), l'effet aligné de A (+0,16 [0,11 ; 0,22]) n'est pas plus
  grand que celui de B, qui ne porte que les actions (+0,14 ; différence
  appariée +0,02 [−0,04 ; 0,08]) ; presque tout l'écart entre A et B vient
  des greffes à événement différent (+0,29 [0,24 ; 0,34]). Ce n'est pas un
  état des seuls besoins.
- Ces données ne montrent donc pas qu'avec 28 couches l'état porté d'un
  tour contienne plus que l'événement de ce tour et ce que disent les
  actions passées, ce que portent aussi les petits modèles à deux couches
  (test 29) ; elles ne l'excluent pas non plus (74 greffes).
- La lecture fixée d'avance (« dépend surtout de son besoin ») est donc
  trop forte : SELF1 et SELF2 passent, mais en grande partie par
  l'événement du tour que l'état porte.

**Ce que le résultat ne dit pas.**
- Rien sur un ressenti.
- La cible du test 26 (le niveau du soulagement) est une supervision
  directe du besoin servi.
- La greffe porte sur les 4 tokens d'un seul tour ; un tour lit aussi
  directement les tokens portés de tout son passé.
- Il ne sépare pas le niveau des besoins de ce qui commande le soulagement
  de l'action écrite.
- Un seul agent (une graine d'apprentissage).

## Test 29 pré-enregistré : sans pré-entraînement, de petits transformeurs portent-ils leurs besoins ?

Protocole `docs/TINY_RELIEF_PROTOCOL.md` (commit `d6c6f03`), écrit avant le
code et toute exécution. Code commis avant toute mesure (`b972387`), essayé
sur 100 pas de la graine 99, hors protocole (512 vies d'apprentissage, A et
B ; précision lue sur les 128 vies tenues à l'écart, survie sur 16 des 256
mondes, contrôles des masques : écarts 0). Seule modification ensuite : la
conversion de la perte pour le journal (`loss.detach()`), sans effet sur
l'apprentissage. Calcul torch sur le
processeur local, le 4 octobre 2026 (deux processus : graines 0 à 4 et 5
à 9). Artefacts : `artifacts/tiny-relief`. Verdicts recalculés en local
(identiques) ; la CI les vérifie à chaque envoi.

**Ce qu'on teste.** Le test 26 (en cours sur le Mac ; publié depuis : valide, INTER1 et
INTER2 passent ; « sans professeur » au sens de sans choix à imiter)
demande si un modèle
de langage pré-entraîné apprend, **sans professeur**, à porter ses besoins
dans ses propres états, en apprenant seulement à prédire le soulagement de
ses actions. Ici, on pose la même question à de **petits transformeurs
appris de zéro** (2 couches, dimension 64, aucun pré-entraînement), sur
**10 graines**. Les vies sont celles du test 26 : actions écrites par une
règle qui ignore les besoins (la règle « événement », 30 % au hasard). La
seule cible est le niveau du soulagement ; **aucune cible de choix**.
Chaque tour s'écrit en 5 tokens : l'événement, « Choix », l'action,
« Soulagement », le niveau.

**Les masques** (ceux du test 22) :
- **A (« porte »)** : un tour ne voit, de son passé, que les tokens
  « Choix » et l'action des tours passés ;
- **B (« actions »)**, le témoin : les tours suivants ne voient que les
  actions passées ; le token de l'action ne voit pas l'événement de son
  tour ;
- **C (« libre »)**, publié sans seuil : tout le passé visible.

**Plafond des actions** : un observateur bayésien exact qui connaît le
monde et la règle qui écrit les actions, et voit l'événement du tour,
l'action jugée et toutes les actions passées, prédit le bon niveau pour
**0,648** des décisions.

| Moyenne sur 10 graines | A, porte | B, actions | C, libre |
|---|---|---|---|
| Précision du soulagement prédit (128 vies tenues à l'écart) | **0,885** | 0,634 | 0,935 |
| Survie en choisissant par le soulagement prédit (256 mondes du test 22) | **0,884** | 0,666 | 0,884 |
| Perte du lot au pas 100 → au pas 3 000 (moyenne des graines) | 1,09 → 0,26 | 1,09 → 0,79 | 1,04 → 0,15 |

| | Critère | Verdict |
|---|---|---|
| **TINY1** | moyenne de (précision de A − plafond) ≥ **0,10**, borne basse > 0, et au moins 8 graines sur 10 au-dessus de 0,05 | **passe** : +0,237 [0,224 ; 0,250] ; 10 graines sur 10 |
| **TINY2** | moyenne de (survie de A − survie de B, même graine) ≥ **0,08**, borne basse > 0 | **passe** : +0,219 [0,196 ; 0,241] |
| | Validité : masques (écart ≤ 1e-6) ; précision moyenne de B ≤ plafond + 0,02 | **valide** : écarts 0 et 0 ; B à 0,634, sous le plafond (0,648) |

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de
liberté ; l'incertitude due aux 128 vies, les mêmes pour toutes les
graines, n'y est pas comptée). Les graines changent l'initialisation et
l'ordre des lots ; l'ordre des lots est tiré du flux
`[270926, 70, graine, 1]` (le protocole disait `[270926, 70, graine]`).

**Critère global (TINY1 et TINY2) : satisfait.**

**Lecture fixée d'avance** (TINY1 et TINY2 passent) : « La mémoire par
l'état ne dépend pas du pré-entraînement. Un petit transformeur appris de
zéro, sous ce masque, en apprenant seulement à prédire ce que ses actes
font à son corps, porte ses besoins dans ses propres états, mieux que tout
observateur de ses seules actions, et s'en sert pour survivre. Et c'est
vrai sur dix graines. »

**Publié sans seuil : graine par graine.**

| Graine | A : précision | B : précision | C : précision | A : survie | B : survie | C : survie |
|---|---|---|---|---|---|---|
| 0 | 0,903 | 0,626 | 0,959 | 0,887 | 0,699 | 0,906 |
| 1 | 0,867 | 0,639 | 0,930 | 0,852 | 0,590 | 0,879 |
| 2 | 0,876 | 0,638 | 0,937 | 0,891 | 0,711 | 0,887 |
| 3 | 0,890 | 0,636 | 0,922 | 0,883 | 0,719 | 0,875 |
| 4 | 0,860 | 0,633 | 0,903 | 0,887 | 0,668 | 0,836 |
| 5 | 0,903 | 0,631 | 0,964 | 0,918 | 0,684 | 0,910 |
| 6 | 0,873 | 0,637 | 0,934 | 0,891 | 0,652 | 0,902 |
| 7 | 0,871 | 0,626 | 0,933 | 0,852 | 0,629 | 0,875 |
| 8 | 0,896 | 0,635 | 0,935 | 0,895 | 0,660 | 0,883 |
| 9 | 0,912 | 0,640 | 0,929 | 0,891 | 0,645 | 0,891 |

- **Chaque graine** de A dépasse le plafond des actions de 0,21 à 0,26.
  Aucune graine de B ne l'atteint (0,626 à 0,640).
- **C (« libre »)**, qui voit tout le passé, prédit mieux que A (0,935
  contre 0,885). Mais A survit autant que C (écart moyen 0,000 [−0,016 ;
  +0,016]) : avec cette lecture, on ne mesure pas de coût, à ±0,016 près,
  à ne voir son passé qu'à travers ses propres états. Les deux moyennes
  sont égales par coïncidence : les vies qui survivent ne sont pas les
  mêmes.
- **Repères sans modèle** (protocole du test 26) : la même lecture,
  appliquée aux niveaux exacts, survit à 0,852 ; la règle « besoins » à
  0,906. A (0,884) est entre les deux, au-dessus de la même lecture
  appliquée aux niveaux exacts ; explication possible, non testée : ses
  probabilités sont plus fines que les quatre niveaux.

**Écarts d'exécution, déclarés.**
- Les 30 résultats étaient d'abord exclus du dépôt par une règle générale
  du `.gitignore` (`runs/`). Une exception a été ajoutée pour les publier.
  Aucun autre écart.

### Ce que cela dit

- **Sans professeur et sans pré-entraînement, la mémoire par l'état
  apparaît, sur toutes les graines.** Un petit transformeur qui ne voit son
  passé qu'à travers ses propres états portés, et qui apprend seulement à
  prédire le soulagement de ses actions, prédit ce soulagement bien mieux
  que tout observateur de ses seules actions (+0,24, 10 graines sur 10).
  Il porte donc, dans ses états, une information sur ses besoins que ses
  actions ne donnent pas.
- **Ses prédictions, lues par une règle fixée par nous, le font
  survivre.** En prenant l'action dont il attend le plus grand
  soulagement, il survit à 0,88, contre 0,67 pour le témoin qui ne
  porte que ses actions, et autant que le modèle qui voit tout son passé.
- **Ce que cela change pour le programme.** Jusqu'ici, la mémoire par
  l'état n'était établie que chez un modèle de langage pré-entraîné
  (Qwen3-0.6B), appris d'un professeur (test 22). Le pré-entraînement n'est
  donc pas nécessaire : sous ce masque, avec une cible qui l'exige, un petit
  transformeur appris de zéro suffit, sur ces tokens simples, pour 10
  graines sur 10 (initialisation et ordre des lots ; mêmes vies). Le test
  ne dit pas ce que le pré-entraînement apporte au modèle de langage (test
  26), ni si une autre architecture ferait de même ; les tests 22 et 29
  diffèrent aussi par la cible, les tokens et la taille.

**Ce que le résultat ne dit pas.**
- Rien sur un ressenti. « Soulagement » est le nom d'un signal du monde.
- « Sans professeur » veut dire : sans choix à imiter. La cible reste une
  supervision directe du besoin : le niveau du soulagement est le besoin
  servi par l'action, rangé par le monde en quatre classes. Le résultat
  montre que ce besoin, ainsi supervisé, passe par les états portés ; pas
  qu'il émerge sans signal qui le nomme. La règle qui choisit l'action la
  plus soulageante est fixée par nous.
- Il ne dit pas **ce que** les états portent (le besoin lui-même, ou ce qui
  commande le soulagement) : c'est la question du test 28, pour le modèle
  de langage.
- **Précision ajoutée le 5 octobre, après lecture (vérifiée ; corrigée
  après la relecture du test 31).** Avec **deux couches**, ce que les tours
  suivants peuvent lire des tokens portés d'un tour (leur sortie de la
  couche 0, seule lue par la couche 1) ne dépend que de **l'événement de ce
  tour** et des actions jusqu'à ce tour. À la couche 0, ces tokens ne voient
  des tours passés que les plongements de « Choix » et des actions, qui ne
  disent rien des événements. Le besoin, qui dépend de tous les événements
  passés, n'est donc **pas porté comme un état mis à jour** de tour en
  tour : chaque décision le recompose en lisant, dans les tokens portés de
  chaque tour passé, l'événement que ce tour y a écrit. Dans une vraie vie,
  un événement plus ancien peut encore les atteindre à travers les actions
  qu'il a fait choisir, un canal que le témoin B a aussi. Vérifié sur les
  agents du test 31 (graine 0, exploration non pré-enregistrée,
  `research/tiny_two_layer_check.py`,
  `artifacts/tiny-survival-ac/exploration/two-layer-check.json`) : à
  actions inchangées, changer un événement plus ancien que le tour j ne
  change pas du tout ce que les tours suivants lisent des tokens portés du
  tour j (écart 0, 64 cas). Pour comparaison, il change l'état final du
  token « Choix » deux tours plus tard (moyenne, sur 64 cas, du plus grand
  écart absolu sur ses 64 dimensions : 0,43), mais peu P(R) (écart moyen
  0,017 ; médiane 6e-6). Chez le modèle libre, ce que les tours suivants
  lisent des tokens du tour j change, jusqu'à 0,74 (plus grand écart sur
  les 64 cas). C'est une propriété du masque et de la profondeur, vraie
  pour tous les poids. La « mémoire par l'état » des tests 29 à 31 est donc
  une mémoire des événements écrite dans ses propres tokens, **pas une
  récurrence**. (Première version : l'événement changé n'était jamais
  celui du tour juste avant j, et l'écart de l'état final à « Choix » était
  appelé à tort un écart de « la décision » ; corrigé, nouveaux tirages.)
- Ces modèles sont petits, le monde aussi. D'autres tailles, d'autres
  durées d'apprentissage ou d'autres tokens pourraient donner autre chose.
- Les tokens sont simples et à positions fixes ; ce n'est pas du langage.

## Test 30 pré-enregistré : seulement survivre (non valide)

Protocole `docs/TINY_SURVIVAL_PROTOCOL.md` (commit `5d57bdb`), écrit avant le
code et toute exécution. Code commis avant toute mesure (`5015d88`), essayé
sur 20 mises à jour de la graine 99, hors protocole (survie non lue).
Calcul torch sur le processeur local, les 4 et 5 octobre 2026. Artefacts :
`artifacts/tiny-survival`. Verdicts recalculés en local (identiques) ; la CI
les vérifie à chaque envoi.

**Ce qu'on teste.** Au test 29, la cible (le niveau du soulagement)
nommait encore le besoin. Ici, plus rien ne le nomme : de petits
transformeurs appris de zéro agissent, et n'apprennent que de leur
**survie** (1 par tour vécu), par renforcement (REINFORCE, 2 000 mises à
jour de 64 vies), sous les masques A (« porte »), B (« actions ») et C
(« libre »), sur 10 graines.

| Moyenne sur 10 graines | A, porte | B, actions | C, libre |
|---|---|---|---|
| Survie (256 mondes du test 22, action la plus probable) | 0,308 | 0,405 | 0,381 |
| Part des décisions où il prend l'action de la règle « besoins » | 0,612 | 0,693 | 0,687 |
| Survie pendant l'apprentissage (100 premières mises à jour → 1 000 → 2 000) | 0.21 → 0.33 → 0.28 | 0.23 → 0.39 → 0.39 | 0.21 → 0.36 → 0.35 |

Repères sans modèle, sur les mêmes mondes : règle « événement » 0,723 ;
règle « besoins » 0,906.

| | Critère | Valeur |
|---|---|---|
| **SURV1** | survie de A − survie de B ≥ 0,08 (borne basse > 0) | −0,097 [−0,182 ; −0,011] |
| **SURV2** | survie de A − 0,723 ≥ 0,05 (borne basse > 0) | −0,414 [−0,506 ; −0,323] |
| | Validité : masques (écart ≤ 1e-6) ; **survie moyenne de B ≥ 0,60** | masques 0 et 0 ; **B à 0,405 : non valide** |

**Le test n'est pas valide** : l'apprentissage par renforcement n'a pas eu
lieu assez pour que la question se pose. Les verdicts SURV1 et SURV2 ne
sont pas revendiqués.

**Lecture fixée d'avance** (l'apprentissage n'a pas eu lieu) : « le test
n'est pas valide ».

**Graine par graine (survie).**

| Graine | A, porte | B, actions | C, libre |
|---|---|---|---|
| 0 | 0,383 | 0,414 | 0,461 |
| 1 | 0,379 | 0,473 | 0,367 |
| 2 | 0,305 | 0,340 | 0,371 |
| 3 | 0,457 | 0,590 | 0,406 |
| 4 | 0,418 | 0,371 | 0,363 |
| 5 | 0,320 | 0,281 | 0,309 |
| 6 | 0,000 | 0,367 | 0,371 |
| 7 | 0,309 | 0,398 | 0,391 |
| 8 | 0,289 | 0,422 | 0,371 |
| 9 | 0,223 | 0,395 | 0,402 |

**Écarts d'exécution, déclarés.**
- **La machine a redémarré pendant le calcul** (vers 23 h 05 le 4 octobre).
  Les apprentissages déjà écrits (graines 0 et 5 entières ; 1 et 6 sans
  C) ont été publiés tels quels. Le code a été modifié pour reprendre sans
  réapprendre les bras déjà mesurés (`b9f444b`), puis
  les graines restantes ont été lancées.
- Un pilote pour le test suivant a tourné en même temps, en priorité
  basse, puis a été suspendu jusqu'à la fin de ce test (il ne touche pas
  ces résultats).

### Ce que cela dit

- **Avec ce budget et cet algorithme, aucun agent n'apprend à survivre**,
  quel que soit le masque : tous restent loin sous la règle « événement »
  (0,72), qu'un témoin bien appris devrait atteindre. Une graine de A
  s'effondre même à 0.
- **Ce n'est pas une réponse à la question** (porter ses états aide-t-il,
  sans aucune cible ?). C'est un échec de méthode : REINFORCE avec une
  récompense de survie lointaine et bruitée apprend trop mal ici.
- **Ce qui suit** : un pilote règle l'apprentissage sur le **témoin seul**
  (`docs/TINY_SURVIVAL_PILOT_PLAN.md`), avant de reposer la question au
  test 31.

**Ce que le résultat ne dit pas.** Rien sur un ressenti ; rien sur la
mémoire par l'état sans cible, ni pour ni contre.

## Test 31 pré-enregistré : seulement survivre, avec un apprentissage qui marche

Pilote (plan `docs/TINY_SURVIVAL_PILOT_PLAN.md`, commit `5edfec6` ; code
`4d70441`), puis protocole `docs/TINY_SURVIVAL_AC_PROTOCOL.md` (commit
`27c7fe6`), écrit après le pilote et avant tout apprentissage de A ou de C.
Code commis avant tout apprentissage (`00b1d2e`). Calcul torch sur le
processeur local, le 5 octobre 2026 (quatre processus). Artefacts :
`artifacts/tiny-survival-ac`. Verdicts recalculés en local (identiques) ;
la CI les vérifie à chaque envoi.

**Ce qu'on teste.** La question du test 30, avec un apprentissage qui
marche. De petits transformeurs appris de zéro (2 couches, dimension 64)
n'apprennent que de leur **survie** (1 par tour vécu, rien d'autre). Rien
ne leur nomme leurs besoins, aucun choix n'est à imiter. Masques : A
(« porte », le passé n'est visible qu'à travers ses tokens « Choix » et
ses actions), B (« actions », le témoin), C (« libre », publié sans
seuil). Dix graines.

**Le pilote** (publié). Quatre façons d'apprendre, réglées sur **le témoin
B seul**, dans 256 mondes à part, graines 100 à 102 ; A n'a jamais été
appris pendant le pilote. Survie moyenne de B : P1 (REINFORCE long) 0,194 ;
**P2 (acteur-critique, γ = 0,9) 0,688** ; P3 (γ = 0,97) 0,637 ; P4 (façon
PPO) 0,681. Selon la règle fixée d'avance, P2 est retenu (seuil 0,65).
Le code du pilote normalise l'avantage dans le lot (P2 à P4), ce que le
plan ne précisait pas ; le protocole du test 31 le fixe.

| Moyenne sur 10 graines | A, porte | B, actions | C, libre |
|---|---|---|---|
| Survie (256 mondes du test 22, action la plus probable) | **0,813** | 0,705 | 0,701 |
| Part des décisions où il prend l'action de la règle « besoins » | 0,837 | 0,805 | 0,778 |
| Survie pendant l'apprentissage (100 premières mises à jour → 1 000 → 2 000 → 3 000 → 4 000) | 0,05 → 0,47 → 0,62 → 0,71 → 0,76 | 0,05 → 0,50 → 0,60 → 0,63 → 0,67 | 0,04 → 0,42 → 0,52 → 0,59 → 0,66 |

Repères sans modèle, sur les mêmes mondes : règle « événement » 0,723 ;
règle « besoins » 0,906.

| | Critère | Verdict |
|---|---|---|
| **SURV1** | moyenne de (survie de A − survie de B, même graine) ≥ **0,08**, borne basse > 0 | **passe** : +0,107 [0,084 ; 0,131] ; A au-dessus de B pour 10 graines sur 10 |
| **SURV2** | moyenne de (survie de A − 0,723) ≥ **0,05**, borne basse > 0 | **passe** : +0,090 [0,058 ; 0,122] |
| | Validité : masques (écart ≤ 1e-6, graine 0) ; **survie moyenne de B ≥ 0,60** ; 10 graines | **valide** : écarts 0 et 0 ; B à 0,705 ; 10 graines |

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de
liberté).

**Critère global (SURV1 et SURV2) : satisfait.**

**Lecture fixée d'avance** (SURV1 et SURV2 passent) : « Sans aucune cible,
sans pré-entraînement, seulement en apprenant à survivre, un petit
transformeur qui ne voit son passé qu'à travers ses propres états survit
mieux qu'un témoin qui ne voit que ses actions, et mieux que ce que permet
l'événement du tour. Il a appris de lui-même à porter, dans ses états,
quelque chose de ses besoins cachés. Rien ne lui a nommé ces besoins. Ce
que ces états portent n'est pas mesuré ici. »

**Nuance (après lecture, exploration non pré-enregistrée, vérifiée ; voir
aussi la précision du test 29).** Avec deux couches, ce que les tours
suivants peuvent lire des tokens portés d'un tour (leur sortie de la
couche 0, seule lue par la couche 1) ne dépend que de **l'événement de ce
tour** et des actions jusqu'à ce tour : à actions inchangées, changer un
événement plus ancien ne le change pas du tout (écart 0, 64 cas, graine 0 ;
`research/tiny_two_layer_check.py`). Dans une vraie vie, un événement plus
ancien peut encore l'atteindre à travers les actions qu'il a fait choisir,
un canal que le témoin B a aussi. Ce que A porte dans ses tokens est donc
surtout **l'événement de chaque tour**, qu'il relit à chaque décision pour
en recomposer ses besoins ; ce n'est pas un besoin porté comme un état qui
dure. « Quelque chose de ses besoins cachés » veut dire ici : de quoi les
retrouver. Avec deux couches, A dispose donc de la même information que C
(tous les événements et toutes les actions passés) ; seul le chemin
diffère : chaque événement passé doit être réécrit dans les tokens portés
de son tour. SURV1 mesure surtout ce que rapporte de voir ses événements
passés sous cette forme, que B ne voit pas.

**Publié sans seuil : graine par graine (survie).**

| Graine | A, porte | B, actions | C, libre | A − B |
|---|---|---|---|---|
| 0 | 0,762 | 0,707 | 0,746 | +0,055 |
| 1 | 0,828 | 0,691 | 0,691 | +0,137 |
| 2 | 0,824 | 0,734 | 0,680 | +0,090 |
| 3 | 0,863 | 0,742 | 0,758 | +0,121 |
| 4 | 0,801 | 0,676 | 0,641 | +0,125 |
| 5 | 0,832 | 0,730 | 0,801 | +0,102 |
| 6 | 0,719 | 0,652 | 0,648 | +0,066 |
| 7 | 0,805 | 0,711 | 0,648 | +0,094 |
| 8 | 0,867 | 0,742 | 0,711 | +0,125 |
| 9 | 0,824 | 0,664 | 0,684 | +0,160 |

- **A dépasse B pour chaque graine** (+0,055 à +0,160). Une graine de A
  (la 6, 0,719) reste juste sous la règle « événement ».
- **C (« libre »), qui voit tout son passé, survit moins bien que A** :
  0,701 contre 0,813, A au-dessus de C pour 10 graines sur 10 (écart moyen
  +0,112 [0,073 ; 0,150] ; publié sans seuil, observation après lecture).
  Avec cet apprentissage et à ce budget, ne voir son passé qu'à travers
  ses propres tokens a aidé à apprendre, à information égale. Les courbes
  montent encore à 4 000 mises à jour pour les trois masques (de 3 000 à
  4 000 : A +0,05, B +0,04, C +0,07) ; un apprentissage plus long pourrait
  réduire l'écart, surtout avec C. Explication possible, non testée : moins
  d'entrées à trier. Au test 29 (cible directe, le soulagement), A et C
  survivaient autant (0,884).
- B survit à peine moins bien que la règle « événement » (0,705 contre
  0,723) : il apprend à peu près ce que permet le tour seul.

**Écarts d'exécution, déclarés.**
- **La machine a redémarré** le matin du 5 octobre, pendant les premiers
  apprentissages ; rien n'avait été écrit. Le code a été modifié pour
  sauvegarder modèle, optimiseur et courbe tous les 250 pas et reprendre
  exactement (`227a76f` ; vérifié : reprise identique, écart des poids 0).
  Tout a été relancé de zéro.
- **Le conteneur a redémarré** à 14 h 07 UTC. Tout a repris des
  sauvegardes. A et B de la graine 0, remesurés depuis leur sauvegarde
  finale pour écrire les contrôles des masques, ont redonné des fichiers
  identiques.
- Les résultats bruts ont été publiés et lus graine par graine dès leur
  écriture (lecture anticipée déclarée ; aucun seuil n'a changé).
- D'autres calculs (tests 26, 28, 32) tournaient en même temps sur la même
  machine ; ils ne touchent pas ces résultats (calcul à un fil,
  déterministe).
- Une exploration non pré-enregistrée (`research/tiny_two_layer_check.py`)
  a été faite sur A et C de la graine 0 et publiée (`ae69a1b`, 18 h 41)
  avant la fin des apprentissages (B et C des graines 4 et 7), après
  lecture des dix résultats de A et de huit de B et de C ; elle ne touche
  pas ces résultats. Corrigée après la relecture (voir la précision du
  test 29).

### Ce que cela dit

- **Sans aucune cible, l'apprentissage par la seule survie suffit** : un
  petit transformeur qui ne voit son passé qu'à travers ses propres tokens
  apprend à s'en servir, et survit mieux qu'un témoin qui ne voit que ses
  actions (+0,107, 10 graines sur 10), et mieux que la règle
  « événement ».
- **Mais ce qu'il garde est une mémoire des événements, pas un besoin qui
  dure.** Avec deux couches, ce que les tours suivants lisent des tokens
  d'un tour ne dépend que de son événement et des actions jusque-là ; le
  besoin est recomposé à chaque décision. Le test 32
  (« la boucle ») pose la question d'un état qui ne passe que d'un tour au
  suivant.
- **Il ne dit pas ce que portent les états** au-delà de cette limite
  structurelle.

**Ce que le résultat ne dit pas.**
- Rien sur un ressenti.
- La récompense (la survie) vient du monde, qui connaît les besoins ; elle
  ne nomme ni le besoin ni l'action à prendre.
- « Sans aucune cible » veut dire : sans cible qui nomme le besoin ou
  l'action. L'acteur-critique apprend toutefois une prédiction : sa tête de
  valeur, au token « Choix », prédit le retour de survie escompté, et cet
  apprentissage passe par le même réseau (chez A, B et C).
- Le réglage a été choisi sur B seulement, dans d'autres mondes ; un
  réglage meilleur pour B ou pour C pourrait exister.
- Ces modèles sont petits, le monde aussi. Ce qui est déjà connu : des
  agents apprennent une mémoire par la seule récompense (agents récurrents
  en apprentissage par renforcement), et des transformeurs à mémoire passent
  leur passé par des tokens ou des états qu'ils écrivent eux-mêmes
  (Transformer-XL, Recurrent Memory Transformer). Aucune recherche
  bibliographique n'a été faite pour ce test ; l'apport revendiqué se
  limite à ce montage précis, à son témoin et au pré-enregistrement.

## Test 32 pré-enregistré : la boucle, un état qui ne passe que d'un tour au suivant (échoue)

Protocole `docs/TINY_LOOP_PROTOCOL.md` (commit `0488b5e`), écrit avant le
code et toute exécution. Code commis avant toute mesure (`973c04c`), essayé
sur 10 mises à jour de la graine 99, hors protocole (rien lu que le temps).
Calcul torch sur le processeur local, le 5 octobre 2026 (deux processus à
un fil, en priorité basse). Artefacts : `artifacts/tiny-loop`. Verdicts
recalculés en local (identiques) ; la CI les vérifie à chaque envoi.

**Ce qu'on teste.** Aux tests 29 et 31, avec deux couches, chaque tour
n'écrit dans ses tokens que son propre événement : le besoin est recomposé
à chaque décision, ce n'est pas une récurrence. Ici, un petit transformeur
appris de zéro (2 couches, dimension 64) **ne voit jamais les tours
passés** : à chaque tour, il lit un état s_t (un vecteur), la dernière
action, l'événement et « Choix » ; la sortie de « Choix » donne le choix,
la valeur et **l'état suivant**. Seul cet état passe d'un tour au suivant.
Il n'apprend que de sa survie (acteur-critique P2 du test 31, le gradient
traversant l'état sur toute la vie). Témoin : le même réseau, état coupé à
chaque tour. Dix graines.

| Moyenne sur 10 graines | A, boucle | B, coupé |
|---|---|---|
| Survie (256 mondes du test 22, action la plus probable) | 0,721 | 0,733 |
| Part des décisions où il prend l'action de la règle « besoins » | 0,826 | 0,831 |
| Survie pendant l'apprentissage (100 premières mises à jour → 1 000 → 2 000 → 3 000 → 4 000) | 0,07 → 0,53 → 0,60 → 0,63 → 0,67 | 0,07 → 0,64 → 0,69 → 0,69 → 0,69 |
| Ce qu'un décodeur linéaire lit de E et N dans l'état transmis (pour B, la sortie de « Choix », non transmise ; part de variance, validation croisée) | 0,68 et 0,69 | 0,42 et 0,46 |

Repères sans modèle, sur les mêmes mondes : règle « événement » 0,723 ;
règle « besoins » 0,906.

| | Critère | Valeur | Verdict |
|---|---|---|---|
| **LOOP1** | survie de A − survie de B ≥ **0,08** (borne basse > 0) | −0,012 [−0,068 ; +0,043] | **échoue** |
| **LOOP2** | survie de A − 0,723 ≥ **0,05** (borne basse > 0) | −0,002 [−0,057 ; +0,053] | **échoue** |
| **LOOP3** | greffes (a) : moyenne des m (ΔP(R) × e) ≥ **0,10**, borne basse > 0, et m > 0,05 pour 8 graines sur 10 | +0,049 [0,003 ; 0,095] ; 3 graines sur 10 | **échoue** |
| **LOOP4** | \|ΔP(R)\| des greffes (b) ≤ la moitié de celui des greffes (a), différence à borne basse > 0 | 0,019 contre 0,064 ; +0,045 [0,003 ; 0,088] | passe |
| | Validité : B ≥ 0,60 ; coupure ≤ 1e-6 ; greffe de soi ≤ 1e-6 ; ≥ 100 greffes (a) et (b) ; 10 graines | B 0,733 ; 0 ; 0 ; 391 et 200 ; oui | **valide** |

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de
liberté).

**Critère global : non satisfait** (LOOP1, LOOP2 et LOOP3 échouent).

**Lecture fixée d'avance** (LOOP1 échoue) : « À cette taille et avec cet
apprentissage, l'état transmis ne le fait pas survivre assez mieux que le
témoin coupé. »

**Publié sans seuil : graine par graine.**

| Graine | A : survie | B : survie | A − B | m (greffes (a)) | \|ΔP\| (b) | Décodeur E / N (A) |
|---|---|---|---|---|---|---|
| 0 | 0,746 | 0,730 | +0,016 | 0,075 | 0,039 | 0,72 / 0,63 |
| 1 | 0,727 | 0,730 | −0,004 | 0,046 | 0,024 | 0,74 / 0,71 |
| 2 | 0,730 | 0,734 | −0,004 | 0,019 | 0,008 | 0,57 / 0,60 |
| 3 | 0,777 | 0,730 | +0,047 | **0,214** | 0,061 | 0,80 / 0,78 |
| 4 | 0,730 | 0,734 | −0,004 | 0,001 | 0,001 | 0,56 / 0,71 |
| 5 | 0,734 | 0,734 | 0,000 | 0,002 | 0,001 | 0,67 / 0,62 |
| 6 | 0,789 | 0,730 | +0,059 | 0,071 | 0,023 | 0,75 / 0,72 |
| 7 | 0,727 | 0,734 | −0,008 | 0,023 | 0,010 | 0,64 / 0,74 |
| 8 | 0,512 | 0,734 | −0,223 | −0,001 | 0,001 | 0,64 / 0,65 |
| 9 | 0,734 | 0,734 | 0,000 | 0,037 | 0,022 | 0,71 / 0,72 |

- **L'état transmis porte une information sur les besoins au-delà du
  tour** : un décodeur linéaire y lit E et N mieux que dans la sortie du
  témoin coupé (0,68 contre 0,42 ; 0,69 contre 0,46), pour chaque graine.
  Le témoin ne voit que l'événement et la dernière action (un décodeur sur
  ces deux seules entrées donne aussi 0,42 et 0,46). Contrôles ajoutés
  après la relecture (exploration non pré-enregistrée,
  `research/tiny_loop_probe_controls.py`,
  `artifacts/tiny-loop/exploration-probe.json`) : le même réseau en boucle
  **non appris** en lit déjà 0,54 et 0,57 ; l'apprentissage n'ajoute
  qu'environ 0,14 et 0,11 (rien pour la graine 2). 0,68 et 0,69, c'est à
  peu près ce qu'un décodeur lit des deux ou trois derniers tours (0,66 et
  0,69 avec deux tours de plus ; 0,71 et 0,74 avec trois) : **une mémoire
  courte**, pas les besoins entiers.
- **Mais l'agent s'en sert peu pour choisir.** Greffé, l'état ne déplace la
  probabilité de choisir R que de 0,05 en moyenne dans le sens des besoins
  d'origine (0,48 au test 23, chez le modèle de langage appris d'un
  professeur) : environ 5 % d'un changement complet. Trois graines
  seulement (0, 3, 6) dépassent le seuil par graine de 0,05 (0,075 ;
  0,214 ; 0,071) ; les graines 3 et 6 sont aussi les seules à survivre
  nettement mieux que le témoin (+0,047 et +0,059 ; 20 vies contre 8 et 21
  contre 6 sur les mêmes mondes), la graine 0 à peine (+0,016 ; 13 contre
  9). Lien observé après lecture, sur trois graines.
- **Quand il s'en sert, c'est le besoin plutôt que l'histoire** : LOOP4
  passe, mais sur des effets petits.
- **La graine 8 s'est effondrée** (0,512) : son apprentissage est resté
  bloqué vers 0,45–0,47 dès 600 mises à jour.
- Le témoin coupé, qui ne voit que l'événement et la dernière action (18
  cas possibles), apprend la même table pour 10 graines sur 10 à partir du
  deuxième tour ; seul le choix du premier tour diffère : survie 0,730 ou
  0,734 (vérifié sur les sauvegardes, à la relecture).
- **Selon g** : effet aligné +0,071 à un tour, +0,024 à deux tours ;
  \|ΔP\| des greffes (b) : 0,029 et 0,010 ; \|ΔP\| des greffes (a0) :
  0,032.
- Vérifié à la relecture (exploration) : LOOP4 ne tient pas au partage
  d'événement. Les donneuses (b) partagent l'événement du tour j dans 57 %
  des cas, contre 26 % pour (a) ; mais à événement égal, \|ΔP\| vaut 0,057
  pour (a) contre 0,018 pour (b) (différence +0,039 [0,0001 ; 0,079]).

**Écarts d'exécution, déclarés.**
- Le tirage n'a donné que 391 greffes (a) (le protocole s'arrêtait à 400 ;
  il en faut 100). Compté et annoncé dans le commit du code, avant toute
  mesure.
- Les résultats bruts ont été publiés et lus graine par graine (survie et
  décodeur) dès leur écriture ; les mesures de greffe n'ont été résumées
  qu'à la fin. Aucun seuil n'a changé.
- Le témoin de la graine 4 a été publié dans le même commit que celui de la
  graine 9 (`8ab68bf`), dont le message ne nomme que la graine 9.
- D'autres calculs (tests 26, 28, 31) tournaient en même temps ; calcul
  déterministe à un fil.

### Ce que cela dit

- **Une vraie boucle, apprise par la seule survie, garde dans son état un
  peu plus que le tour présent** (une mémoire courte, en partie présente
  sans apprentissage), **mais s'en sert peu pour choisir**, à ce budget et
  avec cet apprentissage (la greffe le montre).
- **Une récurrence au sens strict qui serve au choix n'est donc pas obtenue
  de façon fiable** : l'état transmis porte, pour les 10 graines, de
  l'information au-delà du tour, mais trois graines sur dix seulement s'en
  servent un peu pour choisir ; les autres choisissent presque comme le
  témoin coupé (survie à 0,008 près), sauf la graine 8, effondrée.
- **Le contraste avec le test 31** (même apprentissage, même monde) : en
  relisant ses tokens passés, le petit transformeur survit à 0,813 ; en
  devant tout faire passer par un état transmis, il reste à 0,721 (les
  témoins diffèrent aussi : 0,705 au test 31, 0,733 ici).

**Ce que le résultat ne dit pas.**
- Rien sur un ressenti.
- Il ne dit pas qu'une boucle ne peut pas apprendre à s'en servir : un
  autre apprentissage (plus long, ou un état mieux protégé d'un tour à
  l'autre) pourrait réussir. Le réglage a été choisi au pilote pour un
  autre réseau.
- Le décodeur est publié sans seuil ; il lit E et N après l'événement du
  tour, et le témoin en lit déjà une part sans aucune mémoire ; un réseau
  en boucle non appris en lit déjà 0,54 et 0,57.

## Pilote avant le test 33 : faire servir l'état transmis

Plan `docs/TINY_LOOP_PILOT_PLAN.md` (commit `3dfb0ab`), écrit après le test
32 et avant tout code du pilote ; code `e90c4c0`, commis avant toute
exécution. Calcul torch sur le processeur local, du 5 au 6 octobre 2026
(quatre processus à un fil). Artefacts : `artifacts/tiny-loop-pilot`.

**Ce qu'on règle.** L'apprentissage de **la boucle** du test 32, sur les
graines 100 à 102 et 256 mondes à part (flux `[270926, 73]`) ; le témoin
coupé est appris une fois par graine avec le réglage du test 32. Quatre
réglages : L1 (12 000 mises à jour), L2 (taux 1e-3), L3 (**état à porte** :
s_{t+1} = (1 − z) · s_t + z · n_t, z appris), L4 (porte et 12 000 mises à
jour).

| Survie sur les mondes du pilote | Graine 100 | Graine 101 | Graine 102 | Gain moyen sur le témoin |
|---|---|---|---|---|
| Témoin coupé | 0,688 | 0,688 | 0,672 | — |
| L1 (12 000 mises à jour) | 0,695 | 0,785 | 0,773 | +0,069 |
| L2 (taux 1e-3) | 0,672 | 0,680 | 0,684 | −0,004 |
| L3 (état à porte) | 0,766 | 0,809 | 0,738 | +0,089 |
| **L4 (porte, 12 000 mises à jour)** | **0,848** | **0,816** | **0,867** | **+0,161** |

**Choix selon la règle fixée d'avance** (plus grand gain, au moins 0,04,
sans graine sous son témoin − 0,05) : **L4**.

- L3 est exactement le début de L4 (mêmes graines, mêmes tirages, 4 000
  mises à jour au lieu de 12 000).
- Les courbes de L4 montent jusque vers 8 000 à 9 000 mises à jour, puis
  se stabilisent vers 0,82 à 0,84 (survie pendant l'apprentissage, actions
  tirées), avec des effondrements passagers (L4 : 0,30 vers 4 200 mises à
  jour pour la graine 101, 0,10 vers 4 400 pour la 102 ; L3 : jusqu'à
  0,32).

**Écarts, déclarés.**
- Le résultat L3 de la graine 100 a été publié dans le même commit que
  L1 de la graine 100 (`23cc102`), dont le message ne nomme que L1.
- Les résultats ont été lus et commentés au fil de l'eau ; la règle de
  choix n'a pas changé.
- Après la relecture du test 32, une note datée a été ajoutée au plan (la
  boucle du test 32 garde une mémoire courte plutôt que « ses besoins ») ;
  ni le plan ni la règle n'ont changé.

**Ce que cela dit.** Avec un état à porte et un apprentissage plus long,
la boucle survit nettement mieux que le témoin coupé sur les trois
graines du pilote. Ce n'est pas un résultat : ces graines et ces mondes ont
servi à choisir. La question est reposée au test 33, sur des graines
neuves, avec la greffe.

## Test 33 pré-enregistré : la boucle qui s'en sert (échoue sur LOOP3 : 7 graines sur 10, il en fallait 8)

Protocole `docs/TINY_LOOP_GATE_PROTOCOL.md` (commit `e232911`), écrit après
le pilote et avant tout code et tout apprentissage sur ses graines. Code
commis avant tout apprentissage (`f328ba6`), essayé sur 5 mises à jour de
la graine 99, hors protocole. Calcul torch sur le processeur local, le 6
octobre 2026 (cinq processus à un fil). Artefacts : `artifacts/tiny-loop-gate`.
Verdicts recalculés en local (identiques) ; la CI les vérifie à chaque
envoi. Vérification bibliographique faite pendant le calcul :
`docs/LITERATURE_CHECK_LOOP_2026-10-06.md`.

**Ce qu'on teste.** La question du test 32, avec le réglage L4 du pilote
pour la boucle **et** le témoin coupé : un **état à porte**
(s_{t+1} = (1 − z) · s_t + z · n_t, z appris) et 12 000 mises à jour. Un
petit transformeur appris de zéro ne voit jamais les tours passés ; seul cet
état passe d'un tour au suivant. Il n'apprend que de sa survie ; ses besoins
ne lui sont jamais donnés. Dix graines neuves (10 à 19).

| Moyenne sur 10 graines | A, boucle à porte | B, coupé |
|---|---|---|
| Survie (256 mondes du test 22, action la plus probable) | **0,846** | 0,734 |
| Part des décisions où il prend l'action de la règle « besoins » | 0,872 | 0,832 |
| Survie pendant l'apprentissage (100 premières mises à jour → 2 000 → 4 000 → 6 000 → 8 000 → 10 000 → 12 000) | 0,08 → 0,69 → 0,70 → 0,67 → 0,74 → 0,78 → 0,80 | 0,08 → 0,69 → 0,69 → 0,69 → 0,69 → 0,69 → 0,69 |
| Décodeur linéaire de E et N dans l'état transmis (pour B, la sortie de « Choix », non transmise) | 0,82 et 0,83 | 0,42 et 0,46 |
| Le même décodeur sur le réseau **non appris** de la même graine | 0,75 et 0,77 | — |

Repères sans modèle : règle « événement » 0,723 ; règle « besoins » 0,906.

| | Critère | Valeur | Verdict |
|---|---|---|---|
| **LOOP1** | survie de A − survie de B ≥ **0,08** (borne basse > 0) | +0,113 [0,075 ; 0,150] ; 10 graines sur 10 | **passe** |
| **LOOP2** | survie de A − 0,723 ≥ **0,05** (borne basse > 0) | +0,124 [0,087 ; 0,161] | **passe** |
| **LOOP3** | greffes (a) : moyenne des m (ΔP(R) × e) ≥ **0,10**, borne basse > 0, **et m > 0,05 pour 8 graines sur 10** | +0,321 [0,161 ; 0,481] ; **7 graines sur 10** | **échoue** (sur le nombre de graines) |
| **LOOP4** | \|ΔP(R)\| des greffes (b) ≤ la moitié de celui des greffes (a), différence à borne basse > 0 | 0,051 contre 0,355 ; +0,304 [0,162 ; 0,445] | **passe** |
| | Validité : B ≥ 0,60 ; coupure ≤ 1e-6 ; greffe de soi ≤ 1e-6 ; ≥ 100 greffes (a) et (b) ; 10 graines | B 0,734 ; 0 ; 0 ; 391 et 200 ; oui | **valide** |

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de
liberté).

**Critère global : non satisfait.** LOOP1, LOOP2 et LOOP4 passent ; LOOP3
échoue sur une de ses trois conditions : sa moyenne (+0,321) et sa borne
basse (+0,161) passent largement, mais trois graines (11, 14, 15) restent
sous 0,05, alors qu'il en fallait au plus deux.

**Lecture fixée d'avance** (LOOP1 et LOOP2 passent, LOOP3 échoue) :
« L'état sert à survivre, mais la greffe ne montre pas qu'il porte surtout
les besoins. »

**Publié sans seuil : graine par graine.**

| Graine | A : survie | B : survie | A − B | m (greffes (a)) | \|ΔP\| (a) | \|ΔP\| (b) | m à événement égal | Décodeur E / N (A) | Porte z moyenne |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0,867 | 0,734 | +0,133 | **0,529** | 0,544 | 0,064 | 0,516 | 0,88 / 0,90 | 0,26 |
| 11 | 0,805 | 0,734 | +0,070 | 0,040 | 0,050 | 0,019 | 0,051 | 0,78 / 0,81 | 0,20 |
| 12 | 0,883 | 0,734 | +0,148 | **0,416** | 0,451 | 0,050 | 0,426 | 0,87 / 0,88 | 0,26 |
| 13 | 0,898 | 0,734 | +0,164 | **0,453** | 0,491 | 0,072 | 0,489 | 0,89 / 0,88 | 0,24 |
| 14 | 0,797 | 0,734 | +0,063 | −0,004 | 0,082 | 0,038 | 0,000 | 0,57 / 0,58 | 0,27 |
| 15 | 0,770 | 0,734 | +0,035 | −0,002 | 0,047 | 0,027 | −0,010 | 0,72 / 0,77 | 0,25 |
| 16 | 0,891 | 0,730 | +0,160 | **0,377** | 0,413 | 0,057 | 0,413 | 0,86 / 0,87 | 0,24 |
| 17 | 0,781 | 0,734 | +0,047 | **0,344** | 0,371 | 0,069 | 0,349 | 0,88 / 0,85 | 0,29 |
| 18 | 0,898 | 0,734 | +0,164 | **0,498** | 0,520 | 0,059 | 0,555 | 0,88 / 0,89 | 0,28 |
| 19 | 0,875 | 0,734 | +0,141 | **0,558** | 0,580 | 0,055 | 0,568 | 0,92 / 0,85 | 0,30 |

- **Après lecture (non pré-enregistré) : sept graines ont m entre 0,34 et
  0,56, les trois autres à peu près nul (−0,004 ; −0,002 ; 0,040).** Dans
  ces sept graines, greffé dans une autre vie, l'état fait pencher le choix
  de 0,34 à 0,56 dans le sens des besoins de la vie d'origine (au test 23,
  le modèle de langage appris d'un professeur : 0,48 ; moyenne sur les 10
  graines : +0,32). Venu d'une vie aux mêmes besoins, il ne bouge presque
  rien (0,05 à 0,07).
- **Ce n'est pas l'événement du tour** (contrairement au test 28) : à
  événement égal au tour j, l'effet aligné reste de +0,336 [0,169 ; 0,502]
  en moyenne sur les 10 graines (\|ΔP\| 0,362 pour (a), 0,039 pour (b)).
- **Ce n'est pas l'histoire** : sans les 38 greffes (b) qui n'ont pas
  d'autre histoire, \|ΔP\| (b) vaut 0,063 contre 0,355 ; LOOP4 tient
  (différence +0,292 [0,153 ; 0,431]). Avec les greffes (a0) comme
  comparaison, de même : 0,081 contre 0,355.
- **Trois graines (11, 14, 15) survivent un peu mieux que le témoin**
  (+0,070, +0,063, +0,035 ; sur les mêmes mondes, 22 vies contre 4, 27
  contre 11, 18 contre 9 : net pour 11 et 14, pas pour 15) **sans que la
  greffe montre que leur état porte leurs besoins** (m de −0,004 à 0,040 ;
  \|ΔP\| (a) proche de (a0)). Chez les graines 14 et 15, le décodeur appris
  est au niveau du réseau non appris, ou en dessous (0,57 / 0,58 contre
  0,74 / 0,75 ; 0,72 / 0,77 contre 0,75 / 0,78). À l'inverse, la graine
  17, où la greffe est nette (0,344), ne survit guère mieux que le témoin
  (+0,047 ; 31 vies contre 19). Pourquoi n'est pas mesuré.
- **Le décodeur seul ne suffit pas** : le même réseau à porte **non
  appris** laisse déjà lire 0,75 et 0,77 (probablement parce que la porte
  intègre les événements, même sans apprentissage) ; l'apprentissage
  n'ajoute qu'environ 0,07 et 0,05 en moyenne (+0,13 et +0,10 dans les
  sept graines où la greffe agit ; il fait baisser le décodeur dans les
  graines 14 et 15). C'est la greffe qui montre que l'état sert au choix.
- **Selon g** : effet aligné +0,365 à un tour, +0,271 à deux tours ;
  \|ΔP\| des greffes (b) : 0,050 et 0,052 ; (a0) : 0,081.
- **La porte** laisse entrer en moyenne 26 % du nouvel état à chaque tour
  (z moyen sur les décisions et les 64 dimensions, 0,20 à 0,30 selon la
  graine) : l'état dure plusieurs tours.
- **L'apprentissage de la boucle est instable** : quatre graines
  s'effondrent un moment puis se rétablissent (0,00 vers 5 400 mises à
  jour pour la graine 12, 0,00 vers 3 500 pour la 19, 0,09 vers 7 200 pour
  la 17, 0,26 vers 6 800 pour la 13) ; la graine 14 retombe encore à 0,56
  vers 11 800. Le creux de la courbe moyenne à 6 000 (0,67) en vient.
- Le témoin coupé apprend, comme au test 32, presque la même table
  (0,730 ou 0,734).

**Écarts d'exécution, déclarés.**
- Les résultats bruts ont été publiés et lus graine par graine (survie,
  décodeur) dès leur écriture ; les greffes n'ont été résumées qu'après les
  20 apprentissages (verdicts écrits à 5 h 16 UTC, publication à 5 h 17).
  Aucun seuil n'a changé.
- Trois fichiers ont été publiés dans le commit d'une autre graine, dont le
  message ne les nomme pas : la boucle de la graine 10 (`8e9a646`), celle de
  la graine 18 (`858b5d5`), le témoin de la graine 11 (`1893552`). Le
  commit `5246e8f` (« graines suivantes ») ne contient que la boucle de la
  graine 19, sans la nommer.
- La session a redémarré vers 3 h 05 UTC ; les calculs n'ont pas été
  touchés (ils ont continué sans interruption).
- La vérification bibliographique a été faite pendant le calcul, sans lire
  les greffes.

### Ce que cela dit

- **Seulement en apprenant à survivre, une vraie boucle à porte survit
  nettement mieux qu'un témoin sans mémoire** (0,846 contre 0,734, 10
  graines sur 10), et plus que la règle « événement ». La porte et un
  apprentissage plus long ont changé l'issue du test 32.
- **Après lecture, dans sept graines sur dix, l'état transmis greffé fait
  choisir selon les besoins de la vie d'origine**, aussi à événement égal
  et quand la donneuse (b) a d'autres événements récents ; dans les trois
  autres, rien. Ce partage est défini par la statistique même de LOOP3 : il
  décrit, il ne confirme rien.
- **Le critère pré-enregistré demandait huit graines sur dix : il n'est pas
  atteint, et la lecture fixée d'avance s'applique telle quelle** :
  « L'état sert à survivre, mais la greffe ne montre pas qu'il porte
  surtout les besoins. » Trois graines survivent un peu mieux que le
  témoin (nettement pour 11 et 14) sans que la greffe le montre. Qu'une
  récurrence au sens strict, apprise par la seule survie, porte de façon
  fiable le niveau de ses besoins reste à montrer sur des graines neuves.

**Ce que le résultat ne dit pas.**
- Rien sur un ressenti.
- Le réglage a été choisi en regardant la boucle (au pilote, sur d'autres
  graines et d'autres mondes) ; le témoin a reçu le même.
- La récompense (la survie) vient du monde, qui connaît les besoins ; elle
  ne nomme ni le besoin ni l'action. La tête de valeur du critique prédit
  le retour de survie.
- La greffe lit des vies écrites par une règle, pas des vies vécues par
  l'agent ; elle ne sépare pas le niveau des besoins de ce qui en commande
  le choix.
- **Ce qui est déjà connu** (vérification bibliographique) : des agents
  récurrents appris par renforcement (Hayes 2026) ou par stratégie
  évolutionnaire (Chaturvedi et al., 2025, révisé en 2026) suivent un
  niveau interne dans leur état, et forcer ou perturber cet état change
  leur comportement, mais ce niveau leur est **donné en entrée**. L'apport
  que visait ce test est étroit (besoins jamais observés, greffe entre deux
  vies contre un témoin sans mémoire, séparation du besoin et de
  l'histoire, pré-enregistrement) ; LOOP3 échouant, il n'est montré
  qu'après lecture, dans sept graines sur dix.
- Ces modèles sont petits, le monde aussi.

## Second pilote avant le test 34 : rendre la boucle fiable

Plan `docs/TINY_LOOP_PILOT2_PLAN.md` (commit `69d9b8f`), écrit après le
test 33 et avant tout code du pilote ; code `ed0eb6f`, commis avant toute
exécution du pilote ; un essai hors protocole (graine 999, 250 et 500 mises
à jour) a commencé une minute avant ce commit. Calcul torch sur le processeur local, le 6 octobre 2026 (quatre
processus à un fil). Artefacts : `artifacts/tiny-loop-pilot2` (choix
recalculé en local, identique).

**Ce qu'on règle.** L'apprentissage de la boucle à porte du test 33, sur les
graines 100 à 103, cette fois **sur la greffe elle-même** : pour chaque
réglage et chaque graine, la survie sur 256 mondes à part (flux
`[270926, 73]`) et **m**, l'effet aligné des greffes (a), sur 128 vies de
greffe à part (flux `[270926, 79]` ; tirage `[270926, 80, 0]` : 400 (a),
200 (a0), 200 (b)). Le témoin coupé est appris une fois par graine avec R1.
Quatre réglages : R1 (le test 33 tel quel, 12 000 mises à jour), R2 (R1,
norme du gradient bornée à 1), R3 (R1, 24 000 mises à jour), R4 (R2,
24 000 mises à jour).

| Survie sur les mondes du pilote (m) | Graine 100 | Graine 101 | Graine 102 | Graine 103 | Gain moyen sur le témoin | m moyen | Fiable |
|---|---|---|---|---|---|---|---|
| Témoin coupé (R1) | 0,680 | 0,695 | 0,672 | 0,680 | — | — | — |
| R1 (test 33) | 0,848 (+0,408) | 0,816 (+0,466) | 0,867 (+0,463) | 0,852 (+0,392) | +0,164 | +0,432 | oui |
| R2 (gradient borné) | 0,871 (+0,428) | 0,879 (+0,427) | 0,703 (+0,500) | 0,863 (+0,559) | +0,147 | +0,478 | **non** (graine 102 : gain +0,031) |
| **R3 (24 000 mises à jour)** | **0,887 (+0,592)** | **0,887 (+0,455)** | **0,879 (+0,565)** | **0,887 (+0,384)** | **+0,203** | **+0,499** | **oui** |
| R4 (les deux) | 0,871 (+0,485) | 0,883 (+0,455) | 0,859 (+0,487) | 0,879 (+0,460) | +0,191 | +0,472 | oui |

**Choix selon la règle fixée d'avance** (fiable : pour les quatre graines,
m > 0,05 et un gain sur le témoin d'au moins 0,04 ; parmi les réglages
fiables, le plus grand m moyen, le plus simple à 0,01 près) : **R3**, le
réglage du test 33 appris deux fois plus longtemps. R4 est 0,027 en
dessous, R1 0,067 en dessous.

- R1 sur les graines 100 à 102 a été repris des points gardés par le
  premier pilote : ses survies redonnent exactement celles publiées pour L4
  (0,848, 0,816, 0,867 ; ce sont les mêmes réseaux, mesurés à nouveau : cela
  vérifie la mesure, pas l'apprentissage). Ces trois graines sont aussi
  celles sur lesquelles L4 avait été choisi au premier pilote. R3 et R4 prolongent R1 et R2 depuis 12 000 mises
  à jour.
- **Ce pilote départage peu.** Avec R1, le réglage du test 33, les quatre
  graines du pilote portent déjà les besoins (m de +0,39 à +0,47), alors
  qu'au test 33 trois graines sur dix ne les portaient pas. Si trois
  graines sur dix échouent, quatre réussites de suite arrivent environ une
  fois sur quatre par hasard. Le pilote ne montre donc pas que R3 corrige
  l'échec du test 33. Il montre que, sur ces graines, apprendre plus
  longtemps augmente m en moyenne (+0,499 contre +0,432 : nettement sur
  deux graines, 0,408 → 0,592 et 0,463 → 0,565 ; les deux autres baissent
  un peu, 0,466 → 0,455 et 0,392 → 0,384) et la survie sur les quatre (gain
  +0,203 contre +0,164).
- **Borner le gradient n'empêche pas les effondrements.** Le plus bas des
  courbes après 2 000 mises à jour (survie pendant l'apprentissage, moyenne
  sur 100 mises à jour, graines 100 à 103) : R1 0,67 ; 0,30 ; 0,10 ;
  0,67 ; R3 0,67 ; 0,30 ; 0,10 ; 0,44 (après 12 000, R3 redescend à 0,38
  vers 12 300 pour la graine 101 et à 0,44 vers 12 900 pour la 103) ; R2
  et R4 0,68 ; 0,45 ; 0,23 ; 0,64 ; témoin 0,67 à 0,68. En fin d'apprentissage,
  les boucles sont entre 0,80 et 0,86, le témoin vers 0,69.

**Écarts, déclarés.**
- Les résultats de R2 pour les graines 101 et 102 ont été publiés dans le
  commit `c4ca8d9`, dont le message ne nomme que la graine 100 ; celui de
  R3 pour la graine 101 dans le commit `4d6ab31`, dont le message ne nomme
  que R2 graine 103. Les suivants ont été commis un par un.
- Les résultats ont été lus au fil de l'eau ; la règle de choix n'a pas
  changé.

**Ce que cela dit.** Sur ces quatre graines, la boucle à porte apprise par
la seule survie porte ses besoins avec chacun des quatre réglages ; le plus
long (R3) le fait le plus. Ce n'est pas un résultat : ces graines, ces
mondes et ces vies ont servi à choisir. La question est reposée au test 34,
sur dix graines neuves et de nouvelles vies de greffe, avec les critères du
test 33.

## Test 34 pré-enregistré : la boucle fiable (critère global satisfait)

Protocole `docs/TINY_LOOP_RELIABLE_PROTOCOL.md` (commit `c24f777`), écrit
après le second pilote et avant tout apprentissage sur ses graines ; code
commis avant tout apprentissage (`57a862f`), essayé sur 250 et 500 mises à
jour de la graine 998, hors protocole. Calcul torch sur le processeur local,
le 6 octobre 2026, de 10 h 15 à 19 h 07 UTC (quatre processus à un fil).
Artefacts : `artifacts/tiny-loop-reliable`. Verdicts recalculés en local
(identiques) ; la CI les vérifie à chaque envoi.

**Ce qu'on teste.** La question des tests 32 et 33, avec le réglage R3
choisi au second pilote, pour la boucle **et** le témoin coupé : l'état à
porte du test 33 et **24 000 mises à jour** (au lieu de 12 000). Un petit
transformeur appris de zéro ne voit des tours passés que sa dernière action,
comme le témoin ; sinon, seul cet état passe d'un tour au suivant. Il
n'apprend que de sa survie ; ses besoins ne lui sont jamais donnés. Dix
graines neuves (20 à 29) et **128 vies de
greffe neuves** (flux `[270926, 81]` ; tirage `[270926, 82, 0]` : 400 (a),
200 (a0), 200 (b)). Critères du test 33, inchangés.

| Moyenne sur 10 graines | A, boucle à porte | B, coupé |
|---|---|---|
| Survie (256 mondes du test 22, action la plus probable) | **0,896** | 0,733 |
| Part des décisions où il prend l'action de la règle « besoins » | 0,898 | 0,830 |
| Survie pendant l'apprentissage (100 premières mises à jour → 2 000 → 4 000 → 8 000 → 12 000 → 16 000 → 20 000 → 24 000) | 0,08 → 0,68 → 0,73 → 0,79 → 0,80 → 0,82 → 0,84 → 0,84 | 0,09 → 0,69 → 0,69 → 0,69 → 0,69 → 0,69 → 0,69 → 0,69 |
| Décodeur linéaire de E et N dans l'état transmis (pour B, la sortie de « Choix », non transmise) | 0,87 et 0,87 | 0,44 et 0,44 |
| Le même décodeur sur le réseau **non appris** de la même graine | 0,76 et 0,77 | — |

Repères sans modèle : règle « événement » 0,723 ; règle « besoins » 0,906.

| | Critère | Valeur | Verdict |
|---|---|---|---|
| **LOOP1** | survie de A − survie de B ≥ **0,08** (borne basse > 0) | +0,163 [0,146 ; 0,179] ; 10 graines sur 10 | **passe** |
| **LOOP2** | survie de A − 0,723 ≥ **0,05** (borne basse > 0) | +0,173 [0,157 ; 0,189] | **passe** |
| **LOOP3** | greffes (a) : moyenne des m (ΔP(R) × e) ≥ **0,10**, borne basse > 0, **et m > 0,05 pour 8 graines sur 10** | +0,530 [0,510 ; 0,549] ; **10 graines sur 10** (de 0,491 à 0,586) | **passe** |
| **LOOP4** | \|ΔP(R)\| des greffes (b) ≤ la moitié de celui des greffes (a), différence à borne basse > 0 | 0,059 contre 0,534 ; +0,475 [0,452 ; 0,499] | **passe** |
| | Validité : B ≥ 0,60 ; coupure ≤ 1e-6 ; greffe de soi ≤ 1e-6 ; ≥ 100 greffes (a) et (b) ; 10 graines | B 0,733 ; 0 ; 0 ; 400 et 200 ; oui | **valide** |

Intervalles à 95 % sur les 10 graines (loi de Student, 9 degrés de
liberté).

**Critère global : satisfait.**

**Lecture fixée d'avance** (LOOP1 à LOOP4 passent) : « Seulement en
apprenant à survivre, un petit transformeur dont le passé ne passe que par
un état transmis d'un tour au suivant apprend, de façon fiable, à y porter
ses besoins cachés et à s'en servir. Greffé dans une autre vie, cet état
fait choisir selon les besoins de la vie d'origine ; venu d'une vie aux
mêmes besoins mais à l'histoire différente, il change peu le choix. C'est
une récurrence au sens strict, et ce qu'elle porte est surtout le niveau de
ses besoins, que rien ne lui a nommé. »

**Publié sans seuil : graine par graine.**

| Graine | A : survie | B : survie | A − B | m (greffes (a)) | \|ΔP\| (a) | \|ΔP\| (b) | m à événement égal | Décodeur E / N (A) | Porte z moyenne |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 0,906 | 0,734 | +0,172 | **0,586** | 0,590 | 0,050 | 0,533 | 0,87 / 0,88 | 0,24 |
| 21 | 0,914 | 0,730 | +0,184 | **0,528** | 0,532 | 0,055 | 0,445 | 0,87 / 0,86 | 0,29 |
| 22 | 0,914 | 0,730 | +0,184 | **0,560** | 0,567 | 0,075 | 0,553 | 0,87 / 0,81 | 0,24 |
| 23 | 0,879 | 0,734 | +0,145 | **0,529** | 0,540 | 0,051 | 0,441 | 0,88 / 0,86 | 0,26 |
| 24 | 0,844 | 0,734 | +0,109 | **0,510** | 0,511 | 0,083 | 0,487 | 0,88 / 0,87 | 0,25 |
| 25 | 0,914 | 0,734 | +0,180 | **0,537** | 0,539 | 0,041 | 0,464 | 0,88 / 0,90 | 0,23 |
| 26 | 0,883 | 0,734 | +0,148 | **0,515** | 0,519 | 0,044 | 0,462 | 0,86 / 0,88 | 0,31 |
| 27 | 0,895 | 0,730 | +0,164 | **0,530** | 0,531 | 0,070 | 0,473 | 0,88 / 0,90 | 0,19 |
| 28 | 0,902 | 0,730 | +0,172 | **0,513** | 0,516 | 0,060 | 0,454 | 0,85 / 0,87 | 0,22 |
| 29 | 0,906 | 0,734 | +0,172 | **0,491** | 0,497 | 0,063 | 0,447 | 0,88 / 0,84 | 0,29 |

- **Les dix graines portent leurs besoins, et de façon très semblable**
  (m de 0,49 à 0,59). Greffé dans une autre vie, l'état transmis fait
  pencher le choix d'environ la moitié d'un changement complet dans le sens
  des besoins de la vie d'origine ; venu d'une vie aux mêmes besoins, il ne
  bouge presque rien (0,04 à 0,08). Toutes survivent nettement mieux que
  leur témoin : sur les mêmes 256 mondes, de 37 vies gagnées contre 9
  perdues (graine 24) à 50 contre 3 (graine 21).
- **Ce n'est pas l'événement du tour** : à événement égal au tour j (26 %
  des greffes (a), 49 % des greffes (b)), l'effet aligné reste de +0,476
  [0,449 ; 0,503] (\|ΔP\| 0,477 pour (a), 0,052 pour (b)).
- **Ce n'est pas l'histoire** : sans les 24 greffes (b) qui n'ont pas
  d'autre histoire, \|ΔP\| (b) vaut 0,067 contre 0,534 ; LOOP4 tient
  (différence +0,467 [0,443 ; 0,491]).
- **Le décodeur seul ne suffit pas**, comme au test 33 : le même réseau non
  appris laisse déjà lire 0,76 et 0,77 ; l'apprentissage ajoute environ
  0,11 et 0,10. C'est la greffe qui montre que l'état sert au choix.
- **Selon g** : effet aligné +0,613 à un tour, +0,435 à deux tours ;
  \|ΔP\| des greffes (b) : 0,082 et 0,035. Les greffes (a0) (autres
  besoins, mais la règle « besoins » ne changerait pas son choix) bougent
  le choix de 0,128 : l'état porte plus que ce qui fait basculer la règle.
- **La porte** laisse entrer en moyenne 25 % du nouvel état à chaque tour
  (0,19 à 0,31 selon la graine).
- **À 12 000 mises à jour** (le réglage du test 33, mesuré en chemin sur
  les mêmes graines) : survie 0,853 contre 0,734 (+0,119 [0,077 ; 0,161]) ;
  m +0,381 [0,243 ; 0,520], **9 graines sur 10 au-dessus de 0,05**, mais
  la graine 20 de justesse (0,057) et la graine 24 à peu près nulle
  (0,009) ; ces deux graines survivaient à peine mieux que leur témoin
  (+0,008 et +0,012). **À ce point, ces graines satisfaisaient déjà les
  seuils des quatre critères** (calculé après lecture, avec les mêmes
  formules : LOOP1 +0,119 [0,077 ; 0,161] ; LOOP2 +0,130 [0,089 ; 0,172] ;
  LOOP3 ci-dessus ; LOOP4 0,055 contre 0,391, différence +0,336 [0,213 ;
  0,458]) : avec le réglage du test 33, le critère global aurait été
  satisfait sur ces graines. Ce qui change entre 12 000 et 24 000 : les
  graines 20 et 24 rattrapent les autres (0,586 et 0,510), et l'écart entre
  graines se resserre (intervalle [0,510 ; 0,549] contre [0,243 ; 0,520]).
  Avec le test 33 (7 graines sur 10 à 12 000 mises à jour), cela fait 16
  graines sur 20 à 12 000, et 10 sur 10 à 24 000 : une description, pas un
  test ; l'écart entre le test 33 et ce test peut venir des graines.
- **L'apprentissage reste instable** : quatre graines s'effondrent un moment
  après 2 000 mises à jour puis se rétablissent (0,01 vers 2 800 pour la
  graine 20, 0,08 vers 2 200 pour la 23, 0,00 vers 4 700 pour la 27, 0,08
  vers 2 600 pour la 29) ; la graine 21 descend aussi à 0,45 vers 7 500, la
  26 à 0,53 vers 3 100. Le témoin coupé apprend, comme avant, presque la
  même table (0,730 ou 0,734 ; son plus bas après 2 000 : 0,63 à 0,68).

**Publié sans seuil : à 12 000 mises à jour et creux des courbes, graine par
graine.**

| Graine | m à 12 000 | m à 24 000 | A : survie à 12 000 | B : survie à 12 000 | Plus bas de A après 2 000 (vers) |
|---|---|---|---|---|---|
| 20 | 0,057 | 0,586 | 0,742 | 0,734 | 0,01 (2 800) |
| 21 | 0,429 | 0,528 | 0,859 | 0,734 | 0,45 (7 500) |
| 22 | 0,508 | 0,560 | 0,879 | 0,734 | 0,58 (2 100) |
| 23 | 0,417 | 0,529 | 0,902 | 0,734 | 0,08 (2 200) |
| 24 | 0,009 | 0,510 | 0,746 | 0,734 | 0,68 (6 900) |
| 25 | 0,519 | 0,537 | 0,879 | 0,734 | 0,64 (11 200) |
| 26 | 0,568 | 0,515 | 0,883 | 0,734 | 0,53 (3 100) |
| 27 | 0,357 | 0,530 | 0,887 | 0,734 | 0,00 (4 700) |
| 28 | 0,489 | 0,513 | 0,871 | 0,730 | 0,67 (4 800) |
| 29 | 0,461 | 0,491 | 0,883 | 0,734 | 0,08 (2 600) |

**Écarts d'exécution, déclarés.**
- Les résultats bruts ont été publiés dès leur écriture, chacun dans son
  propre commit qui le nomme. Les survies, et m à 12 000 mises à jour
  (affiché par le programme à ce point, comme prévu), ont été lus au fil de
  l'eau ; m à 24 000 n'a été calculé qu'après les 20 apprentissages
  (verdicts écrits à 19 h 07 UTC). Aucun seuil n'a changé.
- Le brouillon du protocole (avec des blancs pour le réglage) et un
  brouillon du code ont été écrits pendant le second pilote, hors du dépôt ;
  le protocole a été complété et commis après le choix du pilote, le code
  après le protocole. La mesure à 12 000 mises à jour (et la phrase « Cela
  dira si… ») a été ajoutée au protocole après le second pilote ; elle
  n'était pas dans le brouillon. Sans seuil, elle a été commise avant tout
  apprentissage.
- Verdicts écrits à 19 h 07 UTC, publication à 19 h 10 ; le texte et le
  script des tables ont été préparés pendant les apprentissages, les
  nombres remplis après les verdicts.
- Vers 16 h UTC, l'outil de la session a redémarré ; les calculs n'ont pas
  été touchés (les quatre processus ont continué depuis 10 h 15).

**Relecture indépendante** (après publication). Tous les nombres publiés ont
été recalculés à part, avec numpy, et retrouvés ; les graines 20 et 24,
réapprises de zéro jusqu'à 12 000 mises à jour, redonnent exactement les
mesures publiées ; un apprentissage coupé puis repris est identique, bit à
bit, à un apprentissage d'une traite. Vérification de la relecture, après
lecture : LOOP4 tient aussi quand on compare des greffes (a) et (b) dont
l'histoire de la donneuse diffère autant (3, 2 ou 1 tours différents parmi
les 3 derniers : \|ΔP\| 0,563 contre 0,064, 0,529 contre 0,081, 0,450
contre 0,044). La relecture a corrigé une conclusion (« apprendre plus
longtemps a rendu le résultat fiable » : à 12 000 mises à jour, ces graines
passaient déjà les seuils) et fait préciser la dernière action, les valeurs
graine par graine et les écarts ci-dessus. Le code de répartition des
apprentissages entre processus a ensuite été rendu plus sûr (une
réservation n'est plus jamais lue vide ; le contrôle de la graine 20 vérifie
qu'il lit le point final) ; rien de cela n'est arrivé pendant le calcul, et
les résultats n'en dépendent pas.

### Ce que cela dit

- **Pour la première fois dans ce projet, une récurrence au sens strict,
  apprise par la seule survie, porte de façon fiable ses besoins cachés et
  s'en sert** : dix graines neuves sur dix, de nouvelles vies de greffe,
  tous les critères pré-enregistrés. Ses besoins ne lui sont jamais donnés ;
  il les reconstruit à partir des événements et de ses actions, les garde
  dans l'état qui passe d'un tour au suivant (sa seule autre entrée venue
  du passé est sa dernière action, que le témoin reçoit aussi), et cet
  état, greffé
  dans une autre vie, fait choisir selon les besoins de la vie d'origine,
  à événement égal et à histoire différente.
- **Apprendre plus longtemps a resserré le résultat ; ce test ne montre
  pas qu'il était nécessaire.** À 12 000 mises à jour (mesuré en chemin),
  ces dix graines satisfaisaient déjà les seuils des quatre critères
  (LOOP3 : 9 graines sur 10 ; la graine 24 à peu près nulle, 0,009, la 20
  de justesse, 0,057, et ces deux graines survivaient à peine mieux que leur
  témoin, +0,008 et +0,012). À 24 000, elles ont rattrapé les autres (m de
  0,49 à 0,59). L'écart avec le test 33 (7 graines sur 10 à 12 000) peut
  venir des graines. Le test ne dit pas pourquoi certaines graines tardent.

**Ce que le résultat ne dit pas.**
- Rien sur un ressenti.
- Le réglage a été choisi en regardant la boucle et la greffe (au second
  pilote, sur d'autres graines, mondes et vies) ; le témoin a reçu le même.
- La récompense (la survie) vient du monde, qui connaît les besoins ; elle
  ne nomme ni le besoin ni l'action. La tête de valeur du critique prédit
  le retour de survie.
- La greffe lit des vies écrites par une règle, pas des vies vécues par
  l'agent ; elle ne sépare pas le niveau des besoins de ce qui en commande
  le choix. « Ce qu'elle porte est surtout le niveau de ses besoins » veut
  dire : plus que l'histoire récente à besoins égaux (LOOP4) ; ce n'est pas
  une lecture fine de E et de N.
- **Ce qui est déjà connu** (`docs/LITERATURE_CHECK_LOOP_2026-10-06.md`) :
  des agents récurrents appris par renforcement ou par stratégie
  évolutionnaire suivent un niveau interne dans leur état, et forcer ou
  perturber cet état change leur comportement, mais ce niveau leur est
  **donné en entrée**, parfois avec du bruit (Chaturvedi et al. 2025,
  révisé en 2026 ; Hayes 2026) ; et qu'un agent récurrent appris par
  renforcement sous observabilité partielle garde une mémoire des variables
  cachées est attendu (DRQN, états à porte). L'apport est étroit : des
  besoins jamais observés, une greffe entre deux vies contre un témoin sans
  mémoire, la séparation du besoin et de l'histoire, le pré-enregistrement.
  Ce n'est pas une découverte de principe.
- Ces modèles sont petits (2 couches, dimension 64), le monde aussi (deux
  besoins, deux actions, six événements).

## Test 35 pré-enregistré : le besoin qui dure (critère global satisfait)

Protocole `docs/TINY_LOOP_DURATION_PROTOCOL.md` (commit `d17938d`), écrit
après la publication du test 34 et avant tout code et toute mesure ; code
commis avant toute mesure (`0befa54`). Mesure torch sur le processeur local,
le 6 octobre 2026, de 20 h 25 à 20 h 28 UTC (deux processus, cinq graines
chacun ; puis un passage sur les dix graines pour écrire les contrôles).
Artefacts : `artifacts/tiny-loop-duration`. Verdicts recalculés en local
(identiques) ; la CI les vérifie à chaque envoi.

**Ce qu'on teste.** Au test 34, on ne greffait l'état transmis qu'à un ou
deux tours de distance. Ici, sans nouvel apprentissage, on greffe l'état
des dix boucles du test 34 (au point final de 24 000 mises à jour) au tour
j, et on lit le choix au tour t = j + g, pour **g = 1 à 6**, sur 128 vies
de greffe neuves (flux `[270926, 83]` ; tirage `[270926, 84, 0]`, trois
tours par receveuse et par g ; 5 282 greffes). m_g = moyenne sur les
greffes (a) de ΔP(R) × e ; une boucle qui suivrait exactement la règle
« besoins » aurait m_g = 1 à toute distance.

| g (tours entre la greffe et le choix) | 1 | 2 | 3 | **4** | 5 | **6** |
|---|---|---|---|---|---|---|
| Greffes (a) / (b) | 335 / 279 | 269 / 297 | 234 / 305 | 200 / 293 | 161 / 292 | 127 / 288 |
| m_g (moyenne sur 10 graines) | +0,550 | +0,407 | +0,307 | **+0,256** | +0,179 | **+0,148** |
| Intervalle à 95 % | [0,529 ; 0,570] | [0,378 ; 0,437] | [0,276 ; 0,338] | [0,229 ; 0,282] | [0,145 ; 0,214] | [0,114 ; 0,182] |
| m_g / m_1 | 1 | 0,74 | 0,56 | 0,46 | 0,33 | 0,27 |
| \|ΔP\| (a) | 0,568 | 0,424 | 0,318 | 0,268 | 0,216 | 0,169 |
| \|ΔP\| (b) (sans autre histoire exclues) | 0,073 (0,084) | 0,060 (0,072) | 0,053 (0,063) | 0,036 (0,045) | 0,036 (0,045) | 0,023 (0,029) |
| \|ΔP\| (a0) | 0,156 | 0,108 | 0,096 | 0,072 | 0,060 | 0,041 |
| m_g à événement égal au tour j | +0,495 | +0,385 | +0,343 | +0,266 | +0,165 | +0,220 |

| | Critère | Valeur | Verdict |
|---|---|---|---|
| **DUR1** | à g = 4 : moyenne des m_4 ≥ **0,10**, borne basse > 0, et m_4 > 0,05 pour au moins **8 graines sur 10** | +0,256 [0,229 ; 0,282] ; **10 graines sur 10** (de 0,188 à 0,299) | **passe** |
| **DUR2** | à g = 4 : \|ΔP\| (b) ≤ la moitié de \|ΔP\| (a), différence à borne basse > 0 | 0,036 contre 0,268 ; +0,232 [0,209 ; 0,255] | **passe** |
| **DUR3** | à g = 6 : moyenne des m_6, borne basse > 0 | +0,148 [0,114 ; 0,182] | **passe** |
| | Validité : ≥ 100 greffes (a) et (b) à g = 4, ≥ 100 (a) à g = 6 ; greffe de soi ≤ 1e-6 ; dix réseaux à 24 000 mises à jour | 200 et 293 ; 127 ; 0 ; oui | **valide** |

**Critère global : satisfait.**

**Lecture fixée d'avance** (DUR1 à DUR3 passent) : « L'état que la boucle
apprend par la seule survie n'est pas une mémoire de un ou deux tours :
quatre tours plus tard, il fait encore choisir selon les besoins de la vie
d'où il vient, plus que l'histoire à besoins égaux, et il en reste quelque
chose à six tours. »

**Publié sans seuil : graine par graine, m_g pour g = 1 à 6.**

| Graine | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 |
|---|---|---|---|---|---|---|---|---|---|---|
| m_1 | 0,602 | 0,572 | 0,551 | 0,556 | 0,518 | 0,581 | 0,542 | 0,536 | 0,530 | 0,509 |
| m_4 | 0,299 | 0,251 | 0,299 | 0,258 | 0,188 | 0,293 | 0,279 | 0,234 | 0,225 | 0,229 |
| m_6 | 0,227 | 0,114 | 0,206 | 0,183 | 0,087 | 0,155 | 0,172 | 0,102 | 0,116 | 0,121 |

- **L'effet s'efface d'environ un quart par tour** (m_g / m_1 : 0,74 ;
  0,56 ; 0,46 ; 0,33 ; 0,27), dans toutes les graines : une mémoire des
  besoins qui s'estompe, pas une mémoire qui garde. Une boucle qui suivrait
  exactement les besoins garderait tout l'effet. Après lecture : ce rythme
  (environ 0,77 par tour) est proche de la part de l'ancien état que la
  porte garde en moyenne à chaque tour (1 − 0,25 = 0,75) ; ce n'est qu'un
  rapprochement, non testé.
- **À toutes les distances, c'est le besoin, pas l'histoire** : les greffes
  (b), aux mêmes besoins, bougent le choix six à huit fois moins que les
  greffes (a) ; à événement égal au tour j, l'effet reste du même ordre.
- Les greffes (a0) (autres besoins, sans que la règle change son choix au
  tour t) bougent aussi un peu le choix (0,16 à 0,04).

**Écarts, déclarés.**
- Les nombres de greffes ont été comptés avant le protocole, sans modèle,
  sur ces vies neuves, pour choisir trois tours par receveuse et par g.
- Les effets à g = 1 et 2 étaient connus (test 34) quand les seuils à g = 4
  et 6 ont été fixés.
- Rien d'autre : chaque fichier brut a été commis seul, sous un message qui
  le nomme.

### Ce que cela dit

- **L'état que la boucle apprend par la seule survie porte les besoins sur
  plusieurs tours** : quatre tours après la greffe, le choix suit encore
  les besoins de la vie d'origine dans les dix graines (+0,26), et il en
  reste à six tours (+0,15). Ce n'est pas la mémoire courte du modèle de
  langage (tests 13 et 22).
- **Mais c'est une mémoire qui s'estompe** (environ un quart par tour),
  pas un compte exact des besoins.

**Ce que le résultat ne dit pas.**
- Rien sur un ressenti.
- Les réseaux sont ceux du test 34, déjà lus ; seules les vies de greffe
  sont neuves.
- La greffe lit des vies écrites par une règle, pas des vies vécues par
  l'agent.
- Petit réseau, petit monde.

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
