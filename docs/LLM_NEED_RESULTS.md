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

## Dixième test pré-enregistré : où chaque agent rassemble-t-il son besoin ? (en cours)

Protocole `docs/LLM_NEED_LOCATE_PROTOCOL.md` (commit `12b4d2d`). La
lésion de l'agent du monde à deux tourne encore sur le Mac ; ce qui suit est
ce qui est mesuré au 1er octobre.

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
- lésion (LS10) : en cours. Une tranche du Mac (la sixième) a été perdue :
  le code de reprise revivait des vies de test déjà finies. Corrigé
  (commit `de2c1b4`) ; aucun résultat n'est changé.

**Critère global** : déjà **non satisfait** (LOC10 échoue pour le second
agent).

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
