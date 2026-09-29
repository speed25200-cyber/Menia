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
