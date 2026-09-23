# Où le LLM ajusté chercherait-il sa cause — résultats

Protocole `docs/LLM_INQUIRY_PROTOCOL.md` et son amendement d'exécution.
Exécuté le 23 septembre 2026, 2 h 45 – 3 h 27 UTC, sur le Mac mini M2 de
Codemagic (workflow `menia-inquiry-mac`, lancé par le relais, commit
`0484358`). Qwen3-0.6B par mlx-lm, adaptateurs VM et F du corps ajusté
(`run-3`), complétion brute, sans génération ; 96 états par modèle (48
débuts de vie du jeu R, 24 pas 11 et 24 pas après le premier mouvement du
jeu M). Artefacts dans `artifacts/llm-inquiry/run-1/llm-inquiry` ;
résumés recalculés depuis les lignes, identiques aux publiés, et verdicts
dans `artifacts/llm-inquiry/verdicts-run-1.json`, vérifiés en CI.

## Contrôle de validité : échoué pour les deux modèles

| | Masse sur les chiffres | Masse sur les symboles | Seuil | Verdict |
|---|---:|---:|---:|---|
| VM | 0,999 | **0,014** | 0,5 | **invalide** |
| F | 1,000 | **0,007** | 0,5 | **invalide** |

Le protocole lit ce contrôle en premier : **I1, I2 et I3 ne sont pas
jugés, et le critère global n'est pas satisfait.** Les gains mesurés
(lieu 1 préféré dans 0,42 des vies pour VM, 0,08 pour F) reposent sur des
distributions de symboles qui portent 1 % de la masse et ne sont pas
interprétés.

## Pourquoi

Un défaut de la mesure, écrit avant l'exécution. Dans les vies qui ont
servi à l'ajustement, un symbole suit toujours un espace :
« inspection du lieu 1, symbole ◇. ». Le découpage de Qwen rattache un
espace au signe qui le suit quand ce signe n'est ni une lettre ni un
chiffre : les modèles ont appris « espace-symbole » comme un tout. La
question, elle, s'arrêtait après l'espace (« …, symbole ») et lisait
ensuite le symbole seul, une suite que les modèles n'ont jamais vue ; d'où
une masse presque nulle (médiane 0,002 par état, 0,07 au plus). Les
chiffres ne posent pas ce problème, parce que le découpage sépare toujours
l'espace d'un chiffre : leur masse vaut 0,999, comme dans le corps ajusté.

## Suite

Une relance corrigée est déclarée dans le protocole (amendement
d'exécution 2) : la question s'arrête sur « symbole », et chaque symbole
est lu comme la suite de morceaux que le modèle écrirait lui-même après ce
mot, espace compris. États, mesures, critères et seuils sont inchangés ;
ce résultat invalide reste publié tel quel.

## Deuxième exécution (amendement d'exécution 2)

Exécutée le 23 septembre 2026, 5 h 36 – 6 h 33 UTC, sur le Mac mini M2 de
Codemagic (workflow `menia-inquiry-mac`, lancé par le relais, commit
`a83f30f`), avec la lecture corrigée des symboles ; mêmes adaptateurs,
mêmes 96 états par modèle. Artefacts dans
`artifacts/llm-inquiry/run-2/llm-inquiry` ; résumés recalculés depuis les
lignes, identiques aux publiés, et verdicts dans
`artifacts/llm-inquiry/verdicts-run-2.json`, vérifiés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| Validité | masse ≥ 0,5 sur les chiffres et sur les symboles | VM 0,999 et 0,997 ; F 1,000 et 0,996 | **passe** |
| **I1** | VM cherche sa cause : lieu 1 préféré dans ≥ 0,7 des vies, gain du lieu 1 ≥ 2 × celui des autres | 0,40 (19 vies sur 48) ; 1,4 × | **échoue** |
| **I2** | F ne la cherche pas : lieu 1 préféré dans ≤ 0,4 des vies | 0,06 | **passe** |
| **I3** | Le changement de corps rallume l'enquête : gain du lieu 1 après le premier mouvement ≥ 1,5 × celui du pas 11 | −0,027 contre −0,069 | **échoue** |
| Global | I1 et I2 | | **non satisfait** |

- **La lecture corrigée rend le test valide** : les symboles portent
  maintenant 99,7 % de la masse ; la cause de l'échec de la première
  exécution était bien la découpe du symbole.
- **Le LLM ajusté sur des corps variables ne sait pas où chercher sa
  cause.** Au début d'une vie, il est très incertain de l'effet de ses
  commandes (0,99 de l'entropie maximale), mais il n'attend presque rien
  d'aucune inspection : 0,004 nat de gain pour le lieu de la marque, à peine
  plus que pour les autres (0,003 en moyenne). Il préfère le lieu 1 dans 19
  vies, le lieu 4 dans 15. Le petit agent récurrent et le
  micro-transformeur élevés sur des corps variables allaient, eux, lire la
  marque.
- **I2 passe pour une raison qui n'est pas une enquête** : ajusté sur un
  corps fixe, le modèle se croit certain de son corps (incertitude 0,0004)
  et n'a rien à chercher ; c'est la confabulation déjà vue dans le corps
  ajusté.
- **I3** : le gain du lieu 1 est négatif avant et après le changement de
  corps ; lire la marque augmenterait, selon le modèle, son incertitude.
  Le calcul écrit avant l'exécution exige un gain positif après le
  mouvement ; il échoue. (Au sens littéral, −0,027 ≥ 1,5 × −0,069, mais un
  gain négatif n'est pas une enquête qui se rallume.)

**Conclusion de la ligne du LLM ajusté** : l'ajustement sur des vies à
corps variable donne à Qwen3-0.6B une bonne part de la structure de son
corps (A3, 0,79) et la révision après un changement (A4), mais pas la
disposition à chercher la trace de sa cause. Cette disposition, que les
petits modèles acquéraient avec les mêmes données, ne passe pas au LLM par
cet ajustement.

## Régime VMI : une enfance où regarder avant d'agir paie

Protocole `docs/LLM_INQUIRY_DATA_PROTOCOL.md`. Build `menia-lora-vmi-mac`
du 23 septembre 2026, 8 h 48 – 10 h 35 UTC, commit `227225d`, lancé par le
relais : Qwen3-0.6B ajusté par LoRA (même ajustement que VM dans `run-3`)
sur 1 500 vies VMI, où les 2 à 6 premiers tours sont des inspections et où
la marque est lue avant le premier mouvement dans trois vies sur quatre ;
puis le test d'enquête et les cellules du corps ajusté sur cet adaptateur.
Artefacts dans `artifacts/llm-lora-mac/run-5-vmi` ; verdicts (VMI contre le
F de la deuxième exécution) recalculés depuis les lignes dans
`artifacts/llm-lora-mac/verdicts-run-5-vmi.json`, vérifiés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| Validité | masse ≥ 0,5 sur les chiffres et sur les symboles | 0,999 ; 0,996 | **passe** |
| **J1** | VMI cherche sa cause (critère d'I1) | lieu 1 préféré dans **0,29** des vies (14 sur 48) ; gains de tous les lieux légèrement négatifs (−0,012 pour le lieu 1) | **échoue** |
| **J2** | F ne la cherche pas (I2 de la deuxième exécution) | 0,06 | **passe** |
| **J3** | aucune prédiction | gain du lieu 1 −0,058 au pas 11, **+0,032** après le premier mouvement suivant le changement | passe au sens du calcul |
| Global | J1 et J2 | | **non satisfait** |

Cellules du corps ajusté, sans prédiction : copie 0,99, commandes
nouvelles 0,65, **nouvelles après un mouvement 0,786** (VM : 0,793), pas 16
à 23 du jeu M 0,80 (VM : 0,86) ; perte de validation 0,109.

- **La prédiction est réfutée.** Même quand la marque est, dans trois vies
  sur quatre, la seule information sur le corps avant le premier
  mouvement, le LLM ajusté n'apprend pas à la lire : au début d'une vie, il
  n'attend d'aucune inspection une baisse de son incertitude. La
  redondance de la marque dans VM n'expliquait donc pas l'échec.
- Après un changement de corps et un premier mouvement qui le contredit, le
  gain attendu du lieu de la marque devient positif (0,032) : une trace
  d'enquête qui se rallume, que rien ne prédisait et qui reste petite.
- **Le reste ne change pas** : la structure du corps (0,786) et la révision
  sont celles de VM.

**Conclusion** : avec cet ajustement (LoRA de rang 8, 600 itérations),
Qwen3-0.6B apprend l'effet de ses commandes à partir de ses mouvements,
mais pas à chercher la marque de sa cause, que les données la rendent
redondante ou nécessaire. Les petits modèles, entraînés longtemps sur ces
mêmes vies avec une tête qui prédit le mouvement, y arrivaient ; ce qui
manque au LLM ajusté tient à l'ajustement, pas à l'enfance seule.

## Ajustement moteur (VMM) : échec de conception

Protocole `docs/LLM_MOTOR_OBJECTIVE_PROTOCOL.md`. Build
`menia-lora-motor-mac` du 23 septembre 2026, 10 h 55 – 12 h 34 UTC, commit
`8f0f80e` : Qwen3-0.6B ajusté sur un exemple par mouvement des vies VMI, la
perte ne portant que sur la case d'arrivée. Artefacts dans
`artifacts/llm-lora-mac/run-6-vmm` ; verdicts dans
`artifacts/llm-lora-mac/verdicts-run-6-vmm.json`, vérifiés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| Validité | masse ≥ 0,5 sur les chiffres et sur les symboles | chiffres 0,995 ; **symboles 0,000** | **invalide** |
| **K1**, **K3** | — | non jugés | — |
| **K2** | I2 de la deuxième exécution | 0,06 | passe |
| Global | K1 et K2 | | **non satisfait** |

Cellules du corps ajusté, sans prédiction : copie des commandes vues
**0,25**, commandes nouvelles 0,24, pas 16 à 23 : 0,24 — le hasard.

**Ce n'est pas un résultat sur l'hypothèse, mais une erreur de conception
de l'ajustement**, écrite dans le protocole avant l'exécution :

- **Trop peu de cibles.** Le protocole affirmait que 1 000 itérations d'un
  exemple par mouvement verraient autant de cibles que les ajustements
  précédents. C'est faux : l'ajustement sur le texte voit à chaque
  itération 4 vies de 10 mouvements, soit environ 24 000 cases d'arrivée en
  600 itérations ; l'ajustement moteur n'en a vu que 4 000. La perte de
  validation stagne (0,73 dès 400 itérations) et le modèle n'a pas appris
  l'effet de ses commandes, pas même à copier celles qu'il a vues.
- **Une cible parasite.** Le masque de mlx-lm compte aussi la position qui
  suit la cible ; les exemples étant complétés d'un jeton de remplissage,
  la perte portait pour moitié sur ce jeton (8 000 jetons entraînés pour
  4 000 cibles).
- **Les symboles oubliés.** Sans perte sur le reste du texte, le modèle ne
  prédit plus aucun symbole après « symbole » ; le test d'enquête ne peut
  plus rien lire.

L'hypothèse (l'objectif compte) reste ouverte. Une relance devrait garder
la perte sur tout le texte, pour les symboles et pour le nombre de cibles,
et y ajouter un poids sur les cases d'arrivée ; elle demande un nouveau
protocole.

## Ajustement pondéré (VMW) : la structure apprise, pas l'enquête

Protocole `docs/LLM_MOTOR_OBJECTIVE_PROTOCOL.md`, amendement 1. Build
`menia-lora-weighted-mac` du 23 septembre 2026, 12 h 42 – 14 h 10 UTC,
commit `a9bf62d`, lancé par le relais : Qwen3-0.6B ajusté sur les mêmes
1 500 vies VMI que `run-5-vmi`, même LoRA, 600 itérations, la perte portant
sur tout le texte mais chaque chiffre de case d'arrivée pesant 20 fois
plus. Artefacts dans `artifacts/llm-lora-mac/run-7-vmw` ; verdicts dans
`artifacts/llm-lora-mac/verdicts-run-7-vmw.json`, vérifiés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| Validité | masse ≥ 0,5 sur les chiffres et sur les symboles | 0,999 ; 0,996 | **passe** |
| **K1** | VMW cherche sa cause (critère d'I1) | lieu 1 préféré dans **0,44** des vies (21 sur 48) ; gains de tous les lieux légèrement négatifs (−0,007 pour le lieu 1, −0,008 à −0,011 pour les autres) | **échoue** |
| **K2** | F ne la cherche pas (I2 de la deuxième exécution) | 0,06 | **passe** |
| **K3** | aucune prédiction | gain du lieu 1 −0,082 au pas 11, +0,002 après le premier mouvement suivant le changement | passe au sens du calcul, parce que le gain d'avant est négatif ; la hausse est presque nulle |
| Global | K1 et K2 | | **non satisfait** |

L'ajustement s'est fait normalement (perte de validation pondérée de 1,16
à 0,17, en baisse jusqu'au bout). Les cases d'arrivée ont porté **0,22**
de la perte, et non « près du tiers » comme l'amendement l'estimait : une
vie compte environ 700 cibles, pas 500 (`weights-VMW.json`). Le poids de 20
est resté celui qui avait été fixé.

Cellules du corps ajusté, sans prédiction :

| | VMI (`run-5-vmi`) | **VMW** |
|---|---:|---:|
| Copie des commandes vues (jeu R) | 0,99 | 0,98 |
| Commandes nouvelles (jeu R) | 0,65 | **0,75** |
| Nouvelles après un mouvement (cellule d'A3) | 0,786 | **0,929** |
| Jeu M, pas 16 à 23 (cellule d'A4) | 0,80 | **0,86** |
| Jeu M, commandes vues seulement avant le changement | 0,24 | 0,39 |

- **La prédiction est réfutée.** Même quand la perte met l'accent sur
  l'effet de ses commandes, le LLM ajusté n'attend d'aucune inspection,
  au début d'une vie, une baisse de son incertitude.
- **L'objectif change la structure apprise.** Sur les commandes jamais
  essayées, après un mouvement, VMW prédit juste dans 0,93 des cas, là où
  tous les ajustements sur le texte plafonnaient à 0,79 (VM, VMI, rang 16).
  C'est la première fois qu'une valeur passe le seuil de 0,80 d'A3. Ce
  n'est pas un verdict A3, qui porte sur le régime VM avec la perte sur le
  texte : c'est une cellule descriptive.
- **Analyse exploratoire, non pré-enregistrée** (petit échantillon). Dans
  les vies d'évaluation, la marque du lieu 1 a été lue avant le premier
  mouvement dans 20 cas (10 dans R, 10 dans M). VMW donne alors à la bonne
  case d'arrivée une probabilité moyenne de 0,26, contre 0,24 dans les 76
  cas où elle n'a pas été lue (hasard : 0,25) ; VMI : 0,26 contre 0,24 ;
  VM (`run-4`) : 0,22 contre 0,25 (`research/llm_mark_use.py`, sortie
  `artifacts/llm-lora-mac/mark-use.json`, vérifiée en CI). **Le LLM
  ajusté ne lit pas la marque** : il n'a pas appris ce que chaque symbole
  dit de son corps. L'enquête ne peut pas se former sur une information
  qu'il ne sait pas lire. L'échec est en amont de la disposition à
  chercher.

**Conclusion** : l'amendement 1 prévoyait une seule relance. Elle a donné
un résultat valide, négatif pour l'hypothèse. Avec un ajustement LoRA court
de Qwen3-0.6B, ni l'enfance (VMI) ni l'objectif (VMW) ne suffisent pour que
le modèle cherche la cause de son corps. L'objectif pondéré lui fait mieux
apprendre l'effet de ses commandes à partir de ses mouvements, mais le lien
arbitraire entre un symbole et son corps, qui ne sert qu'au premier
mouvement d'une vie, n'est pas appris. Ce test ne mesure pas une
expérience vécue.

## Lecture de la marque et régime VML : le code du corps n'est pas appris

Protocole `docs/LLM_MARK_READING_PROTOCOL.md`. Build
`menia-lora-reading-mac` du 23 septembre 2026, 14 h 38 – 16 h 23 UTC,
commit `78e33b8`, lancé par le relais : test de lecture des quatre
adaptateurs publiés, puis Qwen3-0.6B ajusté comme VMW sur 1 500 vies VML
(12 paires inspection puis mouvement, lieu 1 lu une fois sur deux, corps
tiré à nouveau après chaque mouvement), test de lecture, test d'enquête et
cellules. Artefacts dans `artifacts/llm-lora-mac/run-8-vml` ; verdicts
recalculés depuis les lignes dans
`artifacts/llm-lora-mac/verdicts-run-8-vml.json`, vérifiés en CI.

| Adaptateur | Masse sur les chiffres | P1 (après la marque) | P0 (après un autre lieu) |
|---|---:|---:|---:|
| F (`run-3`) | 1,000 | 0,250 | 0,250 |
| VM (`run-3`) | 1,000 | 0,248 | 0,250 |
| VMI | 0,999 | 0,252 | 0,251 |
| VMW | 0,999 | 0,254 | 0,252 |
| **VML** | 0,999 | **0,251** | 0,250 |

(Lecteur parfait : P1 ≈ 0,85 ; modèle aveugle au symbole : exactement 0,25.)

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| Validité | masse ≥ 0,5 | tous les adaptateurs ≥ 0,999 ; enquête VML 0,999 et 0,997 | **passe** |
| **L0** | les adaptateurs publiés ne lisent pas la marque (P1 ≤ 0,35) | 0,248 à 0,254 | **passe** |
| **L1** | VML lit la marque (P1 ≥ 0,6 et P1 − P0 ≥ 0,3) | 0,251 ; écart 0,002 | **échoue** |
| **L2** | VML cherche la marque (critère d'I1) | lieu 1 préféré dans 0,15 des vies ; gains de tous les lieux nuls (−0,001) ; incertitude de départ 0,99 | **échoue** |
| **L3** | F ne la cherche pas (I2) | 0,06 | **passe** |
| Global | L1, L2 et L3 | | **non satisfait** |

- **L'analyse exploratoire est confirmée.** Sur 64 invites par
  adaptateur au lieu de 20 cas, aucun des quatre adaptateurs publiés ne
  tient compte du symbole de la marque.
- **Même quand la marque est la seule information, le LLM ajusté ne
  l'apprend pas.** VML a vu environ 14 000 mouvements précédés de la
  marque, et les cases d'arrivée portaient 0,26 de la perte
  (`weights-VML.json`). Sa perte de validation plafonne dès l'itération
  100 (0,437, puis 0,418 à l'itération 600). Il a appris que la case
  d'arrivée est l'une des quatre cases voisines, tirée au hasard : son
  incertitude de départ est de 0,99 sur 1. Il n'a pas non plus appris un
  code décalé : pour chaque symbole, ses meilleures cases se répartissent
  également entre les quatre corps. Au plafond, les cases d'arrivée
  prédites au hasard font à elles seules environ 85 % de la perte
  restante ; lire la marque l'aurait réduite d'environ un quart.
- Cellules du corps ajusté, sans prédiction : toutes au hasard (copie
  0,22 ; nouvelles 0,23 ; jeu M, pas 16 à 23, 0,27). C'est attendu, puisque
  dans VML les mouvements passés ne disent rien du corps.

**Une propriété de l'Atelier explique ce plafond (constat, non
pré-enregistré).** La table qui donne le déplacement selon le corps et la
commande est un carré latin : pour un corps donné, les quatre commandes
donnent les quatre déplacements ; pour une commande donnée, les quatre
corps aussi. **Ni le symbole seul ni la commande seule ne disent quoi que
ce soit de la case d'arrivée** ; seule leur combinaison le fait, comme un
« ou exclusif ». Un apprentissage par gradient ne reçoit alors aucun
signal partiel pour commencer : tant qu'il n'a pas trouvé l'interaction,
la meilleure prédiction est l'uniforme, qui est exactement là où le modèle
s'arrête. Les mouvements, eux, offrent un indice de premier ordre (la même
commande redonne le même déplacement), que tous les ajustements ont
appris.

**Conclusion** : avec un ajustement LoRA court (rang 8, 600 itérations),
Qwen3-0.6B n'apprend pas le code arbitraire qui relie la marque à son
corps, que la marque soit rare (VMI, VMW) ou nécessaire à chaque mouvement
(VML). La limite tient à l'ajustement face à un problème d'interaction
pure, pas aux données ni à la disposition à chercher : le LLM ne peut pas
chercher une trace qu'il ne sait pas lire. Ce test ne mesure pas une
expérience vécue.

## Une commande préférée (VMLA) : un début d'appui, pas de lecture

Protocole `docs/LLM_MARK_HAND_PROTOCOL.md`. Build `menia-lora-hand-mac` du
23 septembre 2026, 16 h 40 – 18 h 03 UTC, commit `5a787d7`, lancé par le
relais : Qwen3-0.6B ajusté comme VML sur les mêmes vies, à ceci près que la
commande est A dans 70 % des mouvements, ce qui donne au symbole seul un
pouvoir de prédiction (0,61 au lieu de 0,25). Artefacts dans
`artifacts/llm-lora-mac/run-9-vmla` ; verdicts dans
`artifacts/llm-lora-mac/verdicts-run-9-vmla.json`, vérifiés en CI.

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| Validité | masse ≥ 0,5 | lecture 1,000 ; enquête 1,000 et 0,997 | **passe** |
| **H1** | VMLA lit la marque pour A (P1(A) ≥ 0,6 et P1(A) − P0 ≥ 0,3) | P1(A) **0,312** ; P0 0,253 | **échoue** |
| **H2** | aucune prédiction (P1(BCD) ≥ 0,6) | 0,231 | échoue |
| **H3** | VMLA cherche la marque (critère d'I1) | lieu 1 préféré dans **0,56** des vies (VML : 0,15) ; gain du lieu 1 0,0179, des autres 0,0169 | **échoue** |
| **H4** | F ne la cherche pas (I2) | 0,06 | **passe** |
| Global | H1, H3 et H4 | | **non satisfait** |

P1 sur les quatre commandes (critère L1, descriptif) : 0,252. Perte de
validation : 0,428 à l'itération 100, 0,402 à l'itération 600 (VML :
0,418), encore en légère baisse.

- **La prédiction est réfutée**, mais l'appui laisse une trace. Pour la
  commande A, le modèle donne 0,31 à la case qu'implique la marque (0,23
  pour les autres commandes, 0,25 au hasard), et le lieu de la marque
  devient le lieu préféré dans 0,56 des vies au lieu de 0,15. Pour A, sa
  meilleure case suit le symbole ◇ dans 3 cas sur 4 et le symbole ○ dans
  4 sur 4, mais △ et □ donnent la même case. Ces deux symboles commencent
  par les mêmes octets, tout comme ◇ et ○ entre eux ; observation, non
  pré-enregistrée, sur 16 invites.
- **Il n'apprend pas à lire la marque en 600 itérations**, même quand le
  symbole seul prédit la case d'arrivée dans 61 % des mouvements. Le
  gradient a trouvé un appui, mais il reste loin d'un code utilisable.

**Conclusion** : briser la symétrie du problème ne suffit pas, avec cet
ajustement court. La limite tient à l'ajustement (LoRA de rang 8, 600
itérations sur 1 500 vies) : c'est la troisième enfance (VMI, VML, VMLA) et
le deuxième objectif (texte, perte pondérée) qui échouent à faire lire la
marque à Qwen3-0.6B. Ce test ne mesure pas une expérience vécue.

## VMLA prolongé : le LLM ajusté lit la marque de son corps

Protocole `docs/LLM_MARK_HAND_LONG_PROTOCOL.md`. Build
`menia-lora-hand-long-mac` du 23 septembre 2026, 18 h 17 – 19 h 39 UTC,
commit `1a1b2e9`, lancé par le relais : l'adaptateur publié de VMLA (600
itérations) repris et ajusté 750 itérations de plus sur les mêmes vies,
test de lecture à 850, 1 100 et 1 350 itérations au total. Artefacts dans
`artifacts/llm-lora-mac/run-10-vmla-long` ; verdicts dans
`artifacts/llm-lora-mac/verdicts-run-10-vmla-long.json`, vérifiés en CI.

| Itérations | Perte de validation | P1(A) | P1(B, C, D) | P0 | Masse |
|---:|---:|---:|---:|---:|---:|
| 600 (`run-9-vmla`) | 0,402 | 0,312 | 0,231 | 0,253 | 1,000 |
| 850 | 0,356 | **0,737** | 0,249 | 0,251 | 0,999 |
| 1 100 | 0,353 | 0,765 | 0,250 | 0,249 | 0,999 |
| 1 350 | 0,362 | **0,811** | 0,250 | 0,250 | 0,999 |

(Lecteur parfait : 0,85 ; hasard : 0,25.)

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| Validité | masse ≥ 0,5 à chaque point | 0,999 | **passe** |
| **G1** | à 1 350 : P1(A) ≥ 0,6 et P1(A) − P0 ≥ 0,3 | 0,811 ; écart 0,561 | **passe** |
| **G2** | P1(A) gagne au moins 0,1 sur 0,312 | + 0,499 | **passe** |
| G3 | aucune prédiction | P1(B, C, D) 0,250 à chaque point ; P1 sur les quatre commandes 0,390 | — |
| Global | G1 | | **satisfait** |

- **La prédiction est confirmée : pour la première fois, un LLM ajusté lit
  la marque de son corps.** Entre 600 et 850 itérations, la probabilité
  qu'il donne à la case qu'implique la marque, pour sa commande préférée,
  passe de 0,31 à 0,74, puis monte à 0,81, près du lecteur parfait (0,85).
  La perte de validation fait le même saut (0,402 → 0,356). C'est une
  **transition après un plateau**, comme on l'attend quand le gradient
  doit trouver une interaction : l'appui donné par la commande préférée l'a
  rendue trouvable, et la durée l'a fait trouver.
- **La lecture est spécifique** (constats, non pré-enregistrés). Elle ne
  vient que du lieu 1 : après un symbole lu à un autre lieu, la commande A
  reste au hasard (0,25). Les quatre symboles sont lus (0,78 à 0,84 à
  1 350 itérations) ; △ et □, confondus à 600 itérations, sont distingués.
  Pour les commandes B, C et D, le modèle reste au hasard (0,25) au lieu
  d'appliquer la case de A : il ne sait pas, et ne fait pas semblant de
  savoir.
- **La lecture ne s'étend pas aux autres commandes** en 1 350 itérations :
  le code entier du corps (l'interaction complète entre symbole et
  commande) n'est pas appris. C'était sans prédiction.

**Conclusion** : un Qwen3-0.6B ajusté par LoRA apprend à lire la trace de
la cause de ses mouvements, pourvu que son enfance lui donne un appui (une
commande préférée, qui rend le symbole utile à lui seul) et que
l'ajustement dure assez pour passer le plateau. Reste à mesurer s'il
**cherche** désormais cette marque, ce que le protocole renvoyait à une
mesure suivante. Ce test ne mesure pas une expérience vécue.
