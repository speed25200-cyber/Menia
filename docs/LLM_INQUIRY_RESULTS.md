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
