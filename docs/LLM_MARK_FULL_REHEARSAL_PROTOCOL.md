# Protocole pré-enregistré — Le code entier du corps par ajouts répétés

Rédigé le 24 septembre 2026, après la répétition de deux lectures
(`docs/LLM_INQUIRY_RESULTS.md`), **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## Ce qui a été vu

Le LLM ajusté apprend à lire la marque de son corps pour une commande
quand elle est préférée (A, puis B) ; appris seul, B efface A, mais
répétées ensemble, les deux lectures tiennent (0,83 et 0,83, chacune pour
sa commande), et une troisième commence à naître pour C (0,30). Une
commande jouée dans 25 % des mouvements garde sa lecture (A dans les vies
VML), une commande à 10 % la perd (A à l'étape B).

## Hypothèse

**En ajoutant une commande préférée à la fois et en répétant les
précédentes, le LLM ajusté apprend le code entier qui relie la marque à
son corps.**

## Étapes

Deux étapes, dans cet ordre, chacune dans un build, sur les vies de VML
(mêmes graines, mêmes corps, mêmes lieux inspectés) où la commande
préférée (jouée dans 70 % des mouvements, les autres dans 10 % chacune)
est tirée pour chaque vie : **la nouvelle commande est préférée dans la
moitié des vies, les commandes déjà lues se partagent l'autre moitié**.

- **Étape C** (régime VMLABCC) : préférée A, B, C ou C (un quart, un
  quart, une moitié) ; C dans environ 40 % des mouvements, A et B dans
  25 %, D dans 10 %. Reprend `run-15-rehearsal/adapters-VMLAB-3600`.
- **Étape D** (régime VMLABCDDD) : préférée A, B, C, D, D ou D ; D dans
  environ 40 % des mouvements, A, B et C dans 20 %. Reprend l'adaptateur
  final de l'étape C.

Chaque étape : 750 itérations, mêmes réglages (LoRA de rang 8 sur 16
couches, lots de 4, 1 024 tokens, taux 1e-4, perte pondérée, 1 500 vies,
graine 17, mlx-lm 0.31.3 ; poids repris, pas l'état de l'optimiseur) ;
test de lecture après 250, 500 et 750 itérations.

**Règle d'arrêt** : l'étape D n'est lancée que si l'étape C a satisfait
son critère ; si une étape échoue, le plan s'arrête et le résultat est
publié tel quel.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **K-C** | Étape C : C est lue, A et B restent lues | au dernier point : P1(A), P1(B) et P1(C) ≥ 0,6. |
| **K-D** | Étape D : le code entier est lu | au dernier point : P1(A), P1(B), P1(C) et P1(D) ≥ 0,6, et P1 − P0 ≥ 0,3 (critère L1). |
| **K-S** | Il cherche sa marque | test d'enquête sur l'adaptateur final de l'étape D, critère d'I1, dans un build suivant. |
| **K-F** | F ne la cherche pas | I2 de la deuxième exécution (0,06). |

**Critère global : K-C, K-D, K-S et K-F.** Contrôle de validité : masse
≥ 0,5 sur les chiffres à chaque point (et sur les symboles pour
l'enquête).

## Ce que le résultat dira

Si le plan aboutit, un LLM ajusté a appris le code entier qui relie la
trace de la cause de ses mouvements à son corps, par une enfance qui
ajoute une habitude à la fois en répétant les anciennes, et il cherche
cette trace : c'est le modèle de soi actif des petits modèles, entier,
dans un LLM. Le test suivant, à pré-enregistrer, sera de le faire agir
d'après ce qu'il sait de lui-même et de mesurer, par ablation de la
marque, ce que ce savoir lui rapporte. Ce test ne mesure pas une
expérience vécue.

## Exécution

Code : `research/llm_lora_body.py` (régimes VMLABCC, VMLABCDDD) ;
verdicts par `python -m research.llm_mark_reading stage` (étape C :
`--preferred C --learned A B` ; étape D : `--preferred D --learned A B C`,
avec L1 lu dans la sortie) ; workflow `menia-lora-stage-mac`, lancé par
le relais ; artefacts dans `artifacts/llm-lora-mac/run-16-rehearsal-c` et
`run-17-rehearsal-d` ; verdicts vérifiés en CI ; résultats dans
`docs/LLM_INQUIRY_RESULTS.md`.

## Précisions fixées avant l'exécution

Écrites avec le code, avant tout lancement. Dans les 1 500 vies
d'entraînement : VMLABCC joue A dans 0,25 des mouvements, B 0,26, C 0,39,
D 0,10 ; VMLABCDDD joue A 0,21, B 0,20, C 0,21, D 0,39. Les vies VMLAB et
VMLB exportées restent identiques octet pour octet à celles de
`run-15-rehearsal` et `run-14-stage-b`. Étape C : `STAGE_REGIME=VMLABCC`,
`STAGE_START=artifacts/llm-lora-mac/run-15-rehearsal/adapters-VMLAB-3600`,
`STAGE_BASE=3600`, points 3 850, 4 100 et 4 350. Étape D :
`STAGE_REGIME=VMLABCDDD`, départ `run-16-rehearsal-c/adapters-VMLABCC-4350`,
`STAGE_BASE=4350`, points 4 600, 4 850 et 5 100.
