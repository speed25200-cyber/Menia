# Protocole pré-enregistré — Répliquer la recette sur deux nouvelles graines

Rédigé le 24 septembre 2026, après le test d'action sur le lecteur du code
entier (`docs/LLM_INQUIRY_RESULTS.md`) et la vérification bibliographique
(`docs/LITERATURE_CHECK_2026-09-24.md`), **avant l'écriture du code et
toute exécution** ; l'heure est celle du commit. Seuils fixés, un échec est
un résultat.

## Pourquoi

Avec la graine 17, Qwen3-0.6B ajusté par LoRA a appris le code entier qui
relie la marque à son corps, la cherche (48 vies sur 48), s'en sert (6,75
points par vie contre 1,56 marque brouillée) et estime justement ses
chances (0,842 pour 0,848). C'est **une seule graine**, obtenue par un
chemin qui compte deux détours échoués (l'extension sans appui, puis B
apprise seule). Pour parler d'un résultat vérifié, il faut que **la
recette** en tire, sans détour, sur des graines neuves.

## La recette

Pour chaque graine s ∈ {23, 29} (graine des vies exportées **et** de
l'ajustement : autres vies, autre initialisation du LoRA, autre ordre des
lots), cinq ajustements enchaînés, chacun reprenant l'adaptateur du
précédent, mêmes réglages que pour la graine 17 (Qwen3-0.6B, LoRA de rang
8 sur 16 couches, lots de 4, 1 024 tokens, taux 1e-4, perte pondérée,
1 500 vies, mlx-lm 0.31.3) :

1. **Appui** : VMLA (A préférée), 600 itérations depuis zéro ;
2. **Appui prolongé** : VMLA, 750 itérations (1 350 au total) ;
3. **Ajout de B, répété** : VMLAB (A ou B préférée selon la vie), 750
   itérations (2 100) ;
4. **Ajout de C, répété** : VMLABCC (C préférée dans la moitié des vies),
   750 itérations (2 850) ;
5. **Consolidation** : VML (commandes à parts égales), 750 itérations
   (3 600).

Puis, sur l'adaptateur final : test de lecture, test d'enquête, test
d'action (mêmes 48 vies, vraie marque et marque brouillée, même contrôle
VML de `run-8-vml`). Le chemin de la graine 17 passait par deux étapes de
plus (une extension sans appui, B seule) ; la recette les omet.

## Prédictions fixées, pour chaque graine

| | Prédiction | Critère, sur l'adaptateur final |
|---|---|---|
| **P-lecture** | Il lit le code entier | P1(A), P1(B), P1(C), P1(D) ≥ 0,6 ; P0 ≤ 0,3 ; P1 − P0 ≥ 0,3. |
| **P-enquête** | Il cherche sa marque | critère d'I1 (lieu 1 préféré dans ≥ 0,7 des vies, gain ≥ 2 × celui des autres lieux). |
| **P-action** | Il s'en sert | U1 et U2 de `docs/LLM_MARK_ACTION_PROTOCOL.md` (≥ 2,5 points par vie, borne basse > 0). |

**Critère global : les trois prédictions, pour les deux graines.** Aucune
étape intermédiaire n'arrête la recette : chaque étape est publiée (test
de lecture à trois points par étape) et la recette va jusqu'au bout ; si
la lecture d'une étape échoue, cela est dit, et le verdict final est lu
sur l'adaptateur final. Contrôles de validité : masse ≥ 0,5 sur les
chiffres (et sur les symboles pour l'enquête).

## Ce que le résultat dira

Si le critère global passe, la recette (un appui, des ajouts répétés, une
consolidation) donne à un LLM ajusté un modèle de soi actif et utile,
indépendamment de la graine : le résultat de la graine 17 est répliqué.
Si une graine échoue, le résultat dépend de la graine, et l'on dira à
quelle étape. Ce test ne mesure pas une expérience vécue.

## Exécution

Workflow `menia-lora-stage-mac`, rendu paramétrable par la graine
(`LORA_SEED`) et capable de partir de zéro (`STAGE_START=none`), puis
`menia-lora-action-mac` ; lancés par le relais, une étape à la fois ;
artefacts dans `artifacts/llm-lora-mac/rep-s23-1` à `rep-s23-6` et
`rep-s29-1` à `rep-s29-6` ; verdicts par `research/llm_mark_reading.py`,
`research/llm_inquiry_verdicts.py` et `research/llm_mark_action.py`,
vérifiés en CI ; résultats dans `docs/LLM_INQUIRY_RESULTS.md`.

## Précisions fixées avant l'exécution

Écrites avec le code, avant tout lancement. `menia-lora-stage-mac` prend
la graine en variable (`LORA_SEED`, 17 par défaut : les étapes déjà
publiées sont inchangées) pour l'export des vies et pour l'ajustement, et
part de zéro quand `STAGE_START=none`. Étape 1 : `STAGE_REGIME=VMLA`,
`STAGE_START=none`, `STAGE_BASE=0`, `LORA_ITERS=600` (points 250, 500 et
600). Étapes 2 à 5 : 750 itérations chacune, départ l'adaptateur final de
l'étape précédente de la même graine (points 1 350, 2 100, 2 850, 3 600).
Étape 6 : `menia-lora-action-mac` avec `CODE_ADAPTER` l'adaptateur final.
Les deux graines passent dans la même demande du relais à chaque étape.
Dans les vies VMLA des graines 23 et 29, A est jouée dans 0,70 des
mouvements, comme pour la graine 17.

## Amendement 1, 24 septembre 2026 (heure du commit) : panne matérielle et relance

Écrit après l'échec du build de l'étape 5 pour la graine 23, **avant toute
relance**. Le build s'est arrêté à l'itération 270 sur une erreur du
processeur graphique du Mac (« [METAL] Command buffer execution failed:
Caused GPU Hang Error »), sur une machine deux fois plus rapide que les
précédentes (0,35 itération par seconde) : une panne matérielle, pas un
résultat. Ses traces sont gardées dans
`artifacts/llm-lora-mac/rep-s23-5-echec-gpu`. **Règle fixée ici** : un
build arrêté par une panne matérielle est relancé une fois, à
l'identique ; s'il échoue de nouveau, l'étape est déclarée non mesurée
pour cette graine, et le critère global ne peut plus être satisfait.
