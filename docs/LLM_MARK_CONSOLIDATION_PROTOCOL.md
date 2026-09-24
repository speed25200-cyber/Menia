# Protocole pré-enregistré — Consolider trois lectures en un code entier

Rédigé le 24 septembre 2026, après l'échec de l'étape D et le test
d'action (`docs/LLM_INQUIRY_RESULTS.md`), **avant l'écriture du code et
toute exécution** ; l'heure est celle du commit. Seuils fixés, un échec
est un résultat.

## Ce qui a été vu

Le LLM ajusté de l'étape C lit la marque de son corps pour A, B et C (0,87
à 0,88), s'en sert pour agir (6,73 points par vie contre 1,50 sans la
marque) et la cherche ; pour D, sa meilleure case suit déjà la carte de D
dans 0,62 des invites (P1(D) 0,28), sans que D ait jamais été préférée.
Ajouter D comme commande préférée (D dans 39 % des mouvements, A, B et C
dans 20 %) a tout effacé. Deux constats antérieurs bornent ce qu'une
commande doit recevoir pour garder sa lecture : A l'a gardée à 25 % des
mouvements (vies VML après VMLA), l'a perdue à 10 % (étape B) ; C est née à
10 % quand A et B étaient déjà lues.

## Hypothèse

**Sur des vies où les quatre commandes sont également fréquentes, les
trois lectures tiennent et celle de D, déjà naissante, s'achève** : le code
entier se consolide sans commande préférée.

## Ajustement

L'adaptateur de l'étape C (`run-16-rehearsal-c/adapters-VMLABCC-4350`)
est repris et ajusté **750 itérations sur les vies VML** (commandes à parts
égales ; les 1 500 vies de `run-8-vml`, graine 17), mêmes réglages : LoRA
de rang 8 sur 16 couches, lots de 4, 1 024 tokens, taux 1e-4, perte
pondérée, mlx-lm 0.31.3 ; poids repris, pas l'état de l'optimiseur. Test
de lecture après 250, 500 et 750 itérations (4 600, 4 850, 5 100 au
total).

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **Q1** | La lecture de D s'achève | au dernier point : P1(D) ≥ 0,6. |
| **Q2** | A, B et C restent lues | au dernier point : P1(A), P1(B), P1(C) ≥ 0,6. |
| **Q3** | La lecture reste attachée au lieu de la marque | au dernier point : P0 ≤ 0,3 et P1 − P0 ≥ 0,3 (critère L1). |

**Critère global : Q1, Q2 et Q3.** Contrôle de validité : masse ≥ 0,5
sur les chiffres à chaque point.

## Ce que le résultat dira

Si le critère passe, le LLM ajusté a le code entier qui relie la marque à
son corps, par une enfance en trois temps : une commande préférée, des
ajouts répétés, puis une consolidation sans préférence. Le test d'enquête
et le test d'action seront alors refaits sur cet adaptateur (à
pré-enregistrer). Si Q1 échoue, la quatrième lecture ne naît pas sans
appui ; si Q2 échoue, même une consolidation à parts égales efface les
lectures acquises. Ce test ne mesure pas une expérience vécue.

## Exécution

Workflow `menia-lora-stage-mac` (`STAGE_REGIME=VML`,
`STAGE_START=artifacts/llm-lora-mac/run-16-rehearsal-c/adapters-VMLABCC-4350`,
`STAGE_BASE=4350`), lancé par le relais ; artefacts dans
`artifacts/llm-lora-mac/run-19-consolidation` ; verdicts par
`python -m research.llm_mark_reading stage --preferred D --learned A B C`
(E1 = Q1, E2 = Q2, L1 et P0 lus dans la sortie), vérifiés en CI ;
résultats dans `docs/LLM_INQUIRY_RESULTS.md`.
