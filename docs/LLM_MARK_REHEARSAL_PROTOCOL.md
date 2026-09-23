# Protocole pré-enregistré — Garder deux lectures en les répétant

Rédigé le 24 septembre 2026, après l'étape B de l'enfance par étapes
(`docs/LLM_INQUIRY_RESULTS.md`), **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## Ce qui a été vu

Le LLM ajusté qui lisait la marque pour A (0,89), repris sur des vies où B
est préférée, apprend à la lire pour B (0,79) mais oublie A (0,25) ; entre
les deux, il applique à A le déplacement de B. Il ne garde qu'une
association entre la marque et le déplacement de la commande fréquente du
moment. Deux explications restent possibles : **l'interférence** (les vies
de l'étape B ne montrent presque plus A préférée, et rien ne retient
l'ancienne association) ou **une limite** du modèle ajusté, qui ne saurait
tenir qu'une association à la fois.

## Hypothèse

**Si les deux enfances sont répétées ensemble, le LLM ajusté garde les
deux lectures** : l'effacement venait de l'interférence.

## Régime VMLAB

Les vies de VML (mêmes graines, mêmes corps, mêmes lieux inspectés), dont
la commande préférée est tirée **pour chaque vie** : A ou B avec une
probabilité 1/2 ; dans la vie, la commande préférée est jouée dans 70 % des
mouvements, chacune des trois autres dans 10 %. Le générateur des
commandes est celui de VMLA à VMLD, ce qui laisse ces régimes inchangés.

## Ajustement

L'adaptateur final de l'étape B (`run-14-stage-b/adapters-VMLB-2850`, qui
lit B) est repris et ajusté **750 itérations** sur 1 500 vies VMLAB
(graine 17), mêmes réglages : LoRA de rang 8 sur 16 couches, lots de 4,
1 024 tokens, taux 1e-4, perte pondérée, mlx-lm 0.31.3 ; poids repris,
pas l'état de l'optimiseur. Test de lecture après 250, 500 et 750
itérations (3 100, 3 350 et 3 600 au total).

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **R1** | Il lit la marque pour A et pour B | au dernier point : P1(A) ≥ 0,6 et P1(B) ≥ 0,6. |
| R2 | aucune prédiction | P1(C), P1(D), et à chaque point la commande dont le modèle applique le déplacement (constat, comme pour l'étape B). |

**Critère global : R1.** Contrôle de validité : masse ≥ 0,5 sur les
chiffres à chaque point.

## Ce que le résultat dira

Si R1 passe, l'effacement de l'étape B venait de l'interférence : répétées
ensemble, deux associations tiennent, et l'on pourra ajouter C puis D de la
même façon, vers le code entier. Si R1 échoue, le LLM ajusté dans ce
dispositif ne tient qu'une association entre la marque et un déplacement ;
ce qu'il apprend est un indice de son geste habituel, pas une description
de son corps. Ce test ne mesure pas une expérience vécue.

## Exécution

Code : `research/llm_lora_body.py` (régime VMLAB) ; verdicts par
`python -m research.llm_mark_reading stage --preferred A --learned B` ;
workflow `menia-lora-stage-mac` (`STAGE_REGIME=VMLAB`,
`STAGE_START=artifacts/llm-lora-mac/run-14-stage-b/adapters-VMLB-2850`,
`STAGE_BASE=2850`), lancé par le relais ; artefacts dans
`artifacts/llm-lora-mac/run-15-rehearsal` ; verdicts vérifiés en CI ;
résultats dans `docs/LLM_INQUIRY_RESULTS.md`.

## Précisions fixées avant l'exécution

Écrites avec le code, avant tout lancement. Dans les 1 500 vies VMLAB
d'entraînement, A est jouée dans 0,41 des mouvements, B dans 0,39, C et D
dans 0,10 chacune ; les lieux inspectés sont ceux de VML. Les vies VML,
VMLA et VMLB exportées restent identiques octet pour octet à celles de
`run-8-vml`, `run-9-vmla` et `run-14-stage-b`. L'adaptateur final
s'appelle `adapters-VMLAB-3600`. Verdicts :
`python -m research.llm_mark_reading stage --preferred A --learned B
--reading 3100=… 3350=… 3600=…` (E1 y est P1(A) ≥ 0,6 et E2 P1(B) ≥ 0,6 :
leur conjonction est R1).
