# Protocole pré-enregistré — Du code d'une commande au code entier du corps

Rédigé le 23 septembre 2026, après le test d'enquête sur VMLA-1350
(`docs/LLM_INQUIRY_RESULTS.md`), **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## Ce qui a été vu

Sans appui, Qwen3-0.6B ajusté n'apprend pas à lire la marque de son corps
(VML : 0,25 sur la case impliquée, le hasard) : ni le symbole seul ni la
commande seule ne disent rien du déplacement. Avec une commande préférée
(VMLA) et 1 350 itérations, il la lit pour cette commande (0,81) et la
cherche (lieu de la marque préféré dans 48 vies sur 48), mais reste au
hasard pour les trois autres commandes (0,25).

## Hypothèse

**Une fois la marque lue pour une commande, la lecture s'étend aux autres
quand l'enfance les rend aussi fréquentes** : le modèle dispose déjà d'une
représentation du corps tirée du symbole, et apprendre ce qu'elle implique
pour les autres commandes ne demande plus de trouver une interaction pure.
C'est une enfance par étapes : d'abord une main préférée, puis les deux.

## Ajustement

L'adaptateur publié VMLA-1350 (`run-10-vmla-long`) est repris et ajusté
**750 itérations de plus sur les vies VML** (commandes à parts égales ;
les 1 500 vies de `run-8-vml`, graine 17), mêmes réglages : LoRA de rang 8
sur 16 couches, lots de 4, 1 024 tokens, taux 1e-4, perte pondérée,
mlx-lm 0.31.3 ; poids repris, pas l'état de l'optimiseur. Points de
contrôle après 250, 500 et 750 itérations de plus.

## Mesures

Le test de lecture, P1(A) et P1(BCD), sur chaque point de contrôle ; puis
le test d'enquête sur l'adaptateur final.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **C1** | La lecture s'étend aux autres commandes | au dernier point : P1(BCD) ≥ 0,6. |
| **C2** | La lecture de A se maintient | au dernier point : P1(A) ≥ 0,6. |
| **C3** | Le modèle cherche sa marque | test d'enquête : critère d'I1. |
| **C4** | F ne la cherche pas | I2 de la deuxième exécution (0,06), repris tel quel. |

**Critère global : C1, C2, C3 et C4.** Contrôles de validité, lus en
premier : masse ≥ 0,5 sur les chiffres (lecture) ; sur les chiffres et sur
les symboles (enquête). P1 sur les quatre commandes et P0 publiés à chaque
point.

## Ce que le résultat dira

Si C1 et C2 passent, le LLM ajusté a appris le code entier qui relie la
marque à son corps, par une enfance en deux étapes, là où l'enfance
uniforme (VML) échouait : l'ordre de l'enfance décide de ce qu'il apprend.
Si C1 échoue, la lecture reste liée à la commande qui l'a fait naître.
Ce test ne mesure pas une expérience vécue.

## Exécution

Workflow Codemagic `menia-lora-full-mac`, lancé par le relais ; artefacts
dans `artifacts/llm-lora-mac/run-12-full` ; verdicts par
`research/llm_mark_reading.py`, vérifiés en CI ; résultats dans
`docs/LLM_INQUIRY_RESULTS.md`.
