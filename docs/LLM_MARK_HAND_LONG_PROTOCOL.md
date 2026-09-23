# Protocole pré-enregistré — L'appui de VMLA grandit-il avec l'entraînement ?

Rédigé le 23 septembre 2026, après le régime VMLA
(`docs/LLM_INQUIRY_RESULTS.md`), **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## Ce qui a été vu

Avec une commande préférée (A dans 70 % des mouvements), Qwen3-0.6B ajusté
600 itérations commence à lire la marque pour cette commande : il donne
0,31 à la case qu'implique la marque (0,25 au hasard ; 0,23 pour les autres
commandes), et le lieu de la marque devient le lieu préféré dans 0,56 des
vies (0,15 sans commande préférée). Sa perte de validation baissait encore
(0,428 à l'itération 100, 0,402 à l'itération 600). Trois enfances et deux
objectifs ont échoué avec le même ajustement court : reste à savoir si
l'échec tient à la durée.

## Hypothèse

**L'appui trouvé par VMLA grandit avec l'entraînement : prolongé, le LLM
ajusté apprend à lire la marque pour sa commande préférée.**

## Prolongation

L'adaptateur publié de VMLA (`run-9-vmla/adapters-VMLA`, 600 itérations)
est repris et ajusté **750 itérations de plus** (1 350 au total), sur les
mêmes 1 500 vies VMLA (graine 17), avec les mêmes réglages : LoRA de rang 8
sur 16 couches, lots de 4, 1 024 tokens, taux 1e-4, perte pondérée
(`research/llm_weighted_lora.py`), mlx-lm 0.31.3. L'état de l'optimiseur
n'est pas repris (mlx-lm ne reprend que les poids). Points de contrôle
après 250, 500 et 750 itérations de plus (850, 1 100 et 1 350 au total).

## Mesures

Le test de lecture (`docs/LLM_MARK_READING_PROTOCOL.md`), P1(A) et P1(BCD)
compris, sur chaque point de contrôle. Pas de test d'enquête dans ce
build (il ne tiendrait pas dans la durée d'un build) : si la lecture
s'installe, il fera l'objet d'une mesure suivante.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **G1** | Prolongé, VMLA lit la marque pour A | au point 1 350 : P1(A) ≥ 0,6 et P1(A) − P0 ≥ 0,3 (critère de H1). |
| **G2** | L'appui grandit | P1(A) à 1 350 ≥ P1(A) à 600 (0,312) + 0,1. |
| **G3** | Aucune prédiction | P1(BCD) et P1 sur les quatre commandes à chaque point, publiés. |

**Critère global : G1.** Contrôle de validité, lu en premier : masse
moyenne ≥ 0,5 sur les chiffres à chaque point.

## Ce que le résultat dira

- **G1 passe** : la lecture de la marque est à la portée d'un LLM ajusté,
  pourvu que l'enfance lui donne un appui et que l'ajustement dure ; il
  faudra alors mesurer s'il la cherche.
- **G2 passe, G1 échoue** : la lecture progresse mais lentement ; la durée
  nécessaire se lit sur la courbe.
- **G2 échoue** : l'appui ne grandit pas ; avec LoRA de rang 8 sur ce
  modèle, la lecture de la marque n'est pas atteinte, et la suite
  demanderait un autre ajustement ou un autre modèle.

Ce test ne mesure pas une expérience vécue.

## Exécution

Workflow Codemagic `menia-lora-hand-long-mac`, lancé par le relais ;
artefacts dans `artifacts/llm-lora-mac/run-10-vmla-long` ; verdicts par
`research/llm_mark_reading.py`, vérifiés en CI ; résultats dans
`docs/LLM_INQUIRY_RESULTS.md`.

## Précisions fixées avant l'exécution

Écrites avec le code, avant tout lancement. Le build réexporte les vies
VMLA et vérifie qu'elles sont identiques octet pour octet à celles de
`run-9-vmla` avant d'ajuster. La reprise charge les poids publiés
(`--resume-adapter-file`) ; le compteur d'itérations repart de 1, si bien
que les points 250, 500 et 750 de ce build sont les points 850, 1 100 et
1 350 au total. La validation a lieu toutes les 250 itérations. Les tests
de lecture arrivent dans `run-10-vmla-long/reading-850`, `reading-1100` et
`reading-1350` ; seul l'adaptateur final est gardé. Verdicts :
`python -m research.llm_mark_reading long --baseline
artifacts/llm-lora-mac/run-9-vmla/reading-VMLA --reading 850=… 1100=…
1350=…`.
