# Protocole pré-enregistré — Une enfance par étapes pour le code entier du corps

Rédigé le 23 septembre 2026, après l'extension aux quatre commandes
(`docs/LLM_INQUIRY_RESULTS.md`), **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## Ce qui a été vu

Qwen3-0.6B ajusté lit la marque de son corps pour la commande qu'on lui a
fait préférer (A : 0,89 sur la case impliquée, lecteur parfait 0,85,
hasard 0,25) et la cherche (48 vies sur 48). Sans commande préférée, il
n'apprend pas à la lire (VML : 0,25), et une fois A lue, 750 itérations sur
des vies à commandes égales ne l'étendent pas aux autres (0,26) : dans
l'Atelier, ni le symbole seul ni la commande seule ne disent rien du
déplacement, et seule une commande fréquente donne au gradient un appui.

## Pourquoi maintenant

Faire agir le LLM d'après ce qu'il sait de lui-même ne peut rien montrer
tant qu'il ne lit la marque que pour une commande sur quatre : dans ce
monde, inspecter coûte un tour, et savoir où mène A seulement ne rapporte
pas plus de points par tour que bouger sans regarder (environ 0,23 contre
0,25). Un lecteur du code entier en rapporterait près de deux fois plus.
Le code entier vient donc d'abord.

## Hypothèse

**La commande préférée qui a fait naître la lecture de A la fera naître
pour chaque autre commande, l'une après l'autre**, et plus vite, puisque le
modèle dispose déjà d'une représentation du corps tirée du symbole.

## Étapes

Trois étapes, dans cet ordre, chacune dans un build : **B, puis C, puis D**
préférée. À l'étape X, les vies sont celles de VML (mêmes graines, mêmes
corps, mêmes lieux inspectés), la commande étant X avec une probabilité
0,7 et chacune des trois autres avec 0,1 (régimes VMLB, VMLC, VMLD, tirés
comme VMLA). Chaque étape reprend l'adaptateur final de la précédente
(l'étape B reprend `run-12-full/adapters-FULL-2100`) et l'ajuste **750
itérations**, mêmes réglages que VMLA (LoRA de rang 8 sur 16 couches, lots
de 4, 1 024 tokens, taux 1e-4, perte pondérée, 1 500 vies, graine 17,
mlx-lm 0.31.3 ; poids repris, pas l'état de l'optimiseur). Test de lecture
après 250, 500 et 750 itérations de chaque étape.

**Règle d'arrêt** : une étape n'est lancée que si la précédente a satisfait
son critère (E1 et E2). Si une étape échoue, le plan s'arrête et le
résultat est publié tel quel.

## Prédictions fixées, pour chaque étape X

| | Prédiction | Critère |
|---|---|---|
| **E1** | La lecture naît pour X | au dernier point de l'étape : P1(X) ≥ 0,6. |
| **E2** | Les commandes déjà lues restent lues | au dernier point : P1 ≥ 0,6 pour chaque commande lue aux étapes précédentes (A à l'étape B ; A et B à l'étape C ; A, B et C à l'étape D). |
| E3 | aucune prédiction | P1 des commandes pas encore préférées, et le premier point où P1(X) ≥ 0,6, publiés. |

**Critère de l'étape : E1 et E2.** Contrôle de validité : masse ≥ 0,5
sur les chiffres à chaque point.

## Critère final, après l'étape D

| | Prédiction | Critère |
|---|---|---|
| **F1** | Le LLM lit le code entier | P1 sur les quatre commandes ≥ 0,6 et P1 − P0 ≥ 0,3 (critère L1 de `docs/LLM_MARK_READING_PROTOCOL.md`). |
| **F2** | Il cherche sa marque | test d'enquête sur l'adaptateur final, critère d'I1, dans un build suivant. |
| **F3** | F ne la cherche pas | I2 de la deuxième exécution (0,06). |

**Critère global du plan : les trois étapes, F1, F2 et F3.**

## Ce que le résultat dira

Si le plan aboutit, le LLM ajusté a appris le code entier qui relie la
marque à son corps, et l'ordre de l'enfance (une commande préférée après
l'autre) en est la condition : l'enfance uniforme n'y arrive pas. Le test
suivant, à pré-enregistrer, sera de le faire agir d'après ce qu'il sait de
lui-même et de mesurer, par ablation de la marque, ce que ce savoir lui
rapporte. Si une étape échoue, la lecture reste liée aux commandes déjà
apprises. Ce test ne mesure pas une expérience vécue.

## Exécution

Code : `research/llm_lora_body.py` (régimes VMLB à VMLD) et
`research/llm_mark_reading.py` (P1 par commande, verdicts d'étape) ;
workflow Codemagic `menia-lora-stage-mac`, lancé par le relais avec le
régime et l'adaptateur de départ en variables ; artefacts dans
`artifacts/llm-lora-mac/run-14-stage-b`, `run-15-stage-c`,
`run-16-stage-d` ; verdicts vérifiés en CI ; résultats dans
`docs/LLM_INQUIRY_RESULTS.md`.

## Précisions fixées avant l'exécution

Écrites avec le code, avant tout lancement. Les régimes VMLB à VMLD tirent
leurs commandes comme VMLA (générateur à part, la commande de VML tirée
quand même) : mêmes vies, mêmes corps, mêmes lieux inspectés que VML ; la
commande préférée est dans 0,70 des mouvements, chacune des autres dans
0,10. Les vies VML et VMLA exportées restent identiques octet pour octet à
celles de `run-8-vml` et de `run-9-vmla`. Un seul workflow sert aux trois
étapes, le régime (`STAGE_REGIME`), l'adaptateur de départ
(`STAGE_START`) et le nombre d'itérations déjà faites (`STAGE_BASE`) étant
passés en variables par le relais. Points de l'étape B : 2 350, 2 600 et
2 850 itérations au total. Verdicts :
`python -m research.llm_mark_reading stage --preferred B --learned A
--reading 2350=… 2600=… 2850=…`.
