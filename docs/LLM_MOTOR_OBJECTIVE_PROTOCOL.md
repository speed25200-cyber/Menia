# Protocole pré-enregistré — Ajuster le LLM à prédire l'effet de ses commandes

Rédigé le 23 septembre 2026, après le régime VMI
(`docs/LLM_INQUIRY_RESULTS.md`), **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## Ce qui a été vu

Ajusté sur le prochain mot de vies entières, Qwen3-0.6B apprend l'effet de
ses commandes à partir de ses mouvements (0,79 sur les commandes jamais
essayées) mais **n'apprend pas à lire la marque de son corps**, même quand
une enfance où l'on regarde avant d'agir en fait la seule information avant
le premier mouvement (régime VMI : lieu de la marque préféré dans 0,29 des
vies). Les petits modèles qui y arrivaient apprenaient autrement : une tête
dédiée prédisait, à chaque pas, le déplacement que produirait la commande,
et rien d'autre. Dans l'ajustement sur le prochain mot, les chiffres des
cases d'arrivée ne sont qu'une petite part de la perte, et la marque n'en
explique qu'une plus petite encore.

## Hypothèse

**Si l'ajustement porte seulement sur la prédiction de l'effet de ses
propres commandes, le LLM apprend à lire la marque de son corps quand elle
est la seule information, et à la chercher.** L'enfance ne suffit pas ;
l'objectif d'apprentissage compte aussi.

## Ajustement

Mêmes 1 500 vies VMI que `run-5-vmi` (graine 17), même format de texte,
Qwen3-0.6B, LoRA de rang 8 sur 16 couches, lots de 4, 1 024 tokens, taux
1e-4, mlx-lm 0.31.3. Une seule différence d'objectif : **un exemple par
mouvement**, le début de la vie jusqu'à « …, de la case p à la case »
comme entrée, le chiffre de la case d'arrivée comme cible, en complétion
brute (sans gabarit de conversation), **la perte ne portant que sur ce
chiffre** — exactement la question que posent les tests. 1 000 itérations,
pour que le nombre de cibles vues reste du même ordre que dans les
ajustements précédents.

## Mesures

Dans le même build, sur cet adaptateur (« VMM ») : le test d'enquête
(lecture corrigée des symboles), mêmes 96 états, puis les cellules du corps
ajusté.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **K1** | Le LLM ajusté à prédire l'effet de ses commandes cherche sa cause | VMM, pas 0 du jeu R : lieu 1 préféré dans ≥ 0,7 des vies, et gain moyen du lieu 1 ≥ 2 × la moyenne des gains des lieux 2 à 4 (critère d'I1). |
| **K2** | Le LLM élevé sur un corps fixe ne la cherche pas | I2 de la deuxième exécution du test d'enquête (F : 0,06), repris tel quel. |
| **K3** | L'enquête se rallume après un changement | critère d'I3 ; **aucune prédiction**. |

**Critère global : K1 et K2.** Contrôle de validité, lu en premier : masse
moyenne ≥ 0,5 sur les chiffres et sur les symboles. La masse sur les
symboles n'est pas entraînée par cet objectif : si elle tombe sous 0,5, le
test est invalide et le dit. Les cellules du corps ajusté sont publiées
sans prédiction.

## Ce que le résultat dira

Si K1 passe, la disposition à chercher la cause de son corps est à la
portée d'un LLM ajusté, pourvu qu'on l'ajuste à prédire l'effet de ses
propres actions et que son enfance rende la marque utile : c'est le modèle
de soi actif des petits modèles, porté dans un modèle de langage. Si K1
échoue, ni l'enfance ni l'objectif ne suffisent avec cet ajustement court.
Ce test ne mesure pas une expérience vécue.

## Exécution

Code : `research/llm_lora_body.py` (export des exemples de mouvement) et
`research/llm_motor_lora.py` (mlx-lm en complétion brute, perte sur la
cible) ; workflow Codemagic `menia-lora-motor-mac`, lancé par le relais,
artefacts dans `artifacts/llm-lora-mac/run-6-vmm`, verdicts par
`research/llm_inquiry_verdicts.py`, résultats dans
`docs/LLM_INQUIRY_RESULTS.md`.
