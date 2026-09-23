# Protocole pré-enregistré — Un LLM peut-il apprendre à lire la marque de son corps ?

Rédigé le 23 septembre 2026, après l'ajustement pondéré (VMW,
`docs/LLM_INQUIRY_RESULTS.md`), **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## Ce qui a été vu

Ajusté par LoRA sur des vies de l'Atelier, Qwen3-0.6B apprend l'effet de
ses commandes à partir de ses mouvements (jusqu'à 0,93 sur les commandes
jamais essayées avec la perte pondérée, VMW), mais **ne cherche pas la
marque de son corps** (VM, VMI, VMW : lieu 1 préféré dans 0,40, 0,29 et
0,44 des vies). Une analyse exploratoire, non pré-enregistrée et sur 20 cas
par modèle, en donne une raison : **il ne lit pas la marque**. Quand il l'a
vue avant son premier mouvement, il ne prédit pas mieux ce mouvement (0,26
sur la bonne case ; hasard 0,25). Dans toutes ces enfances, la marque ne
sert qu'au premier mouvement d'une vie ; ensuite, les mouvements eux-mêmes
disent le corps.

## Hypothèses

1. **Les adaptateurs publiés ne lisent pas la marque** : l'analyse
   exploratoire se confirme sur un test fait pour cela.
2. **Si la marque est la seule information sur le corps à chaque
   mouvement, le LLM ajusté apprend à la lire.**
3. **S'il la lit, il la cherche** : au début d'une vie, ses propres
   prédictions font du lieu de la marque le lieu à inspecter.

## Régime VML (« lire »)

Chaque vie compte 24 tours : **12 paires inspection puis mouvement**. Le
lieu inspecté est le lieu 1 (la marque) avec une probabilité 1/2, sinon
l'un des trois autres, au hasard. La commande est tirée au hasard. **Le
corps est tiré à nouveau, au hasard, après chaque mouvement** : seul un
symbole du lieu 1 lu juste avant un mouvement dit le corps qui le fera ;
les mouvements passés n'en disent rien. Même Atelier sinon (marque juste
dans 80 % des cas, sinon au hasard ; même format de texte, mêmes cibles).

**Ajustement : celui de VMW**, seule la vie diffère : Qwen3-0.6B, LoRA de
rang 8 sur 16 couches, 1 500 vies (graine 17), 600 itérations, lots de 4,
1 024 tokens, taux 1e-4, perte sur tout le texte, chaque chiffre de case
d'arrivée pesant 20 fois plus (`research/llm_weighted_lora.py`),
mlx-lm 0.31.3.

## Test de lecture

Pour chaque adaptateur, 256 invites au début d'une vie : l'en-tête, puis
« Tour 1 : inspection du lieu L, symbole S. Position p, cible g. », puis
« Tour 2 : commande C, de la case p à la case », avec L de 1 à 4, S parmi
les quatre symboles, C parmi les quatre commandes, p dans {0, 2, 4, 6} et
g = p + 4. On lit la distribution du modèle sur les quatre cases
d'arrivée possibles (renormalisée) et la probabilité qu'il donne à la case
qu'**impliquerait** le symbole S s'il disait le corps. Un modèle qui ne
tient pas compte du symbole donne exactement 0,25 en moyenne (pour p et C
fixés, les quatre symboles impliquent les quatre cases). **P1** : moyenne
sur L = 1 ; **P0** : moyenne sur L = 2 à 4. Un lecteur parfait donnerait
P1 ≈ 0,85 et P0 = 0,25.

Adaptateurs testés, dans le même build : F et VM (`run-3`), VMI
(`run-5-vmi`), VMW (`run-7-vmw`), tels que publiés, puis VML.

## Mesures sur VML

Le test de lecture ; le test d'enquête (`docs/LLM_INQUIRY_PROTOCOL.md`,
lecture corrigée des symboles), mêmes 96 états ; puis les cellules du
corps ajusté, à titre descriptif.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **L0** | Les adaptateurs publiés ne lisent pas la marque | F, VM, VMI et VMW : P1 ≤ 0,35 chacun. |
| **L1** | VML lit la marque | P1 ≥ 0,6 et P1 − P0 ≥ 0,3. |
| **L2** | VML cherche la marque | test d'enquête, pas 0 du jeu R : lieu 1 préféré dans ≥ 0,7 des vies, et gain moyen du lieu 1 ≥ 2 × la moyenne des gains des lieux 2 à 4 (critère d'I1). |
| **L3** | F ne la cherche pas | I2 de la deuxième exécution du test d'enquête (0,06), repris tel quel. |

**Critère global : L1, L2 et L3.** L0 est jugé à part : il confirme ou
infirme l'analyse exploratoire. Contrôles de validité, lus en premier :
masse moyenne ≥ 0,5 sur les chiffres dans le test de lecture (pour chaque
adaptateur) ; masse ≥ 0,5 sur les chiffres et sur les symboles dans le
test d'enquête. Un adaptateur invalide n'est pas jugé. Les cellules du
corps ajusté de VML sont publiées sans prédiction : dans ce régime, les
mouvements passés ne disent rien du corps, et la copie des commandes vues
n'a pas de raison de réussir.

## Ce que le résultat dira

- **L1 et L2 passent** : un LLM ajusté apprend à lire la trace de sa cause
  et à la chercher quand son enfance la rend utile à chaque mouvement ;
  l'échec de VMI et de VMW tenait à la rareté de son usage, pas à une
  impossibilité. C'est le modèle de soi actif des petits modèles, dans un
  modèle de langage, sous une condition précise.
- **L1 passe, L2 échoue** : il lit la marque sans que ses prédictions en
  fassent le lieu à inspecter ; lire et chercher sont dissociés.
- **L1 échoue** : même quand la marque est la seule information à chaque
  mouvement, un ajustement LoRA court de Qwen3-0.6B n'apprend pas ce code
  arbitraire ; la limite tient à l'ajustement, pas aux données.

VML est un monde extrême : un corps qui change à chaque mouvement n'a pas
de structure stable à apprendre. Ce test porte sur la lecture et la
recherche de la marque, pas sur la structure du corps. Il ne mesure pas une
expérience vécue.

## Exécution

Code : `research/llm_lora_body.py` (régime VML) et
`research/llm_mark_reading.py` (test de lecture, verdicts) ; workflow
Codemagic `menia-lora-reading-mac`, lancé par le relais : d'abord le test
de lecture des quatre adaptateurs publiés, puis l'ajustement VML, son test
de lecture, le test d'enquête et les cellules. Artefacts dans
`artifacts/llm-lora-mac/run-8-vml` ; verdicts recalculés depuis les lignes
et vérifiés en CI ; résultats dans `docs/LLM_INQUIRY_RESULTS.md`.

## Précisions fixées avant l'exécution

Écrites avec le code, avant tout lancement. Dans les 1 500 vies VML
d'entraînement, 5,95 mouvements par vie suivent une lecture du lieu 1 (0,77
dans VMI, le premier seulement) ; la case qu'implique la marque est la case
d'arrivée dans 0,85 de ces mouvements, et dans 0,25 après un autre lieu.
Les corps tirés à nouveau viennent d'un générateur à part ; les vies VMI
exportées restent identiques octet pour octet à celles de `run-7-vmw`. Le
découpage autour des chiffres d'arrivée garde partout le pré-tokeniseur
de Qwen (12 chiffres par vie). Dans le build, le test de lecture des quatre
adaptateurs publiés passe en premier, pour être fait même si
l'ajustement échouait ; les cellules du corps ajusté passent en dernier.
Le relais retire le préfixe `llm-lora` : les tests de lecture arrivent dans
`run-8-vml/reading-<adaptateur>`, l'enquête dans `run-8-vml/inquiry-VML`.
