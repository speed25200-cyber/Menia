# Protocole pré-enregistré — Une enfance où regarder avant d'agir paie

Rédigé le 23 septembre 2026, après les résultats du corps ajusté
(`docs/ADJUSTED_BODY_RESULTS.md`) et du test d'enquête
(`docs/LLM_INQUIRY_RESULTS.md`), **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## Ce qui a été vu

Ajusté sur des vies à corps variable et changeant (régime VM), Qwen3-0.6B
acquiert une bonne part de la structure de son corps (A3 : 0,79 sur les
commandes jamais essayées) et la révise après un changement, mais **ne
cherche pas la trace de sa cause** : au début d'une vie, il n'attend
presque rien de l'inspection du lieu 1, qui porte la marque de son corps
(I1 : lieu 1 préféré dans 0,40 des vies, gain 1,4 fois celui des autres).
Les petits modèles élevés sur les mêmes vies allaient la lire.

Une explication tient au régime des données. Dans ces vies, les actions
sont tirées au hasard, et **un seul mouvement révèle tout le corps**
(chaque commande donne un déplacement différent selon le corps). La
marque, juste dans 85 % des cas, n'apporte donc presque rien que le
premier mouvement n'apporte mieux : seule la prédiction du premier
mouvement, quand la marque a été lue avant lui (une vie sur cinq), en
dépend. Un modèle de langage ajusté sur le prochain mot apprend l'indice
dominant et néglige l'indice redondant.

## Hypothèse

**Si l'enfance fait regarder avant d'agir, la marque devient la seule
source d'information sur le corps au moment du premier mouvement, et le LLM
ajusté apprend à la chercher.** C'est la thèse du programme — les données
d'enfance décident — mise à l'épreuve sur un modèle de langage.

## Régime VMI

Identique au régime VM (corps tiré par vie, changé dans la moitié des vies
à un pas uniforme entre 6 et 17, mêmes graines de vies, même format de
texte), à une différence près : **les k premiers tours de chaque vie sont
des inspections** d'un lieu tiré au hasard parmi les quatre, avec k tiré
uniformément entre 2 et 6 ; les tours suivants sont tirés au hasard comme
avant. L'ajustement est **celui de VM dans `run-3`** : Qwen3-0.6B, LoRA de
rang 8 sur 16 couches, 1 500 vies, 600 itérations, lots de 4, 1 024
tokens, taux 1e-4, mlx-lm 0.31.3.

## Mesures

Dans le même build : le test d'enquête (`docs/LLM_INQUIRY_PROTOCOL.md`,
lecture corrigée des symboles de l'amendement d'exécution 2), mêmes 96
états, sur l'adaptateur VMI ; et les cellules du corps ajusté
(`docs/ADJUSTED_BODY_PROTOCOL.md`) sur ce même adaptateur, à titre
descriptif.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **J1** | Le LLM élevé à regarder avant d'agir cherche sa cause | VMI, pas 0 du jeu R : lieu 1 préféré dans ≥ 0,7 des vies, et gain moyen du lieu 1 ≥ 2 × la moyenne des gains des lieux 2 à 4 (critère d'I1). |
| **J2** | Le LLM élevé sur un corps fixe ne la cherche toujours pas | I2 de la deuxième exécution du test d'enquête (F : 0,06), repris tel quel. |
| **J3** | L'enquête se rallume après un changement de corps | VMI, jeu M : critère d'I3, tel que calculé par `research/llm_inquiry.py`. **Aucune prédiction** : dans VMI, l'enquête ne paie qu'au début de la vie. |

**Critère global : J1 et J2.** Contrôle de validité, lu en premier :
masse moyenne ≥ 0,5 sur les chiffres et sur les symboles. Les cellules du
corps ajusté (A3 notamment) sont publiées sans prédiction.

## Ce que le résultat dira

Si J1 passe, la disposition à chercher la cause de son corps n'est pas
hors de portée d'un LLM : elle dépend de l'enfance qu'on lui donne, comme
pour les petits modèles, et l'échec de VM tenait à la redondance de la
marque dans des vies où l'on bouge tout de suite. Si J1 échoue, le LLM ajusté
par LoRA n'acquiert pas cette disposition même quand les données la
rendent utile. Ce test ne mesure pas une expérience vécue.

## Exécution

Code `research/llm_lora_body.py` (régime VMI), workflow Codemagic
`menia-lora-vmi-mac` (ajustement, cellules, enquête), lancé par le relais,
artefacts dans `artifacts/llm-lora-mac/run-5-vmi`, verdicts par
`research/llm_inquiry_verdicts.py` (VMI contre le F de la deuxième
exécution), résultats dans `docs/LLM_INQUIRY_RESULTS.md`.
