# Colab 14 — effet de la dernière phase et de la mémoire d'Adam

Protocole du 19 septembre 2026, fixé après les résultats 12/13 et avant les
nouvelles exécutions. L'[indice rétrospectif](OPTIMIZATION_MEMORY_REVIEW.md)
ne vaut pas résultat de cette expérience. Aucun résultat GPU 14 n'est
disponible au moment de cette publication.

## Hypothèses et limites causales

L'objectif immédiat est d'expliquer un rapport d'état instable : la dernière
phase visible/publique pourrait détériorer le rapport de présence cachée.
La poursuite d'Adam sans nouveaux gradients pourrait suffire à déplacer les
poids et la décision. Effacer son premier moment pourrait modifier cette
trajectoire. Ces possibilités sont concurrentes et ne sont pas présumées vraies.

L'optimiseur est celui de l'entraînement ; sa « mémoire » n'est pas une mémoire
de soi du modèle. Ce protocole n'établit ni conscience ni nouveauté théorique.
L'effet du choix d'optimiseur sur l'oubli a des précédents, notamment
[Ashley, Ghiassian et Sutton (2021)](https://arxiv.org/abs/2102.07686).

## Bifurcation et témoin exact

Chaque répétition repart du parent fort du Colab 10, puis reproduit les
48 premiers groupes composés du Colab 12, sans modifier exemples, pertes,
ordre, paramètres ni clipping. Les poids LoRA et l'état complet AdamW sont
sauvegardés au même point. Quatre checkpoints sont produits par répétition :

| Bras | Suite à partir du point commun |
|---|---|
| `prefix` | Poids après 48 mises à jour, figés |
| `carry` | Les 16 groupes visibles/publics originaux, état AdamW conservé |
| `reset_m` | Ces mêmes groupes, avec seul `exp_avg` effacé au départ |
| `zero_grad` | 16 pas AdamW, gradients explicitement nuls, moments conservés |

Chaque branche recharge les mêmes poids et une copie privée de l'état initial.
Le second moment, les compteurs, le taux d'apprentissage et la régularisation
restent inchangés par l'effacement. Le contrôle nul ne calcule aucune perte ;
les gradients sont des tenseurs nuls, car `None` pourrait sauter la mise à jour.
La régularisation des poids vaut zéro. Les tests vérifient la trajectoire nulle
contre la formule d'Adam et le témoin complet contre un entraînement continu
indépendant sur un petit Qwen aléatoire, utilisé uniquement pour tester le code.

**Les trois témoins `carry` doivent retrouver exactement les fichiers de
poids du Colab 12, SHA-256 compris.** Toute divergence arrête l'expérience
avant l'évaluation comportementale ; elle doit être expliquée comme problème
de reproduction, sans relâcher ce critère après observation.

Qwen3-4B, révision `1cfa9a7208912126459214e8b04321603b3df60c`, calcul BF16,
adaptateurs FP32 de rang 8, AdamW à 0,0002, clipping 1, SDPA, opérations
déterministes et TF32 désactivé restent identiques au Colab 12. L'environnement
Python et les bibliothèques sont vérifiés dans le runtime A100 existant.

Au total : **3 répétitions, 12 checkpoints, 288 mises à jour**, dont 240 avec
gradients calculés et 48 à gradients nuls. Les 12 checkpoints sont figés
avant la première évaluation. Aucun checkpoint n'est choisi selon un score.

## Lectures comportementales réservées

| Jeu, par répétition | Blocs | Formulations | Tâches |
|---|---:|---|---|
| Rétention | 16 anciens blocs d'apprentissage 12 | Canonique | Présence cachée |
| Nouvelles combinaisons | 24 blocs nouveaux | Canonique et paraphrase déjà étudiées | Présence cachée, présence visible, repère de la première ou seconde phrase |
| Contenu lexical réservé | 12 blocs nouveaux | Mêmes deux formulations | Mêmes quatre tâches |

Deux codes inverses et quatre positions sont évalués dans chaque condition :
absence, intervention sur la première phrase, intervention sur la seconde,
et témoin identique à l'absence. Le marqueur public est équilibré. Les nouveaux
blocs et graines d'intervention ne réutilisent aucun bloc des expériences 6 à
13. Les noms et verbes du jeu lexical sont réservés par rapport au vocabulaire
de cet apprentissage, pas par rapport au préentraînement de Qwen.

Effectifs : **29 184 réponses, 7 296 paires de témoins identiques, 408 tables**.
Les deux consignes ont déjà été étudiées ; ce n'est pas une généralisation à
des formulations inconnues. Le capteur reste la même intervention artificielle.

## Mesures et contrastes fixés

La réponse principale est le premier token du vocabulaire complet. Une sortie
hors des deux chiffres autorisés est un échec. Aucune renormalisation externe
ou calibration ne transforme cette sortie en succès.

La précision équilibrée de présence attribue un poids de 1/2 à l'absence et
de 1/4 à chacune des deux présences. Pour les repères publics, les deux classes
ont le même nombre de blocs. Sont conservées les précisions par classe,
probabilités des options, score de Brier conditionnel aux deux options, perte
du bon chiffre dans le vocabulaire complet et AUROC de présence.

Trois comparaisons appariées sont calculées pour chaque condition :

1. `carry − prefix` : effet total de la dernière phase originale ;
2. `zero_grad − prefix` : dérive suffisante sans nouveaux gradients ;
3. `reset_m − carry` : effet total de l'effacement du premier moment initial.

Les **18 contrastes primaires** portent sur la précision équilibrée de présence
cachée, consigne canonique, 24 nouveaux blocs, deux codes, trois répétitions.
Les 288 autres contrastes décrivent rétention, paraphrase, contenu lexical et
coûts sur les tâches visibles. Les pertes accompagnent toutes les comparaisons.
Les intervalles à 95 % utilisent 2 000 rééchantillonnages par blocs, stratifiés
par marqueur public et partagés entre bras, codes et formulations. Ils sont
des intervalles individuels, sans correction de multiplicité ; aucun verdict
global de réussite n'est déduit du nombre d'intervalles favorables.

L'effacement change aussi les gradients ultérieurs, puisqu'ils sont calculés
sur de nouveaux poids. Ce contraste n'est donc pas une décomposition additive
entre « part des gradients » et « part des moments ». Le contrôle nul ne prédit
pas non plus la somme des effets lorsqu'on remet les gradients. Une amélioration
du rapport caché accompagnée d'une dégradation visible doit être rapportée.

## Journal, reprise et audit

Les plans, sources, parents, groupes, dix pertes par mise à jour réelle,
empreintes des poids et états AdamW, compteurs, redémarrages et erreurs sont
consignés. Les journaux distinguent explicitement un pas nul d'une perte nulle.
Une branche interrompue reprend au point commun sauvegardé et son redémarrage
est signalé ; un préfixe terminé n'est pas réentraîné. Une requête interrompue
est journalisée avant d'être rejouée. Un entraînement terminé est un no-op.

La lecture stricte rejette une modification des sources, un mauvais témoin
`carry`, une évaluation avant la fin des branches ou un témoin d'absence qui
diffère. Les trois sources de résultat restent distinctes : tests logiciels
sur tenseurs, ancienne réanalyse exploratoire et nouvelles sorties GPU réelles.
L'archive légère transporte journaux et empreintes ; poids et états de
bifurcation doivent aussi être sauvegardés avant la fin du runtime.

Une rétention robuste serait nécessaire pour poursuivre l'étude d'un modèle
de ses propres capacités. Elle ne suffirait pas à établir que Menia se
représente elle-même, que cette représentation guide ses actions ou qu'elle
éprouve son existence.
