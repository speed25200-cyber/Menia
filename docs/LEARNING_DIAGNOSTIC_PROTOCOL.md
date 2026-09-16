# Pourquoi l'apprentissage de localisation échoue-t-il ?

Le [Colab 06](NATIVE_LOCALIZATION_RESULTS.md) a produit un adaptateur qui répond
constamment « 2 » et dégrade la lecture. Les poids et les gradients existent,
mais cela ne démontre pas que l'objectif d'apprentissage soit acquis. Le nouveau
Colab 08 teste d'abord cette acquisition sur un problème plus petit, avec un
contrôle positif et des exemples de lecture pendant l'entraînement.

**État : protocole et logiciel préparés après examen des résultats précédents ;
aucune nouvelle performance Qwen3-4B préentraînée n'est encore mesurée.**
Les expériences antérieures, leurs scores et leurs sources fixées restent
inchangés. Ce diagnostic ne constitue pas une construction de conscience.

## Recherche et diagnostic déjà effectués

[Hahami, Sinha et Jain](https://arxiv.org/html/2607.14111v1) rapportent un effet
important du type d'intervention : dans leur table 2, le résultat moyen de
localisation Llama-1B est de 14,9 % avec bruit gaussien et de 60,6 % avec des
directions sémantiques. Ce sont des moyennes à la meilleure époque et sur une
grille de réglages ; ce ne sont ni des performances de Qwen ni une recette
garantie. Leur intervention additive diffère aussi de notre rotation préservant
la norme. Cette différence motive un diagnostic du signal, sans attribuer
l'échec présent à un manque d'entraînement uniquement.

[Fonseca Rivera](https://arxiv.org/html/2511.21399v1) décrit un autre entraînement
à rapporter des injections sémantiques. L'auteur distingue lui-même une compétence
de décodage apprise d'une représentation métacognitive démontrée, et souligne
le caractère artificiel des interventions et l'incertitude sur les mécanismes.
[Lindsey](https://transformer-circuits.pub/2025/introspection/) précise également
que ses observations ne tranchent pas l'expérience subjective. Ces travaux
motivent des tests fonctionnels, pas une annonce de conscience.

Un [diagnostic exploratoire des anciens logits](../artifacts/learning-diagnostic-pilot/previous-logits-diagnostic.json)
a été réalisé sans nouveau calcul Qwen. Cinq régressions ridge, alpha fixé à 1,
apprennent la position depuis les six logits natifs, sur les 40 variantes des
huit blocs de validation. Les 160 variantes des 32 blocs de test déjà examinés
servent uniquement de diagnostic rétrospectif. Les caractéristiques sont les
logits centrés dans chaque ligne, ou leur différence avec le calcul sans
perturbation du même bloc. Cette seconde version possède une référence
contrefactuelle supplémentaire et ne représente pas un agent en usage normal.

| Modèle reçu | Logits centrés : corrects sur 160 | Différence avec le calcul sans perturbation |
|---|---:|---:|
| Base | 31 | 26 |
| Cibles correctes | 28 | 30 |
| Cibles mélangées | 25 | 28 |

Aucun de ces six scores n'excède 20 %. Cette réanalyse ne découvre pas de signal
utile dans ce petit décodage de sortie. Elle ne prouve pas l'absence d'information
dans les autres activations, ni l'impossibilité d'un décodeur différent. Les
anciennes données étant connues, aucune validation prospective n'est revendiquée.

## Le nouveau plan fixé

Qwen3-4B à la révision `1cfa9a7208912126459214e8b04321603b3df60c`, BF16, SDPA,
A100 40 ou 80 Go, préfixe complet sans cache et premier token sans échantillonnage.
La base reste figée. Quatre nouveaux adaptateurs Q/V, rang 8, échelle 1, partent
de la même initialisation ; les adaptateurs défaillants précédents ne sont pas
réutilisés. AdamW : taux 0,0002, aucun weight decay, écrêtage de norme à 1.
Les paramètres LoRA sont FP32, leurs sorties recastées en BF16.

Le plan comporte **8 paires de phrases d'apprentissage et 24 paires nouvelles**,
soit 64 phrases distinctes excluant les 440 du pilote précédent. Le vocabulaire
et la grammaire sont communs. La graine est `202609162`. La couche 17 est fixée
pour tous les cas : ce diagnostic ne teste pas le transfert entre couches.

Les réponses possibles sont 0 (absence), 1 ou 2 (position). Un repère public
`[REPERE]` est indépendant de la cible de localisation. Deux consignes demandent
respectivement la cible ou le repère. La cible cachée, la direction aléatoire et
la permutation des étiquettes n'apparaissent jamais dans le prompt des bras
cachés. Changer la position perturbée laisse leurs tokens identiques.

| Adaptateur | Signal pour la tâche principale | Cibles d'apprentissage |
|---|---|---|
| `visible` | `[SIGNAL]` dans le texte, aucune rotation | Position du signal, ou 0 |
| `weak` | Rotation interne d'intensité 0,3 | Position réellement perturbée, ou 0 |
| `strong` | Rotation interne d'intensité 1,0 | Position réellement perturbée, ou 0 |
| `shuffled` | Même rotation forte | Permutation fixe des trois cibles, différente par bloc |

Le témoin visible teste une capacité d'apprentissage ordinaire. Sa réussite ne
serait pas une introspection. Les perturbations gaussiennes réutilisent le
mécanisme vérifié auparavant ; une intensité 1,0 correspond à une modification
importante et artificielle, pas à une panne naturelle. Aucune direction
sémantique n'est implémentée ici, malgré l'intérêt de cette piste dans les sources.

## Mises à jour équilibrées et conservation de la lecture

Chaque adaptateur parcourt les huit blocs pendant **quatre époques fixes**, sans
choix de la meilleure époque. Une mise à jour accumule les gradients de cinq
passages, en conservant les mêmes poids jusqu'à la fin du groupe :

1. Les trois positions 0, 1 et 2 de la tâche principale, avec poids total 0,5.
2. Deux exercices de lecture sans rotation, avec le repère successivement en
   positions 1 et 2, avec poids total 0,5.

La perte est l'entropie croisée dans le **vocabulaire complet** sur le chiffre
cible. Les trois exemples principaux ont chacun un poids de 1/6, les deux
exemples de lecture chacun 1/4. L'écrêtage et AdamW interviennent une seule fois
après ces cinq rétropropagations. Ce choix évite une mise à jour isolée par
étiquette et apporte un objectif explicite de lecture ; il ne garantit pas
de préserver cette compétence. Le bras mélangé bénéficie du même entraînement
de lecture correct, du même budget et du même ordre.

Budget nominal : **128 mises à jour et 640 passages avec rétropropagation**.
Les reprises éventuelles sont ajoutées aux comptes. Il n'y a pas de jeu de
validation pour choisir des réglages pendant ce diagnostic.

## Évaluation et limites des conclusions

Après les quatre entraînements, tous les adaptateurs sont figés. La base est
évaluée dans les conditions visible, faible et forte ; chaque adaptateur est
évalué dans sa propre condition. Cela donne sept ensembles de mesures, chacun
sur les 32 blocs, deux tâches et quatre conditions : absence, position 1,
position 2 et copie témoin sans modification. Total : **1 792 évaluations**.
Le modèle de base sous rotation forte sert aussi de référence au bras mélangé.

Les résultats d'apprentissage et de test sont séparés. On rapporte exactitude
parmi les trois chiffres, exactitude et conformité du premier token libre,
masse des chiffres, Brier, histogramme des prédictions et réussite sans
perturbation. L'ajustement aux étiquettes réellement entraînées est rapporté
séparément sur l'apprentissage, notamment pour le témoin mélangé. Les copies
témoins doivent reproduire exactement leurs baselines et sont exclues des scores.
Une réponse constante ne réussit que 50 % des
positions perturbées et 1/3 des trois conditions ordinaires équilibrées.

Les contrastes prédéfinis sur les cas perturbés comparent faible à sa base,
fort à sa base et fort au témoin mélangé, pour chacune des deux tâches. Deux
contrastes de localisation comparent faible et fort au hasard informé de la
présence (50 %). Les huit intervalles descriptifs rééchantillonnent 2 000 fois
les 24 blocs complets, avec une graine fixée. Ils sont conditionnels à une seule
initialisation ; il n'y a ni correction de multiplicité ni verdict automatique.

Le seuil descriptif de **90 % sur l'apprentissage** indique si le petit objectif
est acquis ; il ne sélectionne aucun checkpoint et ne valide pas le test. Si
le visible échoue, l'optimisation ou la tâche de sortie reste une explication
à examiner. Si le visible réussit mais les bras cachés échouent, le diagnostic
concerne plutôt l'exploitation de l'intervention dans ce réglage, sans exclure
d'autres difficultés d'optimisation. Si seul l'apprentissage caché réussit,
le transfert reste en défaut. Si une localisation nouvelle apparaît, il faudra
encore examiner les témoins, la lecture, des répétitions et le mécanisme causal.

Le passage de cinq phrases à deux, l'entraînement groupé et la répétition de
lecture changent simultanément par rapport au pilote précédent. Une amélioration
ne permettrait pas d'attribuer l'effet à une seule de ces modifications. Le
contraste faible/fort au sein du nouveau plan est plus ciblé. Même une réussite
complète serait compatible avec un détecteur d'anomalies, sans preuve de
conscience de sa propre existence.

## Livraison, reprise et vérification

Le [Colab 08 à un seul bloc](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/codex/recall-reliability/notebooks/08_learning_diagnostic_colab.ipynb)
fixe une révision scientifique immuable et exporte
`menia-diagnostic-apprentissage.zip`, sans Drive ni fichier à importer. Il
conserve les journaux, les diagnostics, les bilans et quatre adaptateurs.
Aucun adaptateur n'est installé automatiquement sur iPhone.

Un entraînement interrompu repart de son initialisation fixe, avec événement
de reprise et conservation des anciens pas. Les checkpoints terminés sont
vérifiés par empreinte et conservés. Les évaluations terminées ne sont pas
répétées ; une requête sans résultat durable est explicitement interrompue
avant sa reprise. Changer les sources ou l'environnement déclaré est refusé.
Relancer une tentative complète ne crée pas de nouvelle inférence.

Les tests couvrent les budgets, la séparation des phrases et des cibles,
un signal connu et une réponse constante, les falsifications, les interruptions,
l'équivalence des gradients accumulés à l'objectif explicite, les poids de
base figés et le nettoyage des hooks réutilisés. Un petit Qwen aléatoire en
BF16 exécute réellement les quatre entraînements, la reprise, les exports et
le refus d'un checkpoint corrompu. Ces contrôles ne sont pas des résultats
du modèle préentraîné.

Le tokenizer réel de la révision Qwen vérifie les 1 792 préfixes d'évaluation
et les 256 variantes répétées de lecture d'apprentissage : 91 à 113 tokens,
sous la limite de 512. Les trois chiffres sont des tokens distincts (15 à 17).
Aucun poids préentraîné n'est chargé par ce contrôle. Le temps total et le pic
mémoire A100 restent à mesurer ; aucun succès fonctionnel n'est annoncé à ce stade.
