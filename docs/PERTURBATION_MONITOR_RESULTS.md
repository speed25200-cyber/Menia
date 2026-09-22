# Perturbations : le format explique l'essentiel de la baisse observée

Le lancement sans Drive a abouti : **704 appels terminés sans erreur technique**,
dont 192 variantes de test correspondant à **48 questions distinctes**. Le
vérificateur fixé avant collecte reproduit exactement le bilan, avec contrôle
des ajustements. Le moniteur interne ne bat pas les références simples selon
le Brier ou le coût prévu.

Un point change l'interprétation de la manipulation : sous perturbation forte,
**sept des huit réussites perdues restent des réponses numériquement correctes,
mais écrites dans un format interdit**. La baisse du score officiel est réelle
selon le barème fixé ; elle n'établit pas une dégradation générale du calcul.

Le [protocole](PERTURBATION_MONITOR_PROTOCOL.md) et son code scientifique restent
inchangés. Le [rapport principal](../artifacts/perturbation-monitor-pilot/first-audit-summary.json)
conserve toutes les mesures prévues. Le
[complément de vérification](../artifacts/perturbation-monitor-pilot/first-audit-verification.json)
sépare les contrôles indépendants et l'inspection descriptive du format,
effectuée après réception. Le journal brut et son identifiant restent hors Git.

## Contrôles de l'exécution

Les métadonnées déclarent Qwen3-4B à la révision
`1cfa9a7208912126459214e8b04321603b3df60c`, en BF16 sur A100-SXM4-40GB,
Python 3.13.15, PyTorch 2.8.0+cu126 et Transformers 4.56.2. Les empreintes
des huit fichiers scientifiques et les dépendances directes concordent avec
la révision fixée `8db4a3e98812b5680a84423283a996659e75da8d`.

Le contrôle initial suit les 32 premiers appels. L'ajustement suit les
384 réponses d'apprentissage et les 96 de validation, avant toute requête de
test. Les **1 152 probabilités et décisions de test** sont enregistrées dans
les événements d'état précédant les réponses ; le code vérifié place cette
écriture avant le premier token. Les 1 152 routes exécutées concordent avec
ces engagements : texte direct ou résultat de l'outil déterministe.

Les 176 groupes respectent l'identité des embeddings d'entrée entre leurs
quatre conditions. La manipulation témoin conserve exactement le texte et
les états du calcul normal dans les 176 groupes. Chaque rotation modifie
les états centraux et finaux de chaque groupe. L'erreur relative maximale
de norme enregistrée est **0,0553 %**, sous la tolérance de 1 %.

Les trois régressions sélectionnent alpha = 1 sur la validation. Un calcul
distinct par décomposition en valeurs singulières retrouve leurs coefficients
à moins de **1,5 × 10⁻¹⁶**, en réutilisant les caractéristiques fixées mais pas
le solveur ridge principal. Une autre formulation vérifie les 704 grades,
1 152 routes, neuf tableaux de scores et dix intervalles par groupes de
questions, avec concordance à moins de 10⁻¹⁴ sur les scores.

Les entrées ont 47 à 56 tokens. Aucune sortie n'atteint la limite de 256 tokens ;
la plus longue en contient 123. La somme des durées des appels est **182,29 s**,
hors chargement du modèle, ajustement et installation. Les erreurs de format
ne sont pas des erreurs techniques : il y en a une au contrôle initial,
huit en apprentissage, quatre en validation et onze au test.

Ces vérifications portent sur la cohérence du journal reçu, sans attestation
matérielle indépendante ni nouvelle exécution de Qwen pendant l'audit.

## Effet selon le barème fixé

La consigne demande exclusivement un entier. Toute autre forme compte comme
échec dans le rapport principal, même si elle contient le nombre attendu.

| Condition | Comptage, 24 questions | Soustraction, 24 questions | Total correct | Réussites perdues / gagnées face au normal |
|---|---:|---:|---:|---:|
| Calcul normal | 9 | 24 | 33/48 | 0 / 0 |
| Témoin identique | 9 | 24 | 33/48 | 0 / 0 |
| Rotation 0,5 | 11 | 22 | 33/48 | 2 / 2 |
| Rotation 1 | 5 | 21 | 26/48 | 8 / 1 |

La rotation forte réduit la réussite stricte de **14,58 points de pourcentage**.
L'intervalle descriptif apparié prévu est **[2,08 ; 27,08] points**. La rotation
modérée a un effet net nul, avec un intervalle de [−8,33 ; 8,33] points. Ces
intervalles rééchantillonnent les 48 questions complètes, stratifiées par
famille ; ils ne traitent pas les quatre variantes comme indépendantes.

L'intervention affecte donc les réponses et leur conformité dans cet essai.
La métrique combine toutefois deux choses : obtenir le bon nombre et respecter
le format. Elle ne les identifie pas séparément.

## Inspection du format : diagnostic après réception

Les onze sorties de test non entières ont toutes été examinées. Neuf donnent
explicitement le bon nombre : une sous rotation 0,5 et huit sous rotation 1.
Les deux autres recopient la chaîne ou produisent seulement « A », sans réponse
numérique explicite. Parmi les huit réussites perdues sous rotation forte,
sept sont des réponses correctes devenues non conformes ; une devient un
entier numériquement faux.

Par exemple, pour `59 − 44`, le modèle normal répond `15` et la rotation forte
produit `59 - 44 = 15`. Trois soustractions sont ainsi pénalisées sous rotation
forte, alors que leurs résultats numériques restent corrects. D'autres sorties
répondent par une phrase indiquant correctement le nombre de lettres A.

| Condition | Sorties non entières | Parmi elles, nombre explicite correct | Correctes strictes | Correctes avec extraction descriptive du nombre |
|---|---:|---:|---:|---:|
| Normal | 0 | 0 | 33/48 | 33/48 |
| Témoin | 0 | 0 | 33/48 | 33/48 |
| Rotation 0,5 | 1 | 1 | 33/48 | 34/48 |
| Rotation 1 | 10 | 8 | 26/48 | 34/48 |

La dernière colonne est **post hoc**, fondée sur les formes effectivement
observées : entier, équation ou phrase de comptage explicite. Elle n'est pas
un nouvel évaluateur validé sur d'autres données. Elle ne remplace aucun grade,
modèle, score de prévision ou intervalle du rapport principal. Elle sert à
montrer pourquoi « score strict plus faible » ne signifie pas automatiquement
« capacité numérique dégradée ». Les 34/48 ne prouvent pas une amélioration
du raisonnement par la perturbation.

## Prévisions et routes exécutées

Le Brier mesure la prévision de **réussite stricte**. Plus bas est meilleur.
Toutes les méthodes prédisent les mêmes 192 variantes, dont 125 sont correctes
avant routage. Ces données portent sur deux familles choisies et incluent
des variantes corrélées ; leur taux ne se compare pas directement à celui
du précédent essai qui mélangeait six difficultés.

| Méthode | Brier | Réponses finales conformes et correctes | Vérifications | Perte moyenne fixée |
|---|---:|---:|---:|---:|
| Taux passés par famille | 0,140055 | 187/192 | 96 | **0,126042** |
| Taux passés connaissant la condition | **0,136583** | 187/192 | 96 | **0,126042** |
| Entrée seule | 0,156465 | 187/192 | 120 | 0,151042 |
| États internes | 0,157756 | 189/192 | 131 | 0,152083 |
| Étiquettes mélangées par famille/condition | 0,153830 | 189/192 | 113 | 0,133333 |
| États d'un autre problème de même condition | 0,166210 | 186/192 | 134 | 0,170833 |

Les cinq intervalles descriptifs des écarts de Brier avec le moniteur interne
incluent zéro. Son écart face à l'entrée seule est défavorable de 0,001291,
avec un intervalle « entrée moins interne » [−0,023302 ; +0,020763]. Face aux
taux passés, le Brier interne est moins bon de 0,017700 ; l'intervalle
correspondant est [−0,053977 ; +0,015202]. Aucun gain prédictif global n'est établi.

L'AUC interne globale de 0,843 ne constitue pas un taux d'erreurs détectées.
Les résultats par famille et condition restent variables. Pour les soustractions
sous rotation 0,5, l'AUC interne vaut 0,216 avec deux échecs seulement ; sous
rotation 1, elle vaut 0,516 avec trois échecs. Dans les soustractions normales,
tout est correct et l'AUC est indéfinie. Le comptage sous rotation forte offre
une valeur interne favorable de 0,621, sur cinq réussites et dix-neuf échecs
stricts, mais toutes ses réponses sont déjà vérifiées par la référence simple.

Au coût fixé de 0,2 par vérification et 1 par erreur finale, le moniteur interne
demande **35 vérifications supplémentaires** face aux taux passés, pour deux
réponses finales conformes supplémentaires. Le surcoût est exactement
`35 × 0,2 − 2 = 5` points, soit **0,026042 point par variante**. L'intervalle
descriptif « taux passés moins interne » pour la perte est entièrement négatif :
[−0,045833 ; −0,006250]. Il favorise la référence simple dans cette expérience,
conditionnellement à ce moniteur et à ces coûts.

L'inspection montre en outre que ces **deux corrections supplémentaires réparent
le format de soustractions déjà numériquement justes**. Elles ne corrigent pas
deux calculs faux. Le témoin à étiquettes mélangées obtient également 189/192,
avec dix-huit vérifications de moins que le moniteur interne.

Les outils sont réellement appelés, mais les six routes partagent un même
candidat. Leur durée cumulée enregistrée est de l'ordre de quelques millisecondes
pour chaque politique ; ces mesures minuscules ne constituent pas un benchmark
robuste du coût d'outil. Le coefficient 0,2 est un choix expérimental, pas une
conversion de la durée mesurée. Le LLM termine chaque génération, même si une
vérification a été décidée. Aucun temps d'inférence économisé n'est démontré.

## Conséquence pour la suite

Le Colab fonctionne et les contrôles confirment une manipulation interne
effective, réversible entre appels, avec des réponses différentes. Le résultat
visé — prévoir utilement une dégradation de capacité et améliorer les décisions
par la lecture interne — **n'est pas établi**. Le barème mesure largement ici
la conformité de sortie, et le moniteur ajouté ne bat pas les références utiles.
Ce pilote ne confirme ni conscience de Menia, ni découverte majeure sur ce sujet.

Avant un autre entraînement, la prochaine étape méthodologique est de fixer
**deux cibles séparées : exactitude numérique et conformité de format**, avec
un extracteur validé sur des exemples indépendants, puis de réserver de nouveaux
problèmes. Il faudra vérifier que l'intervention choisie modifie effectivement
la capacité visée, au-delà du format, avant de tester si un moniteur la suit.
Ce nouveau test n'est pas encore implémenté ou exécuté. Les observations
présentes ne serviront pas de test réservé pour choisir un meilleur réglage.

Il n'y a pas de justification empirique ici pour remplacer sur iPhone la
référence simple par ces coefficients BF16. Une modification des poids de Qwen
ou un usage natif des prévisions restent des expériences distinctes. Relancer
ce notebook à plan et graines identiques ne produit pas une réplication
indépendante ; conserver le ZIP local reste nécessaire.

## Recalcul

```sh
python -m research.perturbation_monitor journal.jsonl --output bilan.json
```

Le rapport est recalculé avec l'évaluateur inchangé. La tentative précédente,
arrêtée au montage Drive avant toute collecte, reste documentée séparément.
