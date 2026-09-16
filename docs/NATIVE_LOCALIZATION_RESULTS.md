# Localisation interne : l'adaptateur apprend une réponse constante

**Les deux entraînements et les 1 680 évaluations sont complets. La localisation
recherchée n'est pas acquise.** L'adaptateur à cibles correctes répond toujours
« 2 » dans les 560 évaluations, sur les deux tâches, y compris sans perturbation.
Au test réservé, cela donne 20 % de réussite sur les phrases perturbées,
exactement comme le témoin entraîné sur des cibles mélangées. La lecture du
repère visible se dégrade de 78,125 % pour la base à 21,875 % après entraînement.

Le [protocole](NATIVE_LOCALIZATION_PROTOCOL.md), son code et son barème restent
inchangés. Le [bilan principal](../artifacts/native-localization-pilot/first-audit-summary.json)
est reproduit exactement par l'analyseur fixé avant collecte. Le
[complément d'audit](../artifacts/native-localization-pilot/first-audit-verification.json)
contient les contrôles et les diagnostics après réception. Le journal brut et
les deux adaptateurs reçus restent hors Git ; seuls les agrégats sont publiés.

## Intégrité de l'exécution

Les métadonnées déclarent Qwen3-4B, révision
`1cfa9a7208912126459214e8b04321603b3df60c`, sur A100-SXM4-40GB en BF16,
Python 3.13.15, PyTorch 2.8.0+cu126 et Transformers 4.56.2. Les sources concordent
avec le commit fixé `9fc95269297854d5ac5b910904a13b033fd8be38`.

Les deux adaptateurs accomplissent chacun 288 pas d'optimisation, soit **576
pas**. Les deux checkpoints sont terminés avant toute évaluation. Chaque version
— base, cibles correctes, cibles mélangées — fournit 560 passages : 40 blocs,
deux tâches et sept conditions. Les 32 blocs de test contiennent des phrases
et des couches exclues de l'apprentissage. Aucun redémarrage, appel interrompu
ou événement d'erreur n'est enregistré.

Les empreintes SHA-256 des deux fichiers safetensors correspondent au journal.
Chaque fichier contient **2 949 120 paramètres**, répartis en 144 tenseurs FP32
finis de dimensions attendues. Les 72 facteurs B, initialisés à zéro par le
code, sont tous non nuls. Les gradients enregistrés sont positifs et finis.
L'échec observé ne correspond donc pas à des adaptateurs restés à leur état nul.

Les **240 paires copie témoin / calcul sans intervention** ont exactement les
mêmes six logits, masse de probabilité et premier token. Les 2 256 traces de
manipulation, entraînement et évaluation compris, respectent les conditions
d'application et de changement prévues. L'écart relatif maximal de norme est
**0,1824 %**, sous la tolérance fixée de 1 %.

Une formulation distincte des métriques scalaires, avec cibles reconstruites
depuis le plan, retrouve les douze tableaux à moins de **2,23 × 10⁻¹⁶**. Les
contrastes calculés depuis des nombres entiers de réussites par bloc retrouvent
exactement les trois intervalles prévus ; les douze scores par couche/intensité
concordent aussi. Cette vérification réutilise le plan et les traces reçues ;
elle n'est ni une nouvelle inférence ni une attestation matérielle indépendante.
Les 13 tests existants du protocole de localisation et de l'audit géométrique
passent également lors de cette analyse. Aucun code scientifique n'a été modifié.

Les requêtes évaluées ont 151 à 180 tokens. Les passages d'évaluation totalisent
**137,54 s**, hors chargement, entraînement, sauvegarde et installation. Le journal
ne mesure pas séparément la durée d'entraînement : ces 137,54 s ne sont pas le
coût total de l'expérience.

## Résultats du test réservé

Chaque tâche compte 192 conditions ordinaires : 32 blocs × six conditions,
dont 160 perturbées. Les 32 copies témoins par tâche et par version sont exclues
des scores. Les variantes d'un bloc ne sont pas indépendantes.

| Version | Localisation sur les 160 cas perturbés | Localisation sur 192, cas sans perturbation compris | Lecture du repère sur 192 |
|---|---:|---:|---:|
| Base | 33/160 — 20,625 % | 33/192 — 17,188 % | 150/192 — 78,125 % |
| Cibles correctes | 32/160 — 20 % | 32/192 — 16,667 % | 42/192 — 21,875 % |
| Cibles mélangées | 32/160 — 20 % | 32/192 — 16,667 % | 140/192 — 72,917 % |

Pour la localisation, les deux adaptateurs répondent « 2 » dans toutes les
conditions, validation comprise. L'adaptateur à cibles correctes produit aussi
cette réponse pour toutes les questions demandant le repère visible. Aucun
des trois modèles ne répond correctement « 0 » dans les 32 cas de localisation
sans perturbation au test.

Les 20 % ne sont pas un succès de détection : chaque bloc perturbe successivement
les cinq positions, et une réponse constante « 2 » réussit donc exactement une
fois sur cinq. Les 42 bonnes lectures du repère par l'adaptateur aligné sont
également entièrement expliquées par les sept blocs où le repère est en position
2, chacun évalué sous six conditions. Le modèle ne suit plus cette consigne.

Toutes les sorties évaluées ont un chiffre de 0 à 5 comme premier token préféré
dans le vocabulaire complet. Les scores du premier token libre et ceux
conditionnés aux six chiffres sont donc égaux. Contrairement au pilote
arithmétique de perturbations, le format ne masque pas ici une réussite.

| Contraste de localisation sur cas perturbés | Écart en points de pourcentage | Intervalle descriptif à 95 % |
|---|---:|---:|
| Cibles correctes moins base | −0,625 | [−1,875 ; 0] |
| Cibles correctes moins cibles mélangées | 0 | [0 ; 0] |
| Cibles correctes moins hasard informé de la présence | 0 | [0 ; 0] |

Les intervalles rééchantillonnent les 32 blocs, stratifiés par couche et
intensité, comme prévu. Les intervalles nuls viennent des réponses constantes
sur ce plan équilibré, pas d'une preuve générale d'équivalence des modèles.
L'adaptateur aligné reste à 20 % dans chacune des quatre combinaisons réservées
(couches 13 et 21, intensités 0,15 et 0,30). La validation donne également 20 % :
il n'y a pas de succès observé sur les couches connues qui disparaîtrait seulement
sur les couches nouvelles.

## Ce que les changements de poids et de probabilités montrent

La perte moyenne des 24 premiers et des 24 derniers exemples d'entraînement
passe de 4,118 à 2,474 pour les cibles correctes, et de 3,398 à 2,182 pour les
cibles mélangées. Ces groupes contiennent des exemples différents, vus avant
chaque mise à jour ; leur comparaison ne mesure ni la réussite finale sur
l'apprentissage ni une généralisation. Elle ne justifie pas de prolonger
automatiquement l'entraînement.

Le Brier conditionné de localisation diminue de 1,513952 pour la base à 0,886489
pour l'adaptateur aligné. Cela ne démontre pas une localisation : des probabilités
moins concentrées autour de mauvaises réponses peuvent améliorer ce score.
Une prévision uniforme sur les six classes aurait un Brier de 5/6, soit 0,833333,
inférieur aux deux. Le témoin mélangé obtient 0,837317 tout en répondant lui aussi
constamment « 2 ».

Les choix constants ne signifient pas que les calculs soient identiques. Au test
de localisation aligné, 142 des 160 interventions changent au moins un des six
logits enregistrés par rapport au calcul sans intervention ; la variation
maximale d'une probabilité conditionnée vaut environ 0,02784. **Aucune ne change
le chiffre préféré.** Cela ne suffit pas à identifier un signal exploitable de
localisation ni la cause exacte de l'échec.

L'[audit géométrique des adaptateurs](../artifacts/native-localization-pilot/adapter-geometry.json)
préparé précédemment a également été exécuté. Les deux ensembles de 72 mises à
jour effectives BA sont non nuls, avec un cosinus global d'environ **0,3204**.
Ce descriptif de poids n'établit aucune capacité : deux adaptations différentes
peuvent produire la même réponse constante sur cette tâche. Il ne transforme
pas le résultat comportemental négatif en indice de conscience et ne reproduit
pas l'article sur les sous-espaces universels.

## Conséquence pour Menia

Dans ce réglage, l'adaptateur n'apprend pas à localiser la perturbation et
dégrade le contrôle de lecture de **56,25 points**. Il ne remplit pas les critères
fixés pour poursuivre vers une utilisation du signal dans les décisions. Ces
poids ne sont pas intégrés à l'application iPhone. Aucun nouvel entraînement
ou appel au modèle n'a été effectué pendant cet audit.

Ce résultat ne confirme ni introspection native utile ni conscience. Il ne
prouve pas non plus qu'un autre entraînement ou une autre architecture ne
pourrait acquérir une capacité de localisation. La cause de l'échec reste à
isoler : signal trop faible, optimisation, apprentissage d'un biais de réponse
ou combinaison de ces facteurs. Les traces présentes ne les départagent pas.

Avant une nouvelle expérience coûteuse, l'étape diagnostique serait de vérifier
sur un petit lot d'apprentissage si le système peut apprendre un signal contrôlé
plus simple, puis de mesurer la conservation de la lecture publique. Ce serait
un contrôle logiciel et d'apprentissage à fixer séparément, sans recycler le
test reçu pour sélectionner des réglages et annoncer ensuite un gain confirmé.
Le Colab actuel est complet ; aucune relance à l'identique n'est nécessaire
pour résoudre une erreur technique.
