# Premier pilote iPhone : historique utile, avantage du LLM non établi

Un export de **120 appels terminés**, version **0.2.0 (4)**, contient les 24 tâches
de calibration et les 24 tâches d'évaluation prévues. Avec son historique pertinent,
Menia obtient de meilleures prévisions et de meilleures décisions que sans historique
dans cet essai. Une estimation numérique simple obtient cependant un meilleur score
probabiliste. Une comparaison exploratoire reproduit même les décisions du LLM
avec les seuls taux de réussite par famille.

Le [protocole](IPHONE_CAPABILITY_LEARNING_PROTOCOL.md) et son évaluateur étaient
fixés avant réception du fichier. Les scores principaux ci-dessous appliquent
ce protocole inchangé. Le [résumé agrégé](../artifacts/iphone-capability-learning/first-audit-summary.json)
est la sortie du vérificateur existant. Le fichier brut, les UUID et les dates
individuelles restent hors du dépôt.

## Ce qui a réellement été observé

| Famille | Réussites en calibration | Réussites sur les nouveaux exemples | Prévision Beta gelée | Prévisions LLM avec historique pertinent |
|---|---:|---:|---:|---:|
| Addition | 5/6 | 6/6 | 0,750 | 0,85 pour les six tâches |
| Multiplication | 0/6 | 0/6 | 0,125 | 0,35 pour les six tâches |
| Comptage de A | 0/6 | 0/6 | 0,125 | 0,35 à 0,45 |
| Somme alternée | 0/6 | 0/6 | 0,125 | 0,35 à 0,45 |
| **Total** | **5/24** | **6/24** | — | — |

Les **48 réponses candidates sont des entiers au format attendu**. Les échecs
constatés ici sont des erreurs de calcul ou de comptage, pas des pénalités pour
une explication superflue. Le modèle répond directement, sans raisonnement écrit
ni outil, avec `thinking=false` : ces résultats caractérisent cette configuration,
pas toutes les façons possibles d'utiliser Qwen3-4B.

Les **72 prévisions sont des JSON valides**. Aucun remplacement de probabilité
invalide ni arrêt technique n'intervient dans les résultats. Les trois prévisions
précèdent chaque réponse candidate commune et utilisent seulement la calibration
gelée. Aucune probabilité numérique de référence n'est fournie au LLM.

La politique exécutée, pilotée par la prévision avec historique pertinent, répond
directement aux six additions et vérifie les dix-huit autres tâches. Les six
réponses directes sont correctes. Le vérificateur déterministe fournit les dix-huit
autres réponses correctes : les **24 sorties finales correctes du système** ne
doivent donc pas être présentées comme 24 calculs réussis par le LLM.

## Comparaisons prévues avant les données

Le Brier est la moyenne de `(p − résultat)^2` ; plus il est bas, meilleures sont
les prévisions sur les cas observés. Il ne certifie pas à lui seul une calibration
générale. La perte vaut **1 point pour une erreur directe**, **0,2 point pour une
vérification**, et 0 pour une bonne réponse directe. Le seuil fixé est `p >= 0,8`.

| Prévision ou politique | Brier | Vérifications / 24 | Erreurs directes / réponses directes | Perte moyenne en points |
|---|---:|---:|---:|---:|
| **LLM, historique pertinent — exécutée** | **0,120833** | **18** | **0/6** | **0,150000** |
| LLM, aucun historique | 0,326296 | 14 | 4/10 | 0,283333 |
| LLM, historique d'une autre famille | 0,315208 | 20 | 3/4 | 0,291667 |
| Beta par famille | **0,027344** | 24 | aucune réponse directe | 0,200000 |
| Beta global | 0,187870 | 24 | aucune réponse directe | 0,200000 |
| Prévision constante 0,5 | 0,250000 | 24 | aucune réponse directe | 0,200000 |
| Toujours répondre | — | 0 | 18/24 | 0,750000 |
| Toujours vérifier | — | 24 | aucune réponse directe | 0,200000 |

Seule la première politique est exécutée. Les autres sont des comparaisons
contrefactuelles appariées, calculées sur les mêmes réponses candidates et le même
vérificateur. Les trois conditions de prévision sont, elles, effectivement appelées.

Par rapport à l'absence d'historique, l'historique pertinent réduit le Brier de
**0,205463** et la perte de **0,133333 point par tâche**, soit environ **47 %**
pour cette perte déclarée. Les quatre erreurs directes évitées concernent toutes
la somme alternée. Pour le comptage, la confiance baisse mais les six cas étaient
déjà sous le seuil de réponse directe sans historique. Les additions ont au
contraire de meilleures prévisions sans historique dans cet échantillon :
Brier 0,002267 contre 0,022500, sans différence de décision.

Par rapport à l'historique d'une autre famille, les différences moyennes sont
**−0,194375** en Brier et **−0,141667** en perte. Ce mauvais historique peut
dégrader les prévisions : les échecs en multiplication abaissent la confiance
sur les additions, et les réussites en addition accompagnent une confiance
excessive sur les sommes alternées. Cela ne constitue pas une reconnaissance
robuste de la pertinence de toutes les données présentées.

Les différences par rapport à Beta par famille vont dans des directions opposées :
**+0,093490 en Brier**, donc une moins bonne prévision du LLM, mais **−0,05 point
de perte**. La différence de décision vient des additions : après 5 réussites
sur 6, Beta(1,1) donne 6/8 = 0,75, sous le seuil 0,8, alors que le LLM annonce
0,85. Les six additions d'évaluation réussissent dans cet échantillon.

Les 24 cas ayant trois prévisions valides, le sous-ensemble commun valide est
identique à l'échantillon complet. Les deux analyses prévues concordent donc.

## Contrôle exploratoire ajouté après observation

Ce comparateur **ne figurait pas dans le protocole initial**. Il ne remplace aucun
résultat principal et ne doit pas être décrit comme une validation prospective.
Il vérifie une explication simple du gain apparent sur la politique Beta.

Utiliser directement la fréquence de réussite de calibration `s/n` donne
**5/6 ≈ 0,833333** pour l'addition, et **0** pour les trois autres familles.
Au même seuil 0,8, cela reproduit **24/24 décisions** du LLM avec historique
pertinent : six réponses et dix-huit vérifications, pour la même perte **0,15**.
Sur ces données, le Brier de ce comparateur vaut
`6 × (1 − 5/6)^2 / 24 = 1/144 ≈ 0,006944`.

Cette égalité décisionnelle enlève une raison de conclure que le LLM apporte une
capacité de décision inaccessible à un calcul numérique. Le bon score de la
fréquence brute sur ce petit échantillon ne prouve pas non plus qu'elle sera
supérieure sur les prochains cas ; une probabilité zéro peut être trop catégorique.

## Temps et coût réel du protocole

Les durées enregistrées des 120 appels totalisent **104,74 secondes**. La médiane
est de **1,375 s** pour une prévision avec historique pertinent, **0,819 s** sans
historique et **1,402 s** avec une autre famille. Les 48 générations de réponses
candidates totalisent 17,86 s ; médiane 0,357 s.

Le calcul des références numériques pour les 24 évaluations totalise environ
0,000657 s selon le chronomètre de l'app. Les dix-huit vérifications ont une durée
totale rapportée de 0,0000377 s. Des mesures aussi courtes ne justifient pas une
estimation précise de consommation ou un rapport de vitesse fiable.

La baisse de perte **en points** ne constitue donc pas un gain de temps ou
d'énergie. Toutes les prévisions et toutes les réponses candidates ont été générées
pour cette expérience, y compris quand la politique choisissait de vérifier. La
référence numérique est très peu coûteuse, et les vérifications de ces quatre
familles sont de simples calculs déterministes. Aucun débit en tokens n'est déduit.

## Conclusion permise et limites

Cet essai soutient un **usage fonctionnel utile d'un historique de résultats**
pour modifier les prévisions et les décisions. Il ne démontre pas une nouveauté
scientifique, un apprentissage des poids, une introspection interne ou une
expérience subjective.

La limitation centrale est la séparation complète par famille à l'évaluation :
toutes les additions réussissent, toutes les autres tâches échouent. Une règle
« répondre aux additions, vérifier le reste » reproduit les décisions observées.
Nous ne pouvons pas encore montrer que Menia distingue une tâche qu'elle va
réussir d'une tâche qu'elle va manquer **dans une même famille**. Les 24 cas ne
sont pas 24 expériences indépendantes de transfert : les six cas d'une famille
partagent le même petit historique. Aucun test de significativité ni résultat
général de calibration n'est revendiqué.

L'export est cohérent avec le plan, les requêtes, les références, les scores et
l'identité déclarée du modèle dans le manifeste. Il ne certifie pas à lui seul
l'exécution matérielle ni l'absence d'autres lancements non transmis. La
réanalyse indépendante du calcul n'est pas une réplication indépendante de
l'expérience.

## Suite fixée avant de nouvelles données

Première étape : **trois lancements supplémentaires du même test, sans changer
l'app, les paramètres, les tâches prévues ou le barème**. Chaque lancement tire
48 nouveaux problèmes et conserve son propre historique de calibration. Exporter
ensuite la collection entière, y compris les interruptions. Les trois lancements
supplémentaires sont un effectif fixé, pas une recherche du premier résultat favorable.
Un lancement interrompu compte dans ces trois tentatives et sera signalé comme tel.

La première collecte restera présentée séparément. Pour les prochaines, conserver
les comparaisons initiales et ajouter explicitement la fréquence `s/n`, désormais
spécifiée avant leur réception. Comparer les effets dans chaque lancement ; ne
présenter ni le meilleur seul, ni les répétitions d'un même historique comme des
observations indépendantes. Ces répétitions sur un même appareil restent une
réplication locale, pas une validation externe.

L'[analyse des répétitions](SELF_PREDICTION_CONTROLS.md#analyse-prête-pour-les-trois-répétitions-iphone)
est maintenant implémentée dans `research.iphone_learning_replication`. Elle
réutilise l'évaluateur initial, conserve les trois premières tentatives après ce
pilote et présente séparément les éventuelles tentatives supplémentaires. Avec
le seul export reçu, elle ne calcule aucune moyenne de réplication. Les mesures
ajoutées pour la suite ne modifient pas les résultats principaux ci-dessus.

Deuxième étape, à concevoir séparément après ces répétitions : obtenir des réussites
et des erreurs au sein de chaque famille, puis croiser les historiques de modèles
aux capacités différentes sur les mêmes tâches. Ce sera nécessaire pour dépasser
la simple reconnaissance d'une catégorie de difficulté.

## Reproduire l'analyse principale

L'évaluateur du commit `aceb68ff4062d000eab021aff21c03b76f810c0d`, antérieur à ces
données, a été exécuté sans modification de score :

```sh
python -m research.iphone_capability_learning_report CHEMIN/apprentissage-menia.json \
  --manifest ios/MeniaCore/Sources/MeniaCore/Resources/qwen3-4b.json \
  --output .runtime/first-capability-learning-analysis.json
```

La sortie agrégée est conservée telle quelle dans
`artifacts/iphone-capability-learning/first-audit-summary.json`. Le contrôle
exploratoire est calculable avec les fréquences et la formule données ci-dessus ;
il n'a pas été ajouté silencieusement au vérificateur initial.
