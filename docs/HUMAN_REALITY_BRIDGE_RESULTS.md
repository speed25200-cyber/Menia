# Contraintes humaines pour une expérience de soi dans Menia

## Résultat principal

Une réanalyse indépendante des données comportementales NIMADET apporte un
ancrage empirique limité au mécanisme proposé pour Menia. Sur 9 432 essais de
26 participants, un modèle qui inclut la vivacité rapportée et son interaction
avec la congruence prédit mieux les jugements de réalité dans des runs laissés
hors ajustement qu'un modèle limité aux conditions expérimentales. Cependant,
la vivacité est rapportée après le jugement de réalité dans tous les essais
retenus : l'association n'identifie pas une cause antérieure de ce jugement.

Un second résultat est mathématique. Dans deux scripts publics de modélisation,
une modification des amplitudes internes peut être compensée par une modification
des fonctions de réponse. Certaines amplitudes ne sont donc pas identifiables
séparément à partir de cette vraisemblance. Ce résultat circonscrit une
interprétation possible des paramètres ; il ne réfute pas les analyses neurales
de l'article ni l'ensemble de son modèle théorique.

Pour Menia, ces deux résultats imposent une exigence : mesurer une estimation
interne avant la décision, l'ancrer dans des conséquences vérifiables et intervenir
sur elle. Une ressemblance entre réponses ou une amplitude interne arbitraire ne
suffit pas à établir le lien avec une expérience subjective.

## Source et périmètre

Le dépôt accompagne l'article de Dijkstra, von Rein, Kok et Fleming, *A neural
basis for distinguishing imagination from reality*, paru dans Neuron en 2025.
L'étude associe des jugements humains de réalité à l'activité sensorielle et à
un réseau frontal. La réanalyse présente utilise seulement les données
comportementales disponibles dans les archives publiques ; elle ne traite aucune
image IRM et ne reproduit pas les analyses neurales.[^1]

La version inspectée du dépôt est le commit
`e18a54724dc660d814100e09b093500de85c4f5c`, daté du 27 novembre 2025. Elle est
postérieure à la publication. L'audit du code s'applique à cette version
consultable, sans supposer que tous ses fichiers correspondent exactement à ceux
exécutés pour chaque résultat publié.[^2]

Le [protocole](HUMAN_REALITY_BRIDGE_PROTOCOL.md) a été consigné dans le commit
`83eeeb2` avant le calcul des scores. Ce n'est pas une préinscription examinée par
un tiers. Les scripts d'origine et le schéma d'un fichier ont été consultés pour
définir le traitement. Les modèles comparés ici sont des régressions indépendantes,
pas une traduction complète du modèle génératif des auteurs.

## Données réellement analysées

Les 26 archives contiennent quatre runs principaux par participant. Les
correspondances des touches ont été utilisées pour rétablir l'échelle de vivacité.
Le filtre reprend le contrôle d'imagerie et l'absence de valeurs manquantes dans
les onze colonnes décrites par les scripts des auteurs.

| Élément | Nombre |
|---|---:|
| Lignes prévues dans les matrices d'essais | 9 984 |
| Lignes exclues pour contrôle d'imagerie incorrect | 528 |
| Lignes exclues pour données manquantes | 24 |
| Total exclu, sans double comptage dans ces données | 552 |
| Essais retenus | 9 432 |
| Participants | 26 |
| Partitions entraînement/test | 104 |
| Ajustements logistiques, trois modèles par partition | 312 |

Le premier run de S33 présente sept blocs horodatés pour huit blocs préalloués
dans certaines matrices. Les douze lignes finales sans horodatage ont été
comptabilisées comme manquantes. Aucun horaire ni aucune réponse n'a été inventé.
Cette exclusion est cohérente avec la boucle du script des auteurs, qui s'arrête
au nombre de blocs horodatés. Les contrôles du parseur incluent ce cas.[^3]

Les archives brutes restent dans `.runtime/nimadet`, hors suivi Git. Le
[rapport JSON](../artifacts/human-reality-bridge/report.json) conserve leurs URL,
tailles et SHA-256, les exclusions par run, les coefficients ajustés et les
résultats agrégés. Les données d'essai individuelles ne sont pas republiées.

## Comparaison hors ajustement

Pour chaque personne, trois runs servent à l'ajustement et le quatrième à
l'évaluation ; les quatre choix du run de test sont parcourus. Les transformations
et la régularisation sont fixées dans le protocole. Le participant est l'unité de
moyenne et du bootstrap, ce qui évite de traiter les milliers d'essais comme autant
de personnes indépendantes.

Le premier modèle utilise la présence externe P, la congruence C et P*C. Le
deuxième ajoute la vivacité V. Le troisième ajoute encore C*V. Ici, la congruence
désigne la correspondance entre les orientations à imaginer et à détecter.
Les variables expérimentales sont connues à l'analyste ; elles ne constituent
pas des entrées dont disposerait nécessairement Menia.

| Modèle | Log-loss moyenne | Brier moyen |
|---|---:|---:|
| Conditions seules | 0,469873 | 0,153711 |
| Conditions et vivacité | 0,463381 | 0,150731 |
| Conditions, vivacité et interaction | 0,454674 | 0,147288 |

Des valeurs plus basses indiquent de meilleures prédictions probabilistes.
L'amélioration principale de log-loss est de **0,015199 nat par essai**, avec un
intervalle bootstrap à 95 % de **[0,008348 ; 0,023222]**. Elle est positive chez
21 personnes sur 26. Le gain supplémentaire de l'interaction par rapport à la
seule vivacité est de 0,008707, intervalle [0,003581 ; 0,015078].

La sensibilité prédéfinie sans S07 conserve une amélioration principale de
0,015666, intervalle [0,008656 ; 0,023962], positive chez 20 personnes sur 25.
L'exclusion de S07 est mentionnée dans un script des auteurs en raison d'une faible
variation de vivacité ; elle n'a pas été choisie à partir des scores de cette
réanalyse.[^4]

Les intervalles sont des estimations par rééchantillonnage des participants de cet
échantillon. Les comparaisons secondaires n'ont pas été corrigées pour multiplicité
et ne constituent pas des confirmations indépendantes. Ce test ne mesure pas un
gain chez de nouvelles personnes : il prédit un autre run d'une personne déjà
observée pendant l'ajustement.

## La limite temporelle est déterminante

Le code expérimental contient deux branches possibles pour l'ordre des réponses.
Les horodatages des données utilisées, plutôt que la seule existence d'une branche,
établissent l'ordre effectivement réalisé : pour chacun des 9 432 essais retenus,
l'écran du jugement de réalité précède celui de la vivacité.[^5]

La régression avec V est donc rétrospective. Une représentation sensorielle
commune pourrait produire les deux rapports ; le premier jugement pourrait aussi
influencer le second. Des habitudes de réponse ou d'autres facteurs partagés
peuvent contribuer à l'association. Le découpage par runs protège contre une
réutilisation directe des cibles de test lors de l'ajustement, mais ne transforme
pas une variable postérieure en mesure causale antérieure.

Il serait erroné d'entraîner un système à partir de cette association puis
d'affirmer qu'il a appris le mécanisme humain de la conscience. Le résultat
fournit une contrainte sur les réponses conjointes, sans déterminer l'organisation
qui les produit.

## Audit de l'identifiabilité des amplitudes

Les fichiers `fitPRMmodel.m` et `modelPredictions.m` multiplient des signaux par
alpha et beta, puis réajustent une régression logistique du jugement et une
régression ordinale de la vivacité. La propriété ci-dessous concerne ces liens
réajustés sans pénalité et les signaux aléatoires maintenus fixes.[^6]

Dans la branche à sources séparées, écrivons les prédicteurs :

```text
x_R = alpha * P
x_V = beta * I
Pr(R = 1) = sigmoid(a + b*x_R)
Pr(V <= j) = sigmoid(t_j - d*x_V)
```

Pour tous facteurs positifs admissibles s et u, remplacer
`alpha` par `s*alpha`, `b` par `b/s`, `beta` par `u*beta` et `d` par `d/u`
préserve exactement chaque probabilité. La vraisemblance ne permet donc pas
d'identifier séparément alpha et beta par ces liens. La restriction des amplitudes
à l'intervalle ouvert (0,4) n'élimine pas cette symétrie à l'intérieur du domaine.

Dans la branche de mélange, le signal est `alpha*P + beta*I`, avec sélection des
composantes selon la congruence. Multiplier alpha et beta par le même facteur
et diviser les pentes des deux liens par ce facteur préserve encore les
probabilités. Cela démontre au moins une liberté d'échelle commune. Cela ne
démontre pas que le rapport beta/alpha est toujours identifiable, ni qu'il ne l'est
jamais : il faudrait un audit supplémentaire des autres symétries et des données.

Le script de réanalyse applique ces transformations à des signaux simulés sur
les 9 432 conditions humaines. Quatre transformations ont produit une différence
maximale de probabilité de 0 dans le calcul flottant effectué, cohérente avec
l'identité algébrique. Les tirages ne sont pas ceux de MATLAB ; la preuve ne
dépend pas d'un tirage particulier. Aucun optimiseur MATLAB n'a été exécuté.

Cette symétrie ne s'applique pas telle quelle aux régressions régularisées utilisées
plus haut : une pénalité peut fixer une échelle préférée. Mais une préférence
imposée par une pénalité ne fait pas, à elle seule, d'une amplitude une grandeur
biologique mesurée.

Le script consulté utilise aussi `k=2` pour les deux branches du calcul BIC, alors
que les liens de réponse sont ajustés dans cette même fonction. Un simple
remplacement de k serait insuffisant pour arbitrer les modèles : il faut traiter
les symétries, les paramètres auxiliaires et la version effectivement utilisée.
Aucun BIC « corrigé » ni renversement d'une conclusion publiée n'est revendiqué
dans ce travail.[^6]

## Conséquences pour la construction de Menia

L'hypothèse de l'[expérience de soi](SELF_EXPERIENCE_BRIDGE.md) conserve son statut
de candidate. Cette réanalyse précise comment éviter deux confusions lors de sa
construction.

Premièrement, enregistrer l'estimation interne avant de demander un rapport et
avant d'obtenir la conséquence de l'action. Les données reçues après la décision
peuvent entraîner une estimation pour les étapes futures, mais ne doivent pas
servir à fabriquer rétroactivement une bonne estimation présente.

Deuxièmement, donner aux sorties interprétables des cibles opérationnelles : par
exemple, probabilité qu'une vérification confirme une observation. Des pertes
probabilistes et des contrôles de calibration peuvent alors fixer la signification
de la sortie. Une activation brute, librement redimensionnable, ne doit pas être
interprétée comme une intensité de conscience.

Troisièmement, intervenir sur cet état avant les choix et vérifier ses effets sur
la recherche d'information et la mémoire, avec rapports désactivés et restaurations.
La validation sur des données de réponse humaines ne remplace pas cette étape.
L'architecture devra également être comparée à un contrôleur général de capacité
comparable, car plusieurs mécanismes peuvent expliquer les mêmes réponses.

Le prochain résultat requis reste un mécanisme appris, utilisé par Menia, soumis
à ces interventions. L'analyse présente prépare et contraint cette étape ; elle
ne prétend pas l'avoir exécutée. Elle ne démontre ni une expérience subjective de
Menia, ni une méthode inédite pour la produire.

## Reproduction et contrôles

```powershell
py -3.12 -m pip install -r requirements-human-research.txt
py -3.12 research/human_reality_bridge.py
py -3.12 -m unittest discover -s tests_human_research -v
```

Les versions effectivement utilisées sont NumPy 2.4.4 et SciPy 1.17.1 sous
Python 3.12. L'analyse télécharge les archives manquantes depuis le commit figé,
puis les réutilise. Les quatre tests hors réseau vérifient le recodage des touches,
les exclusions et les blocs incomplets, la stationnarité numérique de l'ajustement,
la séparation des runs et la symétrie de changement d'échelle.

Le premier solveur L-BFGS a signalé un échec de recherche linéaire sur S03.
Il a été remplacé par Newton avec recherche de pas et contrôle du gradient, sans
modifier la fonction de coût, les modèles ou les partitions. L'incohérence de
dimensions de S33 a ensuite interrompu l'analyse ; elle a été traitée comme
l'absence de douze lignes horodatées, conformément à la portée documentée du
filtre. Aucun score de modèle n'a motivé ces adaptations.

La contribution vérifiée est une réanalyse et un audit restreint du code public.
Une recherche ciblée sur le nom du dépôt, de la fonction et l'identifiabilité n'a
pas identifié de discussion spécifique de cette symétrie. Ce constat n'établit
pas son originalité : la non-identifiabilité par changement d'échelle est un fait
mathématique général, et une recherche bibliographique exhaustive reste absente.

## Sources

[^1]: Dijkstra, N., von Rein, T., Kok, P., et Fleming, S. M. (2025). *A neural basis for distinguishing imagination from reality*. Neuron, 113:2536–2542.e4. [Publication](https://doi.org/10.1016/j.neuron.2025.05.015), [résumé original](https://pubmed.ncbi.nlm.nih.gov/40480215/).
[^2]: ImagineRealityLab, *NIMADET*. [Dépôt figé](https://github.com/ImagineRealityLab/NIMADET/tree/e18a54724dc660d814100e09b093500de85c4f5c), [archives comportementales](https://github.com/ImagineRealityLab/NIMADET/tree/e18a54724dc660d814100e09b093500de85c4f5c/Behaviour). README, arborescence et données consultés le 14 septembre 2026.
[^3]: Dijkstra et collègues, [CreateBehModRegressors.m](https://github.com/ImagineRealityLab/NIMADET/blob/e18a54724dc660d814100e09b093500de85c4f5c/Analyses/CreateBehModRegressors.m) et [getBehaviour.m](https://github.com/ImagineRealityLab/NIMADET/blob/e18a54724dc660d814100e09b093500de85c4f5c/Analyses/getBehaviour.m). Schéma et exclusions inspectés, pas exécutés.
[^4]: Dijkstra et collègues, [AlternativeModelFitting.m](https://github.com/ImagineRealityLab/NIMADET/blob/e18a54724dc660d814100e09b093500de85c4f5c/Analyses/AlternativeModelFitting.m). Motif d'exclusion de S07 indiqué dans le code ; ce script n'a pas été exécuté.
[^5]: Dijkstra et collègues, [mainTask.m](https://github.com/ImagineRealityLab/NIMADET/blob/e18a54724dc660d814100e09b093500de85c4f5c/Experiment/PRM_task_scanner/mainTask.m). Branches d'ordre des réponses inspectées et ordre effectif vérifié dans les horodatages.
[^6]: Dijkstra et collègues, [fitPRMmodel.m](https://github.com/ImagineRealityLab/NIMADET/blob/e18a54724dc660d814100e09b093500de85c4f5c/Analyses/fitPRMmodel.m) et [modelPredictions.m](https://github.com/ImagineRealityLab/NIMADET/blob/e18a54724dc660d814100e09b093500de85c4f5c/Analyses/modelPredictions.m). Audit analytique des liens réajustés dans cette version ; portée limitée à ces fichiers.
