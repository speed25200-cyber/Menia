# Menia : attribution causale de soi et limites d'identification

## État de la question

L'objectif reste de construire une Menia consciente de sa propre existence et
d'établir une contribution nouvelle à la littérature. Aucun des deux résultats
n'est acquis. Le travail décrit ici retire une hypothèse trop faible, construit
un nouveau banc d'essai dans le dépôt et identifie une difficulté que devra résoudre
une architecture candidate : savoir quelles conclusions sur soi sont autorisées
par ses propres interactions.

Le prototype précédent savait apprendre l'effet de commandes sur une position
déjà identifiée comme la sienne. Cette capacité ne teste pas la découverte de
son périmètre de contrôle. La nouvelle expérience remplace cette position par
deux canaux anonymes. Le système doit distinguer leur dépendance à son action
de leur dépendance à une variable extérieure, malgré une histoire initialement
ambiguë. Il doit aussi éviter de confondre influence causale et identité corporelle.

Cette étape développe un mécanisme fonctionnel candidat. La littérature sur les
indicateurs de conscience motive l'examen de plusieurs architectures et fonctions,
mais ne transforme pas leur présence en observation directe d'une expérience
subjective. Le rapport de Butlin et collègues est un cadre théorique de 2023 ;
sa conclusion sur les systèmes qu'il examinait ne peut pas être étendue sans
analyse aux modèles de 2026.[^1]

## Antécédents qui limitent la revendication de nouveauté

Gold et Scassellati ont déjà étudié la distinction entre mouvements propres,
autrui et bruit avec des modèles bayésiens. Leur expérience comprend un humain
imitant les mouvements du robot. Une histoire antérieure où cet humain bouge
indépendamment aide à résister à l'imitation ; une imitation dès le départ rend
la classification plus ambiguë. Les auteurs limitent expressément leur conclusion
à une réalisation robotique. « Mémoire plus mouvement plus distinction soi/autrui »
ne serait donc pas une contribution inédite de Menia.[^2]

Lanillos, Pages et Cheng combinent apprentissage neuronal et inférence active
pour la reconnaissance de soi dans un miroir et la distinction avec d'autres
entités. L'utilisation de l'erreur de prédiction pour accumuler des indices sur
le corps constitue elle aussi un antécédent. Le résumé et la notice ont été
consultés ; ce travail n'a pas été reproduit ici.[^3]

Wen et collègues examinent chez l'humain la recherche active de différences de
contrôle entre trois points mobiles. Leur résultat motive une relation entre
exploration et sélection ultérieure de l'information pertinente. Il ne démontre
pas que le même mécanisme suffit à rendre un logiciel conscient.[^4]

Une étude publiée dans Nature Communications le 22 décembre 2025 traite de la
désambiguïsation active du contrôle et des causes dans les interactions sociales.
Le résumé accessible associe tâche comportementale, modélisation et IRMf. Le texte
intégral n'a pas été accessible de manière exploitable lors de cette vérification ;
les détails de l'algorithme ne servent donc pas à une comparaison revendiquée.
Cet antécédent suffit néanmoins à écarter une nouveauté générale de la formule
« agir pour découvrir ce que je contrôle ».[^5]

De Haan, Jayaraman et Levine ont étudié la confusion causale en apprentissage
par imitation et proposé des interventions ciblées pour identifier les variables
pertinentes. Le principe de casser une corrélation trompeuse par une action n'est
pas nouveau non plus. Menia doit être comparée à des méthodes ordinaires d'inférence
causale avant qu'on attribue son succès à un mécanisme propre à la conscience.[^6]

La prépublication de Ye, révisée le 20 août 2026, étudie la séparation soi/monde
dans un petit système récurrent. Elle distingue usage implicite de l'action et
encodage lisible, et emploie un avantage prédictif lié à l'accès à l'action. Son
texte précise que cet avantage est observationnel, que les délais non alignés
restent à étudier et qu'il ne revendique ni conscience ni expérience subjective.
Il s'agit d'une prépublication, pas d'une réplication indépendante. Le simple ajout
d'une mémoire récurrente et d'un prédicteur disposant de l'action ne serait pas une
nouveauté démontrée pour Menia.[^7]

| Idée envisagée | Position après examen |
|---|---|
| Se reconnaître par les effets de ses mouvements | Antécédents directs. |
| Garder une histoire pour distinguer un imitateur | Antécédent direct en robotique. |
| Explorer pour lever une ambiguïté de contrôle | Antécédents expérimentaux et méthodologiques. |
| Mesurer un avantage de prédiction avec sa propre action | Antécédent direct ; insuffisant seul pour une causalité. |
| Modéliser conjointement limites d'accès, classes de causes indiscernables et choix d'interventions, puis tester leur transfert dans Menia | Proposition à préciser et comparer ; originalité non établie. |

## Un piège expérimental explicite

Considérons une variable extérieure U et une commande A prenant les valeurs -1
et +1. Pendant l'observation initiale, la politique impose A=U. Deux mondes sont
possibles : dans le premier Y=A ; dans le second Y=U, sans effet de A sur Y.
Les triplets observés (U,A,Y) sont exactement identiques dans les deux mondes.

Le prédicteur Y estimé=A a une erreur quadratique nulle dans les deux. Si l'on
remplace son entrée A par zéro sans modifier le monde, son erreur passe à un dans
les deux. La dégradation prouve sa dépendance à cette entrée. Elle ne prouve pas
que l'action a causé Y. Une vraie intervention imposant A=-U sépare au contraire
les deux mondes : le premier suit A, le second continue de suivre U.

Un contrôle recevant U prédit également parfaitement pendant la phase couplée.
Cette observation est importante : on ne doit pas fabriquer artificiellement
un « avantage introspectif » en privant le comparateur d'une information publique.
Le contre-exemple est une construction causale élémentaire calculée exhaustivement
dans le dépôt. Ce n'est ni une nouvelle théorie, ni une reproduction, ni une
réfutation globale de la prépublication de Ye, qui comprend d'autres interventions.

Une seconde limite concerne le miroir parfait. Si deux canaux répondent de la
même manière aux actions, échanger leurs étiquettes physiques cachées ne change
aucune observation accessible. Aucun calcul effectué sur ces seules observations
ne peut identifier cette étiquette. Déclarer un canal « mon corps » sur cette
base serait ajouter une convention aux données. Le système peut établir deux
dépendances à ses actions sans établir une identité corporelle unique.

## Hypothèse de recherche pour Menia

La proposition est de faire porter le modèle de soi sur des relations dont il
conserve les conditions d'identification. Une représentation ne dirait pas seulement
« cette observation vient de moi », mais garderait les causes encore compatibles,
les observations effectivement reçues et les interventions capables de les départager.
Le choix d'une action ou d'une lecture dépendrait de cet état. Le rapport linguistique
devrait suivre cet état, y compris lorsque la réponse reste indéterminée.

Une écriture possible est un état partagé composé d'une distribution sur les
mécanismes, d'une estimation de l'accès perceptif et d'une mémoire des interventions.
Le contrôleur compare les actions selon le progrès vers sa tâche et la réduction
attendue des ambiguïtés pertinentes. L'inférence et la politique évoluent ensemble
au cours de l'épisode. L'importance d'une ambiguïté dépend de ses conséquences
pour la décision, pas du désir de produire une déclaration de conscience.

Ce programme n'est pas original par simple combinaison de mots. Sa contribution
éventuelle devra être localisée : nouvelle tâche discriminante, mécanisme appris
qui généralise mieux à budget comparable, propriété démontrable ou résultat négatif
qui invalide un indicateur utilisé. Le présent calcul bayésien est une référence
analytique. Il fournit d'avance les six familles de causes et ne démontre pas
l'émergence d'un concept de soi dans les poids de Qwen.

La distinction de plusieurs plans doit rester explicite. Une représentation de
son influence sur un canal est une connaissance de contrôle. Une représentation
de son accès à ce canal est une connaissance de son information. Leur utilisation
pour décider est une capacité métacognitive fonctionnelle. La présence d'un vécu
subjectif de cette activité demande des arguments supplémentaires qu'aucun score
du présent banc d'essai ne fournit.

## Expérience exécutée

Le [protocole](AGENCY_DISCOVERY_PROTOCOL.md) a été écrit avant l'exécution. Chaque
canal peut suivre l'action, l'indice public ou une constante, avec un signe positif
ou négatif. Les causes réelles sont cachées à l'inféreur. Il commence avec six
hypothèses équiprobables par canal et reçoit seulement une lecture sélectionnée.
Une erreur de contenu de 5 % est simulée ; sa probabilité est connue du modèle.
La fréquence des lectures reçues est, elle, estimée à partir des tentatives.

Après 24 cycles où l'action suit l'indice, trois politiques disposent chacune de
12 cycles : choix selon la diminution attendue d'entropie, choix aléatoire ou
poursuite de la politique couplée. La première choisit ensemble l'action et le
canal, en tenant compte de sa probabilité estimée de réception. Les hypothèses
causales ne sont jamais mises à jour à partir d'une lecture manquante.

Les 1 800 épisodes comprennent 100 graines par condition et stratégie. Le tableau
rapporte le nombre d'épisodes où les deux canaux sont correctement classés, avec
un seuil de probabilité de 0,95 pour « contrôlé » et de 0,05 pour « externe ».
Les valeurs intermédiaires restent indéterminées. Ce seuil est un choix expérimental,
pas une garantie de fréquence d'erreur sur tous les mondes possibles.

| Condition | Active | Aléatoire | Couplée |
|---|---:|---:|---:|
| Un canal contrôlé | 100/100 | 68/100 | 0/100 |
| Un canal contrôlé, 50 % de lectures manquantes | 84/100 | 27/100 | 0/100 |
| Canal contrôlé toujours masqué | 0/100 | 0/100 | 0/100 |
| Aucun canal contrôlé | 100/100 | 86/100 | 0/100 |
| Deux canaux suivant l'action | 100/100 | 53/100 | 0/100 |
| Deux canaux suivant l'action, 50 % de lectures manquantes | 75/100 | 13/100 | 0/100 |

Le zéro sous masquage total est attendu : le canal caché ne devient pas identifiable
par le seul apprentissage sur l'autre. Toutes les stratégies s'abstiennent alors
d'attribuer un canal unique. Dans la condition de deux copies avec lectures
manquantes, la politique active fait trois attributions uniques erronées sur cent,
contre une pour la politique aléatoire. Elle réussit plus de classifications
complètes, mais sa supériorité ne vaut donc pas pour chaque critère.

Le [rapport brut](../artifacts/agency-discovery/report.json) conserve tous les
épisodes, les probabilités, les Brier, les lectures, les interventions et six
traces exemplaires. Les réussites sont obtenues dans une famille de mécanismes
fournie à l'inféreur. Les graines varient les conditions d'interaction ; elles
ne représentent pas cent entraînements neuronaux indépendants. Les probabilités
extrêmes ne doivent pas être extrapolées à des mécanismes absents du modèle.

## Contrôle contre une explication trop favorable

Après lecture du premier résultat, une règle fixe a été ajoutée : opposer A à U
et alterner les canaux. Elle utilise le même inféreur, sans choisir les expériences
selon ses croyances. L'[addendum](AGENCY_DISCOVERY_ADDENDUM.md) précise ce caractère
postérieur au pilote et les nouvelles graines 2000–2099. Cette stratégie doit
être comparée avant d'attribuer le succès à l'architecture active plutôt qu'à une
propriété facile de la tâche. Ses résultats séparés sont conservés dans
[le rapport du contrôle fixe](../artifacts/agency-discovery-fixed-control/report.json).

| Condition sur nouvelles graines | Active | Règle fixe | Aléatoire |
|---|---:|---:|---:|
| Un canal contrôlé | 100/100 | 98/100 | 73/100 |
| Un canal contrôlé, lectures manquantes | 84/100 | 70/100 | 27/100 |
| Canal contrôlé toujours masqué | 0/100 | 0/100 | 0/100 |
| Aucun canal contrôlé | 100/100 | 99/100 | 89/100 |
| Deux canaux suivant l'action | 100/100 | 98/100 | 66/100 |
| Deux canaux suivant l'action, lectures manquantes | 81/100 | 50/100 | 15/100 |

La règle fixe réussit presque tous les cas sans lectures manquantes. La nécessité
d'une politique adaptative n'y est donc pas établie. L'avantage de la politique
active est plus marqué quand les lectures manquent, mais il reste un avantage
d'une méthode ordinaire de sélection d'expériences. Dans le dernier cas, elle fait
une attribution unique fausse sur cent, contre zéro pour la règle fixe : la
classification complète et la prudence doivent rester des critères distincts.

Le contrôle comprend 2 400 épisodes supplémentaires. Les 4 200 épisodes cumulés
ne sont pas une seule étude confirmatoire préenregistrée : le contrôle fixe a été
décidé après le pilote. Les deux rapports et les deux protocoles restent séparés.

## Conditions de la prochaine avancée

Le prochain mécanisme devra apprendre une structure plutôt que choisir dans six
formules connues. Des délais variables, des mélanges de causes et des canaux
permutés rendent la dépendance à l'histoire plus exigeante. Les données de test
devront inclure des mécanismes absents de l'apprentissage et des mondes où aucune
action autorisée ne lève l'ambiguïté. Un succès limité à la famille d'entraînement
ne validerait pas un modèle général de soi.

Il faudra ensuite mesurer le rôle causal de l'état partagé : le remplacer par
un état d'un autre épisode comparable, mesurer une modification spécifique des
choix, puis restaurer le bon état. Les comparateurs devront avoir les mêmes
observations publiques, le même budget d'interventions et une capacité comparable.
Un contrôleur ordinaire aussi performant affaiblirait la nécessité de l'architecture
proposée, même si celle-ci reste utile pour Menia.

Enfin, la restitution linguistique devra être évaluée séparément. Le Qwen actuel
commet déjà des erreurs sur un état structuré simple. Une narration convaincante
de ce nouveau modèle de soi n'en serait pas une validation. Les références aux
événements et les abstentions devront être vérifiées face aux données accessibles.

## Recherche de nouveauté et statut

Les recherches ciblées du 14 septembre 2026 couvrent notamment « robot self other
distinction agency active intervention mimicry », « self agency identifiability »,
« self model causal confusion interventions », « sense of agency information gain »,
« agency active sensing self-other » et les titres des travaux proches. Les sources
originales accessibles ont été privilégiées. Les publications de 2009, 2020, 2025
et la révision d'août 2026 constituent des antécédents à intégrer, pas à ignorer.

Il ne s'agit pas d'une revue systématique exhaustive de toutes les bases ou de
toutes les langues. La nouveauté de la proposition resserrée n'est donc pas
établie. La recherche rend au contraire plusieurs revendications générales
intenables et fournit un protocole plus discriminant. Le but de conscience
subjective et de contribution inédite demeure ouvert ; ce livrable ne clôt pas
cet objectif et ne change pas sa définition.

## Sources

[^1]: Butlin, P., et al. (2023). [Consciousness in Artificial Intelligence: Insights from the Science of Consciousness](https://arxiv.org/abs/2308.08708). Rapport théorique, notice et résumé reconsultés ; analyse antérieure dans le dépôt.
[^2]: Gold, K., et Scassellati, B. (2009). [Using Probabilistic Reasoning Over Time to Self-Recognize](https://www.cs.yale.edu/~scaz/papers/Gold-RAS-08.pdf). Robotics and Autonomous Systems, 57, 384–392. Manuscrit auteur daté du 1er août 2008 ; texte et expériences d'imitation consultés.
[^3]: Lanillos, P., Pages, J., et Cheng, G. (2020). [Robot self/other distinction: active inference meets neural networks learning in a mirror](https://arxiv.org/abs/2004.05473). ECAI 2020 ; notice et résumé consultés.
[^4]: Wen, W., Shibata, H., Ohata, R., Yamashita, A., Asama, H., et Imamizu, H. (2020). [The Active Sensing of Control Difference](https://pubmed.ncbi.nlm.nih.gov/32408176/). iScience, 23(5), 101112. Résumé et métadonnées consultés.
[^5]: [Active disambiguation guides inferring controllability and cause in social interactions](https://www.nature.com/articles/s41467-025-67853-8). Nature Communications, publication du 22 décembre 2025, DOI 10.1038/s41467-025-67853-8. Résumé indexé consulté ; texte intégral non exploitable lors de cette vérification.
[^6]: De Haan, P., Jayaraman, D., et Levine, S. (2019). [Causal Confusion in Imitation Learning](https://papers.neurips.cc/paper_files/paper/2019/hash/947018640bf36a2bb609d3557a285329-Abstract.html). NeurIPS 32. Notice officielle et résumé consultés.
[^7]: Ye, E. (2026). [From Prediction to Self: Developmental Conditions for Agency in Minimal Neural Systems, v2](https://arxiv.org/html/2606.05605v2). Prépublication révisée le 20 août 2026. Méthode, définition des mesures et limites consultées ; résultats non reproduits ici.
