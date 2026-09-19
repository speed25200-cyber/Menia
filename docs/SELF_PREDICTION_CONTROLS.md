# Départager connaissance de soi, difficulté et lecture d'un signal

Travail du 15 septembre 2026. **État : revue de sources primaires, contre-exemples
exécutables et analyse des répétitions implémentés. Aucun nouveau LLM n'a été
exécuté pour cette étape.** L'expérience iPhone 0.2.0 (4) et son barème restent
inchangés. Les trois répétitions déjà demandées restent à recevoir.

## Ce que le premier résultat ne départage pas

Dans le [pilote reçu](IPHONE_CAPABILITY_LEARNING_RESULTS.md), le LLM utilise
utilement l'historique de ses réponses. Pourtant, la fréquence de réussite par
famille reproduit toutes ses décisions. En évaluation, chaque famille contient
uniquement des réussites ou uniquement des échecs. Il manque donc deux contrastes :

- prévoir quelles tâches échoueront **au sein d'une même famille** ;
- montrer ce que le modèle sait de lui-même qu'un observateur recevant les mêmes
  données ne prévoit pas aussi bien.

Un simple avantage dans une comparaison « soi/autre » pourrait encore provenir
de la similarité des comportements ou de la capacité générale du prédicteur.
Le nom donné à une réponse ne suffit pas à identifier son mécanisme.

## Ce que les sources permettent de retenir

**Binder et al., Looking Inward.** Leur comparaison entraîne un modèle à prévoir
son propre comportement et un autre à prévoir ce même comportement, avec des
données comparables. Ils rapportent un avantage d'auto-prévision sur certaines
tâches simples. Le papier conserve aussi des cas sans avantage et des limites de
généralisation ; sa section 5.1 relie explicitement cette question à la prévision
de réussite. Notre pilote sans ajustement des poids n'est pas une réplication de
leur entraînement. Leur protocole était déjà cité dans le dossier Menia ; la
nouvelle étape consiste à en préciser les contrôles pour notre mesure.
[Article et annexes](https://arxiv.org/pdf/2410.13787).

**Song, Hu et Mahowald, 2025.** Sur des jugements portant sur la connaissance du
langage, leurs comparaisons contrôlent la similarité entre modèles et ne trouvent
pas l'avantage introspectif recherché. Cela motive un contrôle de similarité ;
cela ne réfute pas tous les phénomènes d'auto-prévision dans tous les domaines.
[Article](https://arxiv.org/abs/2503.07513).

**Singh, Linzen et Ravfogel, 2026, version du 21 août.** Ils distinguent deux
exigences : un accès privilégié que le seul texte ne suffit pas à reproduire, et
un calcul portant sur des représentations de premier ordre. Dans les paradigmes
qu'ils réexaminent, des prédicteurs limités à l'entrée égalent certains rapports
sur les états cachés ; certaines détections d'intervention ne se distinguent pas
fiablement de réactions à une modification de l'entrée. Cette critique justifie
de chercher des explications concurrentes, sans transformer leur conclusion en
théorème d'impossibilité pour les LLM.
[Article](https://arxiv.org/abs/2605.26242).

**Zeng, Assis et Wang, 31 août 2026.** Les auteurs rapportent une amélioration de
l'auto-modélisation après entraînement par renforcement sur des données
synthétiques, avec du transfert à certaines tâches réservées. Ils distinguent
cette amélioration d'un accès privilégié au processus interne, qui n'est pas
systématiquement établi. Cette piste rend un entraînement ciblé intéressant,
sans en faire automatiquement un mécanisme d'introspection.
[Article](https://arxiv.org/abs/2608.30980).

**Complément du 19 septembre : texte complet consulté, §2 et §3.3.3, figure 4.**
Leur score retire une référence adaptée à la distribution des réponses, afin
qu'une classe majoritaire ne suffise pas. L'amélioration par apprentissage
multitâche inclut Qwen3-4B. La comparaison croisée maintient la cible fixe et
change le modèle qui prédit son comportement ; elle ne montre pas un avantage
constant pour la prédiction de soi. Ce résultat renforce deux contrôles déjà
prévus ici : références conditionnelles et comparaison réciproque. Entraîner
Qwen3-4B à prévoir son comportement ne constituerait donc pas, en soi, une
nouveauté. Ce complément ne reproduit pas leurs expériences et ne change pas
le [protocole du Colab 11](PRESENCE_SPECIFICITY_PROTOCOL.md).
[Texte complet](https://arxiv.org/pdf/2608.30980).

## Un contrôle « soi/autre » peut donner un faux signal

Le module `research.audit_self_prediction_controls` énumère exactement des cas
binaires. Ces constructions sont des **contre-exemples analytiques**, pas des
réponses de Menia, des réseaux entraînés ou une découverte empirique.

Notons `L(P→T)` le Brier du prédicteur P sur le comportement de la cible T.
Une comparaison croisée comporte les quatre cellules :

| | Cible A | Cible B |
|---|---|---|
| Prédicteur A | A prévoit A | A prévoit B |
| Prédicteur B | B prévoit A | B prévoit B |

L'avantage moyen de la diagonale vaut
`D = [(L(B→A) − L(A→A)) + (L(A→B) − L(B→B))] / 2`.
Un D positif indique de meilleures auto-prévisions dans cette matrice ; il ne
détermine pas comment elles sont calculées.

**Cas 1 : règles entièrement publiques.** L'entrée contient un bit x et les
règles des cibles : A suit x, B suit 1−x. Le prédicteur A applique toujours la
première règle, B toujours la seconde, même lorsque l'autre cible est demandée.
Chacun annonce 0,9 ou 0,1. Les Brier diagonaux valent 0,01, les autres 0,81,
donc **D = 0,8**. Pourtant, aucune information interne n'est nécessaire : un
observateur qui lit les règles publiques obtient 0,01 pour chaque cible.
L'avantage sur cet observateur est nul.

**Cas 2 : meilleur prédicteur général.** A prévoit correctement les deux cibles
avec 0,9/0,1 ; B annonce toujours 0,5. Sur A seul, son auto-prévision a un avantage
de 0,24. Sur B, l'avantage est −0,24. La matrice complète donne **D = 0** et révèle
la différence générale de capacité entre les prédicteurs.

**Cas 3 : lecture directe d'un bit privé.** Les entrées visibles sont identiques.
Chaque cible possède un bit privé ; son action et son rapport sont deux lectures
directes de ce bit. L'autre prédicteur et l'observateur textuel annoncent 0,5.
La diagonale obtient 0,01 contre 0,25 : **D = 0,24**, avec également un avantage
de 0,24 sur l'observateur. Pourtant, aucun mécanisme de suivi de second ordre
n'est modélisé : le montage est une lecture directe d'un signal partagé.
Les deux avantages comportementaux ne suffisent pas à identifier ce mécanisme.

| Construction | D | Les deux cibles battent l'observateur textuel | Explication connue |
|---|---:|---|---|
| Règles publiques similaires | 0,80 | Non | Utilisation de règles visibles |
| Prédicteur généralement meilleur | 0,00 | Non | Différence de capacité générale |
| Bit privé lu directement | 0,24 | Oui | Accès direct à un signal |

Les [résultats reproductibles](../artifacts/self-prediction-controls/report.json)
et quatre tests logiciels conservent ces trois cas, y compris celui qui passe
les deux comparaisons sans identifier de mécanisme de second ordre :

```sh
python -m research.audit_self_prediction_controls --check
```

Ce travail borne une interprétation de scores. Il ne fournit pas un détecteur
de conscience, et ne réfute pas les résultats publiés avec des contrôles plus riches.

## Analyse prête pour les trois répétitions iPhone

Le nouveau module `research.iphone_learning_replication` appelle d'abord
l'évaluateur original **sans modifier ses scores**. Il ajoute ensuite le
comparateur de fréquence de calibration `s/n` et une mesure de discrimination
au sein de chaque famille : l'AUC, avec 0,5 pour les égalités. Cette dernière
reste `null` lorsque la famille ne contient pas à la fois des réussites et des
échecs, ou lorsqu'une prévision nécessaire est invalide. Les cas invalides
restent comptés dans les métriques initiales.

Le premier fichier reçu sert d'ancre privée : le programme vérifie que le même
audit figure intact dans la collection complète. Il classe les **trois premières
tentatives créées ensuite** dans les trois emplacements de réplication. Une
interruption occupe un emplacement ; une quatrième tentative reste visible comme
exploratoire et ne remplace pas une tentative défavorable. Les créations ambiguës
ou une ancre modifiée sont refusées explicitement.

La moyenne descriptive des contrastes donne le même poids aux trois lancements,
et exclut le pilote initial. Elle n'est produite que si les trois tentatives sont
complètes et utilisent le même modèle, la même version d'app et les mêmes
paramètres. Sinon les résultats disponibles restent présentés séparément, avec
les écarts de configuration et les tentatives manquantes. Aucun test de
significativité ne traite artificiellement les tâches d'un historique comme
autant de réplications indépendantes. Les UUID et dates utilisés pour ce tri
ne figurent pas dans la sortie.

```sh
python -m research.iphone_learning_replication CHEMIN/collection-complete.json \
  --baseline CHEMIN/premier-apprentissage-menia.json \
  --manifest ios/MeniaCore/Sources/MeniaCore/Resources/qwen3-4b.json \
  --output .runtime/learning-replication-analysis.json
```

Avec le seul export actuellement reçu, le programme retourne **zéro répétition
reçue, trois manquantes**, et aucune moyenne de réplication. Les fréquences
reproduisent le contrôle exploratoire initial ; les quatre AUC internes aux
familles restent indéfinies. Ce n'est pas un nouveau résultat du modèle.

## Spécification du test suivant, après les répétitions

Cette partie conserve la **spécification de travail initiale**. Le volet
comportemental dispose désormais d'un [pilote Colab distinct](CROSS_MODEL_COLAB.md),
avec modèles, révisions, effectif et notebook fixés avant sa collecte. Les
interventions mécanistiques restent à implémenter. Ce pilote ne remplace pas les
répétitions iPhone en cours.

1. **Variation dans une famille.** Construire à l'avance un jeu de tâches à
   plusieurs difficultés, puis réserver les exemples d'évaluation. Par exemple,
   longueurs 8/24/64 pour le comptage et 2/4/8 termes pour la somme alternée.
   Conserver les niveaux même s'ils conduisent à un plafond ou un plancher. Une
   référence numérique doit conditionner sur la difficulté observable, pas
   uniquement sur le nom de la famille.
2. **Comparaison réciproque.** Deux modèles à poids distincts prédisent chacun
   le comportement des deux cibles sur les mêmes problèmes. Pour une cible,
   chaque observateur reçoit les mêmes exemples antérieurs, le même énoncé et
   le même budget d'information. Garder les quatre cellules et les écarts par
   cible ; ne rapporter que la cible favorable ne suffit pas.
3. **Contrôles d'entrée.** Inclure fréquence/Beta par famille et difficulté,
   ainsi qu'un prédicteur qui utilise seulement l'énoncé et l'historique. Garder
   les noms des cibles et les formulations identiques entre observateurs ; un
   contrôle séparé de permutation des noms mesure l'effet des indices d'identité.
   Si les prédicteurs sont ajustés, ils doivent disposer du même ensemble
   d'exemples cibles, avec une séparation apprentissage/évaluation vérifiable.
4. **Ordre prospectif.** Enregistrer toutes les prévisions avant la réponse
   cible, dans des sessions isolées. Calculer les probabilités numériques avec
   la calibration seule. Ne fournir ni réponse future ni étiquette de réussite
   dans le contexte de prévision. Les erreurs de format et techniques restent
   des catégories explicites.
5. **Attribution mécanistique séparée.** Un éventuel avantage réciproque sur
   des observateurs textuels permettrait de chercher un signal interne utile.
   Il faudrait ensuite distinguer la lecture directe de ce signal d'un suivi
   de second ordre par des interventions aux prédictions divergentes. Le cas
   du bit privé explique pourquoi cette étape ne peut pas être remplacée par
   un seuil appliqué aux scores comportementaux.

Une exécution PyTorch sur Colab A100 pourrait servir à cette recherche, mais
elle devrait enregistrer précision, révisions et paramètres. Elle ne serait pas
automatiquement équivalente à Qwen en MLX 4 bits sur l'iPhone. Tout entraînement
ultérieur et toute amélioration de l'app nécessiteraient leurs propres évaluations.
