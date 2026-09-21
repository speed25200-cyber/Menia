# Protocole : inférence de cause commune et attribution corporelle

Analyse secondaire exploratoire définie le 15 septembre 2026, après lecture de
Chancel, Ehrsson et Ma (2022), des en-têtes et des paramètres publiés, avant
l'exécution de notre modèle. Aucun préenregistrement indépendant n'est revendiqué.
La question est de savoir quelles contraintes cette étude apporte à une
construction candidate de Menia consciente de sa propre existence. Les jugements
humains d'appartenance d'une main ne constituent pas une mesure de conscience
artificielle.

## Sources et unités

Source primaire : [Uncertainty-based inference of a common cause for body ownership](https://doi.org/10.7554/eLife.77221),
version de référence datée du 25 octobre 2022 sur la page eLife. Trois classeurs
liés par l'article sont téléchargés sans modification. Leurs empreintes et
tailles seront enregistrées. Aucun fichier des auteurs ne sera exécuté.

Les données concernent 15 participants retenus après un test de susceptibilité
à l'illusion (18 recrutés). Pour chaque tâche : trois niveaux de bruit visuel,
sept délais, douze réponses par cellule, soit 252 jugements par personne. Les
classeurs contiennent des comptes agrégés ; aucun ordre d'essais ne sera inventé.
Les identifiants seront vérifiés et joints, pas déduits de l'ordre des lignes.

La tâche d'appartenance utilise les délais −500, −300, −150, 0, 150, 300 et
500 ms. La tâche de synchronie utilise −300, −150, −50, 0, 50, 150 et 300 ms.
Leur distribution de stimulation n'est donc pas identique. Les paramètres communs
du modèle sont une hypothèse de transfert, pas un fait déduit des instructions.

## Calculs fixés

Implémenter séparément :

1. La vraisemblance des signaux sous une ou deux causes.
2. La probabilité postérieure de cause commune.
3. La règle de réponse, avec un seuil explicite et un taux de réponse aléatoire.

Les équations sont celles de l'annexe 1 de l'article : sous une cause, le délai
latent vaut zéro ; sous deux causes il suit une gaussienne centrée ; la mesure
est bruitée par une gaussienne de variance dépendant du niveau de bruit.
Un second modèle utilise un critère fixe en millisecondes, tout en conservant
un bruit d'encodage variable. Il n'est pas un contrôle « sans incertitude ».

Rejouer les paramètres publiés, sans les réoptimiser : 15 modèles BCI et 15 à
critère fixe pour l'appartenance, puis 15 modèles joints à paramètres communs
et 15 à priors distincts. Cela donne 60 vraisemblances. La négative de la
log-vraisemblance (NLL) utilisera les comptes oui/non, sans constante binomiale,
comme l'article. Les comparaisons d'AIC/BIC portent sur ces mêmes observations.
Elles évaluent un ajustement sur les données d'origine, pas une prédiction hors
échantillon. Le gain d'un modèle ne démontrera pas son unicité.

Le bruit de source sera d'abord fixé à 348 ms, valeur imprimée dans le texte.
Une vérification séparée utilisera `sqrt(mean(s²))` sur les six délais non nuls
de la tâche d'appartenance, soit `sqrt(120833.33333333333)` ms, pour examiner
l'effet éventuel de l'arrondi. Ces deux conventions seront conservées dans le
rapport. Une meilleure concordance ne sera pas assimilée à une preuve du code
original, qui n'est pas fourni dans les classeurs examinés. Aucun autre réglage
ne sera choisi pour effacer une divergence.

Pour chaque comparaison, un bootstrap de 10 000 échantillons de 15 participants
rapportera les intervalles percentiles de la somme des différences, avec graines
71500 (BCI moins critère fixe) et 71501 (priors différents moins paramètres
communs). Tous les individus sont conservés. AIC et BIC ont la même différence
pour les deux modèles à cinq paramètres ; pour le modèle joint à six paramètres,
la pénalité supplémentaire est calculée sur 504 jugements par personne.

## Équivalence à vérifier et portée

Dans une règle qui compare le logarithme des chances postérieures à un seuil
`c`, les réponses dépendent de `logit(prior) − c`. Pour une valeur de prior
publiée `p`, la transformation `p0 = 0,5`, `c0 = logit(p0) − logit(p)` doit
conserver les réponses, même si elle modifie les probabilités postérieures
internes. L'article reconnaît déjà une lecture possible du prior comme biais
perceptif ou décisionnel. L'identité n'est pas une découverte revendiquée.

La vérification portera sur les 30 réglages tâche/participant du modèle joint
à priors distincts, les trois niveaux de bruit et 301 mesures de délai entre
−1500 et 1500 ms. Elle comparera les probabilités de réponse et les postérieurs
internes. Les délais supplémentaires sont des calculs du modèle, pas des
observations humaines nouvelles. Le prior 0,5 est une référence algébrique,
pas une estimation de ce que croient réellement les participants.

L'analyse distinguera une modification du calcul d'inférence d'une modification
du seuil de réponse. Aucun signal nommé « appartenance » ni aucune probabilité
postérieure ne sera identifié par définition à une expérience vécue. Le code
sera un module de recherche, sans intégration implicite au chat ou à l'iPhone.

## Vérification

Des tests vérifieront l'intégration gaussienne contre une intégration numérique,
les probabilités extrêmes, l'équivalence prior/seuil, la jointure par identifiant
et la conservation de tous les comptes oui/non. Une seconde exécution reproduira
le rapport complet. Les sources ne seront ni modifiées ni incluses dans Git.

## Amendement après échec de validation, avant les calculs de vraisemblance

La première exécution s'est arrêtée sur les valeurs de synchronie de S4 : 17 des
21 cellules sont fractionnaires (par exemple 3,6 et 7,2). Elles ne peuvent donc
pas être traitées comme des comptes binomiaux sur douze essais sans information
supplémentaire. Aucun dénominateur alternatif ni arrondi ne sera inventé.

Le bloc entier de synchronie de S4 sera exclu uniquement du rejeu binomial.
L'appartenance de S4 reste analysée. La comparaison BCI/critère fixe conserve
15 personnes ; celle des modèles joints porte sur les 14 autres. Le bootstrap
joint tirera 14 personnes, avec la graine initialement fixée. Il y aura donc
58 vraisemblances rejouées, sur 609 cellules utilisables, correspondant à
7 308 jugements si l'on applique le dénominateur de douze indiqué par l'article.
Ce total n'est pas une certification d'effectif obtenue depuis des essais bruts.

Le bloc exclu, ses valeurs originales et les NLL publiées seront conservés dans
le rapport. Les différences de modèle seront confrontées aux NLL publiées des
mêmes personnes, sans prétendre reproduire la somme originale sur quinze. Le
test algébrique prior/seuil peut conserver les 30 réglages publiés : il n'emploie
pas ces valeurs comme des observations binomiales.

## Amendement de qualification après le premier rejeu

Le rejeu retrouve les NLL du critère fixe, mais pas exactement celles du BCI ;
les écarts sont plus importants dans les modèles joints. Les deux conventions
de bruit de source prévues n'éliminent pas ces écarts. Les paramètres publiés
ne seront donc pas présentés comme des maximums vérifiés de notre implémentation.
Les scores recalculés seront nommés « NLL pénalisées de forme AIC/BIC » et
conservés comme diagnostic, sans conclusion sur des modèles réoptimisés.

Un calcul séparé effectuera l'arithmétique AIC/BIC depuis les colonnes NLL des
classeurs, sur les quinze personnes et sur les quatorze retenues pour le rejeu
joint. Il vérifiera la concordance de ces valeurs avec les tableaux de l'article,
sans les présenter comme des vraisemblances retrouvées depuis les observations.
Le supplément de prépublication disponible dans OSF a été examiné pour les
conventions de modélisation ; il ne donne pas le code de calcul requis pour
attribuer une cause précise aux écarts. La reproduction reste partielle.
