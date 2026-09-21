# Protocole de réanalyse de la présence en réalité virtuelle

La question est de déterminer si les données de l'étude 2 de Maymon et al.
permettent de choisir un mécanisme causal reliant état corporel, peur rapportée
et présence rapportée. Cette question contraint la [candidate de Menia](INTEROCEPTIVE_PRESENCE_CANDIDATE.md).
La présence dans un environnement virtuel ne constitue pas une mesure générale
de conscience de sa propre existence.

## Statut et données

Analyse secondaire exploratoire définie le 15 septembre 2026, après lecture de
l'article et du script des auteurs, et après inspection des en-têtes, identifiants,
formules et exclusions du classeur. Les résultats publiés sont donc connus.
Ce fichier précède les ajustements de cette réanalyse ; il ne constitue pas un
préenregistrement indépendant.

Sources : [Maymon et al., 2024](https://doi.org/10.1037/xge0001576),
[données et script publics](https://osf.io/6s3mf/). Les trois feuilles de l'étude 2
portent sur les évaluations, la fréquence cardiaque (HR) et la conductance
cutanée (SCL). Chacune contient 60 identifiants distincts, avec le même ordre.
Les sept exclusions indiquées par le script sont 14, 15, 30, 27, 45, 58 et 38.
Il reste 53 personnes, 27 en condition contrôle et 26 en condition hauteur.
Les données sources restent intactes et hors Git. Les trois fichiers téléchargés
seront identifiés par taille et empreinte SHA-256.

## Transformation et vérifications

Les feuilles seront jointes par identifiant, avec vérification de l'accord des
conditions et de l'unicité. Les lignes sans identifiant ne seront pas des
participants. Toute valeur nécessaire manquante chez un participant retenu
arrêtera l'analyse ; aucune valeur ne sera imputée.

La référence est la moyenne des fenêtres `Curb` et `Bottom`. La période de
planche est la moyenne de `Top`, `Start` et `End`. Les trois changements sont
calculés comme planche moins référence. Pour SCL, on applique `log(1 + SCL)`
à chaque fenêtre avant les moyennes, conformément au script de l'étude 2.
Les questionnaires de l'autre feuille, les commentaires libres, les mouvements
et l'étude 1 ne seront pas analysés.

## Modèles fixés

Pour chaque médiateur candidat `M` (changement de peur, HR, SCL), on ajuste :

```text
M = alpha + a * hauteur + u
présence_planche = beta + c_direct * hauteur + b * M + v
présence_planche = gamma + c_total * hauteur + w
```

Le produit `a*b` est d'abord une décomposition statistique. Sa lecture causale
demande notamment des hypothèses sur les causes communes de M et de la présence.
L'affectation à la hauteur ne randomise pas directement le médiateur.

Une analyse supplémentaire ajoutera la présence de référence à chaque équation.
Elle vérifie une dépendance à l'ajustement initial ; elle n'est pas présentée comme
une garantie d'absence de confusion. Aucune sélection du meilleur résultat ne sera
effectuée entre les six ajustements.

Les intervalles percentiles seront calculés avec 10 000 rééchantillonnages de
participants, stratifiés par condition (27/26 conservés), avec graines 94100 à
94105 dans l'ordre sans puis avec ajustement de référence, et peur/HR/SCL dans
chaque série. Cette procédure ne cherche pas à reproduire les tirages PROCESS
des auteurs. Les estimations ponctuelles seront confrontées aux nombres publiés,
en conservant les divergences au lieu de modifier le filtrage pour les effacer.

## Sensibilité à une cause commune non mesurée

Dans le modèle linéaire sans interaction, `rho` désigne la corrélation supposée
entre les deux erreurs structurelles. La méthode suit
[Imai, Keele et Yamamoto (2010), section 5](https://imai.fas.harvard.edu/research/files/mediation.pdf).
Il s'agit d'un paramètre de sensibilité non identifié par les données.

On calculera le produit indirect pour `rho` de -0,8 à 0,8 par pas de 0,1,
ainsi que la valeur qui annule l'estimation ponctuelle. En utilisant les résidus
des modèles de `M` et de la présence conditionnellement aux covariables `C` :

```text
V_M = Var(M - prévision_M(C))
V_Y = Var(Y - prévision_Y(C))
K = Cov(résidu_M, résidu_Y)
b_rho = K/V_M - rho * sqrt((V_Y - K²/V_M)/V_M) / sqrt(1-rho²)
indirect_rho = a * b_rho
rho_annulation = K / sqrt(V_M * V_Y)
```

Toutes les variances et covariances emploient le même diviseur N. Une seconde
expression issue directement du théorème 4 vérifiera le calcul. Une valeur
d'annulation n'est ni la probabilité d'une confusion ni la quantité de conscience.
Cette sensibilité ne traite pas à elle seule les erreurs de mesure, la causalité
réciproque, la non-linéarité ou les confondants créés par le traitement.

## Direction causale concurrente

Pour la peur, les deux ordres `peur puis présence` et `présence puis peur` seront
ajustés comme deux factorisations du même modèle gaussien conditionnel. On
reconstruira leurs moyennes et covariances pour vérifier leur équivalence dans
cette famille, puis on explicitera leurs effets différents sous intervention
sur le médiateur. Les deux modèles autorisent un effet direct de la hauteur.

Cette égalité concerne les distributions gaussiennes ajustées, pas une preuve
que les évaluations ordinales humaines suivent exactement ces distributions.
Elle ne montre pas que le mécanisme inverse est vrai. Elle demande si les données
observées suffisent à préférer une direction sans hypothèse supplémentaire.

## Portée et critères de conclusion

Les associations répliquées soutiendront leur reproductibilité sur cette livraison.
Les intervalles couvrant zéro ne démontreront pas l'absence de tout effet.
Un effet total de hauteur pourra être distingué de l'identification du chemin
qui le produit. Aucun résultat ne sera converti en certificat de conscience
artificielle, en nécessité de provoquer la peur ou en preuve de nouveauté.

Le livrable comprendra code de réanalyse, tests des calculs, rapport numérique et
interprétation pour Menia. Les fichiers des auteurs seront lus sans exécuter
leur script R. Une éventuelle discordance de nombres sera documentée comme
une limite de reproduction, sans en inventer la cause.

## Amendement de lecture, avant tout ajustement

Le premier chargement a été arrêté : `Study 2 - HR!D23`, mesure `Curb` du
participant 43, possède le format Excel `mmss.0`. Le lecteur la convertit en
date, alors que le XML stocke le nombre `112.06100000000001`, sans formule.
La réanalyse lira cette valeur numérique explicite et consignera cette exception.
Il ne s'agit ni d'une imputation ni d'une modification du classeur source.
Ce traitement a été décidé après l'échec de lecture et avant tout résultat ajusté.
