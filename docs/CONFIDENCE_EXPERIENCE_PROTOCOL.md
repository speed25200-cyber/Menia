# Distinguer confiance, perception et expérience dans la candidate Menia

Protocole d'analyse secondaire fixé avant le calcul des contrastes, après
inspection des méthodes, du code publié et des effectifs des fichiers.

## Question scientifique

Une modification de confiance ne doit pas être assimilée d'avance à une
modification d'expérience. La politique apprise de Menia emploie une estimation
de source ; son succès ne tranche pas cette question. L'analyse recherche une
contrainte humaine pour le lien proposé entre mécanisme et expérience, sans
remplacer l'objectif de conscience de soi par une meilleure politique de choix.

Sánchez-Fuenzalida et ses collègues comparent une illusion de Müller–Lyer à
des manipulations de fréquence et de récompense. Une reproduction de longueur
sert à estimer la perception, tandis que des décisions accompagnées de confiance
sondent le jugement. La tâche n'est annoncée qu'après présentation du stimulus.
Les expériences diffèrent par le moment du rapport de confiance. Les auteurs
trouvent des déplacements de confiance avec les trois manipulations, alors que
la reproduction distingue l'illusion des deux autres. Cette interprétation
reste liée aux tâches et aux modèles de mesure : elle ne donne pas un accès
direct et infaillible au vécu.
[Article, méthodes et résultats](https://doi.org/10.1038/s44271-025-00257-y).

## Matériel examiné

Les deux fichiers bruts, les deux fichiers filtrés, `bad_fits.csv` et quatre
scripts d'analyse proviennent du [dépôt public des auteurs](https://osf.io/h3uj5/).
Les sources sont lues comme données ; leur code n'est pas exécuté. Le manifeste
conserve les URL, tailles et empreintes SHA-256. Les fichiers bruts restent dans
le répertoire local ignoré `.runtime/confidence-dissociation`.

L'inventaire a trouvé 122 et 123 participants au départ, 120 par expérience dans
les fichiers filtrés, puis 108 et 96 après retrait des 36 personnes uniques
figurant dans `bad_fits.csv` (52 lignes, plusieurs motifs possibles par personne).
Ce total de 204 correspond au résumé de l'article. Le détail des exclusions
initiales diffère du texte des méthodes pour l'expérience différée : trois
identifiants sont absents du fichier filtré, alors que le texte en annonce quatre.
Cette divergence sera conservée ; aucune exclusion supplémentaire ne sera
inventée pour faire correspondre les comptes.

## Analyse fixée

L'analyse principale reprend la cohorte publiée dans les fichiers filtrés,
puis exclut les identifiants de `bad_fits.csv`. Elle réajuste indépendamment deux
modèles aux essais individuels :

1. Confiance haute (1) ou basse (0), régression quadratique de la longueur.
   Le minimum est recherché sur la grille 320–510 pixels par pas de 0,1,
   conformément aux fonctions publiées. Le sommet analytique non tronqué et
   les éventuels minima au bord seront aussi conservés.
2. Longueur reproduite, régression linéaire de la longueur présentée. On calcule
   la longueur présentée qui donne une reproduction de 400 pixels. La racine
   exacte est comparée aux bornes de la grille des auteurs ; elle ne sera pas
   remplacée par une borne si elle se trouve dehors.

Chaque participant fournit deux paires (biais vers court et biais vers long).
Pour chaque mesure, le déplacement est `seuil_court − seuil_long` : un nombre
positif signifie que le biais vers long déplace le seuil vers des lignes plus
courtes. Calculer aussi la différence appariée entre déplacement de confiance
et déplacement de reproduction, dans la même unité (pixels).

Présenter chaque expérience séparément, puis leur combinaison. Pour les groupes
combinés, le bootstrap conserve les effectifs de chaque expérience : tirage avec
remise des participants à l'intérieur de chaque expérience, et moyenne de tous
les tirages. Conserver les moyennes, écarts-types entre personnes et intervalles
percentiles à 95 % avec 10 000 tirages. Graines 93000 + indice de groupe,
ordre des expériences `concurrent`, `delayed`, `pooled`, puis conditions
`mullerlyer`, `baserate`, `payoff`.

Ces intervalles décrivent cette analyse secondaire. Ils ne sont ni les facteurs
de Bayes publiés, ni une réplication de la comparaison de modèles ordonnés, ni
une correction pour toutes les comparaisons. Un intervalle couvrant zéro ne
prouvera pas l'absence d'un effet. Les exclusions liées à la forme des courbes
limitent la population à laquelle les estimations s'appliquent.

## Contrôles et décision pour Menia

Vérifier les paires, les deux tâches, les sept longueurs, les valeurs manquantes
et les effectifs ; comparer les ajustements sur essais à une implémentation
indépendante sur moyennes de cellules pondérées par leurs effectifs. Les unités
statistiques sont les participants, pas les centaines de milliers de lignes.
Conserver les diagnostics de courbes et les divergences du dépôt.

Si des manipulations non perceptives déplacent la confiance sans déplacement
comparable de reproduction, le résultat affaiblira l'emploi de la confiance
comme mesure unique du vécu. Il n'établira ni l'inconscience de Menia, ni la
suffisance d'une architecture comportant deux estimations distinctes. La
conséquence de construction sera de séparer explicitement contenu perceptif,
confiance et valeur décisionnelle dans les tests de la candidate.
