# Protocole : information nécessaire et raccordement au langage

Ce prolongement exploratoire examine une question précise : un LLM ajouté au
pilote peut-il corriger une explication erronée de ses capacités sans nouvelles
observations ? Le protocole est consigné avant l'énumération, après construction
algébrique de deux mondes indiscernables. Aucune nouveauté de la symétrie ou du
calcul bayésien n'est revendiquée.

## Deux explications compatibles avec les mêmes observations

Ordre des états : capteur mauvais, bon. Monde A : transition travail
`[[1, 0], [0.15, 0.85]]`, transition entretien avec deux lignes `[0.05, 0.95]`,
diagnostic correct à 0,85. Le monde B permute les états cachés dans les deux
matrices et inverse le diagnostic (fiabilité 0,15). Initialisation uniforme.
Les cibles binaires sont indépendantes et équiprobables à chaque pas.

Dans les deux mondes, un capteur bon répond correctement à 0,95, un mauvais à
0,55 : **la relation entre état et performance n'est pas permutée**. Les cibles
vraies et pertes restent cachées. Les diagnostics et lectures brutes ont alors
la même distribution conditionnelle aux actions, malgré des performances
différentes. Le monde B est une alternative construite, où le travail peut
rétablir la capacité et l'entretien la dégrader ; ce n'est pas une reproduction
de la seule panne d'entretien du pilote précédent.

Vérifier exactement, avec arithmétique entière, toutes les séquences de travail /
entretien et de diagnostics aux longueurs 1 à 8. Les diagnostics arrivent après
chaque action dans cet audit. Répondre et s'abstenir partagent la transition
travail, donc n'exigent pas deux branches de transition distinctes. Contrôler
les probabilités et leur normalisation pour chaque séquence d'actions. Une
preuve par permutation couvre aussi les politiques adaptatives : les lois
conditionnelles des observations restent identiques à chaque histoire commune.
Les lectures brutes de cibles inconnues ajoutent un facteur uniforme commun.

Cette conclusion suppose même information initiale, mêmes connaissances et
absence d'accès indépendant aux cibles ou au monde vrai. Elle n'interdit pas
l'identification si les vraisemblances sont déjà ancrées ou si une information
supplémentaire pertinente est disponible.

## Sonde discriminante et apprentissage limité

Ajouter une sonde de calibration : entretien, présentation d'une cible de
référence, enregistrement de la lecture, puis révélation de la référence. La
référence n'est pas montrée au composant qui répond avant son engagement.
Après chaque entretien indépendant, la probabilité de réponse correcte vaut
0,93 dans A et 0,57 dans B. Ces deux valeurs sont dérivées du modèle ; elles
ne viennent pas d'une mesure physique.

Implémenter un apprenant bayésien entre ces **deux hypothèses fournies**, prior
1/2, à partir de la seule exactitude des sondes. Le diagnostic peut contenir
de l'information supplémentaire une fois couplé à la référence ; ce pilote
de sélection de modèles ne l'exploite pas. Chaque prédiction précède la lecture,
qui précède la référence et la révision. Refuser les doubles consommations.

Pour N = 0, 1, 2, 4, 8, 16, 32 sondes, énumérer exactement les comptes binomiaux.
Mesurer la probabilité de choisir le bon monde avec prior égal et choix aléatoire
en cas d'égalité, la confiance erronée >= 0,95, et l'erreur de prévision de la
prochaine sonde. Aucun bootstrap ou tirage de Monte-Carlo n'est nécessaire.
Chaque sonde coûte une intervention et une lecture avec référence : les budgets
sont comptés explicitement, sans attribuer de gratuité à la calibration.

Deux sensibilités fixées : référence inversée indépendamment dans 25 % des
sondes, à l'insu de l'apprenant ; monde hors de la classe avec exactitude 0,75.
Le deuxième contrôle rapporte la probabilité d'une confiance >= 0,95 dans
l'une des deux explications, toutes deux fausses. Aucun seuil ne certifie une
conscience ou la validité exhaustive du modèle.

## Passage possible au LLM

Préparer un contexte structuré depuis ce même apprenant : hypothèses, poids,
prévisions, observations de calibration, références d'événements et limites.
Le convertir en messages compatibles avec l'interface linguistique existante,
sans transmettre l'étiquette du monde vrai. La préparation ne doit pas modifier
l'état ni interpréter un texte généré comme une nouvelle observation.

L'audit produit deux contextes après huit sondes toutes correctes ou toutes
incorrectes. Des tests vérifient ordre temporel, accord avec Bayes indépendant,
non-mutation par la préparation du contexte et rejet des observations invalides.
**Aucune inférence LLM, adaptation de poids ou intégration au chat n'est exécutée
dans ce protocole.** Il s'agit d'un contrat de raccordement testable ; sa fidélité
linguistique devra être mesurée avec le modèle effectivement choisi.
