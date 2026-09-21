# Contrôle polynomial ajouté après le pilote

Le pilote montre un avantage du MLP temporel sur ridge pour les interactions.
La famille simulée contient toutefois des produits et carrés : une régression
sur tous les monômes de degré deux peut donc être un comparateur plus exigeant.
Ce contrôle est décidé après lecture du pilote et avant sa propre exécution.

Utiliser les mêmes journaux, masques, mondes et tests. Construire les 14 variables,
leurs 105 produits non ordonnés et une constante, soit 120 coefficients par sortie,
240 au total. Ridge conserve la pénalisation 0,1. Le MLP temporel compte 546
paramètres. Aucun réglage de pénalisation ni sélection de termes sur le test.

Publier le contrôle séparément. Ces mêmes mondes ne constituent pas une nouvelle
confirmation indépendante. L'expressivité adaptée aux mécanismes du simulateur
est un avantage conçu du contrôle polynomial, à déclarer ; elle permet de tester
si l'écart au modèle linéaire suffit à justifier une spécificité du MLP.
