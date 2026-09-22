# Pilote d'estimation de ses propres erreurs

Plan fixé le 14 septembre 2026, avant l'exécution de ce pilote.

## Question

Un estimateur appris à partir de l'état réel de la mémoire récurrente prévoit-il
mieux la réussite de cette mémoire qu'un observateur disposant des entrées
publiques, mais ignorant les perturbations privées ? Cette question teste un
accès fonctionnel à une information interne. Elle ne teste ni un vécu subjectif,
ni une introspection spontanée du modèle linguistique.

## Comparaison et absence de fuite

Les poids de mémoire publiés (graines 17, 29, 43) restent gelés. Deux estimateurs
de même architecture et même budget sont entraînés sur les mêmes épisodes et
étiquettes de réussite. Le premier lit l'état réel, ses quatre probabilités de
sortie et l'ancienneté ; le second lit l'état et les probabilités reconstruits
en rejouant les entrées publiques dans une copie intacte de la mémoire. Cette
reconstruction est une représentation compressée de l'historique public, pas
l'ensemble de tous les observateurs externes possibles.

L'estimation est faite avant l'affichage de la réponse de rappel : les logits
réels ne sont pas publics à cet instant. Après affichage, la comparaison ne
permettrait plus de revendiquer la même asymétrie d'information. Ni le symbole
cible, ni l'étiquette d'intervention, ni le résultat futur ne sont des features.
L'ancienneté est publique et disponible dans les deux conditions. Les étiquettes
supervisées ne servent qu'à l'entraînement hors ligne ou au score final.

## Données, budgets et contrôles

- Une observation initiale parmi quatre symboles ; un rappel après délai.
- Entraînement : 12 288 épisodes, délais 1, 2, 4, 8, 16, 32, 64 ; mélange
  équilibré sans intervention, bruit gaussien et suppression de coordonnées.
- Perturbation juste avant le dernier pas de rappel. Les niveaux sont aléatoires
  et indépendants du symbole. Les états observables publics sont identiques
  pour tous les contrefactuels d'un même épisode.
- Validation : 2 048 épisodes, autre graine, même famille. Sélection du checkpoint
  par Brier de validation à intervalles de 100 pas, jusqu'à 1 000 pas.
- Évaluation exploratoire : 2 048 nouveaux épisodes par famille et par mémoire.
  Familles : intacte, bruit, suppression, remplacement par un état étranger
  valide, délais longs 128 et 512. Les deux dernières sont hors entraînement.
- Architecture : features normalisées, couche cachée ReLU de largeur 32, sortie
  sigmoïde. Même initialisation et ordre des minibatches pour la paire.
- Contrôles : estimation remplacée par celle de l'observateur et estimation
  interne permutée entre épisodes. Comparaison descriptive avec la probabilité
  maximale brute du rappel. Aucune modification des poids de mémoire.

Les familles d'évaluation réemploient les mêmes épisodes publics pour permettre
les comparaisons contrefactuelles. Les unités indépendantes sont les épisodes,
pas les familles ni les quatre composantes d'une probabilité. Les graines de
données sont distinctes entre entraînement, validation et évaluation.

## Mesure principale et interprétation

Différence appariée Brier(observateur) - Brier(interne), positive en faveur de
l'estimateur interne. Intervalles bootstrap descriptifs à 95 %, par épisode,
1 000 rééchantillonnages. Rapporter les trois mémoires séparément et chaque
famille, sans sélectionner uniquement celles qui réussissent. Ce pilote n'est
pas un test confirmatoire préenregistré auprès d'un tiers ; les intervalles ne
sont pas simultanés et aucun seuil ne certifie une conscience.

Une décision simulée répond si la probabilité estimée dépasse 0,8, sinon vérifie.
Coût : erreur = 1 ; vérification parfaite simulée = 0,2. Cette règle n'exécute
aucun outil externe. Son effet mesure une utilité décisionnelle dans ce monde
synthétique, pas une agency générale. Le remplacement de la prévision sans
modifier la mémoire permet de dissocier estimation et compétence de rappel.

L'intégration à CognitiveSession expose une prévision avant l'évaluation externe
et permet d'en mesurer le Brier après retour de vérité. Elle ne remplace pas la
politique de rappel existante et ne prétend pas intégrer le modèle linguistique.

Un éventuel résultat positif justifierait une expérience plus large. Un échec,
notamment sur un état étranger valide, doit être conservé et expliquer les
limites de cette construction. Les autres volets du rapport (modèle de soi
continu, identité autobiographique, transfert linguistique) restent ouverts.
