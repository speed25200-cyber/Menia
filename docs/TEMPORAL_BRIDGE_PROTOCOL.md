# Protocole d'audit du lien temporel

Ce protocole fixe un audit algébrique après lecture de SST et de son programme,
avant production des résultats numériques. Ce n'est ni une réplication complète
de la simulation publiée, ni une expérience humaine, ni un test de conscience.

La source inspectée est le dépôt de Jan Bellingrath, commit
`d7c01c65310e3cf932e4c5d4d1235be419a3a352`, fichier
`deep_parametric_generative_model.py`, SHA-256
`e4598a881183e330edfccb07b26bb86ab1be91216f1d7984fd2600cd05f44c82`.
Le programme externe a été lu, sans être exécuté ni incorporé au dépôt Menia.
L'audit utilise une implémentation indépendante des identités mathématiques.

## Comparaisons fixées

1. Conserver les deux transitions attentionnelles publiées, les préférences
   `(0.99, 0.01)` et la matrice d'observation symétrique, y compris l'effet de
   l'epsilon `1e-5` de la normalisation publiée.
2. Énumérer 10 001 croyances `(p, 1-p)`, `p = i / 10000`, et les deux politiques.
   Comparer l'énergie attendue calculée comme risque plus ambiguïté à une perte
   ordinaire, écrite comme entropie croisée moins entropie des observations plus
   l'entropie du canal. Vérifier le maximum des différences sur 20 002 cas.
3. Comparer les probabilités de sélection et les décisions de franchissement du
   seuil `2.2`. Les paramètres de sélection sont ceux du programme : habitudes
   `(0.99, 0.99)`, précision `4`, epsilon `1e-5`.
4. Conserver un contrôle volontairement incomplet : supprimer la constante
   d'ambiguïté avant la normalisation avec epsilon. Mesurer sa discordance, sans
   l'utiliser pour conclure à l'équivalence des contrôleurs.
5. Construire deux contrastes mathématiques distincts du réglage publié :
   risque constant/ambiguïté variable, puis ambiguïté constante/risque variable.
   Pour le premier, comparer les canaux symétriques de diagonales `0.9` et `0.6`
   avec croyance `(0.5, 0.5)`. Pour le second, conserver une diagonale `0.75` et
   comparer les croyances `(0.25, 0.75)` et `(0.75, 0.25)`. Ces contrastes utilisent
   des distributions exactes, sans epsilon.

Les identités sont vérifiées à `1e-12`, les contrastes attendus doivent dépasser
`0.1` nat. Le rapport est rejoué avec une tolérance absolue/relative de `1e-12`.
Il n'y a ni échantillonnage aléatoire, ni ajustement aux données humaines,
ni recherche de paramètres après observation des résultats.

## Limites fixées

L'équivalence est celle d'un calcul local de décision, conditionnellement à la
même croyance. La dynamique attentionnelle, les épisodes sautés, les figures
publiées et les fréquences d'effets humains ne sont pas reproduits. Une identité
algébrique générale explique l'équivalence ; la grille en vérifie le calcul
numérique pour les paramètres spécifiés.

Une équivalence fonctionnelle ne réfute pas une hypothèse d'identité phénoménale.
Supprimer une variable de tracé ne supprime pas une quantité encore calculée
ailleurs. Les contrastes proposés ne produisent pas de jugements de durée : leur
fonction est de vérifier qu'une future comparaison peut séparer deux prédicteurs.
Le [dossier d'interprétation](TEMPORAL_PHENOMENAL_BRIDGE.md) précise les engagements
supplémentaires nécessaires avant une expérience confirmatoire.
