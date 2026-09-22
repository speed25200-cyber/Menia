# Protocole pré-enregistré — Canal d'action propre, enfance mutable

Rédigé le 22 septembre 2026, **avant tout entraînement du régime VM**. Les
critères ne seront pas modifiés après lecture. Un échec est un résultat.

## Ce qui a été vu avant d'écrire ceci

Le protocole du canal d'action propre annonçait : « Si P4 échoue alors que
P1 et P3 passent, la conclusion est que l'inférence en contexte apprise sur
des corps stables ne se met pas à jour dans un transformeur, et l'expérience
suivante, à pré-enregistrer, ajoutera un régime à enfance mutable. » Pendant
que les neuf modèles finissaient, une lecture précoce de la graine 17 des
régimes V et VE a montré : inférence du corps après un mouvement 0,995 et
0,989 ; lecture de la marque au premier pas dans toutes les vies ; puis,
après le changement de corps du pas 12, **exactitude 0,000 et 0,005 contre
le nouveau corps aux pas 16 à 23, et 0,5 % et 0 % de vies retournant à la
marque**. Les autres graines n'étaient pas lues. Les seuils ci-dessous sont
fixés en connaissance de ces deux valeurs pour V ; ils portent sur VM, qui
n'a pas été entraîné.

## Question

Le petit agent GRU à enfance stable se réparait après un changement de
corps ; le transformeur à enfance stable, entraîné par prédiction du
prochain token, ne se répare pas. Hypothèse : **la révision en contexte du
modèle de soi est une disposition que le prédicteur de texte n'acquiert que
si ses données d'enfance contiennent des changements de soi.** Un
transformeur qui n'a vu que des corps stables traite tout le contexte comme
une seule preuve ; un transformeur qui a vécu des corps changeants apprend à
laisser la preuve récente l'emporter, à redevenir incertain et à retourner
lire la marque.

## Régime VM

Identique à V : D et E tirés par vie, perte sur tous les tokens, même
modèle, même budget, graines 17, 29 et 43. Une seule différence : dans la
moitié des vies d'enfance, D est retiré à un pas uniforme entre 6 et 17, la
marque suit ; c'est le régime MS de l'expérience du corps mutable, transposé
au texte. Mêmes jeux de test R, M et C, mêmes graines de test, 200 vies.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **Q1** | L'inférence est conservée | Jeu R, après au moins un mouvement : exactitude VM ≥ 0,90. |
| **Q2** | Le modèle de soi se met à jour | Jeu M, pas 16 à 23 contre le nouveau corps : VM ≥ 0,80. Direction VM > V sur chacune des trois graines. |
| **Q3** | Retour sélectif à la marque | Jeu M : part des vies VM lisant la marque aux pas 13 à 23 ≥ 0,50, et cette part dépasse d'au moins 0,30 celle du jeu C. Direction VM > V sur chacune des trois graines au jeu M. |
| **Q4** | L'enquête est conservée | Jeux M et C réunis : VM fait ≥ 1,0 inspection par vie avec une part vers la marque ≥ 0,50. |
| **Q5** | Le changement rend incertain | Jeu M, P-soi, pas de mouvement : confiance moyenne (probabilité maximale du déplacement prédit) aux pas 13 à 15 inférieure d'au moins 0,20 à celle des pas 9 à 11 pour VM ; baisse ≤ 0,05 pour V. |

**Critère global : Q1, Q2 et Q3 satisfaits.** Q4 vérifie que la vigilance
n'a pas remplacé l'enquête initiale ; Q5 donne le mécanisme. La vigilance
persistante observée chez le GRU, des relectures de la marque dans les vies
témoins, est notée si elle apparaît mais n'est pas un critère.

## Ce que ce résultat dirait

S'il passe : chez un prédicteur de texte, savoir réviser son modèle de soi
en contexte est une propriété des données d'enfance, distincte de savoir
l'inférer. Les corpus écrits par d'autres sont stationnaires à cet égard ;
la recette pour un LLM est un ajustement sur des vies où le corps change.
Si Q2 échoue : le transformeur de cette taille ne sait pas réviser même
quand on le lui a montré, et la différence avec le GRU tient à l'architecture
plus qu'aux données ; l'expérience suivante changerait l'architecture.

## Exécution et audit

- `python -m research.own_action_experiment --root artifacts/own-action-mutable --regime VM`
- `python -m research.audit_own_action --root artifacts/own-action-channel --mutable-root artifacts/own-action-mutable --check`
  recalcule Q1 à Q5 depuis les journaux et les poids des deux racines.
- Résultats dans `docs/OWN_ACTION_MUTABLE_RESULTS.md`.
