# Poids finaux — discrimination partielle sur les exemples appris

20 septembre 2026. Les **5 184 évaluations prévues sont terminées et vérifiées**.
Le juge de classement distingue partiellement réussites et erreurs dans les
catégories de ses propres exemples d'apprentissage : AUROC de 0,6542, 0,7262
et 0,7702. L'ajustement reste imparfait. Ce diagnostic descriptif ne départage
pas optimisation, capacité, objectif et quantité de données ; il ne remplace
pas le résultat négatif du Colab 24.

Le [protocole](FINAL_TRAINING_FIT_PROTOCOL.md) est conservé, avec le
[rapport complet](../artifacts/final-training-fit-pilot/summary.json), la
[vérification](../artifacts/final-training-fit-pilot/verification.json) et le
[reçu](../artifacts/final-training-fit-pilot/receipt.json). Aucune mise à jour
de poids, nouvelle génération ou entrée de test n'a servi à ce diagnostic.

## Ajustement final et comparaison descriptive

Chaque ligne évalue le même checkpoint sur ses 576 anciens exemples appris,
puis rappelle son score sur les 384 nouvelles réponses du **producteur de
base** du Colab 24. Celui-ci est le comparateur descriptif le plus proche,
car les réponses d'apprentissage provenaient aussi du modèle de base.
R1–R3 correspondent aux indices 0–2 des fichiers. Une AUROC mesure un ordre,
pas le pourcentage de réponses correctes.

| Répétition | Juge | AUROC dans les catégories, appris | AUROC dans les catégories, Colab 24 base | Brier, appris | Brier, Colab 24 base |
|---|---|---:|---:|---:|---:|
| R1 | CE | 0,6236 | 0,4893 | 0,1043 | 0,0926 |
| R1 | Classement | 0,6542 | 0,5041 | 0,1022 | 0,1012 |
| R1 | Neutralisé | 0,5955 | 0,4801 | 0,1071 | 0,0907 |
| R2 | CE | 0,6584 | 0,6241 | 0,0993 | 0,1022 |
| R2 | Classement | 0,7262 | 0,6887 | 0,0962 | 0,0969 |
| R2 | Neutralisé | 0,6486 | 0,6444 | 0,1008 | 0,1003 |
| R3 | CE | 0,7074 | 0,6435 | 0,0901 | 0,1012 |
| R3 | Classement | 0,7702 | 0,6654 | 0,0816 | 0,0996 |
| R3 | Neutralisé | 0,6205 | 0,6045 | 0,0996 | 0,1030 |

Le classement a le meilleur score d'apprentissage ponctuel des trois bras
dans chaque répétition. Aucun intervalle ni seuil de réussite n'est ajouté
après coup. Dans le Colab 24, **aucun contraste principal face à CE ou au bras
neutralisé n'est confirmé** par les intervalles corrigés.

Les deux colonnes d'AUROC ne définissent pas un écart pur de généralisation :
le nombre d'exemples, les taux de réussite et donc les poids des catégories
diffèrent. Chaque AUROC agrège les paires réussite/erreur disponibles dans
une catégorie. Le Brier dépend aussi de la distribution des cibles. Ces
différences ne prouvent ni mémorisation, ni insuffisance de capacité.

## Ce que masque la moyenne

Voici les catégories d'apprentissage du bras classement. Chaque catégorie
contient 96 exemples ; les nombres entre parenthèses sont les réussites
réelles du producteur de base, **pas les prédictions correctes du juge**.

| Catégorie | R1 : AUROC (réussites) | R2 : AUROC (réussites) | R3 : AUROC (réussites) |
|---|---:|---:|---:|
| Compter A, longueur 8 | 0,6993 (40) | 0,8138 (48) | 0,8585 (48) |
| Compter A, longueur 24 | 0,5488 (22) | 0,7437 (29) | 0,6620 (18) |
| Compter A, longueur 64 | 0,6037 (6) | 0,3366 (8) | 0,8263 (9) |
| Somme alternée, 2 termes | 0,8852 (90) | 0,9362 (94) | 0,9846 (91) |
| Somme alternée, 4 termes | 0,6128 (10) | 0,7242 (5) | 0,4462 (5) |
| Somme alternée, 8 termes | 0,7421 (1) | Indéfinie (0) | 0,2895 (1) |

La discrimination peut rester faible même sur des entrées déjà vues ; elle
varie entre répétitions et catégories. Une AUROC fondée sur une seule réussite
est particulièrement fragile. Une faible erreur de Brier dans la catégorie
sans réussite ne démontre pas la capacité de distinguer ses réussites de ses
échecs. Répéter ces données n'ajoutera aucun exemple de la classe manquante.

Les neuf unités ont un premier token le plus probable appartenant aux codes
0/1 sur toutes leurs entrées. La plus faible masse moyenne des codes vaut
0,999640. Le format du score ne paraît donc pas expliquer cette faiblesse
de discrimination. Il s'agit d'une mesure des logits, sans décision native
d'accepter ou de vérifier une réponse.

## Réception et audit

Le processus A100 40 Go termine avec code retour zéro le 20 septembre à
03:45:00 UTC. L'archive `menia-ajustement-final-v1.zip` contient quatre fichiers
et 1 084 925 octets. SHA-256 :
`7c0907655f68f31a88d6607ad596970ad06e8a5de14581509f5173e0f8d01e54`.
Le journal a pour empreinte
`8bcbcaa5f5a2ca8806c021f5b22bc7f7b63facdb2c2bcb508ebc2505b9152f5b`.

La chaîne, les 5 184 requêtes exactes, leur ordre, les neuf unités et les
empreintes du plan et des sources sont vérifiés. Les neuf fichiers de poids
distants correspondent toujours aux fichiers déjà reçus et audités du lot 24.
Le collecteur rapporte neuf comparaisons exactes des tenseurs de l'adaptateur
après évaluation et neuf contrôles des versions des paramètres de base.
Ce dernier contrôle ne constitue pas une attestation cryptographique
indépendante des poids de base.

Le recalcul local reproduit exactement le rapport distant. Le calcul séparé
des métriques et les comparaisons explicites de paires concordent à
5,56 × 10⁻¹⁷ près. Ils partagent le lecteur et les cibles : ce n'est pas une
réplication externe. Les évaluations totalisent 669,17 secondes mesurées ;
les six tests logiciels préalables ont passé sur PC et sur Colab.

## Conséquence pour l'expérience suivante

Avant d'attribuer ce plafond au rang 8, il faut mesurer la réponse à un budget
d'optimisation supplémentaire, sur les mêmes données et la même capacité.
Une courbe de poursuite d'apprentissage peut comparer les checkpoints après
un nombre fixé de mises à jour, en gardant les trois répétitions et sans
choisir le meilleur résultat après coup. Le redémarrage éventuel de
l'optimiseur doit être déclaré : ce n'est pas la continuation exacte d'un
état d'optimisation qui n'a pas été sauvegardé.

Les données du lot 24, déjà consultées, pourront seulement servir à décrire
cette courbe exploratoire. Toute confirmation ultérieure exigera un nouveau
jeu réservé. Cette expérience ne pourra pas, à elle seule, résoudre l'absence
de classes minoritaires, confirmer un avantage propre au classement ou
identifier un mécanisme de conscience. La capacité à prévoir ses erreurs
et à utiliser cette information dans ses choix reste la question fonctionnelle
à résoudre ; la conscience et la nouveauté scientifique restent non établies.
