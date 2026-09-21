# Ce que le journal d'entraînement permet déjà de vérifier

20 septembre 2026. Diagnostic ajouté pendant l'évaluation du Colab 24,
**avant inspection de ses réponses ou scores de test**. Il utilise uniquement
la sauvegarde d'entraînement reçue sur PC. Le protocole, les poids figés et
le critère principal restent inchangés.

Le [rapport](../artifacts/confidence-ranking-pilot/training-diagnostic.json)
décrit les valeurs enregistrées **avant chaque mise à jour**. Les paramètres
changent à chaque lot et les paires sont remélangées entre les deux passages.
Ces valeurs ne sont donc ni une évaluation du checkpoint final sur tout le
train, ni une mesure de généralisation, ni une preuve de convergence.

## Classement pendant le deuxième passage

Pour une paire de réponses de la même catégorie, dont une seule est correcte,
on oriente la différence des scores vers `score_correct − score_incorrect`.
Le crédit vaut 1 si elle est positive, 0,5 en cas d'égalité, 0 sinon. Il est
calculé seulement sur les paires éligibles, avec leur effectif explicite.
Ce crédit en ligne n'est pas l'AUROC du test figé.

| Répétition | Paires éligibles par passage | CE seul, passage 2 | Classement, passage 1 → 2 | Neutre, passage 2 |
|---|---:|---:|---:|---:|
| 0 | 85 | 54,71 % | 50,00 → 55,88 % | 46,47 % |
| 1 | 92 | 61,41 % | 54,89 → 69,02 % | 59,24 % |
| 2 | 86 | 59,30 % | 50,58 → 65,12 % | 61,05 % |

Le bras de classement progresse dans ces moyennes, mais ses observations
restent loin d'un ordre presque parfait. Les comparaisons entre bras sont
descriptives : on ne leur ajoute pas un seuil de réussite ni un test
statistique improvisé. L'évaluation indépendante des nouvelles questions
reste nécessaire pour savoir si ce comportement se généralise.

La perte logistique de classement du bras correspondant passe de
0,8997 à 0,7193, de 0,9007 à 0,5752 et de 0,9240 à 0,6299.
Une marge toujours nulle donne `log(2) ≈ 0,6931`. La première répétition reste
donc au-dessus de cette référence pendant son deuxième passage, malgré un
crédit d'ordre supérieur à 50 %. Ordre et amplitude des marges ne sont pas
la même mesure.

## Perte individuelle, diversité des données et gradients

La perte moyenne code/EOS du bras de classement diminue de 0,8827 à 0,1978,
de 0,7868 à 0,1867 et de 0,7425 à 0,1726. Cette perte mélange l'émission du
code et celle du token d'arrêt. Sa baisse ne suffit pas à attribuer un gain
à la discrimination des erreurs au sein d'une catégorie.

Chaque passage présente les 576 exemples, mais seulement 85, 92 ou 86 des
288 paires portent un contraste réussite/erreur. Sur la somme à huit termes,
les effectifs éligibles sont 1, 0 et 1. La cellule sans paire est conservée
avec des métriques indéfinies ; une cellule à une paire ne devient pas une
preuve robuste, qu'elle soit réussie ou échouée. Le rapport publie les six
catégories de chaque bras et de chaque passage.

Le seuil de clipping est dépassé dans 57, 51 et 51 des 72 mises à jour du
deuxième passage du bras de classement, contre 26, 18 et 18 pour CE seul.
Ce constat décrit la dynamique de l'objectif modifié ; il ne prouve pas que
le clipping cause un défaut de généralisation. Augmenter le budget, le rang
ou changer le seuil exigerait une comparaison ultérieure contrôlée.

Les pertes individuelles, différences de scores et nombres de tokens du
premier lot sont exactement identiques entre les trois bras de chaque
répétition, avant leur première mise à jour. Cela confirme le départ apparié.

## Vérification et conséquence expérimentale

Le lecteur historique vérifie la chaîne complète des 1 296 mises à jour.
Deux tests du diagnostic couvrent les égalités, marges extrêmes, valeurs non
finies et groupes sans paire. Un [recalcul numérique séparé](../artifacts/confidence-ranking-pilot/training-diagnostic-verification.json)
utilise des tableaux NumPy, une orientation explicite par label et
`logaddexp`, contre les calculs scalaires `fsum`/`log1p` du rapport. Il vérifie
126 groupes et 1 074 valeurs, avec un écart maximal de 4,45 × 10⁻¹⁶.
Le lecteur et la recette de paires restent partagés : ce n'est pas une
réplication extérieure.

Ce diagnostic empêche de supposer que le modèle maîtrise déjà l'objectif
sur ses données d'apprentissage. Il ne permet pas encore de choisir entre
limite de données, budget, capacité d'adaptation ou optimisation. Si le test
figé échoue, ces causes devront être séparées avant d'attribuer l'échec à
l'absence d'auto-évaluation en général. S'il réussit, l'usage de l'information
dans une décision restera à vérifier. Aucune conscience ni nouveauté n'est
démontrée par ce diagnostic.

```sh
python -m unittest tests_research.test_confidence_ranking_training_diagnostic -v
python -m research.confidence_ranking_training_diagnostic JOURNAL_ENTRAINEMENT --freeze artifacts/confidence-ranking-pilot/training-freeze.json --output artifacts/confidence-ranking-pilot/training-diagnostic.json
```
