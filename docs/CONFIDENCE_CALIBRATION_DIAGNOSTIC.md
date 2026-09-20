# Séparer calibration et discrimination après le Colab 23

20 septembre 2026. Analyse exploratoire décidée après lecture des
[résultats du Colab 23](NATIVE_ANSWER_CONFIDENCE_RESULTS.md). Elle conserve
l'échec du critère principal et ne fournit pas une nouvelle confirmation.
Aucun appel LLM ni apprentissage de ses poids n'est requis.

## Deux questions distinctes

La première est de savoir quelles paires contribuent à l'AUROC globale.
On sépare les paires réponse correcte / incorrecte appartenant à une même
catégorie des paires entre catégories. Chaque égalité de scores vaut un
demi-succès. Le score global est exactement la moyenne des deux AUROC,
pondérée par leurs nombres de paires. Une composante sans paire reste
indéfinie. Ces paires partagent leurs observations ; leur nombre ne constitue
pas un nouvel effectif indépendant. Ce découpage est une identité descriptive,
pas un nouveau test statistique ou une méthode inédite.

La seconde question est de savoir si une transformation des probabilités
améliore leur Brier sans apprendre à mieux classer les réponses. La
[documentation de calibration](https://scikit-learn.org/stable/modules/calibration.html)
distingue ces propriétés et décrit le recalibrage sigmoïde. Nous utilisons
ici une variante régularisée à pente non négative, et non l'implémentation
Platt exacte de scikit-learn. Un changement monotone strict conserve l'ordre
des scores ; une pente nulle ou des égalités numériques peuvent créer des
ex æquo. La méthode ne fournit pas d'information nouvelle au modèle.

## Méthode fixée pour ce diagnostic

- Pour chaque répétition, producteur et juge, ajuster une sigmoïde sur les
  seuls 96 exemples de calibration : 27 ajustements, tous rapportés.
- Entrée scalaire : logit du score natif, avec probabilités bornées à
  `[10⁻¹², 1−10⁻¹²]`. Centrage et échelle calculés sur la calibration.
- Objectif : log-vraisemblance négative moyenne plus `b²/(2n)`, où `b ≥ 0`
  est la pente standardisée et `n = 96`. Interception non pénalisée.
- Newton avec recherche de pas ; au plus 100 itérations ; résidu des
  conditions d'optimalité ≤ 10⁻⁹. Un échec numérique arrête l'analyse.
- Pas de sélection d'hyperparamètre, de meilleur juge ou de répétition.
  Les comparateurs Beta et de confiance de sortie du parent restent inchangés.
- Rapporter Brier et AUROC avant/après, ainsi que la décomposition des paires.
  Aucun nouveau seuil de succès ou intervalle confirmatoire.

Le fichier de données doit correspondre à l'empreinte du journal déjà audité.
Le diagnostic vérifie la reconstruction des métriques originales avec la
tolérance de l'audit parent (10⁻¹⁰, relative aux valeurs supérieures à 1).
Un premier passage s'est arrêté sur une égalité flottante exacte trop stricte :
seul le comparateur de confiance de sortie différait, de 2,23 × 10⁻¹⁶ au plus.
Le contrôle reprend la tolérance déjà utilisée pour l'archive, sans changer
l'ajustement ni consulter de nouveaux scores recalibrés à cette étape.
Les données de test ne sont pas utilisées pour ajuster les
coefficients ; toutefois, leur résultat précédent a motivé cette analyse.
Un gain restera donc exploratoire et externe au LLM.

```sh
python -m unittest tests_research.test_confidence_calibration_diagnostic -v
python -m research.confidence_calibration_diagnostic JOURNAL --output artifacts/confidence-calibration-diagnostic/report.json
```

Les résultats seront ajoutés après exécution. Ce diagnostic ne mesure ni
l'usage d'un état interne pour agir ni une conscience de sa propre existence.
