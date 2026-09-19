# Second calcul du Colab 17

19 septembre 2026, préparé pendant l'inférence avant lecture de ses scores.
Les sources et critères scientifiques restent figés à
`7997c23a417abfde194c3b7281e854ea1173d64c`.

`research.audit_natural_errors` recalcule les 24 tableaux globaux, les 144
tableaux par cellule et les 21 contrastes, dont neuf principaux. Il reconstruit
la réponse correcte depuis les lettres ou les opérandes et applique séparément
la règle de format entier. Le Brier et la calibration sont recalculés par
sommes scalaires ; l'AUROC compte les rangs avec demi-poids pour les égalités.
Le bootstrap tire des positions dans chaque strate, puis reconstruit les
différences appariées ; les quantiles sont interpolés depuis les valeurs triées.
Les prérequis, seuils, intervalles corrigés et critère complet sont recalculés.

Deux tests passent en 12,718 secondes sur un journal synthétique hétérogène :
réponses incorrectes ou mal formatées, intervalles non dégénérés, égalités,
classe unique et probabilités aux frontières des cases de calibration.
L'auditeur rejette un score, un intervalle principal, un verdict ou une origine
altérés. Ces essais ne sont pas des résultats de Qwen3-4B.

Le programme ne réutilise pas les métriques ni les fonctions d'analyse
principales. Il partage le plan, le lecteur d'intégrité, la reconstruction des
ajustements et prévisions, ainsi que NumPy. Cette vérification arithmétique
n'est donc pas une réplication extérieure ni un audit indépendant du collecteur.

La présentation fixée dans `scripts/plot_natural_errors.py` conserve les huit
lecteurs, les trois répétitions et les neuf contrastes principaux. Elle est
vérifiée sur des données simulées avant application aux données réelles.
Le protocole exige l'archive complète et la réussite des deux recalculs avant
l'interprétation des scores ; aucun résultat réel n'est encore annoncé ici.

```sh
python -m research.audit_natural_errors JOURNAL.jsonl --summary RESUME.json --output VERIFICATION.json
```
