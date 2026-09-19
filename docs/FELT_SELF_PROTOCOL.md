# Protocole de réanalyse du soi rapporté au repos

Cette analyse secondaire, exploratoire et non préenregistrée, compare six
mesures chez les 50 participants de Hruby et collègues (2024), après
Floatation-REST et Bed-REST. Le protocole est consigné après lecture de l'article
et inspection du format des données, avant calcul des contrastes ci-dessous.
Il ne porte ni sur une intervention à effectuer ni sur un conseil de traitement.

## Sources et unités

- Article : [10.1038/s41598-024-59642-y](https://doi.org/10.1038/s41598-024-59642-y).
- Données : [OSF 5rzbv](https://osf.io/5rzbv/), fichier `Floatation-REST data.csv`,
  version 1 du 22 février 2024, [téléchargement](https://osf.io/download/qm7y2/).
  SHA-256 : `20127d6cbececaa5a5d8aeb3375b65892a168d67ebd7fa4c58761a573fd967b6`.
- Dictionnaire : `Codebook Variables Floatation-REST.docx`,
  [téléchargement](https://osf.io/download/3ky9v/), SHA-256 :
  `09e7cf77cb0977019d57536c1bdf0d97e599ef33cf3e6d9d3ce3f7f2d0eab787`.

Une ligne est un participant, avec les deux conditions dans des colonnes
différentes. Utiliser l'encodage Windows-1252, le séparateur point-virgule et
la virgule décimale. Ne pas écraser la source et ne pas publier les lignes
individuelles dans le dépôt de Menia.

## Comparaisons retenues

| Mesure | Colonnes Floatation / Bed | Étendue |
|---|---|---|
| Frontières du corps perçues | FPBBS / KPBBS | 1–7 ; élevé = frontières plus fortes |
| Conscience de soi rapportée | FPCI_D15 / KPCI_D15 | 0–6 |
| État altéré rapporté | FPCI_D16 / KPCI_D16 | 0–6 |
| Expérience altérée rapportée | FPCI_D24 / KPCI_D24 | 0–6 |
| Contrôle volontaire rapporté | FPCI_D19 / KPCI_D19 | 0–6 |
| Mémoire rapportée | FPCI_D20 / KPCI_D20 | 0–6 |

Ces intitulés viennent du dictionnaire. Aucune borne n'est considérée comme un
seuil de conscience. Les données contiennent des sous-scores, pas les réponses
aux 53 items du PCI ; une validation de leur structure psychométrique n'est pas
possible à partir de cette livraison.

Calculer les contrastes **Floatation moins Bed**, les médianes par condition,
les différences moyennes et médianes, et les comptes négatifs/nuls/positifs.
Vérifier 50 identifiants distincts, les bornes et l'absence de valeurs manquantes
pour les 12 colonnes. Conserver la précision décimale fournie.

## Inférence et contrôles

Test bilatéral exact par changements de signe des rangs absolus, conditionnel
aux amplitudes des différences. Exclure les différences nulles et attribuer
les rangs moyens aux ex aequo. Une programmation dynamique compte les sommes
de rangs sur les `2^n` signes possibles ; il ne s'agit pas d'une simulation de
ces permutations. Le test suppose l'échangeabilité des signes sous l'hypothèse
nulle, ou la symétrie pertinente pour son interprétation en test de localisation.

Appliquer Benjamini–Hochberg aux six valeurs p de cette famille exploratoire.
Ne pas présenter cette correction comme celle du papier, qui regroupe d'autres
comparaisons. Le calcul exact diffère aussi d'une approximation asymptotique.
Une différence entre résultats significatifs et non significatifs ne constitue
pas à elle seule une différence significative entre effets.

Estimer un intervalle descriptif à 95 % de la différence moyenne par bootstrap
apparié de 20 000 tirages des participants, graine 20260915, percentiles 2,5 et
97,5 %. Les intervalles ne sont pas corrigés pour multiplicité. Les scores sont
bornés et parfois ordinaux ; leur moyenne ne crée pas une échelle physique de
l'expérience.

La colonne `Reihenfolge` contient quatre codes A, B, C et D, sans correspondance
explicite trouvée dans le dictionnaire lu. Rapporter les moyennes des contrastes
par code comme diagnostic de sensibilité, sans attribuer arbitrairement un ordre
aux lettres. Aucun ajustement causal de période ou de report entre sessions n'est
revendiqué. Les conditions diffèrent aussi par les vêtements et la température.

## Périmètre de l'audit

Le rejeu humain reste local et exige le CSV source avec son empreinte. Des tests
indépendants sur de petits exemples vérifient le comptage exact des signes et la
correction multiple, sans besoin des données individuelles en CI. Le rapport
conserve les statistiques agrégées et la provenance.

Aucune médiation thérapeutique, aucun résultat de Tobel et collègues (2026),
aucun ajustement de modèle causal cérébral et aucune attribution de conscience
à Menia ne sont reproduits par ce protocole.

## Ajout après le premier calcul : cohérence des sources

Le fichier [XLSX public](https://osf.io/download/y4gmp/) a ensuite été comparé au
CSV, par identifiants, sur les 600 cellules des douze colonnes retenues.
SHA-256 : `9c71b185d187e3409354fcdf7a1b759df5147618183dbfc3dc10af5ad2eee549`.
La feuille contient 50 lignes avec identifiant ; ses autres lignes sont ignorées.
Les codes d'ordre sont aussi comparés. Aucune donnée source n'est modifiée.

Une discordance de 0,5 sur FPBBS et des différences de précision ont motivé une
sensibilité secondaire. Les scores XLSX sont arrondis à trois décimales
(demi-pair), puis soumis à la même procédure et aux mêmes tirages. Ce contrôle
a été ajouté après les résultats CSV ; il ne remplace pas l'analyse principale.
Le rapport conserve les deux analyses et la discordance sans identifiant personnel.

```bash
python -m research.audit_felt_self --data "Floatation-REST data.csv" --xlsx "Floatation-REST data.xlsx"
python -m research.audit_felt_self --data "Floatation-REST data.csv" --xlsx "Floatation-REST data.xlsx" --check
python -m unittest tests_research.test_felt_self -v
```

NumPy suffit pour la version CSV et les tests ; la comparaison optionnelle XLSX
requiert aussi openpyxl. Le contrôle `--check` compare le rapport à `1e-10` près,
sans tolérance relative ; les comptes et probabilités rationnelles sont exacts.
Le rapport versionné comprend la comparaison XLSX et exige donc les deux fichiers
pour son contrôle complet. Aucun accès réseau n'a lieu pendant le rejeu.
