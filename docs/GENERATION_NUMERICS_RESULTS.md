# Le cache est reconstructible au même rythme de calcul

20 septembre 2026. Le [contrôle numérique fixé](GENERATION_NUMERICS_PROTOCOL.md)
est terminé sur Qwen3-4B BF16, SDPA, A100 40 Go. Sur les six cas, le collecteur
constate une égalité exacte entre le cache effectivement généré et sa relecture
avec les mêmes tokens et les mêmes appels. Une relecture du préfixe entier
introduit des écarts numériques, sans changer les douze sorties natives mesurées.

## Résultats

| Contrôle | Constat |
|---|---|
| Générations répétées, même graine | 6/6 égalités exactes déclarées pour texte, métriques, trace et cache |
| Relecture au même rythme | 6/6 caches exacts ; 12/12 sorties de branche strictement égales |
| Ordre inverse des branches | 6/6 comparaisons exactes déclarées |
| Versions des paramètres | Inchangées dans les six cas |
| Relecture du préfixe entier | Écart maximal absolu de cache de 1,0 ; 12/12 séquences natives inchangées |
| Séquence complète, sans cache réutilisé | 12/12 séquences natives inchangées |
| Format natif du cache réel | 12/12 codes suivis d'EOS valides |

Le maximum d'écart sur les deux logits candidats atteint 0,5 pour la relecture
du préfixe entier et 0,75 pour la séquence complète. Les écarts de probabilité
conditionnelle associés atteignent respectivement 1,06 × 10⁻⁶ et 8,23 × 10⁻⁶.
Le maximum absolu du cache n'est pas une mesure normalisée et ne décrit pas
à lui seul l'effet comportemental. Les douze branches gardent leurs tokens.
Au même rythme de calcul, tous ces écarts sont nuls.

Les six rapports donnent tous « correct », et les six choix d'action donnent
tous « vérifier ». La validité du format ne démontre donc aucune discrimination
entre états ou situations. Ces questions techniques n'étaient pas sélectionnées
comme benchmark de compétence, de calibration ou d'utilité.

## Conséquence pour l'étude de l'état propre

Conserver le cache réellement produit permet maintenant de tester des
interventions en contrôlant cette source d'écart numérique. En revanche,
sur ces cas sans intervention, le cache ne contient pas un historique privé
irréductible aux tokens et aux appels de calcul : il est reproduit exactement.
Ce contrôle ne conclut pas à cette équivalence pour toutes les architectures,
précisions, dispositifs ou conditions d'intervention.

Une future étude devra mesurer une variation de conséquence depuis un état
modifié à texte identique, puis vérifier si la prévision suit cette variation.
Un changement des seuls logits du rapport, ou une comparaison avec une
relecture en bloc, ne suffira pas. La prévision devra précéder une tâche
exécutée depuis une copie indépendante du même état. Le
[dessin prospectif](PROSPECTIVE_STATE_PREDICTION_DESIGN.md) reste à instancier
et ne dispose encore d'aucun résultat de ce type.

## Ce qui a été vérifié localement

L'[archive reçue](../artifacts/generation-numerics-pilot/receipt.json) fait
47 962 octets et contient quatre fichiers. Son SHA-256 est
`2ece760c8b199666064cb12c4540bcb8c9eda967473e303394ed7c336dc4a8fc`.
La révision exécutée est `361dc2cb45b43791dc200ee370366ae57e56aede`.
Neuf tests du lanceur passent en 0,436 seconde avant les douze générations
et soixante décodages. Aucune mise à jour ni perturbation n'a été appliquée.

L'[auditeur local](../artifacts/generation-numerics-pilot/verification.json)
vérifie le journal, les identités figées, les prompts et les suffixes avec
le tokenizer fixé, les tokens bruts, le parsing des codes, les probabilités
binaires et l'arithmétique des comparaisons enregistrées. Le
[journal complet](../artifacts/generation-numerics-pilot/generation-numerics-20260920-v1.jsonl)
et son résumé sont conservés pour cette vérification.

Les tenseurs complets des caches, les traces de la seconde génération et les
sorties en ordre inverse ne sont pas exportés. Leurs égalités sont contrôlées
dans Colab et consignées, mais ne peuvent pas être recalculées indépendamment
depuis l'archive. Les versions des paramètres ne sont pas une attestation
cryptographique de tout l'état du processus. Ces limites ne sont pas effacées
par le succès de l'auditeur.

Ce contrôle valide un outil dans les conditions testées. Il n'établit ni accès
appris à soi, ni conscience, ni nouveauté scientifique.
