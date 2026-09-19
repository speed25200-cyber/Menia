# Premier rapport iPhone : cinq calculs réussis

Le rapport `menia-iphone-capability-v1` transmis par l'utilisateur contient cinq
sorties attribuées à Qwen3-4B MLX 4 bits. Le fichier brut, ses UUID et ses horaires
restent hors du dépôt. Cette analyse vérifie sa cohérence, pas l'authenticité
matérielle de l'appareil ni une exécution observée indépendamment.

| Vérification indépendante | Résultat |
|---|---|
| Références recalculées depuis les questions | 5/5 concordent |
| Réponses au format entier strict | 5/5 correctes |
| Prévisions antérieures | 1/2, 2/3, 3/4, 4/5, 5/6 |
| Prévision du prochain test, Beta(1,1) mis à jour | 6/7 = 85,714 % |
| Score de Brier des prévisions successives | 0,0982777778 |
| Identité déclarée et nombre d'octets pertinents | Concordent avec le manifeste fixé |

L'empreinte est recalculée à partir des hashes du manifeste selon le contrat de
l'installateur. Le champ `model.bytes` compte les poids, la configuration et le
tokenizer utilisés pour cette empreinte ; il ne compte pas l'ensemble des fichiers
téléchargés. Sa différence avec les 2,15 Go du téléchargement est donc attendue.

Les cinq tests sont élémentaires et tous réussis. Sous l'hypothèse de réussites
Bernoulli indépendantes de probabilité stable p, le posterior est Beta(6,1), de
fonction de répartition p^6. L'intervalle crédible central à 95 % est
[0,025^(1/6), 0,975^(1/6)] = **[54,074 %, 99,579 %]**. Cet intervalle dépend de
l'a priori et des hypothèses ; il ne couvre pas les autres tâches ou un changement
de distribution. Le taux observé de 100 % n'est pas une fiabilité établie à 100 %.

Le Brier n'est pas ici la mesure d'une confiance verbalisée par Qwen. Les
prévisions viennent du contrôleur numérique. Sur cette série, une constante
0,5 aurait un Brier de 0,25, mais une constante 1 aurait un Brier nul. Comparer
après coup à la meilleure constante ne valide donc pas une calibration générale.

Le rapport fournit des dates de début, sans fin des générations, nombre de
tokens, pic mémoire, consommation ou température. Aucun débit en tokens/s ni
benchmark thermique n'en est déduit. Il ne contient pas non plus de réponses
de Qwen sur son propre bilan. L'usage du bilan par le langage reste à mesurer.

## Suite fixée avant les nouvelles données

Le [protocole de comparaison](IPHONE_COUPLING_PROTOCOL.md) intervient sur le bilan
fourni au LLM : réel, absent, fictif ; deux référents, soi et autre agent. Il
sépare fidélité aux données fournies et accord avec les mesures réelles. Le
choix textuel suit une règle explicite : ce n'est pas encore une tâche avec coût
réel ni un test d'introspection interne ou de conscience subjective.

Les calculs ont été vérifiés avec `research/iphone_report.py`, indépendamment
du noyau Swift. Quatre tests du vérificateur contrôlent les falsifications de
référence, de réponse, de prévision, de résumé et d'identité de modèle.

```powershell
python -m research.iphone_report "chemin/vers/tests-menia.json" --manifest ios/MeniaCore/Sources/MeniaCore/Resources/qwen3-4b.json
python -m unittest tests_research.test_iphone_report -v
```
