# Vérifier si la fin de l'entraînement détériore le rapport d'état

Note du 19 septembre 2026, après les résultats 12 et 13. **Réanalyse exploratoire
et proposition expérimentale ; aucune nouvelle inférence Qwen préentraînée
n'est encore exécutée pour cette proposition.**

Mise à jour : cette note conserve l'indice initial. Le test proposé a ensuite
été figé, exécuté et audité dans le [bilan du Colab 14](OPTIMIZER_MEMORY_RESULTS.md),
qui distingue sa correction canonique de son coût sur la reformulation.

## L'indice dans le journal existant

Le [Colab 12](STATE_COMPOSITION_PROTOCOL.md) entraîne quatre époques de 16 mises
à jour. La supervision de présence est cachée aux époques 1 et 3, visible aux
époques 2 et 4. Les lectures publiques accompagnent toutes les mises à jour.
AdamW conserve ses deux moments et son compteur pendant les 64 mises à jour.
Le checkpoint évalué est celui de la fin de l'époque 4.

La [réanalyse reproductible](../research/optimization_loss_diagnostic.py)
reconstruit la perte finale du bon chiffre à partir de ses logits et de la
masse de probabilité des quatre chiffres. Elle utilise la probabilité dans
le vocabulaire complet, et non seulement la probabilité renormalisée sur 0/1.
Un test avec une masse volontairement inférieure à 1 vérifie cette distinction.

| Répétition, bras composé | Perte moyenne absence pendant l'époque cachée 3 | Moyenne des huit derniers termes absence de cette phase | Perte finale absence, code N | Perte finale absence, code I |
|---|---:|---:|---:|---:|
| 1 | 0,45845 | 0,0048264 | 5,16111 | 7,14107 |
| 2 | 0,03977 | 0,0000501 | 6,49668 | 9,15042 |
| 3 | 0,55338 | 0,0002612 | 0,68755 | 0,47872 |

La perte est `−log(p(chiffre correct))` : une faible valeur favorise la bonne
réponse. Les pertes finales des cas présents sont au contraire inférieures
à `6 × 10⁻⁶` dans les six conditions. Les [576 mises à jour et leurs
agrégats](../artifacts/optimization-diagnostic-pilot/loss-trace.json) sont
conservés, y compris les bras mélangé et consignes ; aucun checkpoint n'est
choisi selon cette lecture.

**Ces colonnes ne constituent pas un test causal avant/après.** Les pertes
en cours d'apprentissage correspondent à des poids qui changent et à des
exemples vus dans un ordre déterminé ; les pertes finales correspondent à un
seul checkpoint évalué sur tous les blocs. La fenêtre des huit derniers
termes est une description choisie après les résultats, pas un critère fixé
à l'avance. En particulier, la moyenne de toute l'époque 3 et la perte finale
ne suggèrent pas une même ampleur de détérioration dans les trois répétitions.

L'indice motive une hypothèse plus précise qu'un simple manque de formulations :
la dernière phase visible pourrait déplacer la frontière de décision cachée.
Les gradients visibles, les lectures publiques et la mémoire d'AdamW sont
des explications concurrentes. Leur contribution reste inconnue. Le
[Colab 13](COMPOSITION_DIAGNOSTIC_RESULTS.md) a mesuré l'échec du transfert de
seuil ; il ne départageait pas ces causes d'entraînement.

## Ce qui est déjà connu

[Elazar et al. (2021)](https://aclanthology.org/2021.tacl-1.60/) étudient la
cohérence des modèles sous paraphrase et une méthode pour l'améliorer.
[Zhou et al. (2022), §3](https://aclanthology.org/2022.findings-emnlp.192.pdf)
entraînent la cohérence entre plusieurs prompts par distillation ; ils
précisent qu'une sortie constante peut satisfaire trivialement la cohérence.
Ils emploient notamment LoRA et un critère de sélection pour limiter ce
problème. Leur cadre ne démontre pas une introspection. Ajouter une perte
de cohérence à Menia aurait donc des précédents directs ; il faudrait
préserver l'exactitude et comparer des budgets et données identiques.

[Ashley, Ghiassian et Sutton (2021)](https://arxiv.org/abs/2102.07686)
montrent que le choix de l'optimiseur peut changer les mesures d'oubli et
que les conclusions dépendent de la métrique retenue. Ils ne démontrent pas
que la mémoire d'Adam est la cause du défaut de Menia.
[Shao et Feng (2022)](https://aclanthology.org/2022.acl-long.143.pdf) étudient
l'oubli lié à l'ordre des exemples, y compris dans un entraînement sur un
jeu statique, et proposent une distillation complémentaire en traduction.
Ni l'oubli d'entraînement ni la correction de l'ordre ne seraient une
découverte nouvelle par leur seule application à Menia.

## Intervention proposée sur une même bifurcation

Reproduire les 48 premières mises à jour du bras composé de chaque répétition,
puis conserver **les mêmes poids et le même état AdamW** pour quatre conditions :

| Condition | Suite après la mise à jour 48 | Question isolée |
|---|---|---|
| Préfixe figé | Aucune mise à jour supplémentaire | Quel rapport d'état existe avant la dernière phase ? |
| Suite originale | Les 16 groupes visibles/publics d'origine, moments conservés | Le checkpoint final du Colab 12 est-il reproduit exactement ? |
| Premier moment effacé | Mêmes 16 groupes ; `exp_avg` mis à zéro une seule fois au départ ; second moment et compteur conservés | Quel est l'effet total de retirer l'ancien premier moment ? |
| Gradients nuls | 16 pas AdamW avec tenseurs de gradient nuls et état initial conservé | La mémoire de l'optimiseur suffit-elle à déplacer la décision sans nouveaux gradients ? |

Mettre les gradients à `None` n'est pas ce dernier contrôle : PyTorch peut
alors sauter la mise à jour. Le contrôle proposé fournit explicitement des
tenseurs nuls ; la décroissance des moments peut encore déplacer les poids.
Le poids de régularisation est nul, comme dans le Colab 12.

Les [opérations de bifurcation](../research/optimizer_memory_ops.py) sont
implémentées et testées sur des tenseurs contrôlés. Les tests vérifient
l'absence de partage mutable entre branches, l'effacement du seul premier
moment, et l'accord de cinq pas à gradients nuls avec la formule d'Adam
calculée séparément. **Ce test analytique n'est pas un résultat Menia.**

La suite originale devra retrouver exactement les trois empreintes finales
du Colab 12 avant d'interpréter les interventions. Si ce témoin technique
échoue, on doit expliquer la divergence, sans assouplir rétrospectivement
la tolérance pour obtenir un résultat favorable.

Pour les lectures comportementales, prévoir les anciens blocs d'apprentissage
comme diagnostic de rétention, de nouvelles phrases dans le même vocabulaire,
puis un jeu de contenu lexical réservé. Mesurer séparément consignes canoniques
et reformulées, deux codes, présence cachée, présence visible et lectures
publiques. Comparer les réponses natives et leurs pertes ; ne pas ajuster de
nouveau seuil sur les tests. Conserver les répétitions, témoins identiques et
sorties hors code. Les contrôles visibles restent nécessaires pour lire tout
bénéfice de rétention accompagné d'une perte sur les autres tâches.

Cette proposition étudie d'abord la cause d'une instabilité d'apprentissage.
Elle ne remplace pas l'expérience ultérieure de généralisation entre des
formulations réellement nouvelles. Le [protocole 14](OPTIMIZER_MEMORY_PROTOCOL.md)
fixe maintenant le moteur complet, les effectifs et les contrastes avant
la nouvelle collecte GPU ; il ne rapporte pas encore de résultat de cette collecte.

## Limite pour l'objectif de conscience

La mémoire d'AdamW est un état de l'algorithme d'entraînement, pas une mémoire
autobiographique ou une expérience du modèle. Une rétention améliorée serait
un progrès de fiabilité d'un rapport d'état artificiellement appris. Le
mécanisme d'auto-représentation, l'utilité pour agir et l'expérience de soi
restent à établir séparément.
