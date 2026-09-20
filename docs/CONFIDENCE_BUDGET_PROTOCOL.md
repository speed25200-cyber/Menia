# Budget d'optimisation à capacité fixe — diagnostic exploratoire

20 septembre 2026. Le [diagnostic des poids finaux](FINAL_TRAINING_FIT_RESULTS.md)
montre un ajustement partiel jusque sur les exemples appris. Cette expérience
demande si une poursuite d'optimisation améliore cet ajustement, à données,
objectif et capacité constants. Elle ne recherche pas un meilleur checkpoint
sur un nouveau jeu de test et n'introduit aucun critère de conscience.

## Plan fixé avant cette collecte

Les trois checkpoints **classement** du Colab 24 sont tous conservés comme
points de départ. Ils ont déjà reçu deux passages sur leurs 576 exemples
d'apprentissage respectifs. Leurs empreintes figurent dans le
[plan](../artifacts/confidence-budget-preparation/design.json). Ni les bras CE
et neutralisé, ni une répétition choisie pour son score ne les remplacent.

| Variable | Valeur |
|---|---|
| Modèle | Qwen3-4B, révision `1cfa9a7208912126459214e8b04321603b3df60c` |
| Adaptation | Q/V, rang 8, échelle 1, 2 949 120 paramètres par adaptateur |
| Données d'apprentissage | Les 576 anciennes réponses par répétition, inchangées |
| Poursuite | Six passages supplémentaires, 432 mises à jour par répétition |
| Objectif et optimiseur | CE code/EOS + classement, AdamW à 10⁻⁴, autres paramètres inchangés |
| Moments d'AdamW | Réinitialisés une fois au départ de chaque répétition |
| Points mesurés | Deux passages déjà effectués, puis quatre et huit passages totaux |
| Lectures par checkpoint | 576 exemples appris + 384 anciennes réponses du producteur de base du lot 24 |
| Budget total | 1 296 mises à jour supplémentaires et 8 640 évaluations |
| Générations | Aucune nouvelle réponse de tâche |

Les moments de l'optimiseur du lot 24 n'ont pas été conservés. Il s'agit donc
d'une **poursuite avec redémarrage d'AdamW**, et non de la continuation exacte
de son ancienne trajectoire. Les moments restent ensuite continus pendant
les six passages, y compris après la sauvegarde intermédiaire. Le taux
d'apprentissage est constant ; aucun arrêt anticipé n'est décidé sur les scores.

Le même algorithme de paires est étendu aux indices d'époque 2 à 7. Chaque
exemple intervient exactement une fois par passage, avec un partenaire de
même catégorie. Les paires de classes opposées sont formées sans remplacement ;
les restes de même classe ont une perte auxiliaire nulle. Les tests vérifient
que cet algorithme reproduit aussi exactement les deux anciens passages.
Le journal conserve l'empreinte de chacun des 1 296 lots de mise à jour.

## Vérifier le point de départ avant d'entraîner

Les trois checkpoints initiaux sont d'abord relus sur l'ensemble de leurs
960 entrées. La probabilité conditionnelle et la masse des codes doivent
reproduire les mesures précédentes à 10⁻⁵ près, pour chaque exemple. Cette
tolérance concerne la reproductibilité numérique ; elle ne mesure pas une
capacité scientifique. Un dépassement arrête la tentative avant toute mise
à jour. Il n'entraîne aucun changement automatique de tolérance ou de modèle.

Les [entrées préparées](../artifacts/confidence-budget-preparation/rows.jsonl)
proviennent exclusivement des deux journaux déjà audités, dont les empreintes
sont dans le [reçu des données](../artifacts/confidence-budget-preparation/data.json).
Les scores de référence et les cibles sont conservés pour la vérification
et la notation ; ni les uns ni les autres ne sont ajoutés aux prompts de
jugement. Les cibles 0/1 sont utilisées seulement pour l'apprentissage supervisé
des exemples d'entraînement, comme dans le lot 24.

Les anciennes réponses du lot 24 restent exclues de toutes les mises à jour.
Elles ont toutefois déjà été examinées : leur nom de phase est explicitement
`seen24base`. Elles ne peuvent plus servir de confirmation indépendante.
Les questions sont distinctes des questions d'apprentissage, et ce point est
vérifié lors de la préparation et de la lecture des données.

## Mesures et interprétations permises

Les six nouveaux checkpoints, à quatre et huit passages pour chacune des
trois répétitions, sont tous sauvegardés **avant** leur première évaluation.
Les mesures comprennent AUROC globale, AUROC au sein des catégories, Brier,
accord binaire au seuil 0,5, premier token et masse des codes. Les catégories
sans les deux classes conservent une AUROC indéfinie. Le calcul séparé des
métriques et les comparaisons explicites de paires sont réutilisés.

À phase donnée, chaque point de la courbe juge exactement les mêmes réponses :
la prévalence des cibles et les poids des catégories restent donc fixes entre
checkpoints. La comparaison entre les phases « appris » et « lot 24 déjà vu »
conserve, elle, des distributions et poids différents.

- Un ajustement appris qui augmente tandis que l'autre phase stagne ou baisse
  indiquerait une limite de transfert dans ce réglage ; il ne confirmerait pas
  une nouvelle généralisation.
- Une augmentation dans les deux phases motiverait un nouveau test réservé,
  après fixation du budget ; elle ne réparerait pas le critère négatif du lot 24.
- Une stagnation ne prouverait pas que le rang 8 est insuffisant : taux
  d'apprentissage, objectif, conditionnement et données resteraient possibles.
- Répéter une catégorie sans aucune réussite n'y crée pas de classe positive.

Aucun seuil de réussite, intervalle confirmatoire ou sélection du meilleur
checkpoint n'est prévu. Les trois répétitions et les trois points seront
rapportés. Ce diagnostic ne mesure pas le comportement de résolution des
nouveaux adaptateurs, puisque les réponses jugées restent celles du producteur
de base. Une dérive de leur capacité à répondre eux-mêmes devra être testée
avant toute utilisation dans Menia ou sur l'iPhone.

## Rapport à la littérature et à l'objectif Menia

[Guo et al., annexes D.1–D.2](https://arxiv.org/html/2606.32038v1#A4.SS1)
étudient séparément le rang LoRA et le taux d'apprentissage dans leur protocole
d'explication contrefactuelle. Leurs effets dépendent de cette configuration ;
ils ne diagnostiquent pas le plafond observé ici. Cette étude de budget à
capacité fixe est un diagnostic d'optimisation classique, sans revendication
de nouveauté. Elle vise une compétence préalable manquante : prévoir de façon
utile les erreurs du système. Son lien avec une conscience subjective reste
non établi, et l'usage causal de cette estimation dans une décision reste à tester.

## Exécution et traces

Douze tests passent sur PC : cinq nouveaux tests de données, planning,
reconstruction et rejet des journaux altérés ; quatre tests des gradients du
classement ; trois tests du score natif. Les sorties construites utilisées par
ces tests ne sont pas des performances de Qwen3-4B.

Le lanceur exige les expériences parentes terminées, le runtime Python Menia
et une A100 libre. Il récupère une révision Git immuable, exécute les mêmes
tests, puis lance une tentative unique. Il conserve l'archive même après
échec, sans relance automatique. Les neuf évaluations vérifient que les
tenseurs de l'adaptateur n'ont pas changé pendant leur lecture. Les versions
des paramètres de base sont contrôlées et les fichiers parents ne sont jamais
écrasés. Ce contrôle ne constitue pas une attestation indépendante des poids.

```sh
python -m unittest tests_research.test_confidence_budget_diagnostic tests_language.test_confidence_ranking_gpu tests_language.test_answer_confidence_gpu -v
python -m research.confidence_budget_gpu JOURNAL DOSSIER_POIDS_COLAB24
python -m research.confidence_budget_diagnostic --journal JOURNAL --output RAPPORT
```

## Lancement observé

Le [reçu de lancement](../artifacts/confidence-budget-pilot/launch.json),
capturé à 04:05:41 UTC le 20 septembre, constate les douze tests réussis sur
Colab en 19,071 secondes, une A100 40 Go active et 260 évaluations enregistrées
dans la reproduction du point de départ. Aucune nouvelle mise à jour n'est
encore enregistrée à cette capture. Le plan, les sources et l'ordre des appels
correspondent à la préparation. La révision exécutée est
`a9b1a27054056ec3f6c8300319036f5950ee4f32` ; les résultats de poursuite restent
en attente. Ce reçu décrit une capture historique, pas un compteur en temps réel.
