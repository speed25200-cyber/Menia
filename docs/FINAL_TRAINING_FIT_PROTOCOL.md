# Diagnostic des poids finaux sur les exemples appris

20 septembre 2026. Diagnostic décidé **après** lecture du
[Colab 24](CONFIDENCE_RANKING_RESULTS.md). Il ne change pas ce résultat négatif
et ne constitue pas une nouvelle confirmation. Il n'effectue aucune mise à
jour, ne génère aucune réponse et n'utilise aucun exemple de test en entrée.

## Question à résoudre

Le suivi précédent observait les scores avant chaque mise à jour, pendant que
les paramètres et les paires changeaient. Il ne mesurait pas l'ajustement du
checkpoint final. Ce diagnostic évalue les neuf adaptateurs finaux sur les
exemples précis qu'ils ont appris, afin d'éviter d'attribuer trop vite les
limites du lot 24 à la seule généralisation ou à la seule capacité.

Une mauvaise discrimination sur les exemples connus est compatible avec un
problème d'optimisation, de représentation, de budget ou d'objectif. Elle ne
les départage pas. Une bonne discrimination sur ces exemples, accompagnée
d'un écart sur des exemples nouveaux, orienterait vers la généralisation ;
elle ne démontrerait ni mémorisation ni mécanisme de connaissance de soi.

## Entrées et poids fixés

Chaque bras CE, classement et neutralisé de chaque répétition relit ses
576 exemples, soit **5 184 évaluations** au total. Les entrées sont exactement
les anciennes questions et réponses du Colab 17, préparées puis utilisées
pour entraîner le Colab 24. Les cibles servent seulement à la notation après
inférence. Aucun partenaire de paire, corrigé ou code cible n'est ajouté au
prompt d'évaluation. Les indices et messages sont vérifiés par le lecteur.

L'empreinte sémantique des 1 728 exemples est
`0694f42765731e3ab2031da214855468112c0bdfefe6ddd8019c769efcc403a8`.
Les neuf fichiers de poids sont ceux du
[relevé de fin d'entraînement](../artifacts/confidence-ranking-pilot/training-freeze.json),
identiques à ceux de l'archive finale. Ils restent au rang 8 et à l'échelle 1.
Qwen3-4B et sa révision, BF16, l'attention SDPA et la lecture du score sont
ceux du lot 24. La fonction `assess_answer` est réutilisée sans changement.

Les gradients sont désactivés. Après chaque unité, les tenseurs de l'adaptateur
en mémoire doivent encore correspondre exactement au fichier chargé. Les
empreintes des fichiers et les versions des paramètres de base sont aussi
contrôlées. Le contrôle de versions n'est pas une attestation cryptographique
indépendante de tous les poids de base.

## Mesures et interprétation

Le rapport conserve Brier, AUROC globale et dans les catégories, accord binaire
au seuil fixe de 0,5, masse des codes et premier token le plus probable. Les
catégories sans les deux classes gardent une AUROC indéfinie. Les métriques
descriptives sont recalculées séparément, dont le classement par comparaisons
explicites de paires, sans utiliser le tri du premier calcul. Le lecteur du
journal et les exemples restent communs ; ce n'est pas une réplication externe.

Il n'y a **aucun seuil de réussite**, aucune recherche d'hyperparamètres et
aucun test de significativité. Un score sur les données apprises ne deviendra
pas une preuve de généralisation. La comparaison descriptive la plus proche
dans le lot 24 porte sur les réponses nouvelles du producteur de base ; les
réponses propres des adaptateurs constituent une autre distribution.

Ce contrôle ne compare pas encore deux capacités ni deux tailles de données.
Il donne une mesure manquante pour choisir entre ces expériences possibles.
Il n'établit ni usage natif de la confiance pour agir, ni accès privilégié à
un état de génération, ni conscience ou nouveauté scientifique.

## Exécution et vérification

Le [plan préparé](../artifacts/confidence-training-fit-preparation/design.json)
fixe les empreintes des neuf checkpoints, des sources et de l'ordre des appels.
Trois nouveaux tests vérifient la couverture des exemples, les deux calculs
sur des sorties construites, et le rejet de messages, poids déclarés, ordre
ou complétude incorrects. Les trois tests existants de lecture du score Qwen
sont aussi exécutés. Ces six tests passent sur PC ; ils ne mesurent pas les
capacités du modèle préentraîné.

Le lanceur demande un Colab 24 terminé et une A100 libre. Il conserve le
runtime Python déjà utilisé, récupère une révision Git immuable, exécute les
six tests, puis lance le diagnostic. Une tentative existante est conservée,
sans relance automatique. L'archive de sortie ne duplique pas les poids déjà
sauvegardés : elle contient le journal, le rapport et les traces d'exécution.

```sh
python -m unittest tests_research.test_confidence_final_training_fit tests_language.test_answer_confidence_gpu -v
python -m research.confidence_final_training_fit_gpu JOURNAL DOSSIER_POIDS_COLAB24
python -m research.confidence_final_training_fit --journal JOURNAL --output RAPPORT
```
