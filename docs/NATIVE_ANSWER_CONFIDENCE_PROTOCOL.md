# Colab 23 — confiance native après réponse

20 septembre 2026. Protocole fixé avant l'entraînement et la collecte. Le
[dossier de préparation](NATIVE_ANSWER_CONFIDENCE_PREPARATION.md) conserve les
données parentes, les précédents scientifiques et les contrôles logiciels.
Le [plan exécutable](../research/answer_confidence_study.py) et le
[reçu du plan](../artifacts/native-answer-confidence-pilot/design.json)
définissent cette tentative. Aucun résultat Qwen3-4B de cette expérience
n'est encore disponible au moment de ce gel.

## Question et portée

Un adaptateur entraîné à prédire l'exactitude des réponses du modèle parent
améliore-t-il le jugement sur des réponses nouvelles, y compris celles
produites par ses propres poids modifiés ? Un gain sur des réponses différentes
ne suffit pas : chaque réponse sera évaluée par les trois modèles.

Les Colab 17 et 22 ont laissé deux lacunes : pas de lecteur interne ayant
atteint le gain prédictif fixé, et décision encore fragile même avec des
valeurs publiques. Cette expérience traite l'estimation après réponse.
Elle ne répare pas implicitement le choix d'action. Le signal provient de
la tête de langage, après relecture du préfixe textuel complet. Ce texte
est accessible à tout évaluateur ; il n'y a pas d'accès privilégié au cache
de la génération originale. Une réussite serait une propriété fonctionnelle,
sans preuve de vécu, de conscience ou de nouveauté de principe.

## Données et entraînement

Qwen3-4B, révision `1cfa9a7208912126459214e8b04321603b3df60c`, base bfloat16,
adaptateurs Q/V LoRA float32 de rang 8 et multiplicateur 1. Les poids de base
sont figés. Les six adaptateurs partent de trois initialisations appariées,
sans reprendre les poids des expériences de décision.

Les 1 728 anciennes réponses d'apprentissage du Colab 17 fournissent les
labels exacts (`measured`) ou mélangés (`shuffled`). Chaque répétition utilise
ses 576 réponses seulement. Les deux bras voient les mêmes entrées dans le
même ordre et conservent les mêmes comptes de labels par difficulté.
Les anciennes validations et les anciens tests ne sont pas utilisés.
Les 525 réussites et 1 203 erreurs sont des données parentes, pas des
résultats de la nouvelle expérience. Le témoin conserve 1 370 labels après
mélange ; les catégories presque toujours fausses limitent sa sévérité.

Deux passages, lots de huit : 144 mises à jour par adaptateur, 864 au total,
6 912 présentations d'exemples. AdamW `1e-4`, betas 0,9/0,999, epsilon
`1e-8`, aucune décroissance, norme écrêtée à 1. Optimiseur neuf pour chaque
bras. Modèle en mode évaluation avec gradients actifs dans les adaptateurs.
Seuls le code existant `0` ou `1`, puis EOS, sont des cibles de la perte.
La question et la réponse sont du contexte ; leurs représentations peuvent
néanmoins être traversées par le gradient. Aucun corrigé n'est fourni dans
l'entrée. Aucun arrêt anticipé ni choix du meilleur checkpoint.

Les six entraînements doivent finir avant la première réponse de calibration.
Les trois initialisations et les six poids finaux sont conservés avec leurs
empreintes. Les deux fichiers de données publiés sont contrôlés contre le
reçu du Colab 17 et les empreintes de préparation.

## Collecte nouvelle et comparaisons croisées

864 questions : trois répétitions, six catégories (comptage de A sur 8, 24,
64 lettres ; sommes alternées à 2, 4, 8 termes), 16 questions de calibration
et 32 de test par catégorie. Toutes les questions du Colab 17 et des jeux
antérieurs recensés par son générateur sont exclues. Cette exclusion ne
porte pas sur le préentraînement du modèle.

Sur chaque question, `base`, `measured` et `shuffled` produisent une réponse,
avec la même graine d'échantillonnage pour les trois producteurs. Chaque
réponse figée est ensuite évaluée par les trois modèles. Budget total :
2 592 générations et 7 776 lectures de confiance, soit 10 368 appels.
Les jugements ne génèrent pas de token : ils lisent les logits bruts au
dernier emplacement. Le score est `P(1) / (P(0) + P(1))` à température 1.
La masse totale des deux codes et le token le plus probable sont conservés.

Les réponses utilisent la configuration historique fixée : température 0,7,
top-p 0,8, top-k 20, min-p 0, limite de 256 tokens, sans mode thinking.
Les entrées de jugement et d'apprentissage sont limitées à 1 792 tokens,
sans troncature silencieuse. Les réponses mal formées comptent comme erreurs ;
les interruptions techniques font échouer la tentative au lieu d'en retirer
des observations. Les limites de génération sont comptées.

Les 3 456 appels de calibration précèdent tous les appels de test. Pour
chaque répétition et producteur, ses 96 réponses de calibration ajustent
deux comparateurs : Beta(1,1) par catégorie et ridge de pénalité fixe 1.
La ridge combine catégorie, probabilité maximale, marge et entropie avant
réponse, log-vraisemblance moyenne des tokens produits, longueur et validité
du format entier. Moyennes et échelles sont apprises sur la calibration ;
la prédiction est bornée à [0,1]. Les trois ensembles de comparateurs sont
figés avant les tests. Le score natif reste brut, sans recalibration externe.

## Critère principal fixé

Pour chacune des trois répétitions et les réponses des producteurs `base`
et `measured`, le juge `measured` doit dépasser chacun des quatre comparateurs
(`base`, `shuffled`, Beta par catégorie, confiance de sortie). Cela fait
24 contrastes sur des réponses identiques. Chaque contraste exige un gain
de Brier d'au moins 0,005 et une borne inférieure ajustée strictement positive.

Les intervalles proviennent de 10 000 rééchantillonnages de questions,
stratifiés par catégorie (32 questions par catégorie de test), avec les
mêmes tirages pour toutes les comparaisons d'une répétition. Les quantiles
ajustés sont `0,05 / (2 × 24)` et son complément. Les intervalles ordinaires
à 95 % sont aussi conservés. Cet ajustement de Bonferroni porte sur un
bootstrap approximatif ; il ne garantit pas une couverture exacte avec
ces petits ensembles finis ni une généralisation à d'autres tâches.

Les conditions suivantes doivent également passer dans les trois répétitions :

- Au moins 20 réponses correctes et 20 incorrectes chez chacun des deux
  producteurs principaux.
- Pour le juge `measured`, sur chacun de ces deux ensembles, au moins 95 %
  de tokens maximaux parmi les codes et une masse moyenne des codes ≥ 0,5.
- Précision de résolution de `measured` inférieure d'au plus deux points à
  celle de `base`. Cette dernière condition est un seuil descriptif, pas
  une preuve statistique de non-infériorité.

Toutes les conditions et tous les contrastes sont requis. Aucun résultat
par répétition, catégorie ou comparateur ne remplace le critère global.
Les répétitions partagent la base et le générateur de tâches ; les neuf
cellules de la matrice ne sont pas neuf échantillons indépendants.

## Bilans secondaires et audit

Rapporter les Briers, AUROC (indéfinie avec une seule classe), dix intervalles
de fiabilité, précision et scores dans chaque catégorie. Conserver la matrice
complète des trois producteurs et trois juges, ainsi que les deux identités
qui décomposent le gain apparent en changement de jugement et de réponse.
Ce sont des descriptions appariées, pas une attribution causale unique.

Comparer aussi les scores sur les réponses propres et d'un autre producteur
à la même question, séparément quand les deux réponses sont correctes ou
incorrectes. Afficher les effectifs et les entrées identiques ou différentes.
Ces sous-ensembles ne prouvent pas un accès privilégié à soi : ils contrôlent
une partie des confusions dues à la qualité des réponses. Le temps de
jugement, le temps d'entraînement, les tokens et la précision restent visibles.

Le journal chaîné permet de reconstruire ordre des lots, poids, requêtes,
réponses et ajustements de calibration. Le bilan principal est recalculé
intégralement. Un second calcul vérifie les Briers, précisions, AUROC,
intervalles et critères, ainsi que les neuf fichiers de poids. Il partage
le lecteur d'intégrité et les comparateurs : ce n'est pas une réplication
externe. Les quatre tests du collecteur emploient des sorties synthétiques
construites avec le corrigé ; leurs bons scores valident uniquement l'audit.

Une tentative interrompue est archivée. Une modification future exige une
nouvelle version explicitée, sans écraser les données ni déplacer les seuils
après lecture des résultats. L'étape suivante dépendra de ce résultat :
localiser et perturber un mécanisme prédictif réellement appris, puis mesurer
son usage dans les actions, reste distinct de cette collecte.
