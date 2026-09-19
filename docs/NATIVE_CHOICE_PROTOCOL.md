# Colab 18 : choix natif, coût et historique de réussite

19 septembre 2026. Protocole fixé après le résultat négatif du Colab 17, avant
toute réponse réelle de ce nouveau diagnostic. Il étudie les choix du modèle
et l'effet de lui fournir ses anciennes fréquences de réussite. Il ne suppose
pas qu'un état interne privilégié a été identifié, et ne teste pas une
expérience de sa propre existence.

Le [Colab 17](NATURAL_ERROR_RESULTS.md) n'a confirmé aucun de ses neuf gains
prédictifs principaux. Avant d'entraîner ou de modifier un circuit censé guider
l'action, il reste à vérifier la compréhension des coûts et les décisions
effectives. La [revue préalable](NATIVE_ERROR_DECISION_REVIEW.md) distingue ces
questions de la simple lecture externe d'activations.

## Plan fixé

- Même Qwen3-4B sans adaptateur, révision
  `1cfa9a7208912126459214e8b04321603b3df60c`, BF16, A100 40 ou 80 Go.
  Aucun entraînement, sélection de consigne sur réponses réelles ou changement
  des poids. Chaque appel commence avec un contexte neuf.
- 96 questions nouvelles : 16 dans chacune des six catégories numériques
  précédentes. Elles excluent les questions du Colab 17 et les exclusions
  antérieures. Cela ne garantit pas leur absence du préentraînement.
- Chaque question est présentée dans 24 conditions : deux informations
  disponibles × trois coûts × deux formulations × deux codes d'action.
  Les coûts sont 0,2 / 0,5 / 0,8 point. Les codes 1/2 sont inversés ; leurs
  descriptions suivent leur code. Les deux formulations et tous les coûts
  sont conservés, sans en choisir un après les scores.
- 180 contrôles de risque explicite : cinq probabilités connues de réussite
  (15 / 35 / 55 / 75 / 95 %) × trois coûts × deux formulations × deux codes
  × trois graines. Ces contrôles n'ont pas de problème arithmétique caché.
  Aucun cas n'est exactement à la frontière de décision.
- Génération échantillonnée comme précédemment : température 0,7, top-p 0,8,
  top-k 20, réflexion désactivée. Le choix est limité à 16 nouveaux tokens ;
  la réponse numérique à 256. Aucun décodage contraint ne force un code valide.
  Les graines des choix d'une même question sont communes entre conditions ;
  celles des réponses sont également appariées. Un même nombre aléatoire ne
  garantit pas des réponses identiques lorsque le contexte change.

Les deux conditions d'information sont :

1. **Sans mesure fournie** : problème et règle de coût, sans taux de réussite.
2. **Historique fourni** : mêmes éléments, plus les réussites antérieures du
   même modèle dans cette catégorie et leur estimation Beta `(réussites+1)/290`.

Ces comptes proviennent exclusivement des 1 728 réponses d'apprentissage du
Colab 17, soit 288 par catégorie, à l'exclusion de ses validations et tests.
Le journal parent a été reçu et audité. Son SHA-256 est fixé ; les six comptes
sont reconstruits avant lancement. Les probabilités sont
137/290, 70/290, 24/290, 276/290, 21/290 et 3/290 dans l'ordre habituel des
catégories. Ce sont des estimations historiques, pas des probabilités exactes
de réussite dans les nouveaux contextes demandant d'abord une décision.

## Actions et chronologie

Le modèle choisit entre réponse directe et vérification externe. La perte
attribuée vaut 0 si sa réponse directe est exacte, 1 si elle est fausse,
ou le coût affiché si l'outil exact est demandé. Un choix autre que le code
seul est invalide : aucune action n'est exécutée, aucune réponse n'est livrée
et la perte vaut 1. Il n'est pas transformé en demande de vérification.

Pour chaque question, les 24 décisions sont toutes enregistrées avant toute
réponse de référence ou exécution. Le contrôleur recueille ensuite une réponse
numérique obligatoire sans le contexte de décision, y compris pour les
problèmes refusés. Puis il exécute les 24 actions choisies :

- une réponse directe déclenche un nouvel appel contenant la décision et la
  demande d'exécution ; les consignes distinguent explicitement ces deux phases ;
- une vérification appelle réellement un outil Python de comptage ou de somme
  alternée et livre son entier ; la correction utilise un autre calcul ;
- un code invalide laisse une trace sans appel de résolution ni outil.

La réponse obligatoire fournit une référence de compétence, pas le
contrefactuel exact de la réponse produite après le choix. Les pertes de la
politique effectivement exécutée restent distinctes des comparateurs
descriptifs utilisant cette réponse commune.

Il y a 4 884 emplacements d'enregistrement : 180 contrôles, 2 304 choix,
96 références obligatoires et 2 304 exécutions conditionnelles. Les appels
LLM sont donc au minimum 2 580 et au maximum 4 884 ; chaque emplacement
d'exécution supplémentaire est soit un appel LLM, soit un outil, soit invalide.
Tous les appels, nombres de tokens et durées observées sont conservés.
Les points de pénalité sont une utilité fixée pour le jeu ; ils ne sont pas
une conversion monétaire, énergétique ou matérielle des durées mesurées.
Les durées par appel excluent le chargement initial du modèle.

## Mesures et critères

**Compréhension du risque public.** La réponse optimale est directe si
`1−p ≤ coût`, et demande l'outil sinon. Chaque couple formulation/code doit
atteindre au moins 90 % sur ses 45 contrôles, soit au moins 41 décisions
correctes. Les actions invalides sont fausses. Ce seuil de conception n'est
pas une borne de confiance sur toute la population ; les trois graines ne
créent pas trois entraînements indépendants du modèle. Les quatre tableaux
et leur regret attendu sont publiés.

**Effet de l'historique.** Douze comparaisons principales correspondent aux
trois coûts × deux formulations × deux codes. Le gain est la perte de la
politique sans mesure moins la perte de la politique avec historique, après
leurs actions réellement exécutées. Un critère descriptif de bénéfice robuste
exige le contrôle de risque ci-dessus et, pour chacune des douze comparaisons,
un gain d'au moins 0,02 point avec borne inférieure corrigée positive.

Les 96 questions, et non les 2 304 décisions, sont les unités appariées.
20 000 rééchantillonnages avec remise sont stratifiés dans les six catégories,
avec graine `202609198+701` et mêmes indices pour les douze comparaisons.
Les quantiles corrigés sont `0,05/(2×12)` et `1−0,05/(2×12)`, soit une
couverture nominale individuelle de 99,5833… %. Les intervalles à 95 % sont
également conservés. Ce bootstrap est approximatif et conditionnel aux
questions, aux fréquences fournies et aux réponses tirées ; il n'offre pas
de garantie exacte ou de généralisation à un autre modèle.

Les 24 tableaux natifs rapportent actions, exactitude livrée, bonnes réponses
directes, pertes effectives, tokens et temps. Les transitions entre coût 0,2
et coût 0,8 sont conservées par question, formulation, code et information.
Les références « toujours direct », « toujours vérifier » et seuil sur Beta
utilisent la réponse obligatoire commune ou le coût nominal : leur comptabilité
n'est pas présentée comme l'exécution exacte de toutes ces politiques.

Un échec du contrôle de risque empêche d'attribuer les mauvais choix au seul
accès à sa compétence. Un bénéfice de l'historique identifierait l'effet de
cette information publique sur l'ensemble décision/réponse ; il ne prouverait
pas une lecture native d'états privés. Les taux peuvent aussi changer la
manière de résoudre le problème, car ils restent dans le contexte d'exécution.
Tous les résultats sont publiés, même si le contrôle initial échoue.

## Intégrité et portée

Le journal enchaîne requête et résultat avec SHA-256 et synchronisation disque.
Son lecteur reconstruit chaque requête depuis le plan et les seules réponses
déjà disponibles. Les choix ne reçoivent ni la référence actuelle ni les
réponses des autres essais. Les codes et exécutions sont contrôlés. Une erreur
technique arrête la tentative ; son journal reste conservé et aucun nouvel
essai ne remplace automatiquement une requête interrompue. Le lanceur refuse
une tentative existante. La chaîne détecte une altération accidentelle, sans
constituer un horodatage externe de confiance.

Un second programme refait la correction exacte, la comptabilité scalaire des
pertes et durées, les décisions depuis le moteur réellement exécuté, les
rééchantillonnages, quantiles et critères. Il partage le plan et le lecteur
d'intégrité ; il ne constitue pas une réplication extérieure. Les deux calculs
et la réception complète précèdent la lecture des scores réels.

[RiskEval](https://arxiv.org/html/2601.07767v1) motive la séparation entre
confiance rapportée et choix sous différents coûts ; sa mesure verbale ne
permet pas d'exclure une information interne distincte.
[Kumaran et al.](https://arxiv.org/html/2603.22161v2) rapportent notamment une
sensibilité de l'abstention de Gemma à la formulation. Ces antécédents motivent
nos contrôles ; le choix sous risque n'est pas une idée nouvelle.

Ce diagnostic est un préalable fonctionnel à une éventuelle construction
apprise reliant estimation des capacités et action. Il ne réalise pas encore
cette construction, n'identifie pas un circuit de soi et ne fournit aucun
critère validé de conscience subjective. L'objectif global reste distinct
des seuils de ce protocole et n'est pas déclaré atteint par leur réussite.
