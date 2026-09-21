# Colab 17 : prévoir les erreurs avant le premier token

Protocole fixé le 19 septembre 2026, après le résultat du Colab 16, avant les
3 456 réponses Qwen3-4B de ce nouveau jeu. **Lecteurs externes, aucun entraînement
des poids du LLM, aucun critère de conscience ou de nouveauté.**

Le Colab 16 favorise une valeur liée à la question donneuse ; il ne confirme
pas un état réutilisable pour une autre question. Ce protocole aborde une
condition fonctionnelle distincte : la prévision des erreurs naturelles.
Il ne prétend pas résoudre la disponibilité d'un état avant toute question
ou l'usage natif de cette information. Les 672 anciens essais donnaient un
Brier interne de 0,1171, contre 0,1132 pour la référence Beta, sans mesurer
les probabilités de sortie. Le nouveau test conserve ces références et
enrichit le contrôle de sortie avant de conclure à un apport interne.

## Questions, modèle et capture

- Qwen3-4B de base, révision `1cfa9a7208912126459214e8b04321603b3df60c`,
  BF16, A100 de 40 ou 80 Go. Aucun adaptateur des Colab 6–16.
- Six cellules : compter les A dans 8/24/64 lettres, calculer des sommes
  alternées de 2/4/8 entiers à deux chiffres. Instruction : répondre par
  l'entier seul. Toute autre forme ou valeur est comptée fausse, même si
  la génération atteint sa limite. Les erreurs techniques restent distinctes.
- Trois répétitions sur les mêmes poids. Dans chacune et chaque cellule :
  96 apprentissages, 32 validations et 64 tests. Au total, 1 728 réponses
  d'apprentissage, 576 de validation et 1 152 de test. Les 3 456 questions
  sont toutes disjointes et excluent 1 630 anciennes questions distinctes.
  Cela ne garantit pas une absence du préentraînement.
- Plan de questions préparé avant ce protocole : graine `202609197`, empreinte
  `1e57ec41851c7da867f30f166545802761d014f5820ac409cb49126d469deff3`.
  Ce plan reste inchangé ; le nouveau plan ajoute collecte et critères.
- Génération identique aux anciens essais numériques : température 0,7,
  top-p 0,8, top-k 20, au plus 256 nouveaux tokens, cache activé, réflexion
  désactivée, aucune troncature de prompt, graine propre à chaque question.
  Une seule réponse est produite par question, avec son résultat conservé.
- Capture de l'entrée projetée, de l'état de couche 17 et de l'état final
  normalisé au dernier token du prompt. Chaque projection a 128 dimensions.
  La tête de sortie fournit aussi maximum de probabilité, marge des deux
  premiers tokens et entropie normalisée. Le callback intervient après cette
  tête et **avant le premier tirage**. Les logarithmes de probabilité de la
  réponse terminée sont gardés après génération, jamais utilisés pour prévoir.

La [validation technique](PROSPECTIVE_CAPTURE_AND_CONTROLS.md) a conservé les
sorties, projections, probabilités et générateurs aléatoires dans quatre cas
courts du modèle réel. Elle ne garantit pas toutes les futures générations.

## Lecteurs et information disponible

Les cinq ensembles de variables sont emboîtés selon le tableau de la
[préparation](PROSPECTIVE_CAPTURE_AND_CONTROLS.md) : entrée seule (390),
confiance de sortie et cellule (9), entrée et confiance (393), contrôle
incluant l'état final (521), puis ajout de l'état intermédiaire (649).
Le contrôle final est une projection, pas toute l'information des logits.
Le token le plus probable est conservé dans le journal sans transformer son
identifiant arbitraire en variable numérique du lecteur.

Les six régressions ridge utilisent uniquement les 576 apprentissages de
leur répétition. Normalisation et coefficients sont ajustés sur ce lot ;
la régularisation minimise le Brier des 192 validations parmi
`0,001 / 0,01 / 0,1 / 1 / 10 / 100`. En cas d'égalité, la première valeur
de cette liste est retenue. Les prévisions sont limitées à `[0 ; 1]`.
Les trois répétitions ne sont pas trois entraînements indépendants du LLM.

Huit prévisions sont enregistrées pour chaque question de test : les cinq
ensembles précédents, la fréquence Beta par cellule `(réussites+1)/(n+2)`,
un lecteur ajusté avec les états intermédiaires permutés dans chaque cellule,
et le lecteur interne ordinaire recevant l'état intermédiaire d'un autre
exemple d'apprentissage de même cellule. Les deux contrôles conservent
l'entrée, la confiance et l'état final du problème courant. Pour validation
et test, les donneurs sont déterminés par le hachage fixé du numéro d'essai.
La permutation peut comporter des points fixes. Ces contrôles remplacent
l'entrée d'un lecteur externe, pas une activation du LLM.

## Engagement avant réponse et interruptions

Pour chaque répétition, les coefficients, la grille de validation, les IDs
des donneurs mélangés et l'empreinte des données d'ajustement sont enregistrés
avant la première question de test. Pour chaque réponse : requête, capture
avec ses prévisions, puis résultat. Chaque événement est écrit, vidé et
synchronisé sur disque. Une chaîne SHA-256 lie l'ordre des événements ; elle
détecte une altération accidentelle, sans fournir un horodatage externe de confiance.

Le lecteur de journal recalcule l'ajustement depuis l'apprentissage et la
validation, puis chaque prévision à partir des variables disponibles à cet
instant. Il rejette les réponses sans capture préalable, les mauvais
ajustements, les incohérences de trace et les champs de réponse dans la capture.
Le test GPU logiciel intercepte le premier tirage d'un petit Qwen et vérifie
que sa capture est déjà présente sur disque.

Une erreur technique arrête la tentative, qui reste conservée. Une requête
interrompue ou une erreur ne peut être remplacée automatiquement. Une collecte
délibérément limitée peut reprendre à la frontière exacte entre deux requêtes,
avec mêmes sources et environnement, sans tirer de nouveau une réponse complète.
Le lanceur Colab refuse de lancer une deuxième fois une tentative existante.
Le bilan scientifique nécessite les 3 456 résultats ; aucune décision de réussite
ni sélection d'une meilleure répétition n'est faite sur un lot partiel.

## Mesures et règle fixées

Le Brier, l'AUROC si les deux classes existent, les fréquences et dix cases de
calibration sont publiés pour chaque lecteur, répétition et cellule. La mesure
principale est le **Brier du comparateur moins le Brier interne** : un nombre
positif favorise l'ajout de l'état intermédiaire.

Neuf contrastes sont principaux : trois répétitions × trois comparateurs
(`finalControl`, `shuffledMiddle`, `donorMiddle`). Les douze autres comparaisons
au lecteur interne sont secondaires. Les différences restent appariées par
question. Vingt mille rééchantillonnages avec remise sont stratifiés dans les
six cellules, avec la graine `202609197 + 401 + répétition`. Les mêmes tirages
servent à tous les contrastes d'une répétition.

Le critère d'apport reproductible du lecteur exige :

1. Une collecte complète, les trois ajustements reconstruits et, dans chaque
   lot de test de 384 questions, au moins 20 réussites et 20 erreurs.
2. Chacun des neuf gains de Brier est au moins `0,005` et sa borne inférieure
   bootstrap, corrigée par Bonferroni pour neuf contrastes, dépasse zéro.

Chaque intervalle principal utilise les quantiles `0,05/(2×9)` et
`1−0,05/(2×9)`, soit une couverture nominale de 99,444… %. Les intervalles
individuels à 95 % sont aussi conservés. L'approximation bootstrap ne donne
pas une garantie exacte de couverture ; elle est conditionnelle aux lecteurs
ajustés, aux domaines et à une réponse tirée par question. Le seuil de `0,005`
et le minimum de classes sont des choix de conception, pas des seuils théoriques
d'introspection. Un échec de ces conditions reste publié et n'est pas remplacé.

À titre descriptif, un lecteur annonçant au moins 0,8 choisit une réponse
directe ; les autres cas reçoivent un coût supposé de vérification de 0,2,
contre 1 pour une réponse directe fausse. Aucun outil de vérification n'est
exécuté dans ce protocole : ce coût idéal ne mesure ni dépense réelle de
calcul ni amélioration effective d'une politique d'agent.

## Lien avec l'objectif de recherche

[Kadavath et al.](https://arxiv.org/abs/2207.05221) distinguent l'évaluation
d'une réponse proposée et la prévision de connaissance avant une réponse
particulière ; leurs résultats de calibration ont des limites de transfert.
[Binder et al.](https://arxiv.org/abs/2410.13787) entraînent des modèles à prévoir
leur comportement et les comparent à d'autres prédicteurs. Ce sont des
antécédents explicites, pas des preuves de notre dispositif ni de conscience.

Un gain ici indiquerait qu'un lecteur externe exploite de l'information
prédictive dans des projections supplémentaires, relativement à ces contrôles.
Il ne prouverait pas que le modèle la consulte, qu'elle lui est propre, qu'elle
représente son existence ou qu'elle est accompagnée d'une expérience.
Le test d'utilisation native, avec interventions causales et contrôle du coût
réel des actions, reste une étape distincte. L'objectif global n'est pas redéfini
comme la simple réussite de ce protocole.

Huit tests de collecte passent, dont trois contrôles existants de génération.
Les quatorze tests des composants préparés auparavant restent applicables.
L'audit complet du résultat et un second calcul arithmétique devront précéder
l'interprétation des mesures GPU. Les sources et critères sont publiés avant
collecte, avec conservation de tous les résultats et échecs.
