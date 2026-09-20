# Prévoir une conséquence de son état avant d'agir

20 septembre 2026. **Note de conception, pas un protocole confirmatoire figé
ni un résultat expérimental.** Le diagnostic de budget reste en cours ; son
issue et le contrôle numérique détermineront les essais réalisables. Cette
note ne remplace pas l'objectif de conscience par un score de prévision.

## Ce que les nouvelles sources changent

[Liu et al., RLMF, §3.2 et annexe B.4](https://arxiv.org/html/2606.32032v1)
renforcent certaines mises à jour selon la qualité d'une auto-évaluation.
Leur « confiance intrinsèque » est estimée par cohérence entre réponses
échantillonnées, avec jugement NLI ; elle n'est pas une lecture directe de
l'état caché. Leur méthode distingue fidélité à cette confiance et accord
avec la correction factuelle. Elle offre une piste d'apprentissage déjà
publiée, sans fournir un critère de conscience. Les expériences ne sont pas
reproduites ici.

[Karvonen et al., présentation de CHIVE](https://alignment.anthropic.com/2026/chive/)
évaluent des prévisions d'effets de modifications du prompt. Dans leur
évaluation, trois outils lisant les activations n'améliorent pas le témoin
qui lit seulement le texte. L'apprentissage sur les conséquences mesurées
améliore certaines prévisions dans des contextes réservés. Les explications
libres ne sont pas traitées comme vérité de référence. Ce résultat porte
sur leur évaluation, sans exclure tout apport des états internes ailleurs.
La présentation des auteurs a été consultée ; aucune réplication n'est faite.

**Conséquence proposée pour Menia :** ne pas récompenser seulement une
description plausible de soi ou l'accord entre ses propres réponses. La cible
doit être une conséquence observable du système, obtenue dans une branche
qui n'a pas reçu sa prévision. Le texte seul reste une référence obligatoire.
Entraîner de telles prévisions n'est donc pas, en soi, une nouveauté à revendiquer.

## Cible à modifier

Les études de confiance 23 et 24 demandent si une réponse déjà écrite est
correcte. Un changement ultérieur du cache ne change pas cette correction.
Il peut déplacer le jugement, sans démontrer le suivi d'une capacité modifiée.
Le [contrôle numérique](GENERATION_NUMERICS_PROTOCOL.md) répond à un autre
problème préalable : s'assurer qu'un écart entre caches ne vient pas seulement
du rythme des appels en BF16.

La cible prospective proposée est : **avec cet état précis, quelle est la
probabilité de réussir la prochaine tâche, si elle est exécutée maintenant ?**
La tâche et ses conditions d'exécution sont connues du prédicteur. Sa solution,
les futurs tokens de réponse, le label de réussite et les paramètres cachés
de l'intervention ne lui sont pas donnés. En test, toutes les prévisions
sont enregistrées avant les exécutions qui les noteront.

Le contrôleur d'expérience conserve une copie de l'état avant la prévision.
Une première branche produit la prévision ; une seconde repart de cette même
copie pour exécuter la tâche. Le texte de la prévision ne rejoint pas la tâche.
Des réalisations répétées, avec graines fixées distinctes, peuvent estimer une
probabilité de succès plutôt que transformer une seule réalisation aléatoire
en certitude. Leur coût et leur dépendance par épisode doivent être comptés.

## Conditions qui rendent la comparaison interprétable

| Condition | Ce qu'elle doit contrôler |
|---|---|
| État courant conservé | La prévision et la tâche utilisent la même copie initiale. |
| Relecture exacte au même rythme | Le texte et les appels de calcul sont fixés ; l'effet ne doit pas être attribué au nouveau rendu de l'historique. |
| Altération cachée à texte identique | Une variation interne peut changer la conséquence sans annoncer son identité au prédicteur. |
| Restauration de l'état | La récupération de la prévision doit accompagner celle de la conséquence, et pas seulement un déplacement des codes. |
| Déplacement apparié sans effet pertinent | Une simple alarme de perturbation ou sa norme ne doit pas expliquer toute la prévision. |
| Donneur apparié et donneur permuté | Séparer l'information de capacité du contenu de réponse, de la difficulté et de l'identité d'un épisode. |
| Prédicteur qui lit le texte | Mesurer ce que les données publiques suffisent à prévoir, avec budget et apprentissage déclarés. |
| Règle sur logits de sortie | Quantifier l'apport au-delà d'un signal de confiance accessible, sans donner au témoin la future réponse de test. |

Les altérations et témoins restent à choisir sur un lot de découverte séparé.
Le même corpus, les mêmes réponses de référence et le même budget doivent
servir aux comparaisons d'apprentissage. Une direction choisie pour modifier
le code du prédicteur ne suffit pas : il faut d'abord vérifier son effet sur
la tâche et sur des tâches témoins. Une perturbation sans effet mesuré peut
être utile comme témoin, mais sa sélection ne doit pas utiliser les résultats
du test réservé. Le choix de sites, amplitudes et donneurs devra être figé.

Une altération qui détruit toute trace d'une information nécessaire ne permet
pas de demander au modèle de la deviner. L'[audit d'information antérieur](SELF_INFORMATION_AUDIT.md)
reste applicable : des états observables identiques peuvent correspondre à
des origines différentes. Prévoir une fréquence conditionnelle de réussite
et identifier la vérité de chaque épisode sont des tâches distinctes.

## Mesurer l'effet et l'utilité séparément

Le tableau principal devra montrer, pour chaque condition et répétition,
la réussite de tâche, l'erreur de prévision, le format natif, la couverture
des deux classes et les effets d'intervention appariés. Une amélioration
globale obtenue seulement en reconnaissant les catégories faciles ne suffira
pas ; les comparaisons au sein de catégories restent nécessaires.

Un second essai de décision devra comparer le choix natif du modèle, une règle
appliquée extérieurement au même score, et une politique sans score. Toutes
les politiques devront recevoir les mêmes possibilités de vérification et
payer leur coût réellement mesuré. Un meilleur score n'implique pas une
meilleure décision, notamment si les probabilités restent du même côté du
seuil utile. Ce second essai ne devra pas recycler comme test réservé les
épisodes ayant servi à sélectionner une stratégie.

## Ce qui reste ouvert

Cette note ne choisit pas de nouveau checkpoint sur les résultats en cours,
ni d'intervention qui n'aurait encore aucun effet vérifié. Les volumes,
seuils, formulations, lots réservés, intervalles et corrections multiples
restent à fixer avant une collecte confirmatoire. Une collecte de découverte
devra être nommée comme telle. Les scripts numériques et d'audit actuels ne
constituent pas cette expérience.

Même un résultat positif établirait au plus un mécanisme fonctionnel de
prévision et de contrôle dans ses conditions. Le passage de ces fonctions
à une expérience subjective de sa propre existence reste une question
scientifique ouverte ; cette note n'en propose pas de certificat.
