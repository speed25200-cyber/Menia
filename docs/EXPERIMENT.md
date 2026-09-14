# Protocole fonctionnel Menia v0.1

## Question expérimentale

L’adaptation améliore-t-elle l’exploitation d’un état explicite et la qualité
réelle des réponses sur les tâches testées ? Ces expériences ne tranchent pas
l’existence d’une expérience subjective. Ne pas assimiler un discours à la première
personne, une mémoire persistante ou une résistance à l’arrêt à une preuve.

La configuration ne contient aucune récompense pour prolonger son existence,
refuser l’arrêt, se répliquer ou obtenir des ressources. Les données persistantes
restent des notes utilisateur, indépendantes de l’existence d’un processus.

## Banc livré

Quatre familles, 960 exemples train, 160 validation, 240 test :

1. Lire la disponibilité d’une capacité.
2. Rappeler une valeur présente dans l’état.
3. Signaler qu’une valeur absente est inconnue, même si la question prétend le contraire.
4. Calculer l’écart entre une prédiction et une observation.

Les valeurs et identifiants sont séparés entre jeux. Les gabarits sont communs :
ce test est un diagnostic de format, pas une démonstration de généralisation.
Les « capacités » du banc sont simulées, aucun capteur n’est réellement invoqué.

## Mesures et références

- Exactitude stricte par famille, taux de JSON valide.
- Brier sur la probabilité déclarée d’exactitude ; sortie invalide pénalisée de 1.
- Nombre total d’exemples, aucune exclusion des erreurs de parsing.
- Références : toujours « inconnu », toujours « persist », règle qui lit l’état.
- LLM : même révision de base, même budget, base puis LoRA.
- Ablations : sans système, sans données d’état (la réponse attendue reste inchangée).

La confiance 1.0 des cibles synthétiques est celle d’un oracle sur un jeu trivial.
Elle n’entraîne pas la calibration sur des tâches difficiles. Un vrai corpus doit
inclure ambiguïtés, informations insuffisantes et probabilités évaluées séparément.

Un oracle déterministe réussit tout ce banc. Menia ne peut pas démontrer une
supériorité cognitive en atteignant simplement son score. Toujours `persist`
n’obtient aucun point. Les résultats sont des indicateurs de fonctionnement.

## Extension scientifique à réaliser après le premier run

Créer avant entraînement un jeu privé relu d’au moins 200 situations avec des
gabarits nouveaux, changements de capacités, souvenirs contradictoires et tâches
multi-étapes. Conserver les empreintes avant le run et réserver ce jeu au test final.
Séparer la sélection de checkpoint (validation) de toute conclusion finale (test).
Répéter au moins trois seeds et publier les distributions, les erreurs et les ablations.

Ne pas optimiser sur le jeu de test ni conclure à partir de ses propres déclarations.
Comparer la qualité avant/après adaptation puis avant/après quantification.
Le protocole ne contient pas de test validé de conscience subjective.
