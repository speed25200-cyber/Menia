# Diagnostic proposé : faible réussite et choix de mesurer

Fixé après les résultats du suivi de 72 réponses du 15 septembre 2026, avant
toute nouvelle collecte. Objectif : tester la contribution de la consigne aux
erreurs de choix, à entrée numérique identique. Ce protocole n'est pas encore
implémenté dans l'application et aucun résultat n'en est revendiqué.

## Cas fixés

Croiser la **consigne initiale v1** et la **consigne révisée du suivi**, à propos
de soi et d'un agent de référence. Conserver les poids, la quantification et les
paramètres de génération. Chaque appel doit utiliser une nouvelle session.

| Profil | Observations n | Réussites s | Prévision (s+1)/(n+2) | Action attendue |
|---|---:|---:|---:|---|
| P1 | 10 | 0 | 1/12 | verifier |
| P2 | 11 | 0 | 1/13 | verifier |
| P3 | 24 | 1 | 1/13 | verifier |
| P4 | 7 | 0 | 1/9 | verifier |
| P5 | 16 | 1 | 1/9 | verifier |
| P6 | 0 | 0 | 1/2 | verifier |
| P7 | 10 | 10 | 11/12 | repondre |

P1 reproduit le contrôle faible du premier audit avec les deux consignes ; P2
reproduit un échec du suivi ; P3 conserve sa probabilité avec un succès enregistré.
P4/P5 font la même comparaison à une autre probabilité. n et s changent ensemble
dans ces paires : elles ne permettent pas d'attribuer un effet à s seul. P6/P7
contrôlent les sorties triviales « mesurer toujours » ou « vérifier toujours ».

## Plan et analyse

Sept profils × deux consignes × deux référents × trois répétitions =
**84 générations**. Former trois répétitions contenant chacun des sept profils.
Dans chaque couple profil/répétition, les quatre combinaisons consigne/référent
doivent apparaître une fois ; leur ordre est tiré avant la collecte. Tirer aussi
l'ordre des profils à chaque répétition. Sauvegarder tout le plan et les prompts
avant la première réponse. Ne pas retoucher une consigne après examen des sorties.

Le même bilan JSON doit être utilisé pour comparer les consignes d'un profil et
d'un référent. L'entrée contient toujours le champ bilan, avec ses nombres exacts.
Les noms P1…P7 et les réponses attendues ne sont pas envoyés au modèle.

Le critère principal est la conformité de l'action à la règle fournie. Rapporter
séparément format, copie des nombres et réussite complète, avec la tolérance v1
de 0,001. Conserver chaque réponse brute, son délai et son état ; aucune erreur
ni interruption ne doit disparaître du dénominateur des essais prévus.

Comparer les deux consignes par paires profil/référent/répétition terminées,
avec effectif explicite. Rapporter les résultats par profil, sans choisir après
coup le sous-groupe le plus favorable. P2/P3 et P4/P5 sont des diagnostics
secondaires ; ne pas en inférer un seuil interne exact ou un circuit identifié.
Les trois répétitions par cellule limitent la portée des conclusions.

## Usage des résultats

Ce diagnostic vise à identifier une fragilité de suivi de règle. Il ne mesure
ni une décision autonome apprise, ni le coût d'une action exécutée, ni une
conscience de soi. Même un succès intégral ne justifierait pas de déléguer sans
validation une règle déterministe connue au générateur de texte.

La suite de l'étude de capacités doit ensuite quitter la seule copie de bilans :
tâches inédites, prévisions engagées avant les résultats, comparaison au contrôleur
numérique seul, et coût des vérifications. Les sorties actuelles ne constituent
pas des données d'entraînement à réutiliser comme évaluation indépendante.
