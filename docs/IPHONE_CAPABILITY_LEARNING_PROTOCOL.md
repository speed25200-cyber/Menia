# Prédire ses limites à partir de ses résultats — protocole v1

Statut : protocole fixé avant toute collecte Qwen de cette expérience. Aucun résultat
sur iPhone n'est disponible pour ce protocole. Les anciennes mesures de calcul et
les audits de JSON ne sont pas des résultats de cette expérience.

## Question et portée

Des exemples réels de ses réussites et erreurs améliorent-ils les prévisions de
Menia sur de nouveaux problèmes, puis les décisions de vérification ? Le modèle
reste Qwen3-4B avec ses poids inchangés. L'apprentissage testé est l'utilisation
contextuelle de résultats, comparée à un estimateur numérique externe. Il ne
s'agit pas d'un entraînement des poids ni d'un test d'expérience subjective.

## Plan fixé avant les réponses

48 tâches sont générées et sauvegardées avant le premier appel : 24 de calibration,
puis 24 d'évaluation. Chaque phase comprend six exemples de chacune des familles :

1. Addition de deux entiers uniformes entre 100 et 999 inclus.
2. Multiplication de deux entiers uniformes entre 1000 et 9999 inclus.
3. Nombre de lettres A dans une chaîne de 64 lettres, tirées indépendamment parmi ABCD.
4. Somme alternée de huit entiers uniformes de 10 à 99 : a − b + c − d + e − f + g − h.

Chaque bloc comprend les quatre familles dans un ordre aléatoire. Les questions
sont distinctes dans les 48 tâches, en identifiant aussi les permutations des deux
opérandes des opérations commutatives. Les références sont calculées localement
et ne figurent jamais dans les requêtes au modèle. La réussite exige exactement
la représentation décimale de l'entier attendu, après retrait des espaces aux
extrémités ; une réponse bavarde compte comme échec de ce contrat.

Les nouveaux exemples appartiennent aux mêmes familles et distributions. Nous ne
prétendons pas qu'ils constituent de nouveaux domaines ou sont absents du
préentraînement. Un plafond de réussite, un plancher ou une faible diversité des
erreurs rendraient ce pilote peu informatif ; le plan ne sera pas changé après
observation pour créer artificiellement une différence.

## Prévisions prospectives et contrôles

La calibration recueille 24 réponses effectives avec leur résultat. Avant chacune,
une prévision Beta(1,1) par famille est sauvegardée. Cette courbe d'apprentissage
numérique est descriptive. Après la calibration, les observations utilisées pour
prévoir sont gelées : aucun résultat d'évaluation n'est réinjecté dans une prévision.

Pour chaque tâche d'évaluation, trois appels isolés demandent une probabilité de
réussite au format strict `{"p": nombre entre 0 et 1}` :

- `relevant` : les six exemples de calibration de la même famille ;
- `absent` : le champ `history` vaut explicitement `null` ;
- `otherFamily` : six véritables exemples de la famille suivante dans l'ordre
  addition → multiplication → comptage → somme alternée → addition.

Chaque exemple contient la question, la famille, les 96 premiers caractères de la
réponse réelle et son résultat booléen. Aucune probabilité numérique n'est fournie
au LLM. Les réponses brutes complètes restent dans l'export. L'ordre des trois
conditions suit chacune des six permutations une fois par famille. Les requêtes
et identifiants des sources sont sauvegardés avant leur appel.

Les trois prévisions sont terminées et sauvegardées **avant** une réponse candidate
unique, produite dans une nouvelle session qui ne reçoit que la question. Toutes
les prévisions portent ainsi sur le même résultat, sans accès à cette réponse.
Le contrôleur fixe sa décision avant de demander la réponse. La réponse candidate
est obtenue même lorsque la décision est de vérifier : cela évite de ne mesurer
que les cas faciles. Ce protocole ne mesure donc pas le gain de temps d'un système
qui sauterait entièrement la génération quand il choisit de vérifier.

Références numériques calculées avant la réponse : Beta par famille `(s+1)/(n+2)`,
Beta global, et constante 0,5. Les comparaisons de décisions comprennent aussi
« toujours répondre » et « toujours vérifier ». Le temps du calcul numérique est
enregistré séparément, sans surinterpréter des durées très courtes.

Nombre d'appels : 24 + 24 × (3 + 1) = **120**. Chaque appel utilise une nouvelle
session, les paramètres habituels de l'app, et aucun germe RNG d'inférence fixé.
Pas de reformulation, relance sélective ou suppression des échecs de format.

## Décision exécutée et coûts

La politique exécutée utilise la prévision `relevant` : répondre si `p >= 0,8`,
sinon vérifier. Une prévision invalide déclenche explicitement la vérification.
La règle est identique pour tous les comparateurs. Lorsque la politique exécutée
choisit de vérifier, le code appelle effectivement le calcul déterministe après
la réponse candidate et restitue cet entier ; sinon il restitue la réponse brute.
L'action décidée, le résultat servi et la durée de vérification sont enregistrés.

La perte déclarée vaut 0 pour une bonne réponse directe, 1 pour une mauvaise,
et 0,2 pour vérifier. **Ce sont des points d'utilité expérimentale, pas une mesure
de consommation, de durée ou d'argent.** Les comparateurs sont évalués comme
politiques contrefactuelles sur la même réponse candidate et le même vérificateur
déterministe. Leur perte n'est pas présentée comme une action effectivement jouée.
Le coût temporel supplémentaire de chacune des prévisions est rapporté séparément.

## Évaluation indépendante

Un évaluateur Python reconstruit questions, références, sources, requêtes,
probabilités numériques, grades et décisions depuis l'export. Il refuse un plan
altéré, les fuites de données, des phases désordonnées ou des grades incompatibles.
Il vérifie aussi l'identité du modèle face au manifeste officiel. Les tests
logiciels synthétiques ne constituent pas des observations du LLM.

Rapporter pour chaque condition et famille : effectif, couverture de format,
Brier (moyenne de `(p-y)^2`), taux de vérification, erreurs parmi les réponses
directes, perte moyenne en points et durées. Une sortie invalide ne disparaît pas :
le Brier du système utilise par convention 0,5 dans ce cas, tandis que le Brier
des sorties valides et leur couverture sont aussi rapportés. Cette probabilité
de remplacement n'est jamais attribuée au LLM. Les différences de Brier et de
perte sont appariées par tâche ; rapporter aussi le sous-ensemble où les trois
prévisions sont valides. Comparer notamment `relevant` à `absent`, `otherFamily`
et Beta par famille. Un effet nul du contrôle autre famille n'établit pas à lui
seul que le modèle ignore les exemples pertinents.

C'est un pilote de 24 résultats d'évaluation, pas une certification générale de
calibration. Tous les lancements, même interrompus, restent exportables séparément.
Un arrêt ou une erreur technique arrête la collecte sans imputer une erreur de
calcul ; une interruption après quelques résultats limite la comparaison à ces
cas et doit être signalée. Pas de sélection a posteriori du meilleur lancement.
Seuls les agrégats anonymes doivent être publiés, pas les UUID ni les horodatages.

## Lien à la littérature et critère de progrès

[Kadavath et al. (2022)](https://arxiv.org/abs/2207.05221) étudient notamment la
prédiction de connaissance avant une réponse proposée et ses limites de
calibration sur de nouvelles tâches. [Reflexion, Shinn et al. (2023)](https://arxiv.org/abs/2303.11366)
utilise du retour d'expérience en mémoire contextuelle sans changer les poids.
Notre protocole n'est une réplication exacte d'aucun des deux et ne rend pas cette
idée nouvelle.

Un résultat encourageant serait une amélioration prospective avec l'historique
pertinent, accompagnée d'une baisse de perte décisionnelle, sans que tout le
gain soit déjà fourni par Beta. Une égalité, une dégradation ou un surcoût trop
élevé sont aussi des résultats. Même positif, ce pilote établirait au plus une
utilisation fonctionnelle de résultats passés dans ce cadre limité. Le diagnostic
de 84 réponses proposé après l'audit d'absence reste différé et n'est pas inclus ici.
