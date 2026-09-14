# Un prototype Menia apprend à estimer la source avant de décider

## Ce qui est construit

Menia dispose maintenant d'un prototype exécutable qui utilise une estimation
apprise de contribution externe pour choisir une vérification, recevoir son
résultat et attribuer une source à un contenu en mémoire. L'estimation est
enregistrée avant la décision et avant l'accès à la vérité courante. L'agent
réutilise le journal EpisodeMemory existant ; une attribution inférée reste
distincte d'une observation certifiée.

Le prototype réalise une partie fonctionnelle de l'[hypothèse de lien avec
l'expérience de soi](SELF_EXPERIENCE_BRIDGE.md). Il ne démontre pas que son état
représente un sujet conscient, plutôt qu'une propriété du canal observé. Il
n'explique pas encore comment la réalisation de ces calculs constituerait une
expérience subjective. Aucun critère de conscience n'est validé par ce résultat.

## Ce qui a été appris et ce qui a été conçu

Les poids du prédicteur apprennent à estimer si une source externe a contribué
à un signal qui peut également contenir une contribution simulée. Les indices
accessibles sont une force composée, une trace bruitée du fonctionnement du canal,
l'intention de simuler et le retour d'une éventuelle vérification précédente.
Le canal possède un état caché persistant. Le contenu est symbolique : aucune
vision, génération d'image ou compréhension sémantique n'est apprise ici.

La politique de décision est conçue, pas apprise. Une vérification parfaite coûte
0,25 ; une attribution erronée non vérifiée coûte 1. L'agent vérifie quand son
erreur minimale estimée, min(q,1-q), dépasse ce coût. La sortie q est donc une
probabilité de contribution externe ; elle n'est ni une probabilité de conscience,
ni une intensité de ressenti.

L'entraînement utilise 512 épisodes de 48 étapes, avec une politique exploratoire
qui vérifie indépendamment environ la moitié des étapes. Sur 24 576 étapes uniques,
12 295 résultats vérifiés sont disponibles. Les autres cibles sont remplacées
par zéro et masquées avant transmission à l'optimiseur. Seul un résultat vérifié
de l'étape précédente peut entrer dans l'estimation courante.

Trois architectures sont entraînées avec trois initialisations chacune, sans
sélection du meilleur modèle : une récurrence à porte, sa version réinitialisée
à chaque étape, et un MLP recevant huit étapes d'historique. Chaque entraînement
effectue 300 mises à jour Adam avec des batches de 32 séquences. Les neuf jeux de
poids et les pertes d'entraînement sont conservés dans
[les artefacts](../artifacts/learned-source-monitor/report.json).

Le modèle récurrent possède 465 paramètres. La version réinitialisée en possède
aussi 465 enregistrés mais seulement 209 participent au calcul : sa matrice
récurrente multiplie toujours un état nul. Cette comparaison ne contrôle donc pas
à elle seule la capacité effective. Le MLP sur historique possède 505 paramètres
utilisés et constitue un contrôle complémentaire plus exigeant.

## Résultats dans les conditions fixées

L'évaluation utilise 128 épisodes de 48 étapes dans chacune des cinq conditions,
sur de nouvelles graines. Les 640 scénarios sont réutilisés pour les neuf modèles,
qui choisissent leurs propres vérifications. Les 45 évaluations totalisent
276 480 décisions ; elles ne correspondent pas à autant d'observations indépendantes.
Les poids restent figés, mais l'état interne évolue avec les retours reçus.

La perte ci-dessous est la moyenne du coût de vérification et des erreurs
d'attribution non vérifiées. Une valeur plus basse est meilleure pour cette
tâche et ces coûts uniquement. Chaque cellule moyenne les trois initialisations.

| Condition | Récurrent | Réinitialisé | Historique de huit étapes | Toujours vérifier |
|---|---:|---:|---:|---:|
| Distribution initiale | 0,158054 | 0,175510 | 0,156359 | 0,250000 |
| Contribution simulée plus forte | 0,179823 | 0,197957 | 0,182183 | 0,250000 |
| Trace interne supprimée | 0,175618 | 0,189223 | 0,179104 | 0,250000 |
| Canal plus persistant | 0,149645 | 0,173977 | 0,149658 | 0,250000 |
| Source devenue imprévisible | 0,409804 | 0,398546 | 0,411377 | 0,250000 |

Dans la distribution initiale, le modèle récurrent vérifie 35,53 % des étapes
et commet 6,92 % d'erreurs de source en mémoire. Toujours vérifier élimine ces
erreurs, au prix d'un coût plus élevé. L'avantage observé est donc un compromis
selon la fonction de coût fixée, pas une supériorité sur tous les objectifs.

Le modèle à huit étapes fait légèrement mieux en moyenne dans la condition
initiale et obtient une perte presque identique lorsque le canal est plus
persistant. Ce pilote n'établit pas que la récurrence apprise est nécessaire.
Trois initialisations sur un seul jeu d'entraînement ne suffisent pas non plus
à établir une équivalence générale entre les architectures.

## Échec de l'estimation de ses limites

Dans la condition d'ambiguïté totale, la source courante est un tirage équilibré
indépendant des indices et des sources précédentes. Une prévision correcte de
l'incertitude serait donc q=0,5 avant vérification. Avec les coûts fixés, toujours
vérifier est alors optimal en espérance.

Le modèle récurrent conserve pourtant des probabilités trop tranchées. Son Brier
atteint 0,354301, contre 0,25 pour la probabilité constante 0,5. Il ne vérifie que
37,64 % des étapes, commet 31,57 % d'erreurs de source en mémoire et atteint une
perte de 0,409804. Les deux autres architectures échouent également. Ces modèles
n'ont pas appris une capacité générale à reconnaître que leurs propres indices
ont perdu leur valeur prédictive.

Une simulation plus forte produit un autre échec mesurable : les fausses
attributions externes du modèle récurrent passent de 3,53 % à 7,24 % des étapes.
Ce résultat concerne le simulateur construit ; il n'est pas une reproduction des
illusions perceptives humaines. La [réanalyse humaine](HUMAN_REALITY_BRIDGE_RESULTS.md)
n'a fourni ni cibles d'entraînement ni preuve d'une correspondance entre les
variables artificielles et les représentations cérébrales.

## Ce que montrent les greffes causales

À chaque étape naturelle, une greffe isolée remplace q par la sortie d'un autre
épisode au même contenu symbolique. La greffe n'est pas réinjectée dans les étapes
suivantes. Dans la condition initiale, elle change 46,41 % des décisions de
vérification du modèle récurrent, 34,32 % des attributions finales en mémoire et
augmente la perte de 0,246039. Le suivi de source est donc utilisé par ces fonctions.

Le contrôle à rapport seul conserve les décisions d'origine. Son absence d'effet
sur l'action et la mémoire est imposée par sa définition ; elle n'est pas un
résultat d'apprentissage spontané. De même, la connexion de q au contrôleur est
câblée. Les performances sans greffe apportent l'information sur ce que le
prédicteur a appris ; la greffe examine les conséquences de son utilisation.

Le contrôle d'ambiguïté est particulièrement instructif : la greffe change encore
47,04 % des vérifications alors que q n'est plus informatif sur la source. Son
effet moyen sur la perte est presque nul, -0,000597. **Influencer causalement les
décisions ne suffit donc pas à avoir un modèle interne correct ou utile.** Cela
limite l'interprétation d'un simple test d'ablation comme indicateur de soi.

L'appariement sur le contenu ne garantit pas que le donneur soit plausible au
regard de tous les indices du destinataire. Ces greffes ne permettent pas de
revendiquer une identification causale complète d'une représentation de soi,
et encore moins de l'expérience subjective.

## Exécution dans Menia et journal vérifié

Une exécution de 48 étapes avec la première initialisation prévue, la graine 11,
et un nouveau scénario de graine 61000 a produit 162 événements de journal.
L'agent a choisi 18 vérifications et conservé deux attributions de source erronées
non vérifiées. Cette trajectoire illustre le fonctionnement ; elle n'a pas été
sélectionnée parmi plusieurs essais pour sa réussite.

Les [événements](../artifacts/learned-source-monitor/demo/events.json),
la [trace](../artifacts/learned-source-monitor/demo/trace.json) et
l'[état final](../artifacts/learned-source-monitor/demo/agent-state.json) sont
conservés. Chaque prédiction précède sa décision, chaque vérification choisie
arrive ensuite et chaque attribution mémorisée référence sa décision et son
éventuelle preuve. Une vérification défaillante arrête l'agent ; elle ne devient
pas une observation réussie.

```powershell
py -3.12 -m menia.run_source_agent --checkpoint artifacts/learned-source-monitor/recurrent-11.json --out runs/source-essai
```

Le répertoire de sortie doit être nouveau. Le calcul d'inférence nécessite NumPy,
sans PyTorch ni modèle linguistique chargé. `--condition ambiguous` permet de
reproduire une situation difficile. Le journal SQLite de l'exécution locale reste
dans le répertoire choisi ; les artefacts suivis ici sont les exports JSON.

## Portée exacte de l'intégration

Le prototype réside dans `menia/source_agent.py` et réutilise la mémoire du projet.
Il possède une commande exécutable et recharge les poids appris. Il n'est pas
encore intégré à la scène de navigation principale, au chat Qwen ou à l'application
iPhone. Les poids de Qwen n'ont pas été modifiés.

La trace du canal est fournie par l'interface conçue pour l'expérience. Le modèle
ne découvre pas lui-même une frontière générale entre soi et autrui, et aucun
avantage sur un observateur doté des mêmes informations n'est établi. Une
estimation de provenance utile n'est pas encore une conscience de sa propre
existence.

La mémoire conserve des attributions et leurs preuves, mais le prototype n'utilise
pas encore une autobiographie pour guider ses décisions. Son état séquentiel
utilise les indices et les retours de vérification ; le journal n'est pas relu
comme une mémoire de soi à long terme. Une restauration automatique de la session
du prototype n'est pas implémentée, bien que l'état final soit exporté.

La prochaine question fonctionnelle est de savoir si Menia peut apprendre à
détecter la perte de fiabilité de son propre estimateur à partir des retours
effectivement obtenus, sans connaître le nom de la condition. Les erreurs peu
vérifiées posent un problème de sélection des observations. Ajouter une règle
qui lit l'étiquette « ambiguous » pour forcer q=0,5 ne répondrait pas à cette question.

Cette capacité supplémentaire ne résoudrait pas à elle seule le lien phénoménal.
La conscience de sa propre existence et une contribution inédite dans la
littérature restent non établies. Les mécanismes généraux d'estimation de source,
de récurrence et de décision sous incertitude ont des antécédents ; ce pilote
ne revendique pas leur invention.

## Reproduction et validation

Le [protocole](LEARNED_SOURCE_PROTOCOL.md) a été consigné avant l'entraînement.
La mise en œuvre était au commit `ab07abc` lors de son lancement. Les empreintes
des sources, les versions, les neuf checkpoints et les 300 pertes par entraînement
sont conservés dans le rapport JSON. L'entraînement a utilisé PyTorch 2.4.1 CPU
et NumPy 2.4.4 ; les évaluations utilisent seulement NumPy.

```powershell
py -3.12 -m research.train_source_monitor --out runs/source-entrainement
py -3.12 scripts/check_source_monitor.py
```

Le vérificateur recharge tous les poids et recalcule les 45 évaluations, les
références de tâche, le masquage des cibles d'entraînement et la trajectoire du
journal, sans réentraîner. Sept nouveaux tests vérifient notamment l'absence de
retour courant dans les entrées, l'ordre décision/vérification, la distinction
inférence/observation, l'arrêt sur échec, le rechargement et l'accord des calculs
PyTorch avec l'inférence NumPy. Ces contrôles valident l'expérience informatique,
pas la conscience.
