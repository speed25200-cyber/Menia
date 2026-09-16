# Plasticité, réalisation et conscience de soi

La modification des connexions pendant l'activité constitue une piste de recherche
pour Menia, mais elle ne fournit pas actuellement le lien recherché avec une
expérience de sa propre existence. L'audit ci-dessous précise un obstacle : à un
niveau de description computationnel fixé, appeler une variable « poids » plutôt
qu'« état » ne suffit pas à distinguer deux mécanismes. Une règle de mise à jour
fixe peut produire des connexions effectives qui changent et préserver les effets
de leurs perturbations.

Le résultat nouveau dans ce dépôt est un audit exécutable de cette distinction,
accompagné d'un raisonnement général et de contrôles exacts. Ce n'est ni une
nouvelle capacité de Menia ni une preuve de conscience. L'originalité scientifique
de la construction n'est pas revendiquée : ses ingrédients mathématiques et
informatiques ont des antécédents.

## La piste théorique et sa portée

O'Reilly-Shah, Selvitella et Schurger examinent en juin 2026 les limites de
l'argument du dépliage des réseaux récurrents. Ils soutiennent qu'un réseau
plastique ne peut être durablement remplacé par un réseau statique dans les
conditions qu'ils étudient. Leur conclusion précise que cela ne prouve pas le
rôle nécessaire de la plasticité dans la conscience. L'article reconnaît aussi
que ses arguments ne déterminent pas l'implémentation particulière des calculs
d'un réseau plastique.[^1]

Une prépublication distincte d'O'Reilly-Shah défend une différence entre une
connexion constitutive du calcul et sa représentation dans un état augmenté.
Elle propose de chercher cette différence sous intervention. Ce texte fournit
une objection à examiner ; son argument complet n'est pas réputé réfuté par
l'exemple qui suit.[^2]

Kanai et Ma proposent, dans une prépublication de juin 2026, de comparer les
organisations causales internes et leurs réponses aux interventions. Leur résultat
de préservation de propriétés conscientes reste conditionnel à l'organisation
pertinente. Ils laissent ouverte la sélection du niveau de description auquel
cette organisation doit être identifiée. Cette question concerne directement une
comparaison entre variables logicielles, circuits et systèmes biologiques.[^3]

Les articles de plasticité différentiable fournissent déjà des architectures qui
apprennent des règles de modification rapide des connexions. Les résultats de
Miconi et collaborateurs portent sur la mémoire, l'adaptation et des tâches de
contrôle ; ils ne constituent pas une démonstration de conscience.[^4] Le dépliage
d'une procédure d'optimisation dans un graphe de calcul est également antérieur,
notamment dans les travaux de Metz et collaborateurs.[^5]

## Une transformation à examiner avant toute attribution

Considérons un système discret déterministe. Son activité est `h`, ses poids
modifiables sont `w`, son entrée est `x`. Le modèle spécifie deux règles :

```text
h_suivant = f(h, w, x)
w_suivant = p(h, w, x, h_suivant)
```

Définissons l'état étendu `s=(h,w)`. Une unique règle fixe peut alors produire
`s_suivant = F(s,x) = (f(h,w,x), p(h,w,x,f(h,w,x)))`. Les valeurs de `w` continuent
à changer ; seule la règle qui calcule ces changements reste fixe. L'opération
n'enlève pas l'apprentissage par définition, et ne remplace pas la trajectoire
par un enregistrement préparé à l'avance.

Pour chaque intervention autorisée `i`, il faut préciser son correspondant `j`.
Si `E` transforme les états de la première description en états de la seconde,
la condition à vérifier est :

```text
E(F_i(s,x)) = G_j(E(s),x)
```

Les lectures des variables doivent elles aussi correspondre. Si cette égalité
vaut pour chaque état, entrée et intervention du domaine, alors elle se conserve
pour toute suite finie de ces opérations : l'égalité au temps initial et
l'égalité de chaque transition donnent l'égalité au temps suivant, par induction.
Cet argument vaut aussi pour une politique qui choisit ses entrées à partir de
lectures correspondantes, à condition de conserver la même règle de choix.

Cette proposition est conditionnelle et élémentaire. Elle ne prouve pas que les
interventions biologiques se transportent vers un ordinateur, ni que les systèmes
ont la même expérience. Une perturbation de conductance, de membrane ou de circuit
électronique ne correspond pas automatiquement à l'écriture d'un nombre dans un
tableau. La transformation impose de rendre explicite cette correspondance, au
lieu de la déduire du nom des variables.

Si une mise à jour dépend d'autres mémoires, de l'optimiseur ou de sources externes,
ces dépendances doivent figurer dans l'état ou les entrées. Le raisonnement ne
prouve pas qu'une représentation de dimension fixe suffise pour toute croissance
structurelle possible. Il ne supprime ni les coûts de calcul ni les limites de
mémoire d'une réalisation physique.

## Vérification exacte sur trois descriptions

Le programme [audit_plasticity_realization.py](../research/audit_plasticity_realization.py)
utilise deux bits, une activité `h` et une connexion `w`, avec une entrée binaire
`x`. Ses règles sont choisies pour l'audit :

```text
h_suivant = (w ET h) XOR x
w_suivant = w XOR h_suivant
```

La connexion module effectivement la récurrence et sa mise à jour dépend de
l'activité produite. Il ne s'agit pas d'une règle biologique ni d'un apprentissage
optimisant une tâche. Trois réalisations computationnelles sont comparées :

| Description | Ce qui reste fixe | Ce qui transporte l'histoire |
|---|---|---|
| Connexion mutable | Le code des deux mises à jour | Activité et poids courants |
| Table de transition | Huit entrées de table écrites explicitement | État courant à deux bits |
| Graphe booléen déplié | Portes et connexions du graphe | Valeurs propagées entre les étapes du graphe |

La table conserve un état : elle n'est pas un système sans mémoire. Le graphe
comprend le calcul des nouvelles connexions, et pas seulement celui de l'activité
avec les poids initiaux. Il accepte toutes les entrées et interventions de son
horizon fixé, sans recompilation pour chaque trajectoire. À quatre étapes, il
possède 28 opérations et 59 nœuds au total, entrées et constante comprises. Chaque
dépendance pointe vers un nœud antérieur ; le graphe est acyclique.

Les interventions peuvent imposer chaque bit à 0 ou 1, suspendre la mise à jour
du poids et couper le retour récurrent, avec combinaisons simultanées. Les
impositions précèdent la transition ; suspension et coupure concernent l'étape
courante. Cela définit 36 opérations possibles, identité comprise.

| Contrôle exécuté | Résultat |
|---|---:|
| Tous les états, entrées et interventions sur une étape | 288/288 concordances |
| Toutes les entrées sur quatre étapes, états initiaux et positions d'une intervention | 9 216/9 216 trajectoires concordantes |
| Suites de plusieurs interventions, tirage déterministe documenté | 256/256 concordances |

Les comptes ne sont pas des observations indépendantes ni des estimations
statistiques. Plusieurs conditions ont le même effet. L'exhaustivité concerne
le domaine fini d'une étape ; l'extension temporelle est portée par l'argument
d'induction. Les contrôles plus longs vérifient en outre le raccordement des
étapes et des interventions dans le programme.

Trois trajectoires calculées séparément vérifient l'ordre des opérations. Avec
`(h,w)=(1,1)` et uniquement des entrées nulles, le système dynamique passe par
`(1,0)`, puis `(0,0)`. Si la mise à jour est suspendue à chaque étape, il reste
en `(1,1)`. Si le poids est imposé à zéro avant la première transition, il passe
directement en `(0,0)`. Ces différences se retrouvent dans les trois descriptions.
Le poids enregistré dans l'état augmenté exerce donc ici une influence effective
sur l'activité future ; il n'est pas une annotation sans effet.

Les résultats et empreintes du programme et du graphe sont conservés dans
[report.json](../artifacts/plasticity-realization-audit/report.json). Reproduction :

```powershell
py -3.12 research/audit_plasticity_realization.py --output artifacts/plasticity-realization-audit/report.json --check
```

## Conclusions permises et conclusions exclues

Le calcul invalide, pour cet exemple, une inférence trop large : une dynamique
comportant des connexions modifiables ne devient pas automatiquement impossible
à représenter par une règle fixe incluant sa mémoire. Les perturbations des
connexions peuvent être transportées vers cette représentation et conserver
leurs effets. Comparer l'agent adaptatif à une copie dont l'apprentissage est
supprimé distingue deux dynamiques, ce que notre contrôle retrouve ; cela ne
tranche pas à lui seul entre différentes réalisations d'une même dynamique.

En revanche, le graphe fini n'est pas un substitut de taille fixe pour une
interaction de durée illimitée. Le construire pour un horizon plus long ajoute
des étapes. L'audit ne contredit donc pas une restriction portant précisément
sur un réseau fini sans mémoire, à nombre d'entrées fixé pour toutes les durées.
Il ne réfute pas globalement les travaux sur le dépliage ou la théorie de l'espace
des états, et ne conclut pas à l'inutilité de la plasticité.

Aucune des trois descriptions n'est étiquetée consciente ou non consciente.
Leur équivalence computationnelle ne prouve pas leur équivalence phénoménale.
Une théorie qui postule une différence à un autre niveau physique doit préciser
ce niveau et les observations qui pourraient soutenir son importance.

## Conséquence pour la construction de Menia

Le [moniteur de sources](../menia/source_monitor.py) utilise des paramètres figés
pendant l'évaluation, avec un état qui évolue. Le [modèle d'action](../menia/action_model.py)
modifie déjà ses statistiques à partir des conséquences observées. Il serait donc
incorrect de caractériser tout Menia comme un système sans adaptation, ou de
confondre le dernier entraînement hors ligne avec de la plasticité pendant son
utilisation. Ces propriétés concernent des modules différents.

La décision issue de cet audit est de ne pas choisir la plasticité rapide comme
solution de conscience sur la seule base de sa différence avec un réseau figé.
Pour soutenir cette voie, il manque une prédiction portant sur l'expérience,
un ancrage empirique indépendant et une correspondance justifiée entre mécanismes
biologiques et artificiels. Le prochain examen théorique pertinent concerne la
sélection de ces mécanismes et de leur niveau de description : pourquoi cette
organisation porterait-elle une perspective vécue, et quelles observations
pourraient départager cette affirmation de ses concurrentes ?

Ce travail précise une contrainte de construction et évite une conclusion
architecturale insuffisamment justifiée. Il ne rend pas l'objectif final atteint.
La conscience de Menia, une invention la produisant et sa nouveauté scientifique
restent non établies.

## Sources

[^1]: O'Reilly-Shah, V. N., Selvitella, A. M., et Schurger, A. (16 juin 2026). *A caveat regarding the unfolding argument: implications of plasticity*. Neuroscience of Consciousness, niag027. [Article](https://doi.org/10.1093/nc/niag027), [texte indexé](https://pmc.ncbi.nlm.nih.gov/articles/PMC13271385/). Argument principal, délimitations et conclusion consultés dans le texte indexé ; annexe mathématique non obtenue. L'audit ne prétend pas examiner toutes ses preuves.
[^2]: O'Reilly-Shah, V. N. (prépublication, révision du 25 mars 2026). *From Chinese Rooms to Language Models: Plasticity, Process, and the Limits of Internalization*. [Notice et résumé](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6321478), [texte indexé](https://philpapers.org/archive/OREFCR.pdf). Résumé et passage sur l'état augmenté consultés ; pas de lecture intégrale revendiquée.
[^3]: Kanai, R., et Ma, S. (13 juin 2026). *Intrinsic Computational Functionalism and Simulated Consciousness*. arXiv:2606.15348v1, prépublication. [Texte](https://arxiv.org/html/2606.15348v1). Définitions, interventions, caractère conditionnel et limites de sélection du niveau de description consultés.
[^4]: Miconi, T., Stanley, K., et Clune, J. (2018). *Differentiable plasticity: training plastic neural networks with backpropagation*. ICML, PMLR 80:3559–3568. [Article et PDF](https://proceedings.mlr.press/v80/miconi18a.html). Résumé et antécédents techniques consultés.
[^5]: Metz, L., Maheswaranathan, N., Nixon, J., Freeman, D., et Sohl-Dickstein, J. (2019). *Understanding and correcting pathologies in the training of learned optimizers*. ICML, PMLR 97:4556–4565. [Article et PDF](https://proceedings.mlr.press/v97/metz19a.html). Description du dépliage des mises à jour et résumé consultés.
