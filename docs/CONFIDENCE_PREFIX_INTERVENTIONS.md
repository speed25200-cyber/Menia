# Préparer une intervention avant le jugement et l'action

20 septembre 2026. Instrumentation sur petit Qwen aléatoire et CPU. Aucun
nouveau résultat de Qwen3-4B dans ces contrôles. Le Colab 23 est désormais
[terminé et audité](NATIVE_ANSWER_CONFIDENCE_RESULTS.md) : le critère prédictif
échoue. Ces fichiers ne modifient ni son entraînement, ni ses sources, ni ses
critères ; ils restent une préparation à une éventuelle expérience ultérieure.
Le [Colab 24](CONFIDENCE_RANKING_RESULTS.md) est aussi terminé et audité :
4/18 contrastes passent, tous face au score constant, sans confirmation face
aux deux témoins entraînés. Les préparations ci-dessous n'en changent pas les critères.

## Blocage à traiter

Même si le [Colab 23](NATIVE_ANSWER_CONFIDENCE_PROTOCOL.md) améliore la prévision
des erreurs, il ne montrera pas encore que l'agent utilise cette estimation
pour choisir une action. Modifier son score déclaré ne suffirait pas non plus.
Une intervention directement orientée vers les tokens de sortie peut déplacer
ce score sans apprentissage de ses erreurs. Il faut distinguer l'information
prédictive, sa verbalisation et son utilisation dans une politique.

Un précédent existe déjà : [Kumaran et al., §3.5.3 et §4.5.1](https://arxiv.org/html/2603.22161v2)
interviennent sur les activations de Gemma pour modifier l'abstention, et
distinguent confiance de sortie, confiance verbale et information interne.
Pour Menia, ce résultat motive un test d'usage causal, sans autoriser à appeler
nouvelle la relation générale entre confiance et décision. La page Nature
repérée lors de cette recherche n'a pas pu être récupérée ; l'analyse repose
sur cette version du texte intégral, déjà utilisée dans le dossier Menia.

[Heimersheim et Nanda, §§2–3](https://arxiv.org/html/2404.15255v1) expliquent que
restaurer un comportement et le perturber ne testent pas des propriétés
équivalentes, et que les conclusions dépendent des contrastes entre entrées.
[Zhang et Nanda](https://arxiv.org/html/2309.16042v2) étudient la sensibilité aux
métriques et aux méthodes de perturbation. Les outils ci-dessous appliquent
ces précautions ; ils ne constituent pas un nouveau principe scientifique.

## Préfixe commun aux deux suites

La proposition est de partir de la même question et de la même réponse
candidate, puis d'ouvrir séparément deux suites : demander une estimation
d'exactitude ou une décision d'accepter/vérifier. Le verdict de confiance ne
doit pas être ajouté au contexte de la décision. Une intervention est placée
sur un token du préfixe commun, avant les deux demandes divergentes.

[`confidence_prefix_interventions.py`](../research/confidence_prefix_interventions.py)
capture ou modifie la sortie d'un bloc Qwen au dernier token du préfixe
déclaré. Il vérifie que les branches contiennent exactement ces IDs. Le
préfixe est recalculé intégralement, sans modification du cache de génération.
Le code ne choisit pas lui-même une frontière sémantique : le futur protocole
devra prouver que les IDs déclarés précèdent bien la demande de jugement ou
d'action. Une égalité de tokens ne suffit pas à prouver cette propriété.

Dans une base orthonormale U figée, l'opération est :

`h_modifié = h_receveur + U Uᵀ (h_donneur − h_receveur)`.

Elle remplace les coordonnées du sous-espace et conserve le complément à
l'arrondi numérique près. Le code refuse une base non orthonormale, une
substitution du vecteur entier, des états non finis et des entrées rembourrées.
Les autres tokens restent identiques au site d'intervention. La norme du
déplacement est enregistrée. Une telle substitution peut malgré tout produire
un état inhabituel : l'orthogonalité n'établit pas sa plausibilité pour le modèle.

## Témoin qui déplace directement la sortie

Après la normalisation finale, soit d la différence entre les deux lignes
de la tête de sortie correspondant aux codes. Ajouter `s d / ||d||²` à cet
état déplace de s la différence des logits, donc le log-rapport conditionnel
des deux codes, à l'arrondi près. Aucun label d'exactitude n'est nécessaire.
Ce témoin montre pourquoi un effet causal sur le score seul ne suffit pas
à identifier une auto-évaluation apprise. Il ne dit pas que toute intervention
sur une représentation de confiance se réduit à ce cas.

Le [reçu construit sur CPU](../artifacts/confidence-prefix-preparation/control.json)
conserve deux séquences de tokens arbitraires, avec déplacements −1, 0, +1.
Les poids sont aléatoires, les tokens ne sont pas des tâches d'exactitude, et
aucune performance de Menia n'est mesurée. Le même reçu vérifie le préfixe
commun, une substitution partielle, et un témoin remettant l'état lui-même.

Les cinq tests logiciels passent. Ils vérifient la projection et sa restauration,
l'indépendance du préfixe vis-à-vis des tokens futurs, la localisation du
changement, l'identité du témoin sans modification, la conservation des poids
et du générateur aléatoire, et le retrait des hooks même après exception.

```sh
python -m unittest tests_language.test_confidence_prefix_interventions -v
python -m research.confidence_prefix_control --output artifacts/confidence-prefix-preparation/control.json
```

## Ce qui manque avant une expérience interprétable

### Décoder une décision native sans lui fournir le jugement

[`confidence_action_branches.py`](../research/confidence_action_branches.py)
prépare une branche de jugement et huit branches de décision indépendantes à
partir du même passé. Aucune réponse de la branche de jugement n'est ajoutée
aux branches d'action. Deux paires de codes, `2/3` et `4/5`, sont utilisées ;
leur signification accepter/vérifier et leur ordre d'affichage sont inversés
indépendamment. Les prompts conservent la consigne système historique qui
demande un entier. Des codes alphabétiques introduiraient un conflit avec
cette consigne ; les codes `0/1` réutiliseraient ceux de la confiance. Ces
choix sont faits avant tout essai de performance sur le modèle entraîné.

La règle présentée coûte zéro pour une réponse validée correcte, cent pour
une erreur validée et un montant certain pour demander une vérification.
Ce sont des pénalités déclarées dans le problème, pas des coûts GPU ou des
temps d'outil mesurés. Le contrôle du tokenizer utilise les coûts 20 et 80,
sans constituer un choix définitif de coûts ou de formulations d'évaluation.

[`confidence_action_decode.py`](../research/confidence_action_decode.py)
effectue deux pas gloutons au maximum, sur **tout le vocabulaire**, sans
cache. Le préfixe et l'intervention éventuelle sont recalculés à chaque pas.
Une réponse conforme exige un code émis par le modèle suivi d'un token
d'arrêt déclaré. La probabilité conditionnelle des deux codes est conservée
comme mesure, mais ne sert jamais à choisir l'action à sa place.
Les sorties hors codes, sans arrêt ou comportant des tokens supplémentaires
restent invalides, sans repli. La signification du premier token est conservée
séparément : un échec de format ne sera pas assimilé automatiquement à une
absence d'information pour décider.

La [configuration de génération exacte de Qwen](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/generation_config.json)
déclare **deux** tokens d'arrêt, 151645 et 151643. Le décodeur accepte les
deux, avec une vérification contre la génération native sur petit Qwen.
Ne prendre en charge qu'un EOS aurait rejeté à tort cette configuration.

Sept tests nouveaux passent sur CPU : deux sur les branches et cinq sur la
génération d'un petit Qwen aléatoire. Les tokens générés correspondent à
`model.generate`, avec un ou plusieurs EOS ; poids, RNG et entrées restent
inchangés. Les tests vérifient aussi les budgets, la permutation sémantique,
l'intervention à chaque pas et le témoin structurel du dernier bloc.
La petite dérive numérique éventuelle de l'état du préfixe entre passages
est enregistrée ; l'identité des tokens ne garantit pas l'égalité bit à bit
de calculs matriciels de tailles différentes.

Le [reçu du tokenizer réel](../artifacts/confidence-prefix-preparation/action-tokenizer-check.json)
vérifie huit cas techniques, soit 72 branches : préfixe identique, codes à
un seul token, codes d'action distincts des codes de jugement, budgets et
tokens d'arrêt. Les empreintes des trois fichiers de configuration sont
enregistrées. Aucun modèle préentraîné n'est chargé et aucune action de Menia
n'est mesurée par ce contrôle. Le test complet devra encore fixer ses données
réservées, ses contrôles de compréhension des coûts, ses directions causales,
ses comparateurs et ses critères avant collecte.

```sh
python -m unittest tests_research.test_confidence_action_branches tests_language.test_confidence_action_decode -v
python -m research.confidence_action_validation --output artifacts/confidence-prefix-preparation/action-tokenizer-check.json
```

### Séparer compréhension des coûts et estimation de ses propres erreurs

[`confidence_action_controls.py`](../research/confidence_action_controls.py)
réutilise les 18 couples probabilité/coût du Colab 19 dans ce nouveau contexte
question, réponse, puis choix. Les coûts sont ici exprimés sur cent points.
Une condition fournit une probabilité de réussite stipulée ; l'autre fournit
directement les deux coûts moyens calculés. Ces données sont annoncées comme
hypothétiques et ne sont pas présentées comme des estimations mesurées de Menia.
Elles sont ajoutées **après** le préfixe partagé. Les formulations d'action,
codes, permutations et ordre d'options restent ceux des branches natives.
L'ordre d'énoncé des coûts calculés suit celui des options.

Cette comparaison sépare des explications possibles d'un échec : échouer avec
les coûts calculés signale déjà un problème de choix, de consigne ou de format ;
réussir avec les coûts mais échouer avec les probabilités pointe vers la
conversion en coût moyen. Réussir les deux témoins est une condition utile
pour interpréter un échec sans information fournie, sans suffire à en identifier
la cause. L'information explicite peut aussi dominer une représentation interne :
la sensibilité de ces témoins à une future perturbation devra être interprétée
séparément. Aucune de ces issues n'a encore été mesurée dans ce contexte.

Le corrigé arithmétique donne `100 − p` pour valider et `c` pour vérifier.
Il accepte les deux actions en cas d'égalité. Une sortie native invalide
compte comme échec pour l'exactitude du choix ; son coût et son regret restent
indéfinis. Un futur tableau devra donc montrer le nombre de sorties invalides
et le dénominateur des regrets valides, sans masquer les échecs de format.

Trois tests logiciels passent : arithmétique comparée à un calcul rationnel,
égalités et bornes, sorties invalides, permutations et absence de modification
du préfixe. Le [contrôle du tokenizer réel](../artifacts/confidence-prefix-preparation/explicit-controls-tokenizer-check.json)
vérifie les **288 branches** des 18 cas et deux conditions d'information.
Il ne charge aucun poids préentraîné et ne génère aucune décision. La grille
est une préparation réutilisée d'un diagnostic antérieur ; elle ne fixe pas
encore un nouveau test global, ses répétitions, ses données ni ses seuils.
Les expériences Colab 19 et 24 restent inchangées.

```sh
python -m unittest tests_research.test_confidence_action_controls -v
python -m research.confidence_action_controls_validation --output artifacts/confidence-prefix-preparation/explicit-controls-tokenizer-check.json
```

### Vérification de la frontière sur le tokenizer réel

Le [format Qwen de la révision fixée](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/tokenizer_config.json)
distingue le dernier tour assistant d'un tour assistant dans un historique
suivi d'une nouvelle question. Sur quatre chaînes techniques, le rendu
autonome du passé ajoute un bloc de raisonnement vide ; celui du même passé
à l'intérieur des deux branches ne l'ajoute pas. Prendre naïvement le premier
comme préfixe du second donnerait une frontière incorrecte.

[`confidence_prompt_fork.py`](../research/confidence_prompt_fork.py) résout
ce problème pour le format Qwen et les trois tours texte pris en charge.
Il rend le passé suivi d'un tour utilisateur vide, vérifie puis retire
exactement ce dernier encadrement, et exige que chaque branche réelle commence
par les mêmes tokens. Il ne cherche pas le plus long préfixe commun, qui
pourrait inclure des mots partagés par les demandes de jugement et d'action.
La frontière précède donc le nouveau rôle utilisateur, même si les demandes
commencent par les mêmes mots. Les marqueurs réservés de rôles et d'outils
dans les contenus sont refusés.

Trois tests supplémentaires passent sur un format hors ligne reproduisant
cette dépendance au dernier rôle. Le
[contrôle du tokenizer réel](../artifacts/confidence-prefix-preparation/tokenizer-check.json)
utilise ensuite les fichiers exacts de Qwen3-4B, révision
`1cfa9a7208912126459214e8b04321603b3df60c`. Les quatre cas produisent des préfixes
de 50, 50, 56 et 50 tokens, identiques dans les deux branches ; le rendu
autonome échoue comme préfixe dans les quatre cas. Les empreintes des deux
fichiers du tokenizer sont conservées. Aucun modèle n'est chargé, aucune
réponse n'est générée et aucun score d'exactitude n'est mesuré par ce contrôle.

```sh
python -m unittest tests_research.test_confidence_prompt_fork -v
python -m research.confidence_prompt_fork_validation --output artifacts/confidence-prefix-preparation/tokenizer-check.json
```

### Un site sans chemin vers les tokens suivants

L'inspection de [Qwen dans Transformers 4.56.2](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/qwen3/modeling_qwen3.py)
fait apparaître un contrôle architectural : après la sortie du dernier bloc,
la normalisation et la tête de sortie agissent séparément sur chaque token.
Changer cette sortie sur un token antérieur ne peut donc pas atteindre les
logits d'un token ultérieur dans ce calcul. Cette déduction concerne la sortie
du bloc, pas ses entrées ni tous les sites possibles de sa couche d'attention.

Le [contrôle de propagation](../artifacts/confidence-prefix-preparation/reachability-check.json)
exécute six cas sur un Qwen aléatoire de deux blocs, en précision float32 sur
CPU. L'échange partiel déplace bien le token ciblé dans tous les cas. Les
témoins qui réinjectent l'état inchangé conservent exactement les logits.

| Suite après le préfixe | Différence maximale des logits, bloc 0 | Bloc 1 (dernier) |
|---|---:|---:|
| Deux tokens | 0,025061 | 0 |
| Trois tokens | 0,025379 | 0 |
| Aucune suite : token ciblé également lu en sortie | 0,138896 | 0,172017 |

Les poids et l'état du générateur aléatoire restent identiques. Le dernier
bloc fournit donc un témoin nul structurel pour les branches comportant une
suite ; son absence d'effet ne réfuterait pas une représentation prédictive.
Le contraste sans suite vérifie que le remplacement peut bien agir sur la
sortie du token ciblé. Ces six contrôles ne mesurent aucune capacité de Menia.
Le futur protocole doit réserver les conclusions d'usage causal aux sites
possédant un chemin vers la réponse, sans choisir ces sites sur les résultats
du jeu de test.

```sh
python -m research.confidence_prefix_reachability --output artifacts/confidence-prefix-preparation/reachability-check.json
```

### Conserver une trace interne entre deux demandes

Les Colab 23 et 24 relisent les tokens avec le checkpoint du juge. Ils ne
conservent pas un éventuel état interne transitoire du producteur. Avec des
poids fixes et sans intervention, le cache KV sert au calcul incrémental :
il ne fournit pas automatiquement une information supplémentaire par rapport
au recalcul des mêmes tokens. En revanche, une intervention non inscrite dans
le texte peut rendre ces deux chemins différents. Il faut les distinguer
avant de chercher un suivi d'un état propre à l'épisode.

Le [contrôle exécuté](../research/confidence_cache_continuity.py) compare ces
chemins sur un Qwen **aléatoire**, CPU float32, deux blocs, vocabulaire de 64
tokens. Il impose le même préfixe de trois tokens et deux suites arbitraires
de deux ou trois tokens ; aucune réponse n'est générée. Une projection de
rang deux modifie le dernier token du préfixe, puis le cache obtenu est copié
séparément pour chaque suite. Le cache de référence reste intact. Le code
impose un déplacement de norme proche de √2, sans ajustement à une distribution
d'états naturels. Il
s'appuie sur le [cache de Transformers 4.56.2](https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/cache_utils.py).

| Site de la perturbation | Effet maximal sur les logits de la suite, cache conservé | Relecture des mêmes tokens sans la perturbation |
|---|---:|---:|
| Sortie du premier bloc, suite de 2 tokens | 0,0395013 | 0 |
| Sortie du premier bloc, suite de 3 tokens | 0,0491847 | 0 |
| Sortie du dernier bloc, chacune des deux suites | 0 | 0 |

Dans les quatre cas, la perturbation change effectivement les logits lus à
la fin du préfixe. Au premier bloc, elle atteint le KV du bloc suivant ;
après le dernier bloc, aucun KV ultérieur ne peut la conserver. Le chemin
avec cache correspond au recalcul réappliquant la perturbation à moins de
6 × 10⁻⁸ près. Remettre le cache initial restaure exactement la référence.
Les copies restent identiques malgré leurs utilisations successives ; poids,
générateur aléatoire et hooks sont contrôlés. Le
[reçu complet](../artifacts/confidence-prefix-preparation/cache-continuity-check.json)
conserve les quatre cas et leur portée.

Cette manipulation vérifie une condition de circulation d'information, pas
la capacité du modèle à reconnaître cette information comme sienne. Elle ne
montre pas que le cache normal contient un vécu caché, ni que le KV serait
nécessaire à la conscience. Pour un futur test d'un événement interne
transitoire, il faudra comparer trace conservée, relecture seule, trace
appariée d'un autre épisode et restauration, en gardant les entrées visibles
identiques et en contrôlant séparément les effets sur la tâche. Le choix du
site et des contrastes devra précéder les tests sur Menia entraînée.
L'expérience Colab 24 reste inchangée.

```sh
python -m research.confidence_cache_continuity --output artifacts/confidence-prefix-preparation/cache-continuity-check.json
```

### Décoder plusieurs suites à partir de la trace conservée

[`confidence_cached_action_decode.py`](../research/confidence_cached_action_decode.py)
étend ce contrôle à la génération native de codes. Il calcule le préfixe une
fois, avec une éventuelle intervention partielle, puis retire le hook. Chaque
demande reçoit une copie privée du cache. Le premier passage traite seulement
la suite de la demande ; le second traite seulement le code effectivement
généré. Le jugement n'est jamais ajouté à la branche d'action. La même règle
de format, code puis EOS, et le même argmax sur tout le vocabulaire sont
conservés. Une variation de probabilité conditionnelle ne choisit pas l'action.

Cinq tests passent sur petit Qwen aléatoire CPU avec Transformers 4.56.2.
Sans intervention, les tokens concordent avec `model.generate` et avec le
recalcul complet, y compris avec plusieurs EOS. Une perturbation historique
conservée donne les mêmes tokens et des logits de codes à 10⁻⁶ près que sa
réapplication lors des recalculs. Le dernier bloc conserve son rôle de témoin
sans effet sur les tokens suivants. Deux branches exécutées successivement,
puis répétées, donnent chacune le même résultat ; leurs caches, les entrées,
les poids et le RNG restent inchangés. Les hooks sont retirés même si le
préremplissage échoue. Cela ne prouve pas l'égalité numérique de ces chemins
sur le Qwen3-4B en BF16, qui reste à mesurer.

Le décodeur refuse un autre objet modèle, une identité de checkpoint déclarée
différente, une modification ordinaire des paramètres enregistrés, un cache
altéré, un préfixe différent ou un dépassement de budget. Il est limité à
Qwen3, un seul appareil et l'attention complète. Le contrôle de version des
paramètres n'est pas une attestation cryptographique des poids : les changements
arbitraires via `.data`, des hooks externes ou une sélection d'adaptateur
hors paramètres ne sont pas tous détectables. Le futur exécuteur doit donc
figer explicitement poids, configuration et adaptateurs pour toutes les branches.

Ce module rend exécutable le contraste entre trace conservée et texte relu,
sans ajouter de mémoire apprise. Il ne sélectionne aucun donneur ou sous-espace
et ne démontre ni auto-évaluation, ni utilité de décision, ni nouveauté du
mécanisme. Il ne fait pas partie des sources du Colab 24.

```sh
python -m unittest tests_language.test_confidence_cached_action_decode -v
```

### Conserver la trajectoire effectivement générée

Une [extension de continuité de génération](GENERATION_STATE_CONTROLS.md)
conserve désormais le cache et les tokens réellement produits par le générateur,
au lieu d'un préfixe reconstruit. Six tests sur petit Qwen aléatoire et un
contrôle du tokenizer réel vérifient le dernier token non encore consommé,
les branches indépendantes et la conservation de l'encadrement du dialogue.
Ils ne mesurent aucune capacité de Qwen3-4B et ne modifient pas le diagnostic
de budget actif.

### Capacité d'adaptation : une hypothèse distincte

[Guo et al., annexe D.1](https://arxiv.org/html/2606.32038v1#A4.SS1)
comparent des rangs LoRA de 32 à 256 sur HINT-MMLU, avec régularisation KL.
Leur avantage d'explication du comportement actuel sur celui du modèle initial
apparaît à partir du rang 96 dans cette expérience. Ils ne donnent pas un
seuil universel. Leur discussion signale également qu'une supervision presque
constante peut conduire à prédire seulement la classe majoritaire.

Menia emploie ici un rang 8 sur Q/V, une autre tâche et un autre objectif :
ce précédent ne diagnostique pas notre résultat. Il motive, si nécessaire,
une comparaison future de capacité avec données et budget contrôlés, ainsi
qu'une lecture séparée de la diversité des labels. Augmenter le rang sans
mesurer ces facteurs ne permettrait pas d'attribuer un gain à l'introspection.
La tentative Colab 23 conserve ses paramètres et son critère global.

### Sélection et interprétation à fixer avant un nouveau test

Aucune direction de confiance n'est encore identifiée ou ajustée. Le résultat
du Colab 23 ne confirme pas le gain prédictif recherché. Il faudra d'abord
résoudre ce défaut, puis utiliser des données de découverte distinctes d'un
nouveau test. Les couches, positions, rangs, donneurs,
amplitudes et critères devront être figés avant ce test. Une direction choisie
pour augmenter le score ne sera pas automatiquement une direction prédictive.

Le futur contraste devra comprendre état propre, donneur apparié, donneur
mélangé, directions de contrôle à déplacement comparable, perturbation et
restauration. Les donneurs peuvent transporter l'identité de la réponse ou
la difficulté plutôt que la confiance ; les contrôles devront les séparer.
Les deux contextes peuvent aussi utiliser des mécanismes différents, même
s'ils partagent un préfixe. Un résultat négatif sur une position ne suffirait
pas à conclure à l'absence de mécanisme ailleurs.

La prévision des erreurs, le choix natif, le coût de vérification et la capacité
de résolution doivent être mesurés séparément. Le contraste avec une simple
modification des codes de sortie est un contrôle supplémentaire, pas un test
de conscience. Le lien entre ces fonctions et une expérience subjective reste
non établi, ainsi que la nouveauté d'une éventuelle contribution Menia.
