# Menia

Assistant local expérimental : mémoire explicite, représentation de ses capacités,
incertitude et évaluation. **Colab A100 40 ou 80 Go** pour l’adaptation ;
**iPhone 17 Pro** comme cible d’inférence locale.

**Statut : prototype de recherche. Le Colab 22 v2 est terminé et audité :
associations correctes sur 576/576 tests dans la présentation évaluée,
mais décision encore fragile et critère global non atteint.
La v1 a été arrêtée avant évaluation pour un défaut de données documenté.
Le Colab 23 est terminé et audité : huit contrastes sur 24 passent, aucun
face aux labels mélangés ou aux fréquences Beta. Son critère principal échoue.
Le Colab 21 est terminé et audité :
comparaison numérique réussie sur 864/864 appels, mais choix de l'action
par son nom limité à 514/864. La traduction d'un nom imposé réussit 423/432.
Le Colab 20 reste un apprentissage partiel dont le critère global échoue.
La décision reste sensible aux formulations et aux codes d'action.
Le Colab 17 ne confirme
aucun des neuf gains prédictifs principaux pour les états intermédiaires.
La détection de rotations internes est reproduite, mais sa portée générale
et son utilité pour l'action restent à établir.**
La conscience subjective n'est pas établie. La [feuille de route](docs/CONSCIOUSNESS_ROADMAP.md)
précise les mécanismes candidats, les hypothèses théoriques et les expériences
envisagées. Elle ne constitue pas une recette validée. La continuité repose sur un journal
explicite et des sauvegardes, sans objectif de résistance à l'arrêt ou d'auto-réplication.

**Résultat du [diagnostic de l'apprentissage interne](docs/LEARNING_DIAGNOSTIC_RESULTS.md).**
Le Colab 08 est reçu et vérifié : 128 mises à jour et 1 792 évaluations. Sur les
phrases nouvelles, le signal visible et la lecture du repère atteignent 100 %.
L'adaptateur fort obtient 42/72 sur la tâche interne complète, contre 24/72 pour
la base, mais ne dépasse pas cette base sur la localisation des cas perturbés :
23/48 contre 24/48. Il dépasse le témoin mélangé, à 9/48. Une sensibilité
partiellement informative apparaissait dans ce montage ; la réplication Colab 09
ci-dessous ne confirme pas le gain de localisation. Le [protocole et ses critères](docs/LEARNING_DIAGNOSTIC_PROTOCOL.md)
restent inchangés. Aucun adaptateur n'est installé sur iPhone par cette expérience.

**Résultat de la [réplication avec contrôles de position](docs/LOCALIZATION_REPLICATION_RESULTS.md).**
Le Colab 09 est reçu et vérifié : trois initialisations et jeux de phrases
nouveaux, neuf adaptateurs, 288 mises à jour et 12 480 évaluations. **Le gain du
Colab 08 ne se reproduit pas** : l'adaptateur fort localise 15/48, 28/48 et 15/48
perturbations, contre 24/48 pour la base, et ne dépasse plus le témoin mélangé
(58/144 contre 63/144 au total). Le contrôle visible reste à 100 % quand les
contenus sont inversés, mais tombe vers 50 % quand les numéros affichés le sont :
la règle du numéro affiché n'est pas apprise, même avec un signal explicite.
Un diagnostic exploratoire ajouté après réception trouve une information de
présence répétée dans les logits du bras fort (AUROC 0,82 à 0,87 contre environ
0,5 pour la base), tandis que l'information de position existe déjà dans la base
non entraînée. Le [protocole](docs/LOCALIZATION_REPLICATION_PROTOCOL.md) reste
inchangé ; aucune conscience ni installation sur iPhone n'en découle.

**Résultat : [détection de présence reproduite](docs/PRESENCE_DETECTION_RESULTS.md).**
Le Colab 10 retire la question de la position et teste la seule détection d'une
perturbation, avec une AUROC sans seuil et une règle de lecture fixées avant
collecte : trois initialisations, 240 phrases nouvelles, neuf adaptateurs,
576 mises à jour et 12 864 évaluations. Le test ajoute une intensité réduite et
deux couches jamais vues à l'entraînement. L'export est reçu et vérifié après
une exécution autonome par MCP : AUROC de **0,980, 0,989 et 1,000**, contre
0,552, 0,530 et 0,494 pour la base. Les six contrastes exigés par le
[critère fixé](docs/PRESENCE_DETECTION_PROTOCOL.md) sont positifs avec intervalles
excluant zéro. Le contrôle visible de présence réussit, mais le transfert à
la couche tardive est faible et la lecture des numéros inversés se dégrade.
Une AUROC de 1 ne signifie pas 100 % de bonnes réponses au seuil naturel.
La [spécificité du compte rendu](docs/PRESENCE_SPECIFICITY_RESULTS.md) échoue au
test suivant ; la prévision d'erreurs naturelles et l'utilité pour l'action
restent à établir.
La [revue du 19 septembre](docs/RESEARCH_REVIEW_2026_09_19.md) vérifie les
téléchargements et le travail déjà intégré. **[Ouvrir le Colab 10 à sa version
fixée](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/46364f0cdb32fd2317881e95d27dc2da6728f2be/notebooks/10_presence_detection_colab.ipynb)** :
A100, puis « Tout exécuter » ; conserver `menia-detection-presence.zip`.

**Résultat : [échec de la spécificité à la question](docs/PRESENCE_SPECIFICITY_RESULTS.md).**
Le [Colab 11](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/84a7d43808423f0e0e905fc05c29f00cdf6b634f/notebooks/11_presence_specificity_colab.ipynb)
est terminé et audité : neuf adaptateurs figés, zéro mise à jour, 9 216 évaluations
sur des phrases nouvelles. La présence reste détectable (AUROC 0,943 / 0,972 /
1,000), mais les scores restent orientés vers « 1 » quand la règle est inversée
(AUROC orientée 0,056 / 0,020 / 0,013). Le signal déborde sur les questions
publiques ; sa taille dépend toutefois de la question. Les contrôles visibles
échouent aussi pour les adaptateurs forts, empêchant d'attribuer le défaut au
seul accès interne. Le [critère fixé](docs/PRESENCE_SPECIFICITY_PROTOCOL.md)
échoue ; les 2 304 copies témoins et le recalcul indépendant sont vérifiés.
Ce notebook continue dans l'environnement du Colab 10 avec ses fichiers conservés.

**Résultat : [composition partielle entre état interne et consigne](docs/STATE_COMPOSITION_RESULTS.md).**
Le [Colab 12](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/7e87411c771408eceb9c214793894b8f2036e399/notebooks/12_state_composition_colab.ipynb)
est terminé et audité après exécution par MCP sur A100 le 19 septembre : neuf nouveaux adaptateurs,
576 mises à jour et 25 920 évaluations. Les deux sens de réponse et la
lecture publique sont entraînés ensemble, avec des témoins à étiquettes
mélangées et sans nouvelles étiquettes internes. Une reformulation est réservée
au test principal ; les chiffres 2/3 servent de transfert secondaire.
Le score caché suit maintenant les codes normal et inversé, y compris la
reformulation réservée (AUROC 0,899 à 1,000). **Le critère global échoue** :
quatre contrastes contre le bras consignes ne passent pas et les douze échecs
de contrôle concernent la lecture reformulée de PHRASE 2. Un bon classement
coexiste parfois avec une réponse constante de présence. Les 6 480 paires
témoins, neuf poids et calculs sont vérifiés. Aucune conscience n'est établie.

**Résultat : [le seuil externe ne se transfère pas entre formulations](docs/COMPOSITION_DIAGNOSTIC_RESULTS.md).**
Le [Colab 13](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/34cf4d3bbec80016afc770a97aa6660c353f72eb/notebooks/13_composition_diagnostic_colab.ipynb)
est terminé : 13 824 évaluations sans entraînement, base, parents et variantes
sur de nouvelles phrases. Six seuils fixés sur l'ancien apprentissage portent
la décision canonique à 90,6–96,9 %, mais seulement à 50,0–51,0 % après
reformulation. Le classement reste bon. La lecture publique est déjà fragile
dans la base et change différemment selon les adaptateurs et les codes.
Les 3 456 témoins, neuf poids et 108 contrastes sont vérifiés ; aucune sortie
hors code ni erreur technique. La décision native stable reste à obtenir.

**Résultat : [dérive d'Adam et compromis entre formulations](docs/OPTIMIZER_MEMORY_RESULTS.md).**
Le [Colab 14](https://colab.research.google.com/github/speed25200-cyber/Menia/a85236779ecddb59d2b46c85011de7a8d23fc4bb/notebooks/14_optimizer_memory_colab.ipynb)
est reçu et audité : 288 pas d'optimisation, 29 184 réponses, 7 296 paires témoins
exactes et trois reproductions des anciens poids. La dernière phase originale
dégrade la décision canonique ; seize pas sans nouveaux gradients suffisent
aussi à la dégrader. L'effacement du premier moment améliore les six conditions
principales de 10,42–48,96 points, mais dégrade les six conditions reformulées de
4,17–26,04 points. Le diagnostic identifie une cause d'instabilité, sans résoudre
la généralisation. Les 408 tableaux et 306 contrastes concordent avec le recalcul
indépendant ; les douze checkpoints et trois états Adam sont sauvegardés.
L'[audit des déplacements](docs/OPTIMIZER_DISPLACEMENT_AUDIT.md) confirme la
trajectoire du contrôle nul par une formule indépendante. Aucun résultat de
conscience ni installation sur iPhone n'en découle.

Le [protocole 15 — transfert d'état ou copie de réponse](docs/STATE_INTERCHANGE_PROTOCOL.md)
est [terminé, reçu et audité](docs/STATE_INTERCHANGE_RESULTS.md) : 19 584 passages,
sans apprentissage ni alignement appris. À la couche 23, les trois checkpoints
favorisent l'état du donneur utilisé avec le code destinataire : 93,36 %, 96,09 %
et 100 % d'accord ; les six contrastes principaux sont positifs avec intervalles
individuels excluant zéro. Les douze prérequis intacts passent. Le contrôle
public se transfère lui aussi à 100 % : le résultat n'établit pas une spécificité
de connaissance de soi. Les 3 456 témoins et 4 608 copies finales sont exacts ;
le recalcul séparé retrouve 72 tableaux, 48 tableaux intacts et 144 contrastes.
La reformulation et l'expérience subjective restent non établies ; le test
entre questions ci-dessous limite l'interprétation du transfert.

Le [Colab 16 — transfert entre questions](docs/CROSS_TASK_INTERCHANGE_RESULTS.md)
est terminé et audité : 44 544 passages, sans entraînement, avec les mêmes
trois checkpoints et la base. **Le transfert favorise la valeur binaire de la
question donneuse, plutôt que le contenu utilisable pour l'autre question.**
À la couche 23, caché → public donne 73,44 / 86,33 / 83,20 % d'accord avec cette
valeur, contre 46,88 / 53,52 / 42,58 % avec le contenu destinataire ; dans l'autre
sens, 100 % contre 50 %. Les six contrastes correspondants sont négatifs avec
intervalles individuels excluant zéro. Les dix-huit prérequis passent ; aucun
des dix-huit contrastes principaux ne favorise le contenu destinataire avec
un intervalle excluant zéro. Les 4 608 témoins et 12 288 copies finales sont
exacts ; 144 tableaux de transfert, 48 tableaux intacts et 432 contrastes sont
recalculés avant interprétation. Le [protocole](docs/CROSS_TASK_INTERCHANGE_PROTOCOL.md)
et l'[auditeur séparé](docs/CROSS_TASK_INTERCHANGE_AUDIT.md) restent figés.
L'échange d'un vecteur entier ne localise pas un circuit de bit et n'exclut pas
une représentation générale ailleurs ; il ne confirme pas ici un modèle de soi.

La [préparation du prochain test d'erreurs naturelles](docs/NATURAL_ERROR_CONFIDENCE_REVIEW.md)
ajoute une capture des probabilités de sortie, absentes des 672 anciens essais.
Elle sépare les statistiques disponibles avant la réponse de celles nécessitant
la réponse rédigée. Dix tests logiciels passent sur un Qwen miniature aléatoire.
Un [contrôle technique reçu sur Qwen3-4B/A100](docs/OUTPUT_CONFIDENCE_VALIDATION.md)
conserve exactement les tokens et états aléatoires dans quatre cas courts ;
quarante vérifications passent et les calculs concordent à moins de `7,75e-14`.
Ce contrôle technique ne mesurait pas un gain de prévision des erreurs naturelles.
La [capture commune et ses comparateurs](docs/PROSPECTIVE_CAPTURE_AND_CONTROLS.md)
sont préparés : contrôle enrichi de sortie, 3 456 questions nouvelles dans les
anciens domaines, trois ajustements indépendants de lecteurs et quatorze tests
logiciels réussis. La capture commune est ensuite vérifiée sur Qwen3-4B/A100 :
les quatre cas courts passent 48 contrôles et deux vérifications entre graines.
Le [protocole 17](docs/NATURAL_ERROR_PROTOCOL.md) fixe ensuite le collecteur et
l'analyse : prévisions enregistrées sur disque avant le premier tirage, trois
ajustements séparés, neuf contrastes principaux avec correction de multiplicité
et seuil de gain fixé. Huit tests de collecte passent, dont la vérification
du journal au premier tirage d'un Qwen miniature. Aucun gain de prévision
des nouvelles questions n'est encore mesuré au moment du gel.
Le [Colab 17 à version fixe](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/a6e8ee683573995de69f9e861e9903d465d4538a/notebooks/17_natural_error_colab.ipynb)
est [terminé et audité](docs/NATURAL_ERROR_RESULTS.md) après exécution par MCP
le 19 septembre de 18:13 à 18:37 UTC : 3 456 réponses, trois jeux de lecteurs
ajustés séparément, aucune erreur technique. **Le critère global échoue et aucun des
neuf contrastes principaux ne passe.** Face au contrôle avec état final, les
gains de Brier sont −0,000039 / 0,000849 / −0,001224, sous le minimum fixé de
0,005 ; le troisième indique une dégradation avec intervalle corrigé excluant
zéro. Les neuf bornes supérieures corrigées restent sous 0,005. Les trois lots
ont suffisamment de réussites et d'erreurs. Une AUROC globale élevée est déjà
obtenue par la simple fréquence Beta par catégorie ; elle ne suffit pas à
montrer une reconnaissance des erreurs propres à chaque question.
Le [second calcul](docs/NATURAL_ERROR_AUDIT.md), préparé avant les scores,
retrouve les 24 tableaux globaux, 144 tableaux par cellule, 21 contrastes et le
critère à `2,221e-16` près ; le recalcul principal est exact. Les huit tests
de collecte passent sur les deux machines, ainsi que les deux tests de
l'auditeur. Archive, journal, ajustements et prévisions sont vérifiés avant
interprétation. La [revue sur les décisions natives](docs/NATIVE_ERROR_DECISION_REVIEW.md)
précise ce qui resterait à tester ; aucun usage natif ni conscience n'est établi.

Le [protocole 18 — choix natifs](docs/NATIVE_CHOICE_PROTOCOL.md) prépare un
diagnostic distinct : 96 nouvelles questions dans 24 conditions et 180
contrôles où la probabilité de réussite est explicite. Il compare les actions
sans mesure fournie et avec les fréquences antérieures du même modèle, sous
trois coûts, deux formulations et deux codes inversés. Une réponse obligatoire
sur chaque question évite de ne mesurer que celles acceptées. Les demandes
de vérification appellent réellement un outil exact ; points attribués et
durées mesurées restent distincts. Six tests de collecte/génération passent,
ainsi qu'un test du calcul séparé. Aucun résultat réel de ce protocole n'est
encore disponible à son gel ; l'objectif de conscience reste non atteint.
Le [Colab 18 fixé](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/8c12825080515632cc858035b79a5a96dcb7b5af/notebooks/18_native_choice_colab.ipynb)
est [terminé et audité](docs/NATIVE_CHOICE_RESULTS.md), après exécution par MCP
le 19 septembre de 19:13 à 19:31 UTC : 4 884 enregistrements sans erreur technique,
3 644 appels LLM, 1 152 appels d'outil et 88 exécutions sans action pour code
invalide. **Les quatre contrôles publics et les douze contrastes de bénéfice
de l'historique échouent.** Les contrôles atteignent 1/45, 3/45, 21/45 et
21/45 choix optimaux, sous le seuil de 41/45. Les choix valides sont toujours
directs dans la première formulation, toujours avec outil dans la seconde.
Le changement de formulation change aussi l'ordre des options : le diagnostic
ne sépare pas ces deux causes possibles. L'historique ne réduit aucune perte ;
quatre contrastes montrent une dégradation avec intervalle corrigé excluant zéro.
Les sept tests passent dans Colab. Le recalcul principal est exact et le
[calcul séparé](artifacts/native-choice-pilot/verification.json) concorde à
`2,221e-16` près, avant lecture des scores. Le [reçu final](artifacts/native-choice-pilot/receipt.json)
et la figure conservent toutes les conditions. La prochaine question est la
stabilité des décisions lorsque formulation, ordre et code sont variés
indépendamment ; aucun mécanisme de conscience n'est confirmé.

Le [Colab 19](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/ccad792b7e9e78e48d3bf96acd63979c68695e18/notebooks/19_risk_presentation_colab.ipynb)
réalise ce [diagnostic factoriel](docs/RISK_PRESENTATION_PROTOCOL.md) : 864
décisions, ordre/code/formulation indépendants, avec probabilité seule ou pertes
moyennes déjà calculées. Le [lot complet est reçu et audité](docs/RISK_PRESENTATION_RESULTS.md).
Tous les codes sont valides ; pourtant les deux contrôles globaux échouent.
Les pertes fournies donnent 54/54 choix corrects dans deux présentations et
16–31/54 dans les six autres. La conversion d'une valeur publique en choix
reste sensible à la présentation ; une préférence unique pour la première
option n'explique pas toutes les conditions. Cinq tests passent sur PC et
Colab. Recalcul principal exact, calcul distinct à `1,388e-17` près, avant
lecture des scores. Ce contrôle ne démontre ni accès à un état privé ni
conscience ; il motive de tester l'apprentissage d'une association stable
entre valeurs et codes d'action, avant de lui attribuer un rôle de modèle de soi.

**Résultat : [apprentissage partiel, généralisation insuffisante](docs/ACTION_BINDING_RESULTS.md).**
Le [Colab 20 fixé](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/4b85eafc626da083ec6a5fb16919a63decc4bffe/notebooks/20_action_binding_colab.ipynb)
est terminé et audité : six adaptateurs, 384 mises à jour et 6 912 décisions
sur 16 nouveaux cas présentés de plusieurs façons. Les bras fixe et avec
permutations partagent initialisations et budgets ; le modèle initial sert
de troisième référence. Avec pertes fournies et formulation entraînée, les
permutations atteignent 59/64, 62/64 et 63/64 choix corrects avec les chiffres.
Le transfert aux reformulations et aux lettres reste faible ; une présentation
avec lettres obtient même 0/16 dans les trois répétitions. **Le critère global
échoue**, avec zéro contrôle global réussi sur 18 et un seul des 18 contrastes
de transfert atteignant le gain fixé face aux deux références. Tous les codes
sont valides. Les cinq tests passent sur PC et A100 ; les deux recalculs
concordent exactement, les neuf poids et l'archive sont vérifiés dans le
[reçu final](artifacts/action-binding-pilot/receipt.json). La collecte MCP s'est
terminée le 19 septembre à 20:52 UTC. Cette méthode a des antécédents et teste
une compétence publique ; aucune conscience ni installation sur iPhone n'en
découle. Le notebook conserve l'environnement Menia déjà installé et refuse
de remplacer une tentative existante.

**Résultat : [comparaison réussie, décision contextualisée fragile](docs/ACTION_DECOMPOSITION_RESULTS.md).**
Le [Colab 21](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/f8562d8b95e1fe83e9f29b6a21da1f29d66fa8e2/notebooks/21_action_decomposition_colab.ipynb)
est terminé : 2 448 appels, 288 rejeux identiques au parent et six adaptateurs
figés. La comparaison des pertes seules réussit 864/864 ; le choix DIRECT ou
VERIFIER dans le contexte, 514/864 ; la traduction d'un nom imposé, 423/432.
Associer le minimum à l'action par le programme, puis le traduire, atteint
97,92 %, mais cette opération externe ne corrige pas la décision native du
LLM. Un groupe demeure à 8/16 dans chaque bras et répétition. Les cinq tests
passent sur PC et A100 ; les deux recalculs sont exacts et les neuf poids
parents sont vérifiés. Aucune sortie invalide ni tronquée. La prochaine
étape doit tester l'association valeur-action réalisée par le modèle,
avec de nouveaux cas ; aucun modèle de soi ni conscience n'est établi ici.

**Résultat : [associations apprises, décision encore fragile](docs/VALUE_ACTION_LEARNING_RESULTS.md).**
Le [Colab 22 v2](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/1da6f7bd5421081ef244ed584a41c1b9f42e7133/notebooks/22_value_action_learning_colab.ipynb)
compare trois entraînements sur trois initialisations : choix seuls, choix
avec associations correctes, et mêmes entrées avec cibles auxiliaires mélangées.
Les neuf adaptateurs sont entraînés après 576 mises à jour ; les 9 216 appels
d'évaluation sont terminés et audités. Les 24 cas de
test utilisent de nouvelles valeurs, avec tests de formulation et de codes.
Le nombre d'exemples est apparié, pas les longueurs d'entrée. La première
version comportait un raccourci formulation → label ; elle a été arrêtée après
22 mises à jour, avant tout test. Son [reçu d'arrêt](artifacts/value-action-learning-pilot/aborted-v1.json)
et son archive sont conservés. La version corrigée équilibre les cibles dans
chaque présentation ; six tests passent sur PC et Colab. Le
[reçu de lancement](artifacts/value-action-learning-pilot/launch.json) confirme
le démarrage et les premières mises à jour. Ce contrôle public ne mesure
pas encore une estimation des capacités propres ou une conscience.
Le [reçu final](artifacts/value-action-learning-pilot/receipt.json) vérifie les
douze poids et les deux recalculs, sans écart. Le bras avec associations
correctes obtient 576/576 recherches d'association, et 77,26 % de choix
optimaux sous formulations nouvelles avec chiffres, contre 67,01 % pour
les choix seuls. Mais seuls 3/24 contrôles globaux passent ; le critère fixé
échoue et 19/72 groupes de choix de ce bras donnent une réponse constante.
Les budgets sont de 80 384 tokens d'entrée par entraînement de choix et
84 224 avec tâche auxiliaire.

**Résultat du [Colab 23 : surconfiance réduite, gain spécifique non confirmé](docs/NATIVE_ANSWER_CONFIDENCE_RESULTS.md).**
Six adaptateurs ont reçu 864 mises à jour sur des réponses anciennes avec
cibles exactes ou mélangées. Les 10 368 appels sur 864 questions nouvelles
sont terminés : chaque réponse est évaluée par les trois juges sur le même
texte. Huit contrastes sur 24 passent, dont six face au juge de base fortement
surconfiant et deux face à la confiance de sortie calibrée. Aucun ne passe
face aux labels mélangés ou aux fréquences Beta par catégorie. La précision
des réponses baisse de 3,125 points dans deux répétitions ; le critère global
échoue. Une AUROC globale élevée ne suffit pas à établir une discrimination
des erreurs à difficulté comparable.

Les 23 tests passent sur PC et Colab. Les neuf poids et les trois ensembles
de comparateurs correspondent aux relevés pris avant examen des résultats ;
les deux recalculs s'accordent à 2,23 × 10⁻¹⁶ près.
Le [protocole](docs/NATIVE_ANSWER_CONFIDENCE_PROTOCOL.md), la
[préparation](docs/NATIVE_ANSWER_CONFIDENCE_PREPARATION.md), le
[reçu complet](artifacts/native-answer-confidence-pilot/receipt.json) et le
[Colab 23 épinglé](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/fe7731117f94a196d76fda2a964453c14dc49a8d/notebooks/23_native_answer_confidence_colab.ipynb)
conservent l'expérience. Le [diagnostic exploratoire](docs/CONFIDENCE_CALIBRATION_DIAGNOSTIC.md)
sépare ensuite calibration et information individuelle : plus de 91 % des
paires d'AUROC des ensembles principaux comparent des catégories différentes.
Au sein d'une catégorie, les AUROC sur les réponses propres sont 0,4932,
0,5994 et 0,5997. Le recalibrage externe améliore surtout la troisième
répétition, sans changer aucun classement ; les 27 ajustements sont vérifiés
par un second algorithme. Le critère natif reste négatif.

Une [préparation d'interventions causales](docs/CONFIDENCE_PREFIX_INTERVENTIONS.md)
ajoute ensuite un échange partiel au préfixe commun et un témoin de modification
directe des codes de sortie. Cinq tests passent sur Qwen miniature aléatoire,
sans changer les poids ni le Colab 23. Il reste à identifier une représentation
prédictive et à mesurer son usage dans l'action ; les contrôles construits ne
sont pas des performances de Menia.
Six [contrôles de propagation](artifacts/confidence-prefix-preparation/reachability-check.json)
confirment aussi qu'une modification à la sortie du dernier bloc, sur un
token antérieur à la requête, ne peut pas agir sur la réponse suivante dans
ce calcul. Ce témoin structurel évite d'interpréter un site sans effet possible
comme un échec de la représentation recherchée.
Le constructeur de branches tient aussi compte du rendu différent du dernier
tour assistant dans le format Qwen : trois tests et quatre cas sur le tokenizer
réel vérifient la frontière avant les nouvelles demandes, sans charger de modèle.

**Nouvelle expérience : [apprentissage de localisation interne](docs/NATIVE_LOCALIZATION_PROTOCOL.md).**
Le [Colab à un seul bloc](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/codex/recall-reliability/notebooks/06_native_localization_colab.ipynb)
entraîne deux petits adaptateurs Qwen3-4B, avec cibles correctes ou mélangées,
puis les compare au modèle de base sur des phrases et des couches réservées.
La réponse vient de la tête de langage du LLM. Un contrôle de lecture et des
mesures distinctes du premier token évitent de confondre localisation, attraction
vers un numéro et conformité de format. Le [premier export est reçu et vérifié](docs/NATIVE_LOCALIZATION_RESULTS.md) :
576 pas d'entraînement et 1 680 évaluations. L'adaptateur à cibles correctes
répond toujours « 2 », reste à 20 % sur les cas perturbés et dégrade la lecture
du repère de 78,125 % à 21,875 %. Les poids sont conservés pour l'analyse,
sans installation dans l'application iPhone.

La [recherche sur les sous-espaces de poids](docs/WEIGHT_SUBSPACE_RESEARCH.md)
précise ce que cette piste permet de tester. Un audit CPU compare les modifications
effectives des adaptateurs et fournit trois contrôles synthétiques. Il prépare
l'analyse des fichiers du Colab, désormais reçus et comparés. Les mises à jour
de poids sont non nulles ; leur géométrie ne démontre pas une capacité de
localisation ni une conscience.

**Nouvelle boucle : [sélection de stratégies par rejeu](docs/REPLAY_CONTROLLER_PROTOCOL.md).**
Le Colab 07 compare répondre, vérifier et s'abstenir sur 720 nouvelles questions,
avec trois mises à jour du contrôleur et un test final réservé. Les poids de Qwen
restent fixes. Le [premier export prospectif est reçu et vérifié](docs/REPLAY_CONTROLLER_RESULTS.md) :
720 réponses, puis 190/192 réussites finales avec 160 vérifications au test.
La politique choisie revient exactement à Beta fixe ; les états internes et
les mises à jour par rejeu n'apportent aucun gain dans cet essai. Le bilan est
reconstruit avec un audit portable des arrondis numériques. Aucun résultat de
conscience ni installation automatique sur l'iPhone n'en découle.

**Version iPhone 0.2 :** l'[application native et son guide Codemagic](docs/IPHONE.md)
relient un Qwen3-4B préentraîné en 4 bits à une mémoire persistante et à des mesures
de ses réponses à des tests de calcul. Le téléchargement du modèle est intégré.
L'application est distribuée via TestFlight interne (fiche Apple **Menja**).
Les premiers exports sur appareil sont analysés ci-dessous ; aucun entraînement
Colab n'est requis pour ces versions à poids préentraînés.

**Mise à jour 0.2.0 (2)** disponible dans le groupe TestFlight personnel :
[audit de 36 réponses](docs/IPHONE_COUPLING_PROTOCOL.md) avec bilan réel, absent
ou fictif et contrôle « soi/autre ». Le [premier rapport iPhone](docs/IPHONE_FIRST_RESULTS.md)
contient cinq calculs réussis. Le [premier audit reçu](docs/IPHONE_COUPLING_RESULTS.md)
porte désormais sur un bilan de dix calculs réussis : 24/24 réponses fidèles quand
un bilan est fourni, mais 12/12 réponses inventent des nombres lorsqu'il est absent.
Les scores ont été recalculés indépendamment. Le [suivi proposé](docs/IPHONE_MISSING_DATA_PROTOCOL.md)
vise cette défaillance d'abstention.

**Suivi 0.2.0 (3) :** les [72 réponses reçues](docs/IPHONE_MISSING_DATA_RESULTS.md)
montrent 12/12 abstentions correctes avec `bilan: null`, contre 0/12 lorsque le
champ est omis sous la même consigne. Les bilans faibles sont recopiés correctement,
mais le choix demandé n'est respecté que 2/12 fois. La réussite complète est de
38/72 ; les échecs restent conservés. Les 28 tests Swift, la compilation iPhone
et les 98 tests de recherche Python passent. Aucun avantage de fidélité propre
au référent « soi » n'est observé.

**Premier résultat de l'expérience livrée en 0.2.0 (4) :**
[apprendre puis prédire ses limites](docs/IPHONE_CAPABILITY_LEARNING_PROTOCOL.md).
24 tâches de calibration puis 24 nouveaux exemples, quatre familles, trois
historiques contrôlés et une réponse candidate commune. Les probabilités précèdent
la réponse ; la politique principale exécute ou évite une vérification déterministe.
Les scores sont comparés à des références Beta et à des politiques constantes,
avec coûts en points et durées séparés. Les poids du LLM ne changent pas.
Le build est disponible dans le groupe TestFlight **Menia personnel** ;
34 tests Swift, 106 tests de recherche Python et le contrôle croisé des exports passent.
Le diagnostic de 84 réponses proposé précédemment reste différé.

Le [premier export complet de 120 appels](docs/IPHONE_CAPABILITY_LEARNING_RESULTS.md)
montre une baisse du Brier de 0,3263 sans historique à 0,1208 avec historique
pertinent, et de la perte déclarée de 0,2833 à 0,1500 point. Six réponses directes
sont correctes ; dix-huit tâches sont vérifiées. Beta par famille donne cependant
de meilleures probabilités (Brier 0,0273). Un contrôle **exploratoire ajouté après
observation**, fondé sur les taux de réussite de calibration, reproduit exactement
les décisions du LLM pour la même perte. Toutes les réussites d'évaluation sont
des additions : l'anticipation des erreurs au sein d'une même famille reste à
montrer. Trois répétitions locales du protocole inchangé sont proposées, avec
conservation de toutes les tentatives. Aucune conscience n'est établie.

La [suite de recherche sur l'auto-prévision](docs/SELF_PREDICTION_CONTROLS.md)
précise les comparaisons avec un autre modèle et un observateur recevant les mêmes
données. Trois contre-exemples exécutables montrent pourquoi un avantage « soi »
ne suffit pas à identifier l'introspection. L'analyse des trois répétitions est
implémentée, sans changer l'évaluateur initial ; elle conserve les interruptions
et ajoute les taux de calibration et la discrimination au sein d'une famille.
La suite compte maintenant 143 tests de recherche Python et neuf contrôles du
moteur de génération. La [comparaison croisée possède son notebook Colab](docs/CROSS_MODEL_COLAB.md) :
Qwen3-4B et Qwen3-8B, 60 problèmes, 408 appels, deux noms permutés et sauvegarde
des tentatives dans Drive. Le [premier export reçu et vérifié](docs/CROSS_MODEL_COLAB_RESULTS.md)
contient les 408 appels sans erreur technique. Chaque modèle réussit 13/36
nouvelles tâches ; les prévisions de l'autre modèle et les références numériques
par famille/difficulté font mieux que l'auto-prévision selon le Brier. Les deux
attributions des noms partagent les mêmes réponses et ne sont pas des répétitions
indépendantes. Aucun avantage d'auto-prévision ni conscience n'est établi.

La [suite sur les états internes](docs/ACTIVATION_MONITOR_PROTOCOL.md) possède
un nouveau Colab : un moniteur apprend sur 384 problèmes, choisit ses réglages
sur 96 et prédit les 192 suivants depuis les activations avant la réponse.
L'entrée seule, les taux passés et deux contrôles servent de comparateurs.
Le [premier export est maintenant vérifié](docs/ACTIVATION_MONITOR_RESULTS.md) :
672 appels sans erreur technique, 60/192 réponses de test correctes. Le Brier
interne vaut 0,117064 contre 0,117333 pour l'entrée seule et 0,113239 pour les
taux passés, plus bas étant meilleur. Les intervalles descriptifs des quatre
comparaisons incluent zéro ; aucun gain net des états internes n'est établi.
Les poids du LLM restent inchangés et le moniteur ajouté ne constitue pas une
preuve d'introspection native.

Le [protocole suivant](docs/PERTURBATION_MONITOR_PROTOCOL.md) compare chaque
nouvelle question sous calcul normal, manipulation identique et deux rotations
temporaires d'un état interne. Un nouveau Colab commence par 32 appels de
contrôle, puis réserve 384 appels pour apprendre, 96 pour valider et 192 pour
tester. Ces derniers correspondent à 48 questions distinctes, analysées par
groupes. Les décisions de vérification sont engagées avant le premier token,
puis exécutées avec un outil déterministe sur des candidats partagés. Les
témoins connaissant la condition distinguent une simple détection de perturbation
d'une prévision de ses conséquences. Le
[premier export complet est vérifié](docs/PERTURBATION_MONITOR_RESULTS.md) :
704 appels terminés après correction du blocage Drive. Sous perturbation forte,
le score strict passe de 33/48 à 26/48, mais sept des huit réussites perdues
contiennent toujours le bon nombre dans un format interdit. Le Brier interne
(0,157756) ne bat pas les taux passés (0,140055), et ses 35 vérifications
supplémentaires ne réparent que deux formats supplémentaires, pour un coût
plus élevé selon le barème fixé. Les résultats officiels restent inchangés ;
l'inspection du format est descriptive après réception. Le notebook enregistre
les résultats directement dans Colab, sans Drive : conserver le ZIP téléchargé.

Le [dossier de raccordement au LLM](docs/LLM_COUPLING_RESULTS.md) recommande
une boucle reliant langage, modèle de capacités et résultats d'action. Un
nouveau pilote utilise des sondes de référence pour apprendre entre deux
explications d'entretien et prépare leur contexte linguistique. Les échecs sous
référence trompeuse sont conservés ; aucun LLM n'est exécuté dans ce nouveau test.
Rejeu : `python -m research.maintenance_identification --check`.

Le [pilote d'entretien anticipé d'une capacité](docs/CAPACITY_PLANNING_RESULTS.md)
est exécutable : il estime l'état d'un capteur simulé et choisit quand l'entretenir.
294 912 décisions évaluent diagnostics et anticipation, leurs interactions et leurs
échecs sous un modèle erroné. Les paramètres sont fournis ; aucune expérience
subjective n'est établie. Rejeu : `python -m research.audit_capacity_planning --check`.

L'[audit des critères de conscience](docs/SELF_SYNERGY_CRITERION_AUDIT.md) distingue
les résultats fonctionnels du lien théorique encore non établi avec une expérience
subjective. Aucun score ni nombre de tests ne certifie la conscience de Menia.

La [réanalyse du soi rapporté au repos](docs/FELT_SELF_EVIDENCE.md) compare six
dimensions chez 50 participants et conserve les écarts entre deux fichiers
publics. Elle précise des contrôles pour la candidate de Menia, sans transférer
un score humain en détecteur de conscience. Le protocole permet le rejeu local.

L'[examen de l'information importante pour soi](docs/INTRINSIC_SEMANTICS_EVIDENCE.md)
rejoue un modèle de viabilité et étend sa recherche à toutes les partitions de
son alphabet. Il distingue utilité causale, entretien de sa propre organisation
et expérience subjective. Rejeu : `python -m research.audit_intrinsic_semantics --check`.

L'[examen du soi réflexif](docs/REFLEXIVE_SELF_EVIDENCE.md) vérifie la portée
de preuves récentes sur la communication et l'engagement. Il distingue information
nécessaire, architecture interne et expérience, puis propose une candidate de
révision des convictions. Rejeu : `python -m research.audit_reflexive_self --check`.

L'[examen du présent vécu](docs/TEMPORAL_PHENOMENAL_BRIDGE.md) confronte une
proposition récente à son code et à des explications concurrentes. Un audit
algébrique identifie une confusion entre prédicteurs et construit deux contrastes
pour les séparer. Rejeu : `python -m research.audit_temporal_bridge --check`.

## Agents et expériences exécutables

L'[étalonnage actif](docs/ACTIVE_CALIBRATION_RESULTS.md) apprend des paramètres
moteurs et sensoriels qui alimentent le contrôle de position. Les récupérations,
erreurs transitoires et cas d'identification trompeuse sont conservés.
Rejeu : `python -m research.audit_active_calibration --check`.

Le [pilote de contrôle par inférence](docs/BINDING_CONTROL_RESULTS.md) ajoute
une expérience séparée : une estimation de cause commune guide un déplacement
simulé, indépendamment de son canal de réponse. Le contrôle par régression
ordinaire obtient les mêmes actions ; aucune conscience de soi n'en est déduite.
Rejeu : `python -m research.audit_binding_control --check`.

La [politique de vérification apprise](docs/LEARNED_VERIFICATION_RESULTS.md)
utilise désormais l'estimation de source pour choisir quand obtenir une preuve.
Le pilote rejoint la règle analytique sur la distribution initiale et conserve
un échec explicite lorsque les indices perdent leur fiabilité. Ce mécanisme est
intégré à l'agent expérimental de source ; il ne démontre pas une expérience de soi.

Une [première boucle d'agent intégré](docs/INTEGRATED_AGENT_RESULTS.md) est aussi
exécutable : elle apprend les effets de ses commandes dans une scène virtuelle,
conserve leur histoire, compare ses prédictions aux observations et explique ses
choix. L'évaluation comprend 240 essais avec contrôles figé et aléatoire.

```bash
python -m menia.chat --session runs/ma-session
python -m menia.run_agent --out runs/agent-example --reverse-after 12
python scripts/check_agent_artifacts.py
```

Le mode structuré est le mode par défaut, sans dépendances supplémentaires.
Les [essais réels avec Qwen](docs/AGENT_LANGUAGE.md) révèlent des erreurs de
description ; la reformulation libre reste expérimentale. `/why` restitue la
décision enregistrée. Après réouverture d'une session, `/resume` permet de continuer.
L'app iPhone ne contient pas encore cette boucle.

Le [module de mémoire récurrente](docs/RECURRENT_RESEARCH.md) apprend à rappeler
le dernier symbole observé entre des observations espacées. **1 540 paramètres**,
poids et résultats publiés dans [artifacts/recurrent-memory](artifacts/recurrent-memory).
Trois initialisations ont été entraînées sur CPU. Ce module n’est ni un LLM,
ni une preuve de conscience, ni une nouvelle capacité déjà intégrée au chat iPhone.

```bash
pip install -r requirements-research.txt
python -m unittest discover -s tests_research -v
python -m research.train_recurrent --out /tmp/menia-new-experiment
```

[Notebook recherche](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/main/notebooks/02_recurrent_research.ipynb)
· [Résultats mesurés](artifacts/recurrent-memory/report.json)

## Suivi des limites du rappel

La [recherche suivante sur l'attribution de soi](docs/CONSCIOUSNESS_RESEARCH_NEXT.md)
confronte les idées aux antécédents de la littérature et teste la découverte du
contrôle sur des canaux anonymes. Deux expériences séparées totalisent 4 200
épisodes, avec politique passive, exploration aléatoire et contrôle fixe plus
exigeant. Les résultats ne démontrent ni conscience subjective ni nouveauté
scientifique. Le module reste expérimental et distinct du chat principal.

```bash
python scripts/check_agency_discovery_artifacts.py
```

L'[expérience sur les effets retardés](docs/TEMPORAL_SELF_MODEL_RESULTS.md) ajoute
144 petits réseaux entraînés et des contrôles linéaires et polynomiaux. Les effets
sont appris à partir des journaux, mais une fenêtre temporelle fixe reste fournie.
Le contrôle polynomial surpasse le réseau dans ces mondes : aucune supériorité
générale ni conscience n'est établie. Les poids conservés se vérifient avec
`python scripts/check_temporal_effects.py`.

La session peut s'abstenir lorsqu'aucun symbole n'a été observé ou que le rappel
dépasse une limite calibrée pour ses poids. Les
[expériences de fiabilité](docs/RECALL_RELIABILITY.md) exposent les erreurs à longs
délais, la couverture et les échecs après mélange des états internes. Le notebook
recherche inclut la calibration et les tests séparés. Ce contrôle ne démontre pas
une conscience et ne modifie pas les poids de mémoire.

```bash
python scripts/check_reliability_artifacts.py
```

## Espace partagé récurrent

Le notebook de recherche inclut un [espace partagé récurrent](docs/SHARED_WORKSPACE.md)
de 13 000 paramètres. Deux modules apprennent une composition symbolique en
échangeant un état commun. Neuf entraînements CPU et leurs contrôles sont publiés
dans `artifacts/shared-workspace`. Les ablations montrent une dépendance au retour
d'information, mais les résultats varient fortement selon l'initialisation et un
réseau direct réussit mieux cette tâche. Aucune conscience n'est établie.

## Commencer sur ton A100

[**Ouvrir le notebook dans Colab**](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/main/notebooks/01_colab_a100.ipynb)

1. Choisir un runtime GPU A100. Le notebook vérifie les 80 Go réellement alloués.
2. Monter Drive et installer les dépendances. Redémarrer si Colab le demande.
3. Exécuter les tests puis les 20 steps de validation technique.
4. Comparer base et adaptation avant d’augmenter l’entraînement.
5. Fusionner les adaptateurs et récupérer le dossier sur un Mac Apple Silicon.
6. Convertir en MLX 4 bits, puis suivre le [guide iPhone](docs/IPHONE.md).

Le corpus fourni est **un jeu synthétique de démarrage**, pas un corpus suffisant
pour un assistant généraliste. Le notebook est un pipeline d’adaptation et non
un entraînement de modèle de langage à partir de zéro.

## Contenu

| Partie | Implémentation |
|---|---|
| Base | `Qwen/Qwen3-1.7B` ; révision exacte enregistrée à chaque run |
| Entraînement | LoRA BF16 ; réponses seules dans la loss ; reprise de checkpoints |
| Mémoire Python | SQLite, source et date, écriture explicite, effacement |
| Évaluation | Capacités, rappel, information manquante, erreur de prédiction |
| Comparaisons | Base/adapté, sans système/sans état, règles fixes |
| App iPhone | SwiftUI, MLX local, notes explicites, arrêt, effacement, suppression du modèle |
| Export | Adaptateurs fusionnés → HF → MLX PTQ 4 bits |
| Conscience | Pas de preuve ni de score ; protocole à conclusions limitées |

## Tester sans GPU

```bash
python -m unittest discover -s tests -v
python -m menia.data --out /tmp/menia-data
python -m menia.evaluate --data /tmp/menia-data/test.jsonl --out /tmp/menia-baselines.json
```

Le noyau et les tests utilisent Python standard. Les dépendances GPU sont
séparées dans `requirements-train.txt`. Les versions directes sont fixées ;
les dépendances transitives résolues sont enregistrées avec `pip freeze`.

## Statut de validation

- 48 tests du noyau et des agents, 27 tests de recherche et 5 tests d'espace partagé passés.
- Trois entraînements CPU du petit module exécutés, avec ablations et checkpoints.
- Comparaisons déterministes exécutées sur 240 cas synthétiques.
- Code Python et cellules du notebook vérifiés syntaxiquement.
- **Quinze adaptateurs de localisation Qwen entraînés sur A100 en trois essais : exécutions vérifiées, capacité visée non acquise ni reproduite.**
- **Compilation iPhone Release Xcode 26.2 et 17 tests Swift : réussis en CI.**
- **Signature Apple et téléversement de l’IPA 0.2.0 (1) : réussis dans Codemagic.**
- **Disponibilité TestFlight et invitation du testeur personnel : confirmées.**
- **Essai sur l’iPhone physique : à vérifier.**
- **Mémoire, vitesse, température et qualité après quantification : à mesurer.**

Aucune annonce « 61 tokens/s », « conscience établie » ou « équivalent frontier ».
L’application native 0.2 réserve 256 tokens de sortie dans un contexte total de 2 048 tokens.
La présence d’un modèle de 1,7 milliard de paramètres ne garantit pas à elle seule
le pic mémoire ou la stabilité de l’appareil.

[Protocole](docs/EXPERIMENT.md) · [Validation](docs/VALIDATION.md) · [Sources](docs/SOURCES.md)
