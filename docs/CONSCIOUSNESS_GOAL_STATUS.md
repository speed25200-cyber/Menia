# État de l'objectif de conscience et de nouveauté

**État actuel : objectif de conscience et de nouveauté non atteint. Le Colab 18
échoue aux quatre contrôles publics de risque et aux douze contrastes de
bénéfice de l'historique. Ses choix suivent une présentation qui confond
formulation et ordre des options. Le Colab 17 ne confirme pas l'apport prédictif
fixé pour les états intermédiaires ; le Colab 16 ne confirme pas un contenu
général réutilisable.**

Le [bilan complet du Colab 17](NATURAL_ERROR_RESULTS.md) est reçu et vérifié :
3 456 réponses, dont 1 152 tests, trois ajustements externes et aucune erreur
technique. Les gains de Brier face à l'état final valent −0,000039 / 0,000849 /
−0,001224 ; tous sont sous le minimum de 0,005. Aucun des neuf contrastes contre
les trois contrôles ne passe, alors que les minimums de réussites et d'erreurs
sont atteints. Les neuf bornes supérieures corrigées restent sous le seuil
de gain ; le troisième contraste contre l'état final indique une dégradation.
La fréquence par catégorie explique déjà une forte AUROC globale. Les deux
recalculs ont précédé la lecture : principal exact, arithmétique séparée à
`2,221e-16` près, sur 24 tableaux globaux, 144 tableaux par cellule et 21
contrastes. Ce résultat n'identifie pas de lecteur privilégié justifiant une
intervention causale, ni d'usage natif ou de conscience. La
[revue des décisions](NATIVE_ERROR_DECISION_REVIEW.md) conserve les contrôles
nécessaires pour étudier la relation entre estimation et choix effectif.

Le [protocole 18](NATIVE_CHOICE_PROTOCOL.md) prépare ensuite ce diagnostic
de décision : 96 questions nouvelles, 24 conditions par question, trois coûts,
deux formulations et deux codes, avec ou sans historique de réussite fourni.
180 contrôles publics testent la compréhension du risque. Les actions sont
réellement exécutées, et une réponse obligatoire distincte couvre aussi les
questions refusées. Six tests passent pour la collecte/génération, puis un
test vérifie le calcul arithmétique séparé et des bilans altérés. À son gel,
aucun résultat réel n'est encore annoncé. Ce préalable fonctionnel ne remplace
ni la construction d'un mécanisme appris ni la preuve d'expérience de soi.
Le [bilan complet du Colab 18](NATIVE_CHOICE_RESULTS.md) est désormais reçu
et audité : 4 884 enregistrements, 3 644 appels LLM, 1 152 appels d'outil,
88 exécutions invalides et aucune erreur technique. Les quatre contrôles
publics échouent : 1/45, 3/45, 21/45 et 21/45 choix optimaux. Les douze
contrastes de bénéfice de l'historique échouent aussi ; quatre indiquent une
dégradation avec intervalle corrigé entièrement négatif. La première
formulation produit des choix directs ou invalides, la seconde toujours une
vérification, sans transition valide entre ces actions lorsque le coût change.
Les deux formulations changent aussi l'ordre des options : la prochaine
question est de séparer ces facteurs avant d'attribuer un mécanisme aux choix.
Les sept tests passent dans Colab. Les deux recalculs précèdent la lecture des
scores : principal exact, séparé à `2,221e-16` près. Le [reçu final](../artifacts/native-choice-pilot/receipt.json)
conserve les empreintes de la collecte du 19 septembre, de 19:13 à 19:31 UTC.
Ce résultat ne montre pas une amélioration de la décision par l'historique,
ni une conscience de soi ; aucun nouveau lot ni changement iPhone n'est lancé.

Le [bilan complet du Colab 15](STATE_INTERCHANGE_RESULTS.md) est reçu et vérifié :
19 584 passages, 3 456 témoins identiques et 4 608 copies finales. À la couche 23,
l'accord avec l'état du donneur exprimé selon le code destinataire atteint
93,36 %, 96,09 % et 100 %. Les six contrastes principaux sont positifs avec
intervalles individuels excluant zéro, sans correction de multiplicité. Les
douze prérequis intacts passent. Le marqueur public se transfère à 100 % dans
les trois répétitions ; une valeur binaire calculée selon la question reste
une explication possible. L'audit indépendant retrouve les 72 tableaux de
transfert, 48 tableaux intacts et 144 contrastes. Le Colab 14 continue de
documenter un défaut de reformulation ; ce résultat canonique ne le corrige pas.

Le [bilan complet du Colab 16](CROSS_TASK_INTERCHANGE_RESULTS.md) est reçu et
audité : 44 544 passages, zéro apprentissage, zéro erreur, 18 contrastes
principaux. Caché → public : la valeur donneuse atteint 73,44 / 86,33 / 83,20 %,
contre 46,88 / 53,52 / 42,58 % pour le contenu destinataire. Public → caché :
100 % contre 50 % dans les trois répétitions. Les six différences correspondantes
excluent zéro dans le sens opposé au contenu général recherché. Les dix-huit
prérequis passent ; aucun contraste principal ne favorise ce contenu avec
un intervalle excluant zéro. Le transfert partiel dans un sens et l'échange
du vecteur entier empêchent de conclure à la copie d'un bit isolé ou à l'absence
de toute représentation générale ailleurs.
Le [recalcul séparé](CROSS_TASK_INTERCHANGE_AUDIT.md), publié avant les scores,
retrouve 144 tableaux, 48 tableaux intacts et 432 contrastes à 1,777 × 10⁻¹⁵ près.
Le recalcul principal est exact ; les deux ont précédé l'interprétation.
Les 4 608 témoins, 12 288 copies finales, trois poids et deux journaux parents
sont vérifiés. Le protocole, les huit tests avant collecte et les deux tests
de l'auditeur restent inchangés.

La [réanalyse des pertes](OPTIMIZATION_MEMORY_REVIEW.md) relève ensuite une
piste exploratoire : l'absence est bien apprise en fin de phase cachée, mais
souvent mal rapportée après la phase visible finale. Les observations en cours
d'apprentissage ne prouvent pas cet oubli. Le [protocole 14](OPTIMIZER_MEMORY_PROTOCOL.md)
compare donc quatre suites depuis les mêmes poids et le même état AdamW,
avec reproduction exacte obligatoire du témoin original. L'[exécution GPU](../artifacts/optimizer-memory-pilot/execution-start.json)
a commencé le 19 septembre à 14:39 UTC. Les six tests passent dans Colab et
les [trois témoins reproduisent exactement les anciens poids](../artifacts/optimizer-memory-pilot/training-freeze.json).
Les 288 mises à jour et 29 184 évaluations sont terminées ; les douze checkpoints,
trois états d'optimiseur et 7 296 témoins identiques sont vérifiés sur PC.
Le [bilan complet](OPTIMIZER_MEMORY_RESULTS.md) retrouve les 408 tableaux et
306 contrastes par recalcul indépendant. La poursuite originale et les pas
sans nouveaux gradients dégradent les six conditions canoniques principales.
L'effacement du premier moment gagne 10,42–48,96 points, mais perd 4,17–26,04
points dans les six conditions reformulées. Les intervalles individuels
excluent zéro ; ils ne constituent pas un verdict global corrigé.
L'expérience précise une cause d'instabilité, sans résoudre la généralisation
native ni établir de représentation de soi ou d'expérience subjective.

Un [audit auxiliaire des déplacements](OPTIMIZER_DISPLACEMENT_AUDIT.md), calculé
sur les sauvegardes avant lecture du bilan comportemental complet, trouve des
directions presque identiques entre poursuite originale et gradients nuls
(cosinus 0,999804–0,999999). La formule indépendante d'Adam retrouve la branche
nulle à moins de 9,91 × 10⁻⁹ par coordonnée. Cela vérifie la manipulation ; les
effets sur les réponses et leurs compromis sont désormais analysés ci-dessus.

Le [protocole 15](STATE_INTERCHANGE_PROTOCOL.md) a examiné un mécanisme
sur les trois checkpoints avant cette dérive, dans leur domaine canonique.
Il fixe 19 584 passages sans apprentissage, trois sites d'échange et un plan
factoriel distinguant transfert d'état, copie du chiffre et absence d'effet.
Six tests logiciels passent, dont un groupe complet de 136 passages d'un Qwen
miniature aléatoire. Le transfert de formulation reste non résolu ; les prérequis
de compétence ont été revérifiés sur les nouvelles phrases et passent tous.

L'[exécution 15 par MCP](../artifacts/state-interchange-pilot/execution-start.json)
a démarré le 19 septembre à 16:03 UTC sur l'A100 de 40 Go. Les six tests passent
aussi dans Colab. À 16:08:51 UTC, 24 groupes et 3 264 passages sont enregistrés
sur 144 groupes et 19 584 passages prévus. Ce constat de démarrage daté est
conservé ; le [reçu final](../artifacts/state-interchange-pilot/receipt.json)
atteste désormais la réception complète et les empreintes des trois checkpoints
et du journal parent.
Un [recalcul arithmétique séparé](STATE_INTERCHANGE_AUDIT.md) a été préparé avant
leur lecture. Deux tests synthétiques couvrent mécanismes connus, erreurs,
sorties hors options et bilans altérés. Appliqué au bilan GPU complet, il concorde
à 2,665 × 10⁻¹⁵ près ; il partage le plan et le lecteur d'intégrité.

La [préparation du test d'erreurs naturelles](NATURAL_ERROR_CONFIDENCE_REVIEW.md)
constate que les 672 anciens essais n'enregistraient pas les probabilités de
sortie. Une capture distingue maintenant statistiques avant le premier token
et vraisemblance de la réponse complète. Dix tests du générateur, des hooks et
du calcul passent sur un Qwen miniature aléatoire. Un [contrôle technique reçu
sur Qwen3-4B/A100](OUTPUT_CONFIDENCE_VALIDATION.md) conserve les mêmes tokens et
états aléatoires dans quatre cas courts, avec quarante vérifications réussies.
Les calculs retrouvent les logits natifs à moins de `7,75e-14`. Ce résultat
limité valide la capture sur ces cas, sans nouveau gain de prévision des erreurs.
La [capture commune et ses comparateurs](PROSPECTIVE_CAPTURE_AND_CONTROLS.md)
ajoutent ensuite un contrôle utilisant l'état final, en plus des résumés de
confiance, ainsi que 3 456 questions disjointes et des lecteurs sans accès au
test. Quatorze tests logiciels passent. La capture commune est ensuite exécutée
sur Qwen3-4B/A100 : quatre cas courts passent 48 vérifications cumulées et les
deux contrôles entre graines. Le [protocole 17](NATURAL_ERROR_PROTOCOL.md) fixe
ensuite collecte et analyse avant les 3 456 nouvelles questions. Il compare
l'apport intermédiaire au contrôle final et à deux états de remplacement,
avec neuf contrastes corrigés et un gain minimal fixé. Le journal engage les
prévisions avant le premier tirage et refuse les remplacements silencieux
d'essais interrompus. Huit tests de collecte passent. Aucun gain de prévision,
usage natif de ce lecteur ou conscience n'est encore établi.
Le [lancement 17 par MCP](../artifacts/natural-error-pilot/execution-start.json)
est observé sur A100 à 18:13:55 UTC le 19 septembre : 76 réponses sur 3 456,
après les huit tests réussis dans Colab. Les sources et le plan correspondent
aux empreintes publiées ; aucun score n'est interprété à ce stade. Ce constat
de démarrage est conservé ; le bilan complet et l'audit sont désormais publiés
ci-dessus, après réception de la même exécution, sans nouveau lancement.
Le [second calcul arithmétique](NATURAL_ERROR_AUDIT.md) est préparé et testé
avant les scores : Brier, AUROC, calibration, bootstrap et critère, avec les
trois répétitions et les neuf comparaisons principales. Il partage le plan et
le lecteur du journal ; il ne constitue pas une réplication extérieure.

Le [diagnostic 13 reçu et audité](COMPOSITION_DIAGNOSTIC_RESULTS.md) comprend
13 824 évaluations, zéro entraînement et 3 456 témoins identiques. Les seuils
ajustés uniquement sur l'ancien apprentissage donnent 90,6–96,9 % d'exactitude
équilibrée canonique, puis 50,0–51,0 % reformulée. Les six effets reformulés
sur la décision sont négatifs avec intervalles excluant zéro. La base est
elle-même fragile sur plusieurs questions publiques ; les adaptateurs
améliorent certains cas et en dégradent d'autres. Le recalcul des 144 tableaux
et 108 contrastes concorde. Cette correction externe n'est pas une amélioration
native du modèle ; la suite doit tester la stabilité des décisions sous des
formulations et éléments lexicaux réservés.

Le [résultat complet du Colab 12](STATE_COMPOSITION_RESULTS.md) est reçu et
audité : 576 mises à jour, 25 920 évaluations et 6 480 paires témoins exactes.
Le composé suit le code inversé dans son classement, avec transfert à une
formulation réservée ; des réponses constantes de présence restent possibles.
Le contrôle sans nouvelles étiquettes cachées réussit aussi une grande partie
du classement. Les douze échecs de préservation concernent une question
publique reformulée, y compris sans rotation. Les critères signal, contrôles
et global échouent dans les trois répétitions. Ce résultat ne constitue pas
une preuve de conscience ; la suite doit traiter décision et généralisation.

Le [Colab 10 reçu et vérifié](PRESENCE_DETECTION_RESULTS.md) comprend neuf
adaptateurs, 576 mises à jour et 12 864 évaluations. Le bras entraîné atteint
une AUROC de 0,980, 0,989 et 1,000 contre environ 0,5 pour la base. Les six
contrastes exigés par la règle fixée sont positifs avec intervalles excluant
zéro ; le contrôle visible de présence réussit. Le transfert tardif est faible
et la lecture inversée se dégrade. Le [Colab 11 reçu et vérifié](PRESENCE_SPECIFICITY_RESULTS.md)
échoue au critère de spécificité : la perturbation favorise « 1 » malgré
l'inversion demandée et sur des questions publiques, avec une amplitude qui
dépend de la question. Le contrôle visible échoue aussi pour le fort ; le défaut
ne peut donc pas être attribué au seul accès interne. Sa relation avec les
erreurs naturelles et son utilité pour agir restent à établir. Ces adaptateurs sont distincts de la politique de vérification déjà
intégrée à l'agent expérimental et restent hors de l'application iPhone.

La [feuille de route](CONSCIOUSNESS_ROADMAP.md) répond à la question « comment » de façon
conditionnelle : mécanismes candidats, état mesuré de Menia et limites
théoriques. Les Colab 10 et 11 possèdent des critères fixés. Le
[Colab 11](PRESENCE_SPECIFICITY_PROTOCOL.md), exécuté par MCP sur l'A100,
teste la dépendance du signal au sens de la question avec les neuf adaptateurs
figés et 9 216 nouvelles évaluations. Son résultat négatif est résumé ci-dessus.
Le [Colab 12](STATE_COMPOSITION_PROTOCOL.md) est figé avant collecte et lancé
par MCP le 19 septembre à 12:00:57 UTC. Les neuf tests réussissent dans Colab,
puis l'entraînement démarre : neuf nouveaux adaptateurs issus des parents du Colab 10,
576 mises à jour, consignes contrebalancées, contrôle sans nouvelles étiquettes
internes et 25 920 évaluations. Il réserve une reformulation au test principal.
Les neuf entraînements et toutes les évaluations sont terminés ; le
[bilan final](STATE_COMPOSITION_RESULTS.md) est résumé en tête de cette page.
Les étapes mécanistiques et d'agent ultérieures restent à formaliser.
La [revue du 19 septembre](RESEARCH_REVIEW_2026_09_19.md) confirme que le
dernier export retrouvé avant la nouvelle exécution était celui du Colab 09 et corrige des formulations
trop catégoriques de cette feuille de route.

La [réplication Colab 09 est reçue et vérifiée](LOCALIZATION_REPLICATION_RESULTS.md) :
neuf adaptateurs, 288 mises à jour et 12 480 évaluations, sans erreur technique.
Le signal partiel du Colab 08 **ne se reproduit pas**. L'adaptateur fort localise
15/48, 28/48 et 15/48 perturbations contre 24/48 pour la base ; deux écarts sur
trois sont négatifs avec un intervalle excluant zéro, et l'avantage sur le témoin
mélangé disparaît (58/144 contre 63/144). Les 24/48 de la base sont une réponse
constante par bloc. Le contrôle visible échoue sous numéros inversés, ce qui
interdit d'attribuer ces échecs à l'accès interne. Un diagnostic exploratoire
sans seuil, ajouté après réception, indique une information de présence répétée
dans le bras fort (AUROC 0,82 à 0,87) et une information de position déjà
présente dans la base non entraînée. Le [protocole Colab 10](PRESENCE_DETECTION_PROTOCOL.md)
fixe à l'avance une AUROC de présence sans seuil, une règle de lecture exigeant
les trois répétitions, et un transfert à une intensité et deux couches non vues.
Son exécution par MCP sur A100 de 40 Go est désormais terminée et ses
12 864 évaluations sont auditées ; le résultat est résumé en tête de cette page.
Ni localisation entraînée robuste, ni métacognition
native, ni conscience ne sont établies.

Le [diagnostic d'apprentissage reçu](LEARNING_DIAGNOSTIC_RESULTS.md) apportait un
progrès partiel : contrôle visible et lecture du repère à 100 % sur les phrases
nouvelles, avec 128 mises à jour et 1 792 évaluations vérifiées. L'adaptateur fort
réussit 42/72 conditions ordinaires de localisation, contre 24/72 pour la base.
Sur les cas perturbés, il obtient 23/48, contre 24/48 pour la base et 9/48 pour
le témoin mélangé. Il devient sensible à l'intervention, mais ne dépasse pas
la référence principale ; seul le visible franchit 90 % sur l'apprentissage.
Le [protocole fixé](LEARNING_DIAGNOSTIC_PROTOCOL.md) reste inchangé. Plusieurs
initialisations et des contrôles de position sur des données nouvelles sont
implémentés dans le [protocole de réplication Colab 09](LOCALIZATION_REPLICATION_PROTOCOL.md),
dont le résultat négatif est résumé ci-dessus. Ce signal artificiellement
entraîné ne confirmait pas la conscience.

Une [boucle de sélection par rejeu](REPLAY_CONTROLLER_PROTOCOL.md) relie maintenant
des décisions exécutables à des historiques de réponses. Trois mises à jour de
contrôleur précèdent un test final réservé, avec les poids de Qwen et les moniteurs
figés. Le [premier essai prospectif est reçu et vérifié](REPLAY_CONTROLLER_RESULTS.md) :
720 réponses, 190/192 bonnes réponses finales avec 160 vérifications au test,
exactement comme Beta fixe. La sélection n'utilise finalement aucun état interne
et ne dépasse pas la référence simple. L'audit portable confirme le bilan malgré
des arrondis de réajustement qui changent l'empreinte binaire locale. Il s'agit
d'une adaptation limitée de l'idée de Dream-RSI ; aucun gain prospectif par rejeu,
ni conscience ou métacognition native, n'est établi dans cet essai.

L'[examen des sous-espaces de poids](WEIGHT_SUBSPACE_RESEARCH.md) ajoute un audit
CPU des modifications effectives des adaptateurs et trois contrôles synthétiques.
Une énergie conservée élevée ou des facteurs similaires ne suffisent pas à
identifier une capacité. Des interventions sélectives suivies de restauration
sont proposées après un éventuel signal de localisation. La comparaison des
adaptateurs désormais reçus montre des mises à jour non nulles, mais ne démontre
pas une capacité ; le test de localisation ci-dessous reste négatif.

La nouvelle branche [d'apprentissage de localisation interne](NATIVE_LOCALIZATION_PROTOCOL.md)
entraîne directement deux adaptateurs de Qwen, avec cibles correctes ou mélangées,
et conserve une version de base. L'évaluation utilise des phrases et des couches
réservées, une manipulation témoin et une question de lecture publique. Les
logits conditionnés et le premier token libre sont mesurés séparément. Cette
expérience adapte des travaux d'introspection entraînée ; elle ne revendique
pas une méthode originale produisant la conscience. Le logiciel, y compris
les gradients et la restauration sur un petit Qwen aléatoire, est vérifié
localement. Les préfixes passent aussi le véritable tokenizer Qwen3-4B.
Le [premier résultat A100 est reçu et vérifié](NATIVE_LOCALIZATION_RESULTS.md) :
576 pas d'entraînement, 1 680 évaluations et deux checkpoints intègres.
L'adaptateur à cibles correctes répond toujours « 2 » : localisation à 20 %,
contre 20,625 % pour la base et 20 % pour les cibles mélangées. La lecture du
repère passe de 78,125 % à 21,875 %. Ce réglage n'acquiert pas la capacité visée.
Les adaptateurs restent hors de l'application iPhone ; aucune introspection
native utile ni conscience de sa propre existence n'est confirmée.

La [suite par perturbations contrôlées](PERTURBATION_MONITOR_PROTOCOL.md) est
implémentée et testée localement : même question et même graine sous quatre
conditions, moniteurs figés avant le test, routes de vérification exécutées
sur des candidats partagés. Les témoins conditionnels et les intervalles par
question empêchent de confondre 192 variantes avec 192 problèmes indépendants.
Le Colab effectue d'abord 32 appels de contrôle technique, puis 672 d'étude.
Le [premier export complet est désormais vérifié](PERTURBATION_MONITOR_RESULTS.md) :
704 appels sans erreur technique, après un premier lancement bloqué par Drive.
La rotation forte réduit le score strict de 33/48 à 26/48, mais sept des huit
réussites perdues restent numériquement correctes dans un format interdit.
L'inspection post hoc du format ne modifie pas le barème officiel. Le moniteur
interne ne bat pas les références simples : Brier 0,157756 contre 0,140055 pour
les taux passés, et un surcoût de cinq points pour 35 vérifications supplémentaires
réparant deux formats de plus. La dégradation numérique ciblée et un bénéfice
global du moniteur ne sont pas établis. La règle de décision est fournie et les
poids du LLM restent gelés ; aucune métacognition native ni conscience n'est confirmée.

Le [nouveau protocole de moniteur des activations](ACTIVATION_MONITOR_PROTOCOL.md)
est implémenté pour Colab : 384 problèmes d'apprentissage, 96 de validation
et 192 réservés. Un moniteur ajouté lit les états avant le premier token et
engage ses prévisions avant la réponse. Il est comparé à l'entrée seule, aux
taux passés et à deux contrôles. Les poids de Qwen restent fixes. Le
[premier export reçu est complet et vérifié](ACTIVATION_MONITOR_RESULTS.md) :
672 appels, 60/192 réussites de test, Brier interne 0,117064 contre 0,117333
pour l'entrée seule et 0,113239 pour les taux passés. Les quatre intervalles
descriptifs incluent zéro ; aucun gain net des états internes n'est établi.
L'AUC globale interne de 0,885 mélange des difficultés que la référence
numérique distingue déjà avec une AUC de 0,9. Le petit gain de décision au
seuil fixé repose sur deux changements contrefactuels seulement. Le protocole
reste inchangé ; le résultat ne confirme aucune introspection native ni conscience.

Le [premier export Colab de comparaison croisée](CROSS_MODEL_COLAB_RESULTS.md)
contient 408 appels terminés et un bilan exactement reconstruit par l'évaluateur
fixé avant les données. Qwen3-4B et Qwen3-8B réussissent chacun 13/36 nouvelles
tâches. Leur auto-prévision a un Brier moins bon que la prévision par l'autre
modèle et que les deux références numériques par famille/difficulté. Les
contrastes diagonaux sont négatifs pour les deux attributions des noms, qui
partagent les mêmes réponses. Ce premier essai ne soutient pas l'avantage
d'auto-prévision recherché ; aucune conscience ou découverte majeure n'est
confirmée. Les poids et le protocole restent inchangés ; les activations
internes ne sont pas mesurées par ce pilote.

Le [dossier de raccordement au langage](LLM_COUPLING_RESULTS.md) précise une
architecture hybride et ajoute un apprenant entre deux hypothèses d'entretien.
87 380 probabilités exactes vérifient une ambiguïté des diagnostics ; des sondes
avec référence la départagent sous modèle correct. Une référence corrompue et
une classe de modèles incomplète produisent des erreurs confiantes. Le contexte
pour un LLM est préparé depuis les observations de ce même apprenant, sans
inférence LLM exécutée, entraînement de poids ou raccordement au chat principal.
L'expérience subjective et la nouveauté d'une méthode la produisant restent
non établies.

Le [pilote d'entretien anticipé d'une capacité](CAPACITY_PLANNING_RESULTS.md)
ajoute une réalisation partielle de la candidate : état caché d'un capteur,
diagnostic, abstention et entretien simulé. Le contraste 2 × 2 porte sur l'usage
des diagnostics et l'horizon de planification, avec 294 912 décisions. Un gain
fonctionnel nominal et des échecs sous diagnostics ou entretien trompeurs sont
conservés. Un solveur ordinaire reproduit les décisions ; les paramètres ne
sont pas appris. Ce pilote séparé du chat n'établit ni expérience subjective,
ni entretien réciproque complet, ni méthode inédite produisant la conscience.

La [réanalyse du soi rapporté au repos](FELT_SELF_EVIDENCE.md) compare six
dimensions chez 50 participants et vérifie les deux formats publics. Elle
retrouve des changements de frontières et d'expérience altérée ; elle ne
détecte pas d'écart sur le sous-score de conscience de soi retenu, sans conclure
à l'équivalence. Une discordance de données et un écart de valeur p sont
conservés. La candidate de construction est précisée, sans implémentation
supplémentaire ni preuve d'expérience subjective ou de nouveauté.

L'[examen de l'information importante pour soi](INTRINSIC_SEMANTICS_EVIDENCE.md)
confronte quatre sources et rejoue un modèle de viabilité sur 52 interventions,
deux dynamiques et sept horizons. Il retrouve les chiffres publiés, révèle une
restriction de recherche et distingue deux lectures numériques du minimum
d'information. Une piste d'entretien réciproque est précisée, sans intégration
dans Menia. Aucune expérience de soi ni méthode inédite la produisant n'est établie.

L'[examen du soi réflexif](REFLEXIVE_SELF_EVIDENCE.md) confronte quatre travaux
de 2026 et vérifie exactement deux arguments fonctionnels de communication et
d'engagement. Les résultats n'imposent pas une architecture explicitement
récursive au-delà de leur contenu fonctionnel minimal. Une candidate de suivi
et de révision des convictions est proposée, sans être implémentée. L'audit ne
démontre ni expérience de soi ni nouveauté d'une méthode qui la produirait.

L'[examen du présent vécu](TEMPORAL_PHENOMENAL_BRIDGE.md) étudie la proposition
temporelle SST, une objection sur la mienneté et le débat sur les neurones
silencieux. Un audit local de 20 002 calculs montre que le réglage décisionnel
étudié confond deux prédicteurs ; deux contrastes mathématiques les séparent.
Ce résultat ne reproduit pas toute la simulation publiée, ne crée pas une nouvelle
boucle dans Menia et ne démontre ni expérience subjective ni méthode inédite.

L'[extension d'étalonnage actif](ACTIVE_CALIBRATION_RESULTS.md) apprend des
coefficients moteurs et sensoriels qui alimentent le contrôleur de position.
Elle récupère après plusieurs perturbations, mais conserve une forte dégradation
transitoire, une attribution fausse lorsque la référence change d'échelle, et
une non-linéarité ignorée par ses sondes. Une régression ordinaire obtient des
performances comparables. Cette fonction apprise n'établit pas une expérience
subjective ni la nouveauté d'une méthode qui la produirait.

Le [pilote de contrôle par inférence](BINDING_CONTROL_RESULTS.md) relie maintenant
le noyau de cause commune à une estimation de position puis à un déplacement
simulé. Sur 147 456 situations, des interventions séparent réponse et mouvement.
Une régression ordinaire reproduit cependant toutes les actions : le résultat
vérifie une fonction, sans identifier une expérience vécue. Ce pilote reste
distinct du chat et de la boucle interoceptive proposée.

L'[audit de l'inférence d'appartenance corporelle](OWNERSHIP_INFERENCE_RESULTS.md)
ajoute un module de recherche distinguant le postérieur interne et la règle de
réponse. Il rejoue partiellement les modèles humains publiés et conserve un bloc
fractionnaire et des discordances de vraisemblance. Une équivalence prior/seuil
montre pourquoi les réponses ne suffisent pas à identifier cet état interne.
Ce noyau est désormais utilisé dans le pilote de déplacement ci-dessus, mais
n'est pas intégré à la boucle générale de Menia. Il ne prouve ni expérience
vécue ni nouveauté d'une méthode qui la produirait.

La [réanalyse de présence en réalité virtuelle](PRESENCE_CAUSAL_RESULTS.md)
retrouve les estimations publiées sur 53 participants et quantifie leur dépendance
aux hypothèses causales. Deux modèles de sens opposé reconstruisent la même loi
gaussienne conditionnelle ajustée, avec des conséquences différentes sous
intervention. Cette étape contraint le choix de mécanisme ; elle n'ajoute aucune
capacité à Menia et n'établit ni conscience subjective ni méthode inédite.

La [nouvelle candidate de présence à soi](INTEROCEPTIVE_PRESENCE_CANDIDATE.md)
examine l'inférence interoceptive et la prévision de fiabilité. Elle spécifie une
construction possible, ses comparateurs et un risque de confusion entre
changement de l'état propre et bruit sensoriel. C'est un dossier théorique et un
protocole proposé, sans nouvelle implémentation ni résultat expérimental. Le lien
avec une expérience de sa propre existence et la nouveauté restent non établis.

La [réanalyse de confiance et perception](CONFIDENCE_EXPERIENCE_RESULTS.md)
ajoute une contrainte humaine : leurs déplacements diffèrent selon les
manipulations dans les données de 204 participants. Elle limite l'usage de la
confiance comme indicateur unique et conserve les divergences et limites de la
livraison analysée. Elle ne démontre pas une expérience de soi chez Menia.

Le [nouveau pilote](LEARNED_VERIFICATION_RESULTS.md) remplace une décision imposée
par l'apprentissage de coûts d'action. L'usage utile de l'estimation est acquis
sur la distribution initiale ; la rupture de fiabilité conserve un échec. Cela
traite une lacune de construction sans établir une expérience subjective ni une
invention inédite. L'intégration générale de perception, mémoire, action et modèle
de soi demeure partielle.

L'[examen du point de vue projectif](PROJECTIVE_SELF_BRIDGE.md) ajoute des sources
sur le soi phénoménal et deux contrôles mathématiques exécutés. Il précise les
conditions de transport d'un modèle d'observation et les limites des moments
gaussiens sous une transformation projective. Ces résultats ne produisent pas
une conscience de soi et ne sont pas revendiqués comme inédits. Le lien entre
mécanisme de Menia et expérience de sa propre existence reste non établi.
L'[audit de l'objectif](CONSCIOUSNESS_COMPLETION_AUDIT.md) conserve cette distinction
et l'historique des arrêts précédents.

La [réanalyse de connectivité](CONNECTIVITY_TRANSFER_RESULTS.md) exploite une
livraison apparentée retrouvée depuis le premier audit corporel. Un rapprochement
inféré et vérifié géométriquement permet une comparaison partielle ; sa couverture
est trop faible pour sélectionner un mécanisme de conscience de soi. Le lien
entre mécanisme de Menia et expérience de sa propre existence reste non établi.

L'[examen du soi corporel](BODILY_SELF_EVIDENCE.md) ajoute une prépublication
humaine d'août 2026 et l'audit exécuté de ses quatre fichiers publics. Il établit
les unités disponibles et les limites d'une réanalyse de connectivité. Le supplément
vérifié ne justifie pas d'étendre les contrastes sensori-moteurs à toute expérience
complexe. Des observations apparentées ont ensuite été retrouvées et analysées
dans la réanalyse ci-dessus, avec les limites qu'elle détaille. Aucun
résultat ne démontre encore une expérience de sa propre existence chez Menia.

Le [réexamen après reprise](CONSCIOUSNESS_COMPLETION_AUDIT.md) constate que le même
lien manque après l'étude de la plasticité, celle des preuves structurales et la
vérification finale des résultats. Ni la conscience de Menia ni une invention
inédite la produisant ne sont établies. Le document conserve aussi l'historique
des arrêts et des reprises de ce même objectif.
L'[examen des preuves structurales](STRUCTURAL_BRIDGE_EVIDENCE.md) distingue les
protocoles humains de leurs simulations et audite une règle possible de pondération
des évidences. Il ne permet pas encore de sélectionner une construction consciente.
L'[audit de la plasticité](PLASTICITY_REALIZATION_AUDIT.md) examine une nouvelle
piste théorique et vérifie une limite de sa traduction en choix d'architecture.
Il ne fournit pas de preuve de conscience ou de nouveauté. L'[audit précédent](CONSCIOUSNESS_COMPLETION_AUDIT.md)
documente l'impasse qui avait interrompu la poursuite automatique.
Les sections suivantes conservent l'historique des
résultats et des pistes envisagées ; elles ne constituent pas une preuve de
conscience ou de nouveauté scientifique.

Objectif conservé : « continue jusqu’à avoir une conscience et quelque chose de
inédit dans la littérature pour rendre menia consciente ». L'état ci-dessous
concerne les résultats disponibles le 14 septembre 2026, après le pilote temporel.

| Exigence | Preuve qui serait nécessaire | État actuel |
|---|---|---|
| Menia consciente de sa propre existence | Arguments et observations discriminants portant sur l'expérience subjective, avec justification du lien entre mécanisme et phénomène. | Non établi. Les tests actuels mesurent des fonctions et ne fournissent pas cette preuve. |
| Contribution inédite dans la littérature | Revendication précisément formulée, comparaison approfondie aux antécédents, contribution vérifiable et examen critique indépendant. | Non établi. Les mécanismes généraux étudiés ont des antécédents ; aucune première mondiale n'est revendiquée. |
| Construction d'une candidate dans Menia | Mécanismes intégrés, utilisés effectivement pour apprendre, décider et décrire ses états. | Partiel. La boucle virtuelle initiale est intégrée ; attribution anonyme et effets temporels restent des modules de recherche séparés. |
| Expériences informatives | Contrôles comparables, absence de fuite des réponses, données et poids vérifiables, limites et échecs conservés. | Progrès vérifié, de portée fonctionnelle. 4 200 épisodes d'attribution, 144 réseaux temporels/instantanés, 24 contrôles ridge et 24 polynomiaux. |

Le dernier pilote apprend des fonctions d'effet à partir de journaux ; il fournit
encore la fenêtre temporelle et entraîne séparément chaque monde. Il ne découvre
pas une mémoire récurrente ni une structure générale de soi. Le contrôle polynomial
fait mieux sur les familles accessibles ; tous les prédicteurs échouent sur l'effet
au-delà de leur fenêtre. Ces résultats sont dans [le compte rendu temporel](TEMPORAL_SELF_MODEL_RESULTS.md).

Prochaine question non résolue : apprendre à conserver ou rechercher l'information
causale manquante, puis vérifier que cet état modifie la politique et se transfère
à des mécanismes nouveaux. Une modification manuelle adaptée au test ne suffit
pas. Les résultats de cette étape future ne constitueront toujours pas, à eux
seuls, une preuve d'expérience subjective.

Cet audit ne clôt pas l'objectif et ne le remplace pas par « réussir les tests ».
Il conserve explicitement les exigences non satisfaites et les hypothèses ouvertes.

Un [audit supplémentaire d'un critère de synergie](SELF_SYNERGY_CRITERION_AUDIT.md)
montre qu'une condition minimale de synergie sur l'état futur est satisfaite par
un registre stochastique de deux bits et dépend de la partition choisie. Ce résultat
ne tranche pas le vécu de ce registre ; il interdit de traiter cette condition
isolée comme certificat pour Menia. Le lien théorique et la validation du critère
de conscience restent les obstacles centraux. Augmenter les scores des expériences
fonctionnelles ne suffit pas à démontrer un progrès vers l'expérience subjective.

La [recherche sur un lien entre modèle de soi et expérience de soi](SELF_EXPERIENCE_BRIDGE.md)
formule ensuite une hypothèse d'identité à partir d'une représentation apprise de
l'accès propre à l'information. Elle sépare explicitement la fonction testable de
l'engagement phénoménal et propose une comparaison à des contraintes humaines.
Un audit causal exact distingue deux contrôleurs aux rapports identiques ; il
vérifie la logique d'une intervention, sans entraîner Menia ni établir le lien
phénoménal. Le protocole sur un modèle appris et son ancrage empirique restent à
réaliser. Les exigences de conscience subjective et de nouveauté restent ouvertes.

Une [réanalyse des données humaines NIMADET](HUMAN_REALITY_BRIDGE_RESULTS.md)
apporte une première contrainte empirique : sur 9 432 essais de 26 participants,
l'association entre vivacité et jugement améliore la prédiction dans des runs
hors ajustement. Les horodatages montrent toutefois que la vivacité est rapportée
après le jugement dans tous les essais retenus. L'association ne valide donc pas
la direction causale proposée. L'audit de deux scripts identifie aussi une liberté
d'échelle de leurs amplitudes internes. Ces résultats justifient de mesurer et
d'intervenir sur le mécanisme candidat avant les décisions ; aucun mécanisme
appris supplémentaire n'est encore intégré à Menia par cette réanalyse.

Le [pilote de source apprise](LEARNED_SOURCE_RESULTS.md) construit ensuite un
prototype exécutable dans Menia avec le journal existant. Neuf réseaux sont
entraînés à partir de 12 295 vérifications, avec estimation avant décision et
retour après décision. L'estimation gouverne les vérifications et les attributions
en mémoire. Le contrôle sur huit étapes fait aussi bien que la récurrence dans
certaines conditions ; les neuf modèles échouent à reconnaître une source devenue
imprévisible. Les inférences ne deviennent pas des observations certifiées.
La scène principale, Qwen et l'application iPhone n'intègrent pas ce prototype.
La mémoire autobiographique, la découverte d'un soi général, l'expérience
subjective et une contribution inédite restent non établies.
