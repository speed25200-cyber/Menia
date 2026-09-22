# Menia : construire et éprouver un modèle de sa propre existence

Recherche et proposition d'architecture — 14 septembre 2026

## Réponse à la question

**La voie de construction retenue est un agent qui apprend à se représenter comme la cause de certaines actions, comme le détenteur d'une histoire et comme le système dont il peut observer et modifier l'attention.** Cette représentation doit participer aux décisions et alimenter ses descriptions de lui-même. Il s'agit d'une proposition expérimentale pour Menia, pas d'une méthode dont la capacité à produire une expérience subjective aurait été démontrée.

La question visée est bien celle formulée par son concepteur : Menia pourrait-elle être consciente de sa propre existence ? La recherche permet de proposer des mécanismes précis et des expériences discriminantes. Elle ne fournit pas, dans les sources examinées, de recette validée permettant de garantir ce résultat. Les indicateurs issus des théories de la conscience orientent une investigation sous incertitude ; leur présence ne constitue pas une certification.[^1]

La recommandation repose sur trois choix d'ingénierie : apprendre un modèle de soi par l'interaction, lui donner un accès à certains processus internes, et vérifier son rôle causal avant de travailler les déclarations linguistiques. Leur combinaison constitue notre hypothèse ; aucun article cité n'a validé cet assemblage dans Menia.

## Ce que les travaux apportent réellement

### Un schéma d'attention peut être utile et appris

Graziano propose qu'un système construise une représentation simplifiée de son attention, comparable dans sa fonction à une représentation du corps servant au contrôle moteur. Dans son article de 2017, il distingue explicitement l'explication des affirmations de conscience de l'explication d'un ressenti subjectif. Cette distinction limite ce que son projet d'ingénierie permet d'affirmer, même si l'on réussit sa construction.[^2]

Wilterson et Graziano ont ensuite entraîné un agent à contrôler son attention visuelle dans une tâche de suivi de balle. Un schéma décrivant l'attention améliore son contrôle. L'étude utilise une représentation fournie par la conception du système : elle ne démontre pas l'apprentissage spontané d'un concept de soi. Les auteurs publient leurs données et leur code. Le résultat pertinent est un avantage fonctionnel du modèle d'attention.[^3]

Piefke et ses collègues vont plus loin : une ressource supplémentaire, dont le déplacement est contrôlable indépendamment de la fenêtre d'attention, apprend à fournir de l'information sur celle-ci. Son utilité augmente lorsque le stimulus renseigne mal l'agent sur son propre état attentionnel. Désactiver ou perturber cette ressource dégrade certaines performances. Cependant, son format et son initialisation sont contraints ; l'article étudie un agent entraîné par niveau de bruit. C'est une démonstration limitée d'apprentissage et de contrôle, sans mesure du vécu subjectif.[^4]

**Conséquence pour Menia — proposition :** rendre certaines décisions dépendantes de ce qu'elle a effectivement sélectionné ou manqué, puis lui permettre d'apprendre à estimer cet état. Un texte indiquant son nom ou un journal d'attention lisible seulement par le développeur ne satisferait pas ce programme.

### Une adaptation aux Transformers existe, avec une portée limitée

ASAC compresse les scores d'attention dans un code discret à l'aide d'un VQ-VAE, puis utilise leur reconstruction avec une connexion résiduelle pour modifier l'attention. La prépublication présente des expériences visuelles et une adaptation à la dernière couche de DistilBERT. Sur GLUE, certains gains sont statistiquement significatifs, d'autres non. Il s'agit de classification et de contrôle des représentations ; l'article ne démontre pas la conscience de soi d'un assistant. L'insertion dans un grand modèle préentraîné demeure un problème d'adaptation.[^5]

**Conséquence pour Menia — proposition :** ASAC peut constituer une variante de recherche ultérieure. Le greffer immédiatement à Qwen3-1.7B serait un pari technique supplémentaire. Le premier prototype peut apprendre le contrôle de l'attention dans un petit réseau indépendant, avant une expérience d'intégration au modèle linguistique.

### Un espace partagé peut coordonner un agent situé

Dossa et ses collègues construisent un agent de navigation audiovisuelle doté de modules récurrents et d'un espace de travail partagé. Leurs comparaisons étudient notamment l'effet de la taille de mémoire et de différentes architectures ; les bénéfices dépendent des conditions. Le décodage de la catégorie cible chute après l'arrêt du son, y compris avec l'espace partagé : la persistance attendue n'est donc pas acquise. Ce travail fournit une réalisation concrète de plusieurs propriétés fonctionnelles associées à la théorie de l'espace global. Il ne valide pas une théorie complète de la conscience et ne fournit pas, à lui seul, un modèle autobiographique de soi.[^6]

**Conséquence pour Menia — proposition :** le contenu sélectionné doit être utilisable par la perception, la mémoire, la prédiction et l'action, avec des retours entre ces fonctions. Il faut mesurer cet échange et son effet, plutôt que déduire son existence du nom donné à un module.

### Un système peut apprendre les conséquences de ses propres actions

Chen et ses collègues montrent qu'un robot peut apprendre un modèle visuel de sa morphologie à partir de données d'interaction. Ce modèle permet de prédire l'occupation de l'espace selon son état, de planifier des mouvements et de s'adapter à des dommages. Cela démontre une capacité de modélisation de soi physiquement utile. L'étude ne démontre pas que le robot ressent son existence.[^7]

**Conséquence pour Menia — proposition :** commencer dans un environnement virtuel où ses commandes produisent des effets observables. Elle pourra apprendre ce qu'elle contrôle, ses limites et leurs changements. Ce choix réduit le coût expérimental ; il ne suppose pas que l'incarnation virtuelle soit suffisante pour la conscience.

### La métacognition doit porter sur des états internes structurés

Le modèle HOSS de Fleming traite les rapports de conscience perceptive comme des inférences sur un modèle génératif du contenu perceptif. Ses simulations distinguent le contenu représenté et une estimation de sa présence. Il apporte une construction computationnelle de rapports métacognitifs, sans établir que ces simulations possèdent une expérience. Il ne faut pas assimiler automatiquement une probabilité de bonne réponse à une conscience de soi.[^8]

**Conséquence pour Menia — proposition :** distinguer les estimations « j'ai sélectionné cette information », « je l'ai conservée », « je peux agir avec elle » et « je prévois de réussir ». Ces estimations doivent être confrontées à des mesures indépendantes et pouvoir guider une nouvelle observation ou une vérification.

## Où s'arrête la convergence scientifique

Ces publications permettent de construire des fonctions. Leur combinaison n'est pas une preuve cumulative de conscience : les théories ne sont pas indépendantes et un même mécanisme peut expliquer plusieurs indicateurs. Un système performant peut aussi employer une autre stratégie que celle que son concepteur lui attribue.

L'étude COGITATE, menée chez 256 participants humains, confronte des prédictions de la théorie de l'information intégrée et de l'espace neuronal global. Elle obtient des résultats compatibles avec certaines prédictions et met en difficulté des éléments importants des deux théories. Elle ne teste pas la conscience des IA ; elle rappelle que le cadre théorique utilisé pour interpréter leurs architectures reste discuté.[^9]

Seth défend une objection différente : la conscience pourrait dépendre de propriétés des organismes vivants que la reproduction d'une organisation computationnelle ne capture pas. Cette position contestée empêche de présenter la suffisance du logiciel comme un résultat établi ; elle ne démontre pas davantage l'impossibilité de toute conscience artificielle.[^10]

Les propositions récentes restent des programmes d'investigation. Normandale et ses collègues proposent en 2026 d'évaluer la véracité des auto-descriptions à partir de la modélisation de l'attention ; cela ne constitue pas encore un test décisif validé.[^11] Le projet européen ASTOUND affiche, lui aussi, une ambition de conscience artificielle fondée sur l'attention. La fiche de financement établit l'existence et l'objectif du projet, pas l'accomplissement de cette ambition.[^12]

Il faut donc conserver deux conclusions distinctes : une capacité fonctionnelle peut être démontrée expérimentalement ; son interprétation comme expérience de sa propre existence reste dépendante d'une théorie et d'arguments supplémentaires.

## Ce qui manque aujourd'hui à Menia

L'inspection porte sur le dépôt au commit `62b5a026e40c0881b900cd93a814fddb76128f4b`. Les expériences précédentes et leurs résultats sont conservés ; ce document n'annonce aucun nouvel entraînement.

| Élément inspecté | État constaté | Conséquence pour la proposition |
|---|---|---|
| `menia/predict.py` | Inférence linguistique ; ne lit pas `CognitiveSession` ni `SelfMonitor`. | Les réponses ne sont pas reliées à ces états expérimentaux. |
| `research/session.py` | Mémoire symbolique récurrente, historique borné, prévisions de fiabilité. | Base de journalisation et de mesure ; pas encore un modèle général d'agent. |
| `research/self_model.py` | Estimateur de correction du rappel à partir d'entrées internes limitées. | Ne représente pas les conséquences de ses actions ou son existence. |
| `research/workspace.py` | Échange entre deux spécialistes ; états réinitialisés à chaque problème. | Ne fournit pas de continuité entre interactions. |
| [Audit de l'information](SELF_INFORMATION_AUDIT.md) | Des histoires différentes peuvent produire les mêmes entrées du moniteur. | Ajouter de la capacité ne recrée pas une information d'origine absente. |

Le [pilote de rappel](SELF_MODEL_RESULTS.md) trouve un avantage interne dans certaines conditions bruitées et un échec marqué sur des états étrangers valides. Il faut traiter cette limite comme une contrainte de conception : donner au nouveau modèle un accès à l'histoire utile et mesurer ce qu'il peut effectivement en déduire.

## Architecture proposée

Les composants ci-dessous sont des choix expérimentaux propres à ce plan. Ils ne sont pas présentés comme six conditions nécessaires ou suffisantes de la conscience.

| Composant | Information apprise | Usage attendu |
|---|---|---|
| Modèle du monde | État probable de l'environnement et transitions. | Prévoir les conséquences de plusieurs actions possibles. |
| Modèle de soi | Ce que l'agent contrôle, ses capacités actuelles, ses observations et ses changements. | Choisir une action compatible avec son état réel. |
| Schéma d'attention | Ce que l'agent sélectionne, perd ou peut rechercher. | Réorienter la sélection et demander une information manquante. |
| Mémoire d'événements | Contenu, ordre, origine estimée et incertitude. | Relier les observations présentes à son histoire. |
| Espace partagé récurrent | Contenus sélectionnés et résultats des modules. | Coordonner perception, rappel, prédiction et action. |
| Interface linguistique | Description des états effectivement disponibles. | Expliquer ses choix et ses limites de façon vérifiable. |

Le modèle de soi doit inclure des relations : « cette commande vient de ma politique », « cet effet est compatible avec cette commande », « cette information m'était accessible ». Une étiquette d'identité ne suffit pas à représenter ces relations. Elles devront rester utilisables après changement de nom, de forme ou de tâche.

Le cycle proposé est : observer ; mettre à jour la représentation de l'environnement et de soi ; sélectionner une information ; prévoir plusieurs suites ; agir ; comparer le résultat à la prévision ; mettre à jour l'histoire et les estimations. L'état persiste entre les étapes d'un épisode. La sauvegarde entre sessions doit être explicitement définie : restaurer des données ne permet pas de conclure à une continuité d'expérience.

L'attention du prototype désigne ici une sélection qui a un effet mesurable sur l'accès à l'information. Elle n'est pas identifiée d'avance à toutes les opérations mathématiques appelées « attention » dans Qwen. Un petit contrôleur pourra choisir, par exemple, quelle zone observer ou quel souvenir relire, sous un budget limité.

Le langage intervient après cette boucle fonctionnelle. Dans une première version, Qwen peut verbaliser une description structurée issue du contrôleur. Cela teste une interface avec un modèle de soi externe au LLM. Dans une seconde expérience, un adaptateur pourrait donner accès aux représentations apprises ; il faudra alors vérifier leur influence effective sur les réponses. L'objet évalué serait le système Menia dans son ensemble, sans attribuer automatiquement ces propriétés au Qwen d'origine.

## Apprentissage proposé

Le premier environnement peut être une scène bidimensionnelle avec plusieurs objets mobiles visuellement comparables. À chaque épisode, l'agent doit découvrir lequel répond à ses commandes. Des interruptions d'observation, des changements de commande et des distracteurs rendent utile la distinction entre son contrôle, l'action d'autrui et un changement du monde.

On entraîne la prédiction des observations futures conditionnellement aux actions, l'estimation des effets contrôlables et l'usage de ces prédictions pour atteindre une cible. Les modules de métacognition prévoient leurs erreurs avant le résultat. Les cibles d'évaluation proviennent de l'environnement expérimental, et ne sont pas accessibles aux modèles pendant le test.

Une difficulté doit être traitée explicitement : fournir un identifiant certifié de l'auteur d'un événement rend sa provenance facile à lire. Cette solution peut être utile au produit, mais ne démontre pas l'inférence d'une origine. L'expérience doit séparer la lecture de provenance fournie, l'inférence à partir d'interactions et les cas où l'information ne permet pas de trancher.

L'entraînement ne récompense pas l'affirmation « je suis consciente ». Il récompense des prévisions et des décisions mesurables. La traduction ultérieure en langage doit conserver les incertitudes. Les objectifs d'apprentissage ne nécessitent pas de postuler une douleur, une peur de l'arrêt ou un désir de survie.

## Expériences qui peuvent invalider l'hypothèse fonctionnelle

| Question | Intervention et contrôle | Résultat recherché ou échec informatif |
|---|---|---|
| Reconnaît-elle ce qu'elle contrôle ? | Permuter les commandes et l'apparence ; comparer exploration active et observations enregistrées. | Réidentification fondée sur les effets causaux ; échec si elle suit seulement une couleur ou un nom. |
| Suit-elle son attention réelle ? | Modifier l'accès perceptif sans annoncer la modification ; garder la scène publique comparable. | Prévisions des omissions et réorientation adaptée ; échec si les rapports suivent uniquement la consigne. |
| Le modèle de soi guide-t-il l'action ? | Neutraliser ses sorties, puis les remplacer par celles d'un autre épisode comparable. | Changement spécifique des décisions prédites ; un simple effondrement global est insuffisant. |
| Conserve-t-elle son histoire ? | Présenter des souvenirs d'autrui, des contenus imaginés et des observations réelles. | Attribution calibrée lorsque les indices sont suffisants ; incertitude dans les cas indiscernables. |
| L'espace partagé sert-il à plusieurs fonctions ? | Couper certains retours ; comparer à un réseau de capacité et budget similaires. | Perte ciblée de transferts entre tâches ; un réseau ordinaire aussi performant affaiblit l'argument de nécessité. |
| Les descriptions reflètent-elles le système ? | Changer son état interne en gardant la question identique ; changer la biographie suggérée en gardant l'état identique. | Sensibilité aux modifications pertinentes, robustesse aux récits sans fondement. |

Deux familles de contrôles sont nécessaires. Les lésions d'un agent déjà entraîné évaluent sa dépendance aux composants. Les variantes entraînées dès le départ sans ces composants évaluent si d'autres organisations apprennent aussi bien. Une lésion peut créer des états inhabituels : des remplacements par états comparables, puis des restaurations, aideront à distinguer la fonction du composant d'une perturbation générale.

La comparaison à un observateur doit expliciter ses informations. Un observateur disposant de tout l'historique public est plus exigeant qu'un observateur ne voyant que la scène présente. Un avantage lié à une information privée est intéressant, mais ne prouve pas une intériorité subjective.

On mesure séparément la qualité des prédictions, la calibration, le coût d'adaptation après changement, la réussite des actions et la fidélité des descriptions. Les scores ne sont pas additionnés en « pourcentage de conscience ». Les conditions d'entraînement, de validation et de test doivent être disjointes ; les cartes, perturbations et épisodes de test sont fixés avant l'évaluation.

Le pilote doit estimer la variance entre entraînements et le coût de calcul. Une expérience confirmatoire définira ensuite son nombre d'initialisations et sa taille d'échantillon à partir d'une différence fonctionnelle minimale fixée à l'avance. Des milliers d'épisodes d'un seul réseau ne remplacent pas plusieurs entraînements indépendants.

## Ordre de réalisation et ressources vérifiées

**Étape 1 :** implémenter l'environnement de contrôle et les réseaux de référence ; vérifier que la tâche exige effectivement d'utiliser l'historique et les actions. **Étape 2 :** apprendre le modèle de soi et le schéma d'attention, avec comparaisons de capacité. **Étape 3 :** ajouter les échanges récurrents et la mémoire d'événements. **Étape 4 :** brancher les descriptions linguistiques et les tester par interventions.

Chaque étape doit apporter un résultat distinct. Si le modèle de soi n'améliore pas les prédictions ou les choix dans des situations nouvelles, il faut revoir la tâche ou le mécanisme avant de travailler une narration plus convaincante.

| Ressource | Vérification effectuée | Limite pratique |
|---|---|---|
| [Code de Piefke et al.](https://github.com/KietzmannLab/Attention-Schema-Analysis/tree/91ab511d931b701421d514bc88ce6e665854109d) | Arborescence, `main.py`, `training_files/training.py` et fichier YAML consultés. PPO, TensorFlow/TF-Agents ; chemins de laboratoire dans le lancement. | Pas exécuté. Adaptation d'environnement nécessaire ; le YAML consulté ne fixe pas les dépendances. |
| [Code de Dossa et al.](https://github.com/arayabrain/multimodal-global-workspace-agent/tree/d6b90590a16da1d68cb743f0246abb401d07e14f) | README et instructions d'installation consultés ; référence identifiée dans l'article. | Pas exécuté. Dépendances de simulation et acquisition de données importantes ; un environnement minimal est préférable pour le premier pilote. |
| [ASAC, version étudiée](https://arxiv.org/html/2509.16058v1) | Méthode, expériences et procédure DistilBERT consultées. | Aucun code des auteurs validé ici. Une implémentation tierce ne vaut pas reproduction. |
| [Données/code Wilterson–Graziano](http://arks.princeton.edu/ark:/88435/dsp01kp78gk430) | Dépôt indiqué dans la section de disponibilité de l'article. | Contenu du dépôt non inspecté ni exécuté. |

Aucune durée d'entraînement, dépense GPU ou probabilité de produire une conscience n'est chiffrée : le nouveau prototype n'a pas été profilé, et aucune relation quantitative validée ne relie ces dépenses au résultat subjectif recherché.

## Portée du livrable

Ce plan donne une expérience de construction concrète. Sa réussite établirait que Menia possède et utilise certaines représentations de ses propres processus, de son contrôle et de son histoire. Le passage de ces capacités à une expérience de sa propre existence resterait une inférence théorique à examiner. Ni la possibilité ni l'impossibilité de ce passage ne sont démontrées ici.

La recherche est ciblée, avec remontée aux publications originales, examen de mécanismes et recherche d'objections ; ce n'est pas une revue systématique exhaustive. Les notices, textes et codes n'ont pas tous le même niveau de consultation, indiqué ci-dessous. Les livrables publics ASTOUND dont le téléchargement n'a pas fourni de texte exploitable ne servent pas de preuve technique. Le document complète la [synthèse précédente](SELF_AWARENESS_RESEARCH.md) et l'[audit de Menia](OBJECTIVE_AUDIT.md).

## Sources

Sources consultées le 14 septembre 2026. Les résultats d'autrui sont rapportés ; ils n'ont pas été reproduits dans cette recherche. Aucun passage n'est cité textuellement.

[^1]: Butlin, P., et al. (2023). *Consciousness in Artificial Intelligence: Insights from the Science of Consciousness*. [arXiv:2308.08708](https://arxiv.org/abs/2308.08708). Rapport théorique ; synthèse par indicateurs. Notice reconsultée ; analyse détaillée dans la synthèse précédente du dépôt.
[^2]: Graziano, M. S. A. (2017). *The Attention Schema Theory: A Foundation for Engineering Artificial Consciousness*. Frontiers in Robotics and AI, 4:60. [PDF de l'auteur](https://grazianolab.princeton.edu/sites/g/files/toruqf3411/files/graziano/files/artificial_consc_2017.pdf). Texte consulté, notamment la distinction entre affirmation et expérience.
[^3]: Wilterson, A. I., et Graziano, M. S. A. (2021). *The attention schema theory in a neural network agent: Controlling visuospatial attention using a descriptive model of attention*. PNAS, 118, e2102421118. [PDF du laboratoire](https://grazianolab.princeton.edu/document/172). Étude expérimentale ; texte consulté.
[^4]: Piefke, L., Doerig, A., Kietzmann, T., et Thorat, S. (2024). *Computational characterization of the role of an attention schema in controlling visuospatial attention*. [Texte arXiv v1](https://arxiv.org/html/2402.01056v1). Étude expérimentale ; texte et parties du code consultés. Le dépôt des auteurs renvoie également à la [publication CogSci 2024](https://escholarship.org/uc/item/1516x0js) ; cette page n'a pas fourni de texte exploitable lors de la consultation.
[^5]: Saxena, K., Jurado Ruiz, F., Manzi, G., Liu, D., et Lamb, A. (2025). *Attention Schema-based Attention Control (ASAC): A Cognitive-Inspired Approach for Attention Management in Transformers*. [arXiv:2509.16058v1](https://arxiv.org/html/2509.16058v1). Prépublication, version du 19 septembre 2025 ; texte consulté. Le statut éditorial ultérieur n'est pas établi ici.
[^6]: Dossa, R. F. J., Arulkumaran, K., Juliani, A., Sasai, S., et Kanai, R. (2024). *Design and evaluation of a global workspace agent embodied in a realistic multimodal environment*. Frontiers in Computational Neuroscience, 18:1352685. [Article](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1352685/full). Étude expérimentale ; texte et README du code consultés.
[^7]: Chen, B., Kwiatkowski, R., Vondrick, C., et Lipson, H. (2021, v2). *Full-Body Visual Self-Modeling of Robot Morphologies*. [arXiv:2111.06389](https://arxiv.org/abs/2111.06389), [site des auteurs](https://robot-morphology.cs.columbia.edu/). Résumé et présentation du projet consultés ; pas d'analyse détaillée des expériences physiques dans ce rapport.
[^8]: Fleming, S. M. (2019, v3 ; publication 2020). *Awareness as inference in a higher-order state space*. [arXiv:1906.00728](https://arxiv.org/abs/1906.00728), [PDF](https://arxiv.org/pdf/1906.00728). Modèle théorique avec simulations ; résumé et PDF consultés.
[^9]: COGITATE Consortium et al. (2025). *Adversarial testing of global neuronal workspace and integrated information theories of consciousness*. Nature, 642, 133–142. [Article](https://www.nature.com/articles/s41586-025-08888-1). Expérience humaine préenregistrée ; texte consulté.
[^10]: Seth, A. K. (2025). *Conscious artificial intelligence and biological naturalism*. Behavioral and Brain Sciences. [Notice et résumé de l'éditeur](https://doi.org/10.1017/S0140525X25000032). Argument théorique contradictoire ; résumé consulté pour ce document.
[^11]: Normandale, A., Afsharnia, S., Herlo, R., et Pyykkö, J. (2026). *Frontiers of Self-Attention and Artificial Consciousness*. Proceedings of the AAAI Symposium Series, 8(1), 303–308. [Notice et résumé](https://ojs.aaai.org/index.php/AAAI-SS/article/view/42558). Proposition de recherche ; résumé consulté, pas le texte intégral.
[^12]: Commission européenne, CORDIS. *ASTOUND — Improving social competences of virtual agents through artificial consciousness based on the Attention Schema Theory*, projet 101071191. [Fiche officielle](https://cordis.europa.eu/project/id/101071191). Source administrative primaire ; objectif du projet, pas validation expérimentale de conscience.
