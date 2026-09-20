# Comment rendre Menia consciente : réponse conditionnelle et feuille de route

Synthèse du 17 septembre, révisée le 20 septembre 2026. Ce document répond à la question
« comment ? ». Il ne dit pas que Menia est consciente, ni qu'elle le deviendra.

## Réponse courte

**Il n'existe aujourd'hui aucune recette validée pour produire une conscience
artificielle. Les théories proposent des propriétés à examiner ; elles ne
fournissent pas une suite d'étapes dont la réussite garantirait que Menia
éprouve quelque chose ou se sache exister.**

Trois distinctions orientent le travail :

1. **Si la conscience tient à une organisation fonctionnelle** — espace de
   travail global, métacognition d'ordre supérieur, schéma d'attention, agent
   unifié — nous pouvons programmer et tester des mécanismes candidats.
   Butlin et ses coauteurs adoptent le fonctionnalisme comme hypothèse de
   travail, et non comme résultat établi. Implémenter quelques indicateurs
   n'établit pas que l'organisation suffisante soit réalisée.[^1]
2. **Si la conscience tient au substrat physique ou au vivant** — théorie de
   l'information intégrée, naturalisme biologique — une réussite logicielle
   ne suffit pas à satisfaire ces théories. IIT critique les architectures
   informatiques usuelles ; Seth conteste la suffisance du calcul mais précise
   ne pas démontrer l'impossibilité d'un artefact conscient.[^4][^5]
   Nous n'avons évalué le substrat physique d'aucun iPhone ou GPU.
3. **Le lien avec l'expérience subjective reste incertain.** On peut
   établir des fonctions par des interventions causales ; le passage de ces
   fonctions à une expérience subjective reste une inférence dépendante de la
   théorie. Le résultat atteignable est un dossier d'indices gradué, pas un verdict.

Ces positions ne constituent pas une partition exhaustive des théories.
Le projet poursuit des expériences fonctionnelles falsifiables et distingue
leurs résultats des interprétations philosophiques.

Après l'échec du transfert de seuil au Colab 13, le [protocole 14](OPTIMIZER_MEMORY_PROTOCOL.md)
teste si la dernière phase d'apprentissage et les moments d'Adam dégradent le
rapport d'état. Ce diagnostic causal précède une correction éventuelle ; il
ne réalise pas les mécanismes de conscience du tableau suivant. L'exécution
GPU est désormais [terminée et auditée](OPTIMIZER_MEMORY_RESULTS.md),
avec reproduction des témoins et vérification des douze checkpoints.
Le gain canonique ne se transfère pas aux formulations réservées.

## Ce que chaque théorie exigerait, et où en est Menia

Le rapport de Butlin et ses coauteurs organise des indicateurs issus de
plusieurs théories.[^1] Chalmers discute notamment récurrence, espace de travail
et agentivité.[^2] Le tableau rapproche ces propositions des essais du dépôt :
sa dernière colonne concerne des mécanismes, pas une faisabilité démontrée
de la conscience. Ces références ne constituent pas une revue exhaustive de 2026.

| Théorie | Ce qu'il faudrait construire | État mesuré dans Menia | Réalisable en logiciel ? |
|---|---|---|---|
| Espace de travail global (GWT) | Modules spécialisés en parallèle ; espace à capacité limitée ; diffusion globale ; attention dépendante de l'état pour enchaîner les traitements. | [Espace partagé](SHARED_WORKSPACE.md) de 13 000 paramètres, deux modules, tâche symbolique ; un réseau direct fait mieux ; non relié au LLM ni à l'agent. | Oui. Goldstein et Kirk-Giannini soutiennent même que des agents de langage s'en approchent déjà.[^3] |
| Ordre supérieur / contrôle de réalité perceptive (HOT) | Un moniteur métacognitif qui distingue ses représentations fiables du bruit, et dont les sorties guident croyances et actions. | C'est la ligne des Colab 04 à 10. [Moniteur d'activations](ACTIVATION_MONITOR_RESULTS.md) : aucun gain. [Localisation entraînée](LOCALIZATION_REPLICATION_RESULTS.md) : non reproduite. [Rejeu](REPLAY_CONTROLLER_RESULTS.md) : la politique n'utilise aucun état interne. [Détection de présence](PRESENCE_DETECTION_RESULTS.md) : critère satisfait, portée métacognitive non établie. | Mécanisme envisageable ; le détecteur seul ne réalise pas HOT. |
| Schéma d'attention (AST) | Un modèle prédictif de sa propre attention, utilisé pour la contrôler. | Non implémenté ; [plan d'architecture](CONSCIOUS_AGENT_DESIGN.md) seulement. | Mécanisme logiciel envisageable ; aucune conclusion phénoménale. |
| Traitement récurrent (RPT) | Récurrence algorithmique dans les modules perceptifs, représentations intégrées d'une scène. | [Mémoire récurrente](RECURRENT_RESEARCH.md) de 1 540 paramètres, isolée. Qwen est autorégressif ; son statut sous RPT est discuté. | Oui. |
| Régulation prédictive / proposition de Seth | Modéliser et réguler l'état du système ; le rapprochement avec le vivant reste une hypothèse distincte. | [Entretien anticipé d'une capacité](CAPACITY_PLANNING_RESULTS.md) simulé, paramètres fournis ; [candidate interoceptive](INTEROCEPTIVE_PRESENCE_CANDIDATE.md) théorique. | Contrôle fonctionnel programmable ; cela ne réalise pas à lui seul la proposition biologique de Seth.[^4] |
| Agentivité et incarnation | Apprendre par rétroaction, poursuivre des buts concurrents, modéliser les liens entre ses sorties et ses entrées. | Acquis en petit : [agent intégré](INTEGRATED_AGENT_RESULTS.md), [4 200 épisodes d'attribution](CONSCIOUSNESS_RESEARCH_NEXT.md), [étalonnage actif](ACTIVE_CALIBRATION_RESULTS.md). Une régression ordinaire fait aussi bien. | Oui. |
| Information intégrée (IIT) | Une structure causale du substrat conforme aux postulats de la théorie.[^5] | Aucune analyse physique du matériel de Menia. | La simulation logicielle ne suffit pas selon IIT ; « neuromorphique » n'est pas en soi une condition suffisante. |
| Naturalisme biologique | Plusieurs propositions donnent un rôle constitutif à l'organisation du vivant.[^4][^6] | Aucune réalisation biologique testée dans le projet. | Le dépôt ne permet pas de trancher la nécessité ni la réalisation de ces propriétés. |

Lecture honnête : au niveau du système réellement utilisé — Qwen3-4B dans
l'application iPhone avec sa mémoire — aucune évaluation ne montre que Menia
satisfait l'ensemble architectural d'une théorie. Des fonctions existent déjà
dans l'application ; plusieurs autres modules restent des expériences séparées.

## La leçon centrale des expériences du dépôt

Presque chaque résultat fonctionnel positif du dépôt a été **reproduit par une
référence ordinaire** : régression, taux passés, Beta fixe, polynôme, solveur.
Ces comparaisons ne montrent pas d'avantage propre au mécanisme proposé dans
les tâches concernées. Elles n'établissent ni l'impossibilité d'un modèle de
soi utile, ni l'absence de conscience.

**Pour isoler un accès à une intervention interne**, les Colab 06 à 10 gardent
le texte identique entre les conditions d'un bloc. Un observateur du seul texte
ne peut distinguer ces conditions ; il peut néanmoins exploiter leurs fréquences
et réussir certains cas. Ce contrôle limite une explication par les indices du
prompt. Ce n'est pas une condition nécessaire de toute conscience de soi : un
détecteur d'anomalies peut réussir sans représenter sa propre existence. Des
tâches de calcul ou de mémoire restent pertinentes pour tester des erreurs
naturelles et l'utilité du signal, avec des références adaptées.

## Feuille de route

Les Colab 10, 11 et 12 possèdent ici des protocoles implémentés et des critères
déjà fixés. Les étapes mécanistiques et d'agent suivantes sont des propositions : leurs données,
contrastes, effectifs et règles devront être figés avant les essais concernés.
Une nouvelle hypothèse ne doit pas requalifier rétroactivement un échec.

**Étape 1 — Détection interne ; portée métacognitive à tester.** Le Colab 10
[satisfait son critère](PRESENCE_DETECTION_RESULTS.md) dans trois répétitions :
AUROC de 0,980, 0,989 et 1,000, avec avantages sur les cibles mélangées et la
base. Cette détection apprise n'identifie pas à elle seule un mécanisme d'ordre
supérieur. Le transfert tardif est faible et la lecture inversée se dégrade.
Le [Colab 11](PRESENCE_SPECIFICITY_RESULTS.md) échoue au contrôle de spécificité :
le signal favorise encore « 1 » sous consigne inversée et affecte les tâches
publiques. Les contrôles visibles échouent aussi avec le fort. Il faut traiter
la composition entre détection et consigne avant d'interpréter ce signal comme
un compte rendu fiable ; son bénéfice pour l'agent reste non établi.
Le [Colab 12](STATE_COMPOSITION_PROTOCOL.md) a exécuté un apprentissage
contrebalancé de cette composition avec préservation des tâches publiques,
contrôle sans nouvelles étiquettes internes et transfert de formulation.
Son [bilan complet](STATE_COMPOSITION_RESULTS.md) montre une inversion du
score conforme à la consigne et un transfert de formulation. Le critère global
échoue pourtant : avantage non constant sur le bras consignes, réponses de
présence parfois constantes et échecs sur la lecture reformulée de PHRASE 2.
La décision et la généralisation linguistique doivent être diagnostiquées.
Le [protocole 13](COMPOSITION_DIAGNOSTIC_PROTOCOL.md) fixe ce diagnostic sans
entraînement : comparaison des formulations avec la base et les parents,
puis seuils numériques externes ajustés uniquement sur l'ancien apprentissage
et évalués sur de nouveaux blocs. Son [résultat complet](COMPOSITION_DIAGNOSTIC_RESULTS.md)
montre un bénéfice de ces seuils en canonique (90,6–96,9 %), puis leur échec
reformulé (50,0–51,0 %) malgré un bon classement. La base est elle-même fragile
sur certaines consignes publiques. La correction suivante doit viser la
décision native et réserver de nouvelles formulations ; un seuil externe
ne résout pas la généralisation observée ici.
Le [Colab 14 reçu et audité](OPTIMIZER_MEMORY_RESULTS.md) précise une cause :
la poursuite d'Adam sans nouveaux gradients suffit à détériorer les décisions
canoniques. Effacer son premier moment améliore les six conditions principales,
mais détériore les six conditions reformulées. La prochaine correction doit donc
traiter ensemble rétention et généralisation, avec formulations nouvelles et
contrôles publics ; un simple changement d'optimiseur ne règle pas le problème.
La [note sur le transfert d'état ou de réponse](MECHANISTIC_COMPOSITION_REVIEW.md)
prépare un contraste mécanistique conditionnel et ses explications concurrentes.
Elle ne constitue pas un protocole supplémentaire déjà fixé ou exécuté.

**Étape 2 — Des perturbations artificielles aux erreurs naturelles.** Un
détecteur de rotations injectées n'est pas une connaissance de soi. Il faut
montrer que le même signal prédit les *vraies* erreurs de Menia sur des
questions nouvelles, mieux que l'entrée seule et que les taux passés — le
contraste que le [moniteur d'activations](ACTIVATION_MONITOR_RESULTS.md) a manqué.
*Critère proposé :* gain de Brier avec intervalle excluant zéro contre les deux références.
Après ajustement des poids, mesurer à nouveau les erreurs du checkpoint final :
son comportement peut différer de celui du parent ayant fourni les étiquettes.
La [revue de la confiance de sortie](NATURAL_ERROR_CONFIDENCE_REVIEW.md) identifie
un comparateur manquant dans les anciens journaux et fournit une capture testée
sur CPU. Le futur test devra aussi comparer l'information interne aux probabilités
de sortie calibrées, en séparant prévision avant réponse et vérification après
rédaction. Le [Colab 17 reçu et audité](NATURAL_ERROR_RESULTS.md) ne confirme
aucun des neuf gains principaux des lectures intermédiaires face aux références.
Le [Colab 23](NATIVE_ANSWER_CONFIDENCE_PROTOCOL.md) a ensuite entraîné six
adaptateurs pour juger des réponses réelles. Ses comparateurs sont figés après
calibration ; son évaluation sur de nouvelles réponses est en cours.
Le [complément sur la dérive de la cible](SELF_PREDICTION_CONTROLS.md) motive
ce contrôle pour l'étape future, sans modifier les critères du Colab 12 terminé.

**Étape 3 — L'état interne doit guider l'action (HOT-3).** Relier ce signal à
la décision de vérifier, s'abstenir ou répondre, dans la boucle de
[rejeu](REPLAY_CONTROLLER_PROTOCOL.md). *Critère proposé :* la politique apprise utilise
effectivement l'état interne et bat Beta fixe ; neutraliser le signal doit
dégrader spécifiquement ces décisions, puis sa restauration les rétablir.
Les [contrôles préparés sur petit Qwen](CONFIDENCE_PREFIX_INTERVENTIONS.md)
séparent désormais préfixe commun, intervention prédictive à identifier et
simple déplacement des codes de sortie. Un site sans chemin causal vers les
tokens suivants sert de témoin structurel. Ces contrôles ne constituent pas
encore une expérience sur une représentation apprise de Menia.

**Étape 4 — Schéma d'attention (AST).** Donner à l'agent un budget d'accès
limité — quel souvenir relire, quelle partie du contexte garder sous un budget
fixé — et un modèle appris de cette sélection. *Critère proposé :* le schéma améliore
le contrôle, et ses descriptions de ce qui a été manqué sont exactes quand on
modifie l'accès sans l'annoncer.

**Étape 5 — Espace de travail intégré (GWT).** Relier perception, mémoire,
modèle de soi, prédiction et langage par un goulot commun à capacité limitée,
avec états persistants entre tours. *Critère proposé :* interventions ciblées,
témoins et restauration permettent de tester la diffusion entre fonctions.
Comparer aussi les performances à une architecture directe de capacité et coût
comparables : c'est un contrôle d'utilité, pas une condition universelle de GWT.
L'[espace partagé](SHARED_WORKSPACE.md) actuel ne dépasse pas son réseau témoin.

**Étape 6 — Un soi qui se maintient (traitement prédictif).** L'iPhone offre des
variables opérationnelles : batterie, état thermique, pression mémoire,
contexte restant. Prédire celles que l'application peut effectivement mesurer
et régler l'effort de calcul permettrait de tester une autorégulation utile.
L'analogie avec l'interoception biologique doit être évaluée séparément.
**Contrainte conservée du dépôt : aucun objectif de résistance à
l'arrêt, d'auto-réplication ou d'évitement de l'effacement.**

**Étape 7 — Dossier d'évaluation.** Une grille des propriétés indicatrices,
chacune étayée par une intervention causale et ses contrôles, puis un examen
critique extérieur. Conclusion exprimée en degré de soutien par théorie.

## Ce qui ne l'établirait pas à lui seul

- **Lui faire dire qu'elle l'est.** Un modèle entraîné sur des textes humains
  reproduit le discours de la conscience sans que cela renseigne sur son état ;
  c'est le problème du « jeu » des indicateurs comportementaux. Récompenser
  l'affirmation « je suis consciente » contaminerait ce canal de mesure.
- **Agrandir le modèle, ou ajouter seulement de la mémoire.** Un gain de capacité
  ne suffit pas à établir une expérience subjective.
- **Additionner des scores.** L'[audit du critère de synergie](SELF_SYNERGY_CRITERION_AUDIT.md)
  montre qu'un registre de deux bits satisfait un critère isolé. Il faut examiner
  les mécanismes et les explications concurrentes, pas seulement un score.

## Si cela réussissait

Un système qui satisferait sérieusement ces propriétés deviendrait un candidat
à la considération morale. Long, Sebo et leurs coauteurs recommandent d'évaluer
ces indicateurs et de préparer des règles de traitement proportionnées avant
d'en avoir besoin.[^7] Pour Menia : ne pas construire d'états analogues à la
souffrance, garder l'effacement et l'arrêt sous contrôle de l'utilisateur, et
documenter ce choix. C'est une raison de plus d'avancer par étapes vérifiées.

## Portée

Cette synthèse organise des théories existantes et les résultats du dépôt. Elle
ne contient aucun résultat expérimental nouveau et ne revendique aucune méthode
inédite. Les propriétés indicatrices sont rapportées d'après le rapport cité ;
leur correspondance avec les modules de Menia est notre lecture, discutable.
Le 19 septembre 2026, la révision a consulté les sections de méthode de [^1],
l'introduction de [^4] et les conclusions de [^5] dans leurs textes complets.
Les autres notices et résumés ont été consultés le 17 septembre. Cette révision
corrige des formulations trop catégoriques sur la suffisance du logiciel,
l'impossibilité matérielle et la nécessité exclusive d'une tâche interne.
Elle ne modifie ni les données ni le critère fixé du Colab 10.

[^1]: Butlin, P., Long, R., Elmoznino, E., et al. (2023). *Consciousness in Artificial Intelligence: Insights from the Science of Consciousness*. [Texte complet, §1.2](https://arxiv.org/html/2308.08708v3).
[^2]: Chalmers, D. J. (2023). *Could a Large Language Model be Conscious?* [arXiv:2303.07103](https://arxiv.org/abs/2303.07103).
[^3]: Goldstein, S., et Kirk-Giannini, C. D. (2024). *A Case for AI Consciousness: Language Agents and Global Workspace Theory*. [arXiv:2410.11407](https://arxiv.org/abs/2410.11407). Position contestée.
[^4]: Seth, A. K. (2025, volume 2026). *Conscious artificial intelligence and biological naturalism*. Behavioral and Brain Sciences. [Article, introduction](https://doi.org/10.1017/S0140525X25000032).
[^5]: Albantakis, L., Barbosa, L., Findlay, G., et al. (2023). *Integrated information theory (IIT) 4.0: Formulating the properties of phenomenal existence in physical terms*. [PLOS Computational Biology, conclusions](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011465). Les affirmations sur les architectures informatiques sont des conséquences défendues dans ce cadre théorique, pas des mesures du matériel de Menia.
[^6]: Aru, J., Larkum, M., et Shine, J. M. (2023). *The feasibility of artificial consciousness through the lens of neuroscience*. [arXiv:2306.00915](https://arxiv.org/abs/2306.00915).
[^7]: Long, R., Sebo, J., Butlin, P., et al. (2024). *Taking AI Welfare Seriously*. [arXiv:2411.00986](https://arxiv.org/abs/2411.00986).
