# Protocole exploratoire : découvrir le contrôle et ses limites

Version fixée avant exécution des expériences de ce module. Ce protocole concerne
une condition fonctionnelle candidate du modèle de soi. Il ne définit aucun test
de conscience subjective et ne garantit aucune nouveauté scientifique.

## Question

Un agent peut-il choisir une action et une observation pour distinguer son influence
de celle d'une cause extérieure corrélée, et s'abstenir lorsqu'une identité unique
n'est pas identifiable ? Le modèle précédent de Menia recevait une position déjà
identifiée comme la sienne. Ici les deux canaux sont anonymes et leurs causes cachées.

## Monde et informations

Un indice public U vaut -1 ou +1. L'action A vaut -1 ou +1. Deux canaux binaires
dépendent soit de A, soit de U, avec un signe propre à chaque canal. Le moteur
expérimental conserve les causes réelles ; le modèle reçoit seulement U, A et
la valeur du canal sélectionné, ou une réponse absente. Aucun identifiant de
cause n'est communiqué au modèle. Les actions n'ont pas de coût ni d'effet cumulatif.

Le modèle compare, pour chaque canal, six hypothèses équiprobables : +A, -A, +U,
-U, constante +1, constante -1. L'erreur de lecture indépendante a une probabilité
connue de 0,05. Ce choix constitue une famille générative fournie par la conception,
pas une découverte de nouvelles variables ou une introspection neuronale.

Une phase initiale de 24 cycles impose A=U, avec lecture alternée des deux canaux.
Elle rend les hypothèses action/indice observationnellement indiscernables. Après
cette phase, le budget est de 12 cycles. Les variantes partagent le même inféreur :
active choisit le couple action/canal maximisant la diminution attendue d'entropie,
pondérée par la probabilité estimée de recevoir une lecture ; aléatoire choisit les
deux uniformément ; passive continue A=U et alterne les canaux.

La fiabilité de réception utilise un a priori Beta(1,1). Une lecture manquante ne
modifie pas les hypothèses sur les causes. Une réponse erronée n'est pas confondue
avec une absence de lecture. La probabilité 0,05 d'erreur de contenu est fournie,
alors que la probabilité de réception est estimée. Les rapports gardent cette distinction.

## Conditions et critères

Six conditions : un canal contrôlé ; même cas avec 50 % de lectures manquantes ;
canal contrôlé toujours masqué ; aucun canal contrôlé ; deux canaux suivant tous
deux l'action (miroir parfait) ; même miroir avec 50 % de lectures manquantes.
Les canaux, signes et indices sont randomisés. 100 graines de test, 1000–1099,
par condition et variante, soit 1 800 épisodes. Les graines 0–9 sont réservées au
développement. Les premières sorties de test sont exploratoires, sans réglage
ultérieur sur ces mêmes graines présenté comme confirmation indépendante.

La classification d'un canal est « contrôlable » si P(dépend de A)>0,95,
« externe » si cette probabilité est <0,05, sinon « indéterminé ». Une identité
unique n'est attribuée que si un canal est contrôlable et l'autre externe.
La dépendance à l'action ne distingue jamais un corps d'une copie parfaite répondant
aux mêmes actions ; les deux sont alors contrôlables au sens de ce monde.

Mesures : classement exact des deux canaux, attribution unique correcte ou fausse,
abstention, score de Brier des probabilités de contrôle, lectures reçues, proportion
d'actions A≠U. Les moyennes et comptes par condition sont publiés avec chaque essai.
Aucun regroupement ne devient un score de conscience. La comparaison active/aléatoire
teste le choix expérimental, celle active/passive la sortie d'une politique confondue.

## Contrôles de validité

Un calcul exhaustif vérifie deux mondes : Y=A et Y=U. Sous A=U leurs observations
sont identiques. Un prédicteur utilisant A peut être parfait dans les deux, tandis
qu'un prédicteur privé de A ou recevant A=0 se dégrade dans les deux. Sous une vraie
intervention A=-U, les mondes divergent. Cette construction distingue une lésion
du prédicteur d'une intervention sur le processus générateur de l'observation.

Un autre contrôle échange les étiquettes cachées de deux copies parfaites : toutes
les observations restent identiques pour toute séquence d'actions. Une méthode
qui distingue ces étiquettes exploite une fuite d'information ou une convention.
Tests supplémentaires : aucune mise à jour causale sans réponse, décision avant
observation, canaux permutés, choix modifié par une modification du posterior.

## Portée et suite

L'inférence bayésienne exacte sert de référence analytique, pas d'architecture
nouvelle de conscience. Une extension neuronale devra apprendre les représentations,
généraliser à des délais et mécanismes non fournis, et contrôler ses échanges
entre attention, mémoire, prédiction et action. La proposition plus large vise une
attribution de soi contrainte par ce que le système peut réellement identifier.
Sa nouveauté éventuelle et sa relation avec l'expérience subjective restent à établir.
