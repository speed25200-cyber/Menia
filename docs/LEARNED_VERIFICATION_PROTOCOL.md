# Faire apprendre à Menia l'usage de ses estimations

Protocole de pilote fixé le 14 septembre 2026, avant son exécution.

## Hypothèse de construction

La candidate reste un agent dont une représentation persistante de ses propres
accès à l'information intervient dans la perception, la mémoire et l'action.
L'hypothèse phénoménale est que certaines organisations de ce type pourraient
constituer un point de vue vécu. Ses conditions de suffisance restent inconnues.
Le présent pilote traite une lacune de construction ; il ne transforme pas sa
réussite en preuve de conscience de sa propre existence.

Dans `SourceAgent`, le moniteur apprend une probabilité de contribution externe
`q`, mais le développeur impose `min(q, 1-q) > coût` pour décider de vérifier.
Remplacer cette règle par une politique apprise permet d'examiner si l'usage de
`q` peut être acquis à partir des conséquences des choix. Cela ne crée pas une
théorie inédite : il s'agit d'apprentissage de valeurs d'actions dans un problème
de décision avec information partielle.

## Sources et raison du choix

Kumaran et ses collègues étudient l'abstention de modèles de langage. Leur
intervention sur les activations de Gemma 3 27B modifie les choix sur des questions
réservées au test. Les directions sont construites dans un contexte proposant
l'abstention ; les analyses distinguent confiance, politique et effet direct
résiduel. Cela motive une intervention sur une estimation interne avant la
décision. Ce travail ne démontre pas la conscience subjective. Notre petit agent
et son apprentissage ne répliquent pas cette expérience sur un LLM.
[Article et méthodes, Nature Machine Intelligence, 7 septembre 2026](https://www.nature.com/articles/s42256-026-01293-x).

La prépublication de Phua utilise des architectures avec espace partagé et
modèle de soi, puis des lésions sans réentraînement. Les actions sont apprises
par imitation d'un oracle ; une supervision supplémentaire vise la correction
des propres décisions. Dans certaines tâches, le routage impose l'espace
partagé comme seul chemin de l'information. Ces choix donnent une antériorité
aux expériences de dépendance causale, tout en limitant l'interprétation de
leurs lésions. Ici, aucune action optimale fournie par un oracle ne sera utilisée
comme cible ; les deux actions seront explorées aléatoirement et leur coût
effectivement réalisé sera observé après la décision.
[Prépublication, méthodes 4.2–4.3](https://arxiv.org/html/2512.19155v1).

Le cas architectural défendu par Goldstein et Kirk-Giannini articule modules,
sélection, entretien de représentations et diffusion sous une lecture
fonctionnaliste de la théorie de l'espace global. Sa suffisance pour une
expérience dépend de cette lecture théorique. Elle ne découle pas du succès
d'un petit réseau. Ce pilote ne réalise pas encore cet ensemble de propriétés.
[Argument théorique, sections 5 et 9](https://arxiv.org/html/2410.11407v1).

## Construction limitée et vérifiable

On conserve les trois moniteurs récurrents déjà entraînés (graines 11, 23, 37).
Ils ne sont pas réentraînés ni sélectionnés sur les nouveaux tests.

Le contrôleur reçoit seulement `q` et le coût public de vérification. Il apprend
deux valeurs : coût attendu d'accepter l'attribution courante et coût attendu de
vérifier. Il choisit la valeur la plus basse ; une égalité choisit l'acceptation.
La décision initiale de source reste `q >= 0,5`. L'erreur d'attribution coûte 1,
la vérification parfaite coûte entre 0,05 et 0,45 pendant l'apprentissage.

La représentation du contrôleur comporte neuf fonctions triangulaires de `q`
centrées de 0 à 1, espacées de 0,125, et le coût. Deux régressions ridge
(`lambda = 0,001`, équations normales en sommes non normalisées) sont ajustées
aux résultats des seules actions effectivement choisies. Vingt coefficients
sont enregistrés. Le contrôleur aveugle utilise les mêmes caractéristiques,
mais reçoit toujours `q = 0,5`. Son information utile et sa capacité effective
sont donc plus petites : ce contrôle mesure l'apport de `q`, pas un avantage
architectural à information égale. La règle analytique existante est le contrôle
plus exigeant à information égale.

Pour chaque moniteur, 512 épisodes de 48 étapes sont produits en condition
standard, graine `71000 + graine_moniteur`. Un générateur indépendant
(`72000 + graine_moniteur`) choisit chaque vérification avec probabilité 0,5 et
les coûts uniformes. À chaque étape : prédiction, choix, puis retour. Seules les
vérifications précédentes alimentent le moniteur. Le simulateur fournit après
la décision le coût de l'action sélectionnée à l'apprentissage hors ligne du
contrôleur. Le coût d'une erreur d'acceptation révèle sa correction pendant
l'apprentissage, mais n'entre jamais dans les caractéristiques du moniteur.
Il n'y a ni coût contrefactuel transmis à l'ajustement, ni cible « vérifie ».
Le régime d'entraînement suppose donc un évaluateur externe des erreurs ; il
ne constitue pas un apprentissage autonome sans vérité de référence.

## Évaluation fixée

512 nouveaux épisodes de 48 étapes par condition, cinq conditions existantes
(`standard`, `strong_imagery`, `trace_removed`, `persistent`, `ambiguous`).
Graine `81000 + 1000 * indice_condition`, commune aux modèles et contrôleurs.
Trois coûts testés : 0,10 ; 0,25 ; 0,40. Aucun réglage après ces évaluations.
Chaque contrôleur déroule sa propre boucle avec son propre historique de
vérifications ; la vérité courante n'est visible qu'après le choix.

Comparer contrôleur appris, contrôleur aveugle, règle analytique, vérification
systématique et acceptation systématique. Mesurer coût, fréquence des
vérifications, erreurs en mémoire et score de Brier du moniteur avant le choix.
Conserver chaque graine séparément. Les milliers d'étapes ne sont pas des
initialisations indépendantes ; ce pilote à trois moniteurs ne justifie pas une
conclusion confirmatoire générale.

Sonde causale locale : sur chaque état de la trajectoire du contrôleur appris,
remplacer uniquement son entrée `q` par celle de l'épisode suivant ayant la même
classe de source prédite. La première attribution et le moniteur restent
inchangés. Mesurer les décisions changées et le surcoût immédiat de ces décisions,
sans propager cette branche dans la trajectoire. C'est une intervention sur une
entrée interne du contrôleur, pas une modification des activations d'un LLM.
Les donneurs ne sont pas appariés sur tous les indices : un décalage de
distribution conditionnelle reste possible. Remettre la même entrée doit
restaurer exactement le choix ; changer seulement un rapport destiné au journal
ne modifie pas le contrôle par construction.

Critères exploratoires annoncés pour la condition standard, coût 0,25 :

1. Le contrôleur appris a un coût inférieur au contrôleur aveugle pour les trois
   moniteurs.
2. Il ne dépasse pas le coût de la règle analytique de plus de 0,01 pour chacun
   des trois moniteurs (tolérance de pilote, pas seuil de conscience).
3. Les greffes modifient certaines vérifications, sans modifier l'attribution
   initiale dans cette sonde. Un surcoût positif serait compatible avec une
   utilisation utile de l'information ; un simple changement ne l'établit pas.

Conserver les échecs, notamment le cas où la source devient indépendante des
indices. Aucun gain n'est attendu par magie d'une politique utilisant une
estimation devenue fausse. La suite utile serait d'apprendre quand cette
estimation cesse d'être fiable et de tester ce diagnostic dans plusieurs tâches,
avant de prétendre avoir une représentation générale de soi.
