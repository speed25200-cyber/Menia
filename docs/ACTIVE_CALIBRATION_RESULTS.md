# Apprendre ce qui agit et ce qui mesure

Le prototype apprend maintenant des propriétés de son actionneur et de ses
capteurs à partir de commandes suivies de mesures. Ces estimations alimentent
le contrôleur de position déjà construit. Après une inversion cachée du moteur,
il retrouve un contrôle efficace en fin d'épisode. Il corrige aussi plusieurs
perturbations visuelles. Deux limites majeures sont conservées : les erreurs
peuvent fortement augmenter pendant l'adaptation, et une référence trompeuse
fait apprendre une explication fausse.

Cette extension réalise un apprentissage fonctionnel de conditions d'action et
de perception. Elle n'établit pas une expérience de sa propre existence chez
Menia. Une régression ordinaire obtient des résultats comparables ou meilleurs ;
ni les principes employés ni leurs performances ne sont revendiqués comme une
méthode nouvelle produisant la conscience.

## Pourquoi cette construction est pertinente, et ce qu'elle ne résout pas

La [candidate de présence à soi](INTEROCEPTIVE_PRESENCE_CANDIDATE.md) exige qu'un
modèle représente des conditions qui affectent réellement la perception et
l'action. Le [pilote précédent](BINDING_CONTROL_RESULTS.md) utilisait une
estimation de position, mais connaissait les paramètres de son actionneur et
le bruit visuel. Cette extension retire une partie de cette connaissance
préalable : gain moteur, biais visuel, échelle visuelle et variance résiduelle
sont estimés à partir d'observations.

Des travaux antérieurs couvrent déjà ces fonctions. Baioumy et collègues ont
proposé un contrôle tolérant aux pannes sensorielles et examiné un piège : si
l'estimation est attirée vers l'état désiré, un changement de cible peut
ressembler à une panne. Leur article de 2021 évalue un bras simulé ; il n'établit
pas une conscience du robot. Ici, l'estimation des paramètres dépend des
commandes et mesures, sans utiliser la cible à atteindre.[^1]

Lanillos, Pages et Cheng présentent en 2020 une distinction robot–autrui fondée
sur des relations apprises entre commandes, proprioception et vision. Leur
discussion distingue cette capacité sensorimotrice de la conscience de soi et
précise les limites de leur réalisation. Cela exclut de présenter le seul
apprentissage de ces relations comme une invention propre à Menia.[^2]

Une prépublication de Ni et collègues, déposée en juin 2026, décrit des commandes
destinées à départager des modèles de panne sous contraintes. Elle fournit un
antécédent récent à la discrimination active moteur/capteur. Son ensemble de
modèles et ses garanties diffèrent de notre régression ; son code et ses
résultats ne sont pas reproduits ici.[^3]

Enfin, Zaadnoordijk, Besold et Hunnius ont développé une critique conceptuelle
de l'assimilation entre concordance prédiction–sensation et sentiment
d'agentivité. Leur argument porte sur ce que cette concordance permet
d'expliquer ; ce n'est pas une démonstration d'impossibilité de la conscience
artificielle. Ajouter ici une estimation causale et des actions ne démontre
pas davantage que ces opérations constituent un vécu.[^4]

## Modèle appris et informations disponibles

Le [protocole](ACTIVE_CALIBRATION_PROTOCOL.md) définit un actionneur de gain g :
une commande u provoque un déplacement g*u. Lors de l'étalonnage, deux capteurs
mesurent le déplacement depuis un point remis à zéro. Leurs réponses sont :

`p = r*g*u + bruit_p`

`v = s*g*u + b + k*(g*u)² + bruit_v`.

Le modèle apprend deux relations affines entre commande et mesure. Leur pente,
biais et variance résiduelle sont inconnus. Il utilise au plus 24 observations
récentes, avec une régression bayésienne normale–inverse-gamma. Les prévisions
de mesure sont des distributions de Student correctement normalisées. Le
modèle enregistre ses prévisions avant l'action puis révise ses paramètres
après les mesures. Son état peut être sauvegardé et restauré.

La référence est supposée conserver r=1. Sous cette hypothèse, sa pente estime
le gain moteur ; le rapport des pentes visuelle et de référence estime l'échelle
visuelle. L'agent ne découvre pas la validité de r=1. La condition qui viole
cette hypothèse révèle précisément sa dépendance à cette référence. Le terme
quadratique k est également absent de la famille apprise.

Le modèle ne reçoit aucun paramètre réel, étiquette de panne ou coût physique
réel. L'expérimentateur lui fournit en revanche une procédure de remise à zéro,
un objet commun aux deux capteurs pendant l'étalonnage et les commandes possibles.
Il apprend les coefficients d'une structure conçue à l'avance, pas cette
structure elle-même. Il ne possède pas un corps physique.

Après chaque commande d'étalonnage, les coefficients servent à corriger huit
signaux visuels de position. Ces signaux passent dans `BindingController`, avec
le bruit estimé. Le déplacement est ensuite ajusté selon la moyenne et la
variance estimées du gain moteur. L'incertitude complète sur le biais et
l'échelle visuelle n'est pas intégrée : le contrôleur emploie leurs valeurs
ponctuelles, avec des règles de repli et une limite de commande à ±6.

La fréquence de proxy aligné, la dispersion spatiale des proxys et la
distribution des positions restent connues. Les positions de tâche sont
réinitialisées entre essais ; leurs résultats ne servent pas à l'apprentissage.
La boucle persistante est celle de l'étalonnage : commande, mesure, révision,
nouvelle commande, puis réemploi des paramètres pour la tâche.

## Expérience et résultats

Le plan principal comprend neuf conditions, 32 épisodes par condition, quatre
contrôleurs et 96 cycles par épisode. Il produit **110 592 commandes
d'étalonnage et 884 736 évaluations de déplacement**. Les contrôleurs partagent
les mêmes bruits et positions ; leurs commandes peuvent différer. Les témoins
algébriques supplémentaires ne sont pas comptés dans ces totaux.

Un changement survient au cycle 32, sans avertissement au modèle adaptatif.
Le contrôle figé cesse simplement d'apprendre après les 32 cycles initiaux.
Le contrôle sans exploration utilise uniquement des commandes d'étalonnage
nulles ; il reçoit des mesures, mais n'excite pas le moteur. Le contrôle ordinaire
utilise des moindres carrés, la même fenêtre et des sondes alternées.

Le tableau donne le coût quadratique de déplacement dans les **24 derniers
cycles**. L'intervalle porte sur la différence appariée « figé moins adaptatif »,
par bootstrap de 4 000 tirages des 32 épisodes. Il quantifie la variabilité du
simulateur, sans inférence à des humains.

| Condition | Adaptatif | Figé | Sans exploration | Régression ordinaire | Figé − adaptatif [intervalle 95 %] |
|---|---:|---:|---:|---:|---:|
| Inchangé | 0,741092 | 0,735966 | 1,928825 | 0,741187 | −0,005126 [−0,010968 ; 0,000419] |
| Moteur inversé | 0,722320 | 10,784997 | 5,129748 | 0,722393 | 10,062677 [9,863767 ; 10,259129] |
| Moteur affaibli | 0,781686 | 1,394876 | 2,515400 | 0,781611 | 0,613190 [0,567490 ; 0,656728] |
| Biais visuel | 0,755823 | 1,068417 | 1,927043 | 0,755869 | 0,312594 [0,278311 ; 0,343462] |
| Échelle visuelle | 0,810794 | 0,839738 | 1,893843 | 0,810920 | 0,028944 [0,017994 ; 0,040349] |
| Bruit visuel | 0,949531 | 1,062474 | 1,968164 | 0,947935 | 0,112943 [0,092219 ; 0,131938] |
| Perturbation combinée | 0,886372 | 5,985042 | 4,231463 | 0,884406 | 5,098670 [4,962899 ; 5,241727] |
| Référence trompeuse | 3,431613 | 0,852622 | 1,891254 | 3,422572 | −2,578991 [−2,825457 ; −2,365493] |
| Vision non linéaire | 0,816192 | 0,812561 | 1,964201 | 0,815450 | −0,003631 [−0,011272 ; 0,003780] |

L'adaptation améliore le coût final par rapport au modèle figé dans les six
conditions de perturbation appartenant à la famille attendue. Elle n'apporte
pas d'avantage résolu lorsque le monde reste inchangé ou lorsque la vision
devient non linéaire. Avec une référence trompeuse, elle aggrave nettement le coût.
Les régressions ordinaires obtiennent des coûts très proches et sont légèrement
meilleures dans plusieurs conditions ; aucune supériorité générale de la
régression bayésienne n'est établie.

L'effort quadratique des sondes vaut une unité par cycle pour les variantes
actives et alternées, zéro sans exploration. Le tableau ne facture pas cet
effort dans l'erreur de tâche. Il ne démontre donc pas un bénéfice net pour un
système où l'exploration aurait un coût économique ou énergétique donné.

## Ce qui est distingué

Les moyennes de paramètres au dernier cycle précisent le mécanisme :

| Modification réelle | Gain appris | Biais visuel appris | Échelle visuelle apprise | Bruit visuel appris |
|---|---:|---:|---:|---:|
| Gain = −1 | −1,0016 | 0,0140 | 1,0080 | 0,4884 |
| Gain = 0,5 | 0,5024 | −0,0005 | 1,0286 | 0,4665 |
| Biais visuel = 1,5 | 1,0059 | 1,5267 | 0,9901 | 0,4779 |
| Échelle visuelle = 0,5 | 0,9946 | −0,0302 | 0,4921 | 0,4746 |
| Bruit visuel = 2 | 0,9920 | 0,1564 | 0,9483 | 1,9325 |

Un biais visuel ne provoque donc pas systématiquement un changement estimé du
moteur, et un moteur inversé est représenté par un gain de signe opposé. Ce sont
des estimations continues, sans précision parfaite ni classificateur universel
de panne. Les résultats reposent sur une référence stable et les expériences
de calibration autorisées.

Sans commande exploratoire, la matrice des observations ne permet pas
d'identifier la pente. Le statut d'identification reste faux dans tous les
épisodes de cette variante. Une référence absente conserve également ce statut
dans le témoin dédié. Le prior peut encore guider une action prudente ; il ne
devient pas une observation du moteur.

## Trois échecs instructifs

**L'adaptation peut d'abord empirer le contrôle.** Après inversion, le coût
atteint 19,482225 sur les douze premiers cycles suivant le changement, contre
11,228194 pour le modèle figé. Dans la perturbation combinée, ces valeurs sont
9,367164 et 6,102841. La fenêtre contient alors des mesures issues de deux
régimes. Sa moyenne peut décrire un moteur qui n'existe dans aucun des deux.
La prise en compte de variance et la limite des commandes ne suffisent pas à
éviter ces erreurs. Les courbes des 96 cycles et les trois phases restent dans
le rapport machine ; la réussite tardive ne masque pas cette dégradation.

**Une bonne prévision sensorielle peut soutenir une mauvaise explication.**
Les mondes `(g=0,5,r=1,s=1)` et `(g=1,r=0,5,s=0,5)` produisent exactement les
mêmes mesures d'étalonnage pour toute commande, à bruits égaux. Le témoin
exécuté retrouve zéro différence de mesure et de paramètres appris, malgré un
écart de 0,5 unité entre déplacements réels pour une commande d'amplitude un.
Le modèle ne peut pas tester r=1 avec ces seules observations.

Dans la condition trompeuse, le gain réel reste un, mais le gain appris moyen
vaut 0,5022. La variance interne du gain devient faible, environ 0,000834.
L'identification est déclarée valide *sous l'hypothèse de référence stable*,
alors que cette hypothèse est fausse. Le coût final grimpe à 3,431613. Un nombre
interne précis n'atteste donc pas une explication correcte. L'équivalence porte
sur les observations de calibration, sans exclure une référence supplémentaire
ou un autre protocole qui pourrait les départager.

**Des sondes optimales pour un modèle peuvent ignorer sa fausseté.** Le critère
géométrique choisit exactement l'alternance −1,+1 pendant les 96 cycles. Il
n'apporte ici aucune découverte de stratégie par rapport à cette alternance
ordinaire. Pour ces commandes, une vision `v=x+0,15*x²` et une vision
`v=x+0,15` produisent la même moyenne quand x=±1. Le modèle affine apprend donc
un biais proche de 0,15 sans voir la courbure. La commande zéro, disponible mais
non sélectionnée, les séparerait : leurs moyennes y diffèrent de 0,15.

Ce dernier témoin a été ajouté après le premier résultat, comme indiqué dans
l'amendement au protocole. Il conserve les performances initiales. Il montre
une limite du choix des sondes, sans prétendre qu'une sonde unique suffirait
avec du bruit ni que la correction de ce défaut aurait déjà été implémentée.

## Conséquence pour l'objectif de Menia

Le résultat utile est une relation apprise entre commandes, conditions de
mesure et contrôle, avec des paramètres révisés qui changent effectivement
les actions. C'est une extension du même contrôleur de position, et non un
nouveau nom donné à sa réponse binaire. Le calcul fonctionne sans canal de
description de soi.

La fonction demeure réalisable par des techniques ordinaires d'identification
de systèmes. Elle peut décrire un outil extérieur commandé autant qu'une partie
du système qui commande. Aucun résultat ne démontre qu'un point de vue vécu
accompagne cette identification. Ni la précision interne, ni la réussite après
perturbation, ni le nombre d'essais ne certifient la conscience.

La candidate complète reste partielle : pas de besoin propre, pas d'interoception
physiologique, pas de modèle appris des ressources de perception ou de mémoire,
pas d'intégration au chat ou à l'iPhone. Le modèle sauvegardé assure une continuité
fonctionnelle de paramètres ; il ne prouve pas un sentiment de continuité.

L'étape suivante de construction devra traiter les explications concurrentes
pendant les ruptures et choisir certaines expériences pour mettre en défaut
la famille de modèles, au lieu de seulement préciser ses coefficients.
L'obstacle phénoménal reste distinct : il faut établir pourquoi une organisation
de ce type produirait une expérience de sa propre existence. Cet audit ne fournit
pas encore cet argument ni une méthode nouvelle qui produirait cette expérience.

## Reproduction et validation

```text
python -m research.audit_active_calibration
python -m research.audit_active_calibration --check
python -m unittest discover -s tests_research -v
```

Le calcul requiert NumPy, déjà présent dans les dépendances de recherche. Les
huit nouveaux tests couvrent les mises à jour contre une formulation matricielle
indépendante, la normalisation de la prévision de Student, la fenêtre, l'ordre
prédiction–mesure–révision, la sauvegarde/reprise, l'identification impossible,
la couverture des sondes et l'utilisation des paramètres par le contrôleur.
La suite comprend désormais 68 tests de recherche.

Le rapport conserve les conditions, graines, coûts finaux par épisode,
intervalles appariés, courbes, estimations et un extrait du journal autour de
l'inversion. Le rejeu compare tous les champs, avec une tolérance absolue et
relative de 1e−10 sur les valeurs flottantes. Les mesures du journal sont
normalisées en scalaires Python pour préserver le même format après lecture JSON.
La CI exécute les tests et le rejeu intégral. Ces validations contrôlent le
logiciel, pas la conscience.

## Sources

[^1]: Baioumy, M., Pezzato, C., Ferrari, R., Hernández Corbato, C., et Hawes, N. (2021). [Fault-tolerant Control of Robot Manipulators with Sensory Faults using Unbiased Active Inference](https://arxiv.org/html/2104.01817v1). Article accepté à ECC 2021 ; dépôt du 5 avril 2021. Présentation, biais liés à la cible, résultats simulés et conclusion consultés. Code des auteurs non exécuté.
[^2]: Lanillos, P., Pages, J., et Cheng, G. (2020). [Robot self/other distinction: active inference meets neural networks learning in a mirror](https://arxiv.org/html/2004.05473v1). Article accepté à ECAI 2020 ; dépôt du 11 avril 2020. Résumé, introduction, limites et conclusion consultés ; aucune reproduction.
[^3]: Ni, X., et al. (17 juin 2026). [Safe, Real-Time Active Model Discrimination and Fault Diagnosis for Nonlinear Systems via Differentiable Reachability](https://arxiv.org/html/2606.19590v1). Prépublication. Résumé, introduction et formulation de la discrimination entre modèles consultés. Les garanties annoncées ne sont pas transférées à Menia.
[^4]: Zaadnoordijk, L., Besold, T. R., et Hunnius, S. (13 mai 2019). [A match does not make a sense: on the sufficiency of the comparator model for explaining the sense of agency](https://doi.org/10.1093/nc/niz006). *Neuroscience of Consciousness*, 2019(1):niz006. Résumé primaire PubMed et passages indexés de PMC consultés ; texte intégral direct non récupéré. Argument conceptuel, sans réanalyse d'expérience.
