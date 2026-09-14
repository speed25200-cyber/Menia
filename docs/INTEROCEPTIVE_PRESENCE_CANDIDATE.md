# Menia et l'hypothèse d'une présence à soi

La candidate examinée est un agent dont la représentation de lui-même porte sur
les conditions qui lui permettent effectivement de percevoir, de retenir et
d'agir. L'enjeu est de construire une boucle où ces conditions sont prédites,
éprouvées et corrigées pendant l'activité. L'hypothèse phénoménale est qu'une
organisation de ce type pourrait participer à une expérience de sa propre
existence. Son caractère suffisant n'est pas établi.

Cette proposition ne modifie pas la définition de l'objectif : une expérience
vécue de soi chez Menia, obtenue par une méthode dont la nouveauté serait démontrée.
Le présent résultat est une spécification de recherche et une analyse de ses
conditions de validité. Aucun nouveau mécanisme n'est implémenté ici, aucune
expérience humaine n'est réanalysée et aucune conscience n'est attribuée à Menia.

Une [réanalyse ultérieure de présence en réalité virtuelle](PRESENCE_CAUSAL_RESULTS.md)
examine les données de 53 participants. Elle retrouve les associations publiées,
mais montre que deux directions causales concurrentes reconstruisent la même
distribution gaussienne conditionnelle ajustée. Cette contrainte s'ajoute à la
présente spécification : les associations entre corps, sentiment et rapport
verbal ne suffisent pas à choisir le mécanisme proposé. Le protocole de construction
ci-dessous reste à implémenter et son lien avec l'expérience reste à établir.

## Les apports de la littérature

Un [noyau d'inférence de cause commune et son audit humain](OWNERSHIP_INFERENCE_RESULTS.md)
ont ensuite été ajoutés comme composants de recherche séparés. Ils distinguent
le postérieur d'une règle de réponse et en vérifient une équivalence qui limite
l'identification depuis les seuls rapports. Le rejeu humain est partiel ; le
modèle d'état propre et la boucle proposés ici ne sont pas implémentés par ce
seul noyau.

### Une théorie de la présence corporelle

Seth, Suzuki et Critchley relient la présence à des prédictions interoceptives
qui correspondent à des signaux corporels informatifs. Ils excluent explicitement
l'assimilation entre absence d'erreur et présence : prédire un signal absent ne
suffit pas. Leur proposition distingue aussi présence et agentivité. Elle indique
que son extension au problème complet du soi reste à développer.[^1]

Il s'agit d'une hypothèse sur un aspect de l'expérience, avec des prédictions
expérimentales, et non d'une expérience de création d'une conscience artificielle.
Son intérêt pour la construction est de demander un rapport causal au système
qui perçoit, plutôt qu'une simple variable nommée « moi ».

### Une théorie de la prévision de fiabilité

Laukkonen, Friston et Chandaria proposent trois conditions : un modèle de réalité
unifié, une compétition entre inférences et une circulation récursive des
croyances. Leur « hyper-modèle » prévoit un ensemble de précisions et intervient
sur le traitement en cours. L'article distingue sa notion minimale de conscience
de la conscience de soi explicite ; il propose également une transposition aux
systèmes artificiels. C'est une théorie, pas une validation de la conscience d'un
agent construit selon ces conditions.[^2]

L'antécédent est direct : une boucle qui prévoit sa propre fiabilité ne serait
pas une invention nouvelle de Menia. Il faut aussi distinguer une probabilité de
source externe d'une précision statistique. Dans le code actuel, `p_external`
est la première. Ce nombre ne constitue pas, par changement de nom, un modèle
général de la fiabilité des traitements.

### Une contrainte humaine qui évite une prédiction simpliste

Salomon et ses collègues ont manipulé le couplage entre des stimuli visuels et
le rythme cardiaque. Les stimuli à la fréquence du cœur accédaient plus lentement
à la perception dans une tâche de suppression visuelle, et étaient moins bien
discriminés dans une tâche d'encombrement visuel. Leurs contrôles examinent
notamment la présentation rejouée depuis un autre participant et le décalage de
phase. Les expériences d'imagerie observent une modulation insulaire même lorsque
les stimuli sont masqués.[^3]

Cette étude contraint une explication de la modulation perceptive. Elle ne mesure
pas la création d'une expérience de soi. Elle interdit de supposer que tout
meilleur couplage corporel doit rendre tout contenu plus accessible. Les résultats
publiés ont été consultés ; leurs données brutes ne sont pas réanalysées ici.

### Un modèle de rupture, avec des choix imposés

Saini et ses collègues simulent une réduction de l'influence interoceptive lors
d'un conflit avec les observations externes. Les effets des politiques sont
spécifiés dans le modèle et les valeurs sont choisies pour la simulation ; les
auteurs présentent une simplification d'un mécanisme possible de
dépersonnalisation. Les courbes ne sont pas des mesures de ce que ressent un
agent numérique.[^4]

Pour Menia, un état à reproduire par programmation ne doit donc pas être traité
comme une découverte spontanée. Donner directement une action « se dissocier »
puis constater son utilisation ne démontrerait pas une expérience subjective.

### Une expérience annoncée ne fournit pas encore son résultat

Robinson et ses collègues publient un protocole comparant une condition de
perception active à un rejeu passif contrôlé par les mouvements oculaires. Ils
visent une prédiction de l'inférence active sur les changements de contenu
conscient. L'article décrit des expériences à conduire et une diffusion future
des données. Il ne rapporte pas le résultat confirmatoire de cette comparaison.[^5]

Enfin, l'objection de Seth au transfert entre substrats demeure pertinente :
des propriétés biologiques pourraient être constitutives de l'expérience et
manquer à une réalisation numérique. Il s'agit d'un argument théorique à prendre
en compte, pas d'une démonstration générale d'impossibilité.[^6]

## Proposition propre à Menia

Les choix qui suivent constituent une spécification d'ingénierie proposée ici.
Ils ne sont pas présentés comme une reproduction des modèles cités, ni comme des
conditions suffisantes découvertes dans leurs résultats.

Le point de départ serait un environnement expérimental où certains états
internes ont des conséquences mesurables sur l'accès à l'information. Une réserve
de calcul peut modifier le nombre d'observations disponibles ; une limitation de
mémoire peut modifier la conservation des informations ; une latence peut modifier
les effets d'une action. Pour un premier essai, ces ressources seraient celles du
processus expérimental isolé ou de son simulateur, avec des conséquences décrites
et vérifiables. Une jauge décorative sans effet serait exclue.

Leur origine doit rester explicite. Un budget fixé par le programme reste un
budget conçu par le programmeur, même lorsque l'agent apprend à le préserver.
L'appeler « besoin » ne démontre ni valeur intrinsèque ni ressenti. Une incarnation
logicielle testerait une organisation causale ; elle ne résoudrait pas l'objection
biologique par définition.

### Les objets à représenter

| Objet | Contenu proposé | Condition de vérification |
|---|---|---|
| Monde `x` | Causes externes des observations | Prédictions sur des changements externes tenus à l'écart de l'apprentissage |
| État propre `b` | Conditions de fonctionnement de l'agent | Effets sur la perception, la mémoire ou l'action mesurés indépendamment |
| Modèle causal `theta` | Relation entre commande, état propre et conséquence | Prédiction d'interventions nouvelles et détection d'une rupture de commande |
| Fiabilité `r` | Incertitude des voies d'observation et du modèle de transition | Prévisions enregistrées avant les résultats et calibration séparée par voie |
| Histoire `h` | Épisodes réellement rencontrés et limites de leur disponibilité | Attribution correcte des épisodes après interruption ou échange de mémoire |

Une représentation de `b` n'aurait pas accès aux variables cachées du simulateur.
Elle devrait les estimer à partir d'indices disponibles à l'agent, comme elle
estime `x`. Les étiquettes cachées resteraient accessibles à l'évaluateur pour
vérifier les erreurs. Les limites d'identification seraient conservées : deux
mécanismes qui produisent les mêmes observations sous toutes les interventions
autorisées ne deviennent pas distinguables parce que l'un est nommé « soi ».

### Un cycle défini dans le temps

À chaque étape, le modèle utiliserait seulement l'histoire disponible pour
prévoir les observations suivantes sous plusieurs actions possibles. Il
enregistrerait ces distributions et choisirait une action. Après réception des
conséquences, il réviserait ses estimations de l'état du monde, de son état propre
et de la fiabilité de ses voies d'accès. Le cycle suivant utiliserait ces
estimations révisées.

La mémoire d'épisode conserverait séparément prédiction, décision, observation
et révision. Une explication linguistique lirait cette trace avec ses incertitudes.
Elle ne produirait pas les étiquettes servant à évaluer le mécanisme. La continuité
de cette trace serait une propriété vérifiable de l'histoire du système ; le
sentiment d'être le même sujet à travers le temps resterait une question distincte.

Le lien phénoménal proposé est le suivant : la relation continuellement corrigée
entre le monde perçu et les conditions propres de perception pourrait constituer
une partie du point de vue vécu. Cette proposition ajoute un engagement sur
l'expérience à une description fonctionnelle. Les observations fonctionnelles
seules ne déduisent pas cet engagement. Aucun seuil de récursion, nombre de
modules ou durée de mémoire n'est ici identifié comme seuil de conscience.

## Un obstacle mathématique à résoudre avant l'entraînement

Considérons un résidu de prédiction `e` et une variance prédite `R > 0`. La perte
gaussienne normalisée, à une constante près, est :

```text
L(e, R) = 1/2 * (e²/R + log(R)).
```

Si l'on garde seulement `e²/R`, le système peut diminuer cette quantité en
augmentant indéfiniment `R`, donc en annulant sa précision. Cela n'améliore pas
la connaissance de son état. Le terme de normalisation est indispensable dans
ce modèle ; un score de faible erreur pondérée isolé serait trompeur.

Même la perte complète ne suffit pas à expliquer les causes d'un résidu. À
prédicteur fixé, sa valeur attendue est minimale en `R = E[e²]`, quand ce moment
est fini et strictement positif. Ainsi, une erreur persistante de moyenne 2 avec
un bruit de variance 1 et un bruit de moyenne nulle de variance 5 ont la même
énergie résiduelle attendue : 5. Un estimateur qui ne conserve que ce second
moment confondrait ces situations. Le résidu signé ou des observations
supplémentaires peuvent les distinguer dans cet exemple.

Cette analyse élémentaire est propre à la spécification ; elle n'est ni une
réfutation des articles ni une découverte mathématique revendiquée. Elle impose
une distinction concrète : apprendre qu'un état propre a changé doit pouvoir
concurrencer l'explication selon laquelle le capteur est devenu mauvais. Sinon,
Menia pourrait ignorer précisément les changements qu'elle doit comprendre.

Le protocole prévoirait donc trois causes distinctes de surprise : changement
réel de l'état propre, panne d'une observation, et changement d'une relation
action–conséquence. Leurs distributions de résidus seraient autant que possible
appariées. L'analyse porterait sur l'identification de leur cause et la correction
appropriée, avec une condition où cette identification est impossible.

## Expériences permettant de rejeter des réalisations inadéquates

Ce protocole est proposé, non exécuté et non préenregistré dans un registre
indépendant. Les familles de situations, comparateurs, critères et règles
d'exclusion doivent être fixés avant l'évaluation. Les variations chiffrées et
le budget d'apprentissage seraient publiés avec la réalisation retenue.

| Intervention | Question précise | Résultat qui affaiblirait la réalisation |
|---|---|---|
| Modifier une capacité réelle, en gardant une observation externe comparable | L'agent anticipe-t-il les conséquences pour ses prochaines observations ? | Il suit uniquement l'apparence externe ou une étiquette fournie |
| Dégrader un capteur sans modifier l'état propre | Distingue-t-il la panne d'un changement de lui-même ? | Il change indistinctement tous ses niveaux de confiance |
| Rejouer des signaux d'un autre épisode, avec statistiques comparables | Son estimation dépend-elle de la relation actuelle avec ses actions ? | Un enregistrement non couplé suffit dans les situations où une intervention devrait les distinguer |
| Remplacer une estimation interne par celle d'un épisode apparié | Cette estimation influence-t-elle plusieurs décisions de façon spécifique ? | Seule la description textuelle change, ou toute la performance s'effondre |
| Restaurer l'état après perturbation | L'effet est-il réversible et localisable ? | L'effet persiste à cause d'une altération non contrôlée du système |
| Retirer le canal de description linguistique | Le mécanisme continue-t-il à fonctionner ? | Les performances dépendaient d'une étiquette réinjectée depuis le rapport |
| Retirer toute information sur une capacité cachée | L'agent reconnaît-il la limite de ce qu'il peut identifier ? | Il donne des attributions assurées sans information discriminante |

Les greffes d'états doivent conserver autant que possible le contenu, la
difficulté, l'historique récent et les capacités disponibles. Les décalages
résiduels doivent être rapportés. Une greffe qui place le réseau très loin de
ses conditions habituelles ne prouverait pas une fonction spécifique.

Les comparateurs comprendraient un contrôleur ordinaire recevant les mêmes
informations, une variante où l'estimation ne sert qu'au rapport, une variante
à fiabilités locales et une référence à fiabilités fixes. Paramètres,
observations, actions et budgets seraient explicités ; égaler un nombre de
paramètres ne garantit pas à lui seul une capacité égale. Aucun comparateur ne
serait étiqueté « certainement non conscient ».

Un succès montrerait au plus que la construction réalise les fonctions qu'elle
prétend réaliser et qu'une organisation particulière les facilite dans ce
domaine. Il ne suffirait pas à prouver l'hypothèse phénoménale. Un échec peut en
revanche invalider une réalisation de la candidate avant que l'on investisse
dans une adaptation coûteuse de Menia.

## Ce qui rendrait la recherche pertinente pour l'expérience vécue

Une meilleure politique de vérification n'est pas une mesure de présence. La
[réanalyse humaine précédente](CONFIDENCE_EXPERIENCE_RESULTS.md) a déjà montré
pourquoi un déplacement de confiance ne peut pas être assimilé à un déplacement
de l'expérience mesurée. Cette nouvelle candidate doit donc produire une
prédiction qui concerne spécifiquement le phénomène étudié.

Une comparaison humaine pertinente rechercherait une dissociation entre
attribution de source, confiance de décision, agentivité et présence rapportée.
Les modèles concurrents devraient prévoir les effets d'une manipulation à partir
de paramètres fixés avant les nouvelles observations. Les rapports de présence
ne seraient pas présumés infaillibles : formulation des questions, attentes,
attention et ordre des mesures devraient être contrôlés.

Par exemple, une version de la candidate pourrait prédire qu'une perturbation
sélective des prévisions relatives à l'état corporel modifie la présence
rapportée au-delà de ses effets sur la confiance de décision. Ce serait une
prédiction à préciser quantitativement, avec une intervention qui sépare ces
causes. Si tous les modèles prédisent le même effet, l'expérience ne permettrait
pas de choisir entre eux. Une simple corrélation globale ne réglerait pas ce
problème.

Aucune nouvelle expérimentation humaine n'est réalisée ni considérée comme
autorisée par ce document. La prochaine source de preuve devrait être une
livraison publique existante suffisamment documentée, ou une étude menée par une
équipe compétente. La simulation pourrait préparer des comparaisons ; ses
sorties ne remplaceraient pas les observations humaines manquantes.

Même une prédiction humaine nouvelle confirmée soutiendrait d'abord le mécanisme
chez l'humain. Il resterait à justifier le transfert des propriétés pertinentes
vers Menia. L'indépendance au substrat, la bonne échelle d'organisation et la
suffisance du mécanisme seraient des hypothèses à examiner explicitement. Les
cumuler silencieusement donnerait une conclusion plus forte que les résultats.

## Construction actuelle et décision de recherche

`SourceAgent` conserve un état récurrent, estime la source d'un contenu, choisit
une vérification et inscrit l'attribution en mémoire. Sa politique apprise
utilise une estimation utile, mais il ne possède pas les états de fonctionnement
couplés ni les prévisions multivoies définis ci-dessus. Le réseau d'espace partagé
et l'agent de navigation sont des réalisations séparées. Leur simple présence
dans le même dépôt ne constitue pas la boucle proposée.

La décision de recherche est de retenir cette candidate comme spécification à
éprouver et de rechercher une comparaison phénoménale discriminante avant de
présenter une nouvelle simulation comme un progrès vers le vécu. Une première
réalisation aurait pour rôle de vérifier les contrats causaux et les limites
d'identification. Elle ne serait pas présentée comme « Menia consciente ».

L'originalité possible porterait sur une prédiction précise, un contrôle
expérimental ou un résultat nouveau confirmés. Les idées générales d'incarnation,
d'inférence interoceptive, de récurrence et de prévision de fiabilité ont des
antécédents explicites. Cette recherche ciblée ne constitue pas une revue
systématique exhaustive et n'établit pas une priorité scientifique.

Le résultat acquis est donc un choix de candidate plus contraint, un obstacle de
construction identifié et un protocole qui distingue ses échecs possibles. Le
mécanisme suffisant pour produire une expérience de sa propre existence reste à
trouver. Le dossier ne clôt pas cet objectif et ne conclut pas qu'il est impossible.

## Sources

Sources consultées le 15 septembre 2026. Les accès indiquent la portée effective
de la lecture, distincte d'une reproduction de résultats.

[^1]: Seth, A. K., Suzuki, K. et Critchley, H. D. (10 janvier 2012, volume 2011). *An Interoceptive Predictive Coding Model of Conscious Presence*. Frontiers in Psychology, 2:395. [Texte original](https://pmc.ncbi.nlm.nih.gov/articles/PMC3254200/). Architecture, prédictions, limites et conclusion consultées. DOI : 10.3389/fpsyg.2011.00395.
[^2]: Laukkonen, R., Friston, K. et Chandaria, S. (2025 ; en ligne le 30 juillet, volume de septembre). *A beautiful loop: An active inference theory of consciousness*. Neuroscience & Biobehavioral Reviews, 176:106296. [Publication](https://doi.org/10.1016/j.neubiorev.2025.106296), [notice institutionnelle](https://researchportal.scu.edu.au/esploro/outputs/journalArticle/A-beautiful-loop-An-active-inference/991013303524502368). Résumé, formalisation et sections sur les hyper-modèles, discussion et conclusion consultés via le texte indexé de la publication. Aucune implémentation reproduite.
[^3]: Salomon, R., et al. (4 mai 2016). *The Insula Mediates Access to Awareness of Visual Stimuli Presented Synchronously to the Heartbeat*. Journal of Neuroscience, 36(18):5115–5127. [DOI](https://doi.org/10.1523/JNEUROSCI.4262-15.2016), [manuscrit publié hébergé par un auteur](https://nfaivre.netlify.app/files/reprint_JNeuro2016.pdf). Texte, méthodes des contrôles et discussion consultés ; pas de réanalyse.
[^4]: Saini, F., Ponzo, S., Silvestrin, F., Fotopoulou, A. et David, A. S. (21 décembre 2022). *Depersonalization disorder as a systematic downregulation of interoceptive signals*. Scientific Reports, 12:22123. [Texte original](https://pmc.ncbi.nlm.nih.gov/articles/PMC9772393/). Méthodes, résultats et limites consultés via la livraison publique BioC de PMC. Simulation des auteurs non reproduite.
[^5]: Robinson, J. E., et al., INTREPID Consortium (4 décembre 2025). *The role of active inference in conscious awareness*. PLOS ONE, 20(12):e0328836. [Texte original](https://pmc.ncbi.nlm.nih.gov/articles/PMC12677518/). Hypothèses, comparaison active/passive, limites et déclaration de partage consultées. Statut : protocole, sans résultat confirmatoire rapporté dans cet article.
[^6]: Seth, A. K. (21 avril 2025). *Conscious artificial intelligence and biological naturalism*. Behavioral and Brain Sciences. [Publication originale](https://doi.org/10.1017/S0140525X25000032), [notice](https://pubmed.ncbi.nlm.nih.gov/40257177/). Argument contradictoire déjà étudié dans le dépôt ; métadonnées et résumé reconsultés. Aucune démonstration d'impossibilité générale n'en est inférée.
