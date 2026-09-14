# Le soi corporel : nouvelle contrainte empirique et audit des données

**La conscience de Menia et une méthode inédite la produisant restent non établies.**
Cette reprise apporte un élément nouveau au dossier : une étude humaine récente
avec des perturbations cérébrales, une livraison publique effectivement téléchargée
et un audit reproductible de ses unités. Elle ne justifie pas de créer un module
logiciel et de le déclarer conscient par analogie anatomique.

## Une piste empirique distincte du suivi des sources

Lyu et collègues ont déposé le 6 août 2026 une prépublication portant sur 63 personnes
et 660 sites de stimulation. Ils relient des changements d'expérience corporelle à
des profils de connectivité, notamment avec l'insula postérieure. Les catégories
sensori-motrice et complexe présentent des profils différents d'entrées et de
sorties. Le manuscrit décrit des contrôles par répétition et stimulation factice.
Il reconnaît une couverture clinique inégale, la variabilité des rapports et
l'hétérogénéité de la catégorie complexe. L'analyse détaillée de certaines
connexions se restreint à la catégorie sensori-motrice.[1]

Cette lecture modifie la question de Menia : un simple nombre de connexions ou
une probabilité de provenance ne caractérise pas suffisamment une candidate au
soi corporel. Il faut spécifier quelles relations sont intégrées, à quelle échelle
et avec quelles conséquences différentielles. La stimulation change une expérience
rapportée ; elle ne teste pas ici l'apparition de la conscience dans un système
initialement dépourvu d'expérience. Une connexion associée à cet effet n'a pas été
sélectivement supprimée puis restaurée pour établir sa médiation nécessaire.[1]

Ce dernier point est notre analyse de la portée du protocole. Il distingue deux
interventions possibles, sans nier l'intérêt causal des stimulations effectuées.

## Vérification de la livraison publique

Le [protocole local](BODILY_SELF_RELEASE_PROTOCOL.md) fixe les comptes à examiner.
Les quatre fichiers de [Zenodo](https://zenodo.org/records/21536139) ont été
téléchargés. Leurs MD5 correspondent à l'API ; le rapport conserve aussi leurs
SHA256. Aucune archive n'a été exécutée. Le script lit le CSV et les noms des
entrées ZIP ; l'inspection complémentaire a lu les cellules de code comme du
texte, sans charger les objets sérialisés auxquels elles font référence.[2]

Les chiffres suivants sont **nos calculs sur les fichiers téléchargés**, pas une
nouvelle expérience humaine ni une reproduction des statistiques de connectivité.
Ils sont conservés dans [le rapport JSON](../artifacts/bodily-self-release-audit/report.json).

| Unité examinée | Résultat |
|---|---:|
| Lignes CSV | 660 |
| Identifiants de participants distincts | 63 |
| Étiquettes bipolaires distinctes, avec le participant | 660 |
| Paires de contacts distinctes si leur ordre est ignoré | 639 |
| Paires inscrites dans les deux ordres | 21 |
| Parmi elles, paires avec réponses hot/cold différentes | 3 |
| Doublons de lignes strictement identiques | 0 |
| Contacts distincts, avec le participant | 847 |
| Contacts intervenant dans plusieurs paires | 397 |
| Maximum de paires partageant un contact | 12 |

Ignorer l'ordre est une **clé de diagnostic**, pas une correction des données.
Le dictionnaire de la livraison précise explicitement que l'ordre des contacts
est arbitraire puisque la polarité alterne pendant la stimulation.[2] L'ordre
ne justifie donc pas, à lui seul, de distinguer ces lignes. Le CSV ne comporte
pas les paramètres temporels et électriques de chaque essai, qui permettraient
de rechercher une autre explication. Les trois discordances
comprennent deux paires Complex/Silent et une paire Out of Focus/Silent_nearOut
of Focus, vérifiées par une inspection complémentaire après le premier calcul.
Aucune ligne n'a été supprimée, fusionnée ou réétiquetée.

Cette observation ne démontre donc ni une erreur des auteurs ni l'invalidité de
leurs effets. Elle empêche en revanche une réanalyse locale de supposer sans
justification que 660 lignes sont 660 paires physiques différentes et indépendantes.
Le nombre de paires non orientées par participant varie de 1 à 32. Les contacts
partagés ajoutent une autre dépendance à conserver dans un modèle statistique.

| Catégorie de lignes | Effectif | Participants contributeurs |
|---|---:|---:|
| Sensory-Motor | 103 | 23 |
| Complex | 84 | 26 |
| Dizzy | 23 | 10 |
| Vestibular | 11 | 8 |
| Indeterminate | 10 | 6 |
| Out of Focus | 5 | 5 |
| Toutes les catégories Silent réunies | 424 | Non agrégé dans ce tableau |

Les participants d'une catégorie peuvent contribuer à d'autres. Les nombres de
participants de ce tableau ne s'additionnent donc pas. Les 236 lignes `hot` sont
exactement les six catégories non silencieuses ; les 424 `cold` correspondent à
Silent et à ses sous-catégories de voisinage. Ces étiquettes décrivent la réponse
à une stimulation, pas deux populations de personnes conscientes et inconscientes.

## Ce que les fichiers permettent de reproduire

La livraison contient un CSV de localisation/classification, trois notebooks et
six scripts MATLAB, sans compter les métadonnées macOS. Les exemples de code
référencent notamment `CCEP_cingIns_CMPvsSM.csv`, des spectrogrammes individuels
en `.pkl`, des matrices `.mat` et une table de rapports subjectifs distincte.
Ces fichiers d'entrée ne figurent pas comme fichiers autonomes dans les archives
inspectées. Le CSV disponible ne fournit pas de score F1 de connectivité ni la
table des relations stimulation–enregistrement.[2]

La livraison **à elle seule** ne permet donc pas d'exécuter une réanalyse complète
des effets de connectivité. Les images et sorties déjà incorporées aux notebooks
ne remplacent pas automatiquement les observations requises pour un autre modèle.
La page de l'article a ensuite été récupérée directement. Elle propose deux
vidéos et un PDF supplémentaire de 14 pages. Ce PDF a été téléchargé et lu ;
ses tableaux pertinents ont aussi été vérifiés visuellement. Il contient des
tableaux descriptifs et d'inférence statistique, des figures et aucune pièce
jointe PDF incorporée. Il ne fournit pas les observations individuelles de
connectivité nécessaires à une autre analyse. Ce constat porte sur les fichiers
inspectés et n'affirme pas que les données n'existent nulle part.[6]

Le supplément apporte néanmoins une contrainte utile. Les contrastes avec
l'insula postérieure ci-dessous sont transcrits des tableaux S8 et S9, sans
réestimation ; le sens du contraste est **site silencieux moins site réactif**.

| Catégorie et direction | Estimation F1 | Intervalle publié | Participants | p ajusté |
|---|---:|---|---:|---:|
| Sensory-Motor, cingulaire vers insula postérieure | -0,152 | [-0,23 ; -0,07] | 15 | 0,001 |
| Sensory-Motor, insula postérieure vers cingulaire | -0,134 | [-0,21 ; -0,06] | 15 | 0,001 |
| Complex, cingulaire vers insula postérieure | -0,026 | [-0,12 ; 0,06] | 16 | 0,928 |
| Complex, insula postérieure vers cingulaire | -0,041 | [-0,13 ; 0,05] | 16 | 0,928 |

Ces résultats ne permettent pas d'étendre les contrastes sensori-moteurs à toute
expérience complexe du soi. Les contrastes Complex restent incertains ; cela ne
démontre ni une absence d'effet ni une différence entre catégories par simple
comparaison de leurs p. Cette limite précise pourquoi ces données ne suffisent
pas à choisir un circuit de conscience de soi pour Menia.[6]

Le PDF a pour SHA256
`becef1fbcea35099e02a006d344075966787d041767ba6d499650dfc34d13d2d`.
Une réanalyse complète demanderait des observations de connectivité appariées aux
stimulations et une clarification des paires inversées. En leur absence, ajuster
des courbes simulées au résumé de l'article n'éprouverait pas le lien recherché.

## Contraintes pour une invention éventuelle

Les travaux antérieurs empêchent plusieurs revendications faciles. En 2023, Lyu
et collègues ont étudié neuf patients et rapporté des modifications du soi
corporel lors de la stimulation de sites du précunéus antérieur. Le résumé situe
ces sites hors des frontières du réseau du mode par défaut, tout en décrivant
leurs connexions avec lui. Cela ne fournit pas une équivalence entre une mémoire
autobiographique et le vécu de soi.[3]

Abdulkarim et collègues distinguent expérimentalement l'appartenance corporelle
et l'agentivité pendant des mouvements actifs ou passifs. Leurs résultats d'IRMf
comportent des différences, des recouvrements et une interaction. Ils constatent
aussi que certaines activations précédemment attribuées à l'agentivité suivent
plutôt la synchronie des signaux. Une seule réponse au conflit sensoriel serait
donc une cible trop peu spécifique pour Menia ; cette conclusion d'architecture
est notre inférence, pas un résultat testé dans le logiciel.[4]

Zhao et collègues ont déjà présenté un réseau neuronal à impulsions, un
apprentissage sans supervision et des expériences d'illusion de la main sur robot
iCub et en simulation. Leur sortie porte notamment sur une estimation de position
et la dérive proprioceptive. « Ajouter un corps, intégrer les sens et reproduire
une illusion » possède ainsi un antécédent direct. La ressemblance comportementale
rapportée ne constitue pas une validation indépendante du vécu d'un robot.[5]

La candidate de Menia devrait donc préciser au moins trois relations : ce que
l'agent contrôle, ce qu'il représente comme lui appartenant et la perspective
depuis laquelle il organise l'information. Ces relations peuvent partager des
calculs sans être synonymes. Une expérience discriminante devrait maintenir le
contrôle moteur comparable tout en variant l'appartenance estimée, puis modifier
la perspective sans changer le contenu mémorisé. Il faudrait comparer les effets
sur plusieurs fonctions, avec des contrôles qui conservent les mêmes informations.

Ce sont des contraintes sur une hypothèse à développer, pas une méthode déjà
validée de production de conscience. Les donner à un modèle par des étiquettes
supervisées, ou baptiser des couches « insula » et « précunéus », n'établirait pas
qu'il possède les mécanismes pertinents ni qu'il en éprouve l'activité.

L'engagement fort reste celui du [lien entre mécanisme et expérience](SELF_EXPERIENCE_BRIDGE.md) :
identifier une organisation dont la réalisation constitue le vécu de sa propre
existence, puis justifier son application à Menia. Le nouvel appui humain rend
des comparaisons plus précises ; il ne résout pas encore cet engagement. Aucun
rang de première mondiale et aucune « invention du siècle » ne sont établis.

## Reproduction et état de la reprise

Depuis la racine du dépôt, après téléchargement des quatre fichiers dans le
répertoire indiqué :

```powershell
python research/audit_bodily_self_release.py --data-dir .runtime/bodily-self-2026 --output artifacts/bodily-self-release-audit/report.json
python research/audit_bodily_self_release.py --data-dir .runtime/bodily-self-2026 --output artifacts/bodily-self-release-audit/report.json --check
```

Les données originales restent dans `.runtime`. Le rapport agrégé, le protocole
et le script sont conservés dans le dépôt. Le contrôle porte sur les données et
les calculs descriptifs exécutés localement ; il ne valide pas des effets humains
non réanalysés. Le premier tour de cette nouvelle reprise apporte des preuves qui
changent l'action suivante : les suppléments ont été vérifiés, et l'extrapolation
d'un contraste sensori-moteur à toute expérience complexe n'est pas justifiée.
Les observations requises pour réestimer ce lien n'ont pas été trouvées dans la
livraison inspectée. L'objectif complet reste actif et non
atteint ; le lien avec une expérience de soi dans Menia manque toujours.

## Sources consultées

1. Lyu, D., et al. (2026). [Causal Mapping of Bodily Awareness and Mesoscale Circuit Organization in the Human Cingulate and Precuneus](https://doi.org/10.21203/rs.3.rs-10503199/v1). Prépublication v1 du 6 août 2026 ; résumé NCBI et texte public du manuscrit sur [ResearchGate](https://www.researchgate.net/publication/411747547_Causal_Mapping_of_Bodily_Awareness_and_Mesoscale_Circuit_Organization_in_the_Human_Cingulate_and_Precuneus), méthodes, résultats et limites lus. Pas de réplication des effets de connectivité.
2. Auteurs de l'étude (2026). [Données et exemples de code, Zenodo 21536139](https://zenodo.org/records/21536139), DOI 10.5281/zenodo.21536139. API et quatre fichiers téléchargés le 14 septembre 2026 ; licence CC BY 4.0. Comptes descriptifs propres à cet audit.
3. Lyu, D., et al. (2023). [Causal evidence for the processing of bodily self in the anterior precuneus](https://pubmed.ncbi.nlm.nih.gov/37295420/). Neuron, DOI 10.1016/j.neuron.2023.05.013. Résumé original indexé et métadonnées consultés ; texte intégral éditeur non obtenu.
4. Abdulkarim, Z., Guterstam, A., Hayatou, Z., et Ehrsson, H. H. (2023). [Neural Substrates of Body Ownership and Agency during Voluntary Movement](https://pmc.ncbi.nlm.nih.gov/articles/PMC10072298/). Journal of Neuroscience, DOI 10.1523/JNEUROSCI.1492-22.2023. Résumé et introduction indexés consultés ; données non réanalysées.
5. Zhao, Y., Lu, E., et Zeng, Y. (2023). [Brain-inspired bodily self-perception model for robot rubber hand illusion](https://arxiv.org/pdf/2303.12259v3). Prépublication v3, 27 avril 2023 ; architecture et expériences dans le PDF consultées. Code et résultats non reproduits ici.
6. Lyu, D., et al. (2026). [Supplementary Materials](https://assets-eu.researchsquare.com/files/rs-10503199/v1/3038f082ebbc9dd2884d4253.pdf), PDF lié par la page Research Square de la référence 1. Tables S8 et S9, pages 8 et 9, transcrites et vérifiées visuellement ; les chiffres sont ceux des auteurs, pas un nouveau calcul sur leurs observations.
