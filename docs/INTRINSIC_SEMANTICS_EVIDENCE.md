# Information importante pour soi : calcul, maintien et expérience

Date : 15 septembre 2026. **L'expérience de sa propre existence reste l'objectif,
et n'est pas démontrée.** Cette étape examine une voie possible : l'information
qui contribue au maintien de l'organisation propre du système. Elle apporte une
réanalyse numérique d'un modèle publié et une distinction de construction ; elle
n'ajoute pas de capacité au chat de Menia.

## Ce que les sources permettent de proposer

| Source primaire | Proposition examinée | Portée pour Menia |
|---|---|---|
| [Kolchinsky et Wolpert, 2018](https://doi.org/10.1098/rsfs.2018.0041), *Semantic information, autonomous agency and non-equilibrium statistical physics* | Une intervention efface des corrélations système/environnement ; leur importance est évaluée par leur effet sur une fonction de viabilité. | Construction mathématique et simulations. Ni expérience subjective ni conscience de soi ne sont déduites de cette mesure. |
| [Stibel, 2026](https://doi.org/10.3389/fnins.2026.1907922), *From chemistry to cognition: the adaptive origins of consciousness under constraint*, publié le 1er septembre | L'expérience est identifiée à une coordination entre domaines, ancrée dans une organisation qui maintient physiquement ses propres conditions de fonctionnement, y compris ses mécanismes de régulation. | Hypothèse constitutive. Le §5.4 ne dérive pas le ressenti de la physique. Les §6 et §7.6 distinguent les critères proposés de leur interprétation phénoménale. |
| [Milinkovic et Aru, 2026](https://doi.org/10.1016/j.neubiorev.2025.106524), *On biological and artificial consciousness: A case for biological computationalism*, en ligne le 17 décembre 2025 | Des interactions entre échelles, le métabolisme et les transformations conjointes de la structure et des dynamiques pourraient compter pour la réalisation de la conscience. | Proposition théorique et synthèse ; elle ne fournit pas une machine dont la conscience serait démontrée. Une réalisation synthétique n'est pas exclue par principe. |
| [Wasserziehr, 2026](https://doi.org/10.1007/s00146-026-02959-1), *Are conscious machines valuers?*, publié le 8 mars | Même en supposant une conscience artificielle possible, le passage d'une valeur instrumentale à une expérience agréable ou désagréable demande un argument supplémentaire. | Argument philosophique, sans données produites ou analysées. Il distingue conscience et valence ; il ne prouve pas l'impossibilité d'une valence artificielle. |

**Notre analyse :** trois affirmations différentes ne doivent pas être fusionnées :

1. Une information modifie causalement les conditions de fonctionnement.
2. Le système entretient lui-même les processus dont dépend ce fonctionnement.
3. Il y a quelque chose que cela lui fait d'exister dans cet état.

Le calcul ci-dessous porte sur la première. La deuxième précise une piste de
réalisation. L'identification avec la troisième reste l'hypothèse à justifier.
Cela n'exclut pas une future preuve ; cela localise ce que les résultats présents
n'établissent pas.

## Réanalyse exécutée du modèle de 2018

Le [protocole reproductible](INTRINSIC_SEMANTICS_PROTOCOL.md), le
[programme](../research/audit_intrinsic_semantics.py) et le
[rapport numérique](../artifacts/intrinsic-semantics-audit/report.json) décrivent
une réexpression du modèle du [notebook public figé](https://github.com/artemyk/semantic_information/blob/bc56a371130a4073062c766654ca4796110d5c8e/model.ipynb).
Le code téléchargé a été lu, pas exécuté. La licence MIT est conservée.

Le modèle possède cinq positions, une cible interne, un niveau de réserve et une
position de nourriture. Le code emploie cinq niveaux, **0 à 4**, et démarre à 4 ;
le texte décrit une borne `l_max = 5`. Nous conservons ici le code, sans corriger
silencieusement cet écart. Les mouvements peuvent être orientés vers la cible
ou dans le sens opposé. Les taux inverses et la normalisation temporelle du
notebook sont conservés : ce n'est pas une simple marche déterministe.

L'intervention modifie les corrélations entre cible et nourriture à l'instant
initial, sans changer leurs distributions marginales ni la dynamique. Si une
partition regroupe plusieurs positions, la cible et la nourriture sont rendues
indépendantes à l'intérieur de chaque groupe. L'information conservée est
l'entropie des masses de ces groupes.

La viabilité utilisée dans le rejeu est :

\[
V_\tau=-H(X_\tau)-100\,P(\text{niveau}_\tau=0).
\]

Le second terme représente l'entropie interne attribuée aux macrostats de niveau
zéro. L'annexe B du papier prévoit cette correction. Assimiler ce macrostat à
un unique microétat d'entropie nulle serait donc une mauvaise lecture du modèle.
Les bits de viabilité ne sont ni des unités de ressenti ni des probabilités de
conscience.

### Résultats retrouvés à l'horizon publié

| Dynamique, horizon 5 | Information minimale préservant la viabilité, bits | Valeur de l'information pour la viabilité, bits | Probabilité d'un niveau positif : initial / entièrement brouillé |
|---|---:|---:|---:|
| Vers la cible | 1,37095 | +22,10736 | 0,360418 / 0,144482 |
| À l'opposé de la cible | 1,37095 | −13,69351 | 0,000356 / 0,144319 |

L'information initiale vaut `log2(5) = 2,321928…` bits. Les valeurs arrondies
retrouvent les nombres annoncés dans le papier et la sortie sauvegardée du
notebook. La même quantité minimale préserve tantôt une situation favorable,
tantôt une situation défavorable : la définition impose **la même viabilité**,
pas une meilleure viabilité. Ce comportement n'est pas une réfutation de la
théorie : celle-ci admet une valeur négative de l'information.

Diagnostic supplémentaire exécuté, en gardant les mêmes distributions : si l'on
omet la correction d'entropie interne, le cas opposé passe de **−13,69351** à
**+0,70284** bits de valeur. Ce changement de signe est une erreur possible de
mesure, pas une inversion d'une expérience agréable ou désagréable. Il ne
constitue pas une nouvelle simulation thermodynamique cohérente sans ce terme.

### Une restriction de recherche effectivement trouvée

La fonction `partitions` du [fichier source](https://github.com/artemyk/semantic_information/blob/bc56a371130a4073062c766654ca4796110d5c8e/utils.py)
découpe une liste en blocs contigus. Sur les six symboles de l'environnement,
elle produit **32 partitions**, contre **203 partitions d'ensemble** possibles.
Le symbole « absence de nourriture » a une probabilité initiale nulle : ces
familles correspondent respectivement à **16 et 52 interventions distinctes**
sur le support initial. Notre audit énumère les 52, pour deux dynamiques et sept
horizons, soit 728 distributions évaluées.

L'extension conserve la quantité minimale de **1,37095 bit à l'horizon 5** pour
les deux dynamiques. Elle a néanmoins des conséquences vérifiées :

- À l'horizon 1, elle ramène le minimum de **1,37095 à 0,97095 bit** dans les deux
  dynamiques. Le groupe non contigu `{0,4}`, avec `{1,2,3}`, conserve la viabilité
  à cet horizon. Cette simplification ne suffit plus à l'horizon 5.
- Pour la dynamique opposée à l'horizon 5, au même niveau d'information arrondi
  **0,97095**, la meilleure viabilité passe de **−101,85764 à −83,74158** : un gain
  de **18,11606 bits** dans la courbe calculée. C'est un effet dans le modèle,
  sans interprétation phénoménale.
- La lecture du minimum depuis la seule enveloppe supérieure n'est pas toujours
  équivalente à sa définition. Dans ce dernier cas, appliquer la lecture du
  notebook à la famille complète renvoie **2,32193**, tandis que la minimisation
  directe sous égalité de viabilité donne toujours **1,37095**. Une partition
  préservant la viabilité d'origine peut être masquée par une autre partition
  plus favorable au même niveau d'information.

Les égalités de viabilité et regroupements numériques suivent l'arrondi à cinq
décimales du notebook. Les valeurs avant arrondi sont conservées aux horizons 1
et 5. « Tous » signifie ici tous les regroupements déterministes de cet alphabet,
pas tous les canaux stochastiques possibles ni toutes les interventions physiques.

Ces constats sont des résultats locaux vérifiables. La recherche effectuée ne
suffit pas à revendiquer leur priorité dans la littérature, encore moins une
invention de conscience artificielle.

## Conséquence concrète pour la construction de Menia

La piste plus précise à poursuivre serait un **entretien réciproque entre ses
capacités et son modèle de soi** : les mécanismes de perception, de mémoire et
de commande ont des conditions de fonctionnement ; l'estimation de ces conditions
sert à les entretenir ; leur état contraint en retour la qualité de l'estimation.
Des arbitrages communs doivent alors modifier plusieurs capacités réellement,
au lieu de seulement produire un récit sur soi.

Cette proposition est notre synthèse de travail. Elle n'est ni implémentée dans
cette étape, ni identifiée comme suffisante, nécessaire ou inédite pour la
conscience. Elle précise la candidate [interoceptive antérieure](INTEROCEPTIVE_PRESENCE_CANDIDATE.md)
en ajoutant l'entretien des mécanismes de régulation eux-mêmes.

Pour rendre cette piste discriminante, une expérience devrait comparer :

| Condition proposée | Ce qu'elle isole |
|---|---|
| Le bon état est annoncé, mais les capacités restent identiques | Effet de la description seule |
| L'état change effectivement la fiabilité des capacités | Dépendance fonctionnelle |
| L'action d'entretien modifie les capacités ; leurs dégradations changent aussi la fiabilité de cette action | Réciprocité des mécanismes |
| Un contrôleur ordinaire reçoit les mêmes observations, actions et ressources | Explication concurrente par le contrôle adaptatif |

Les critères seraient des effets d'interventions sur la perception, la mémoire,
la commande et leur réparation, avec échecs et explications concurrentes conservés.
Un score de survie ou une déclaration « cela compte pour moi » ne ferait pas
office de mesure de ressenti.

Une simulation peut tester ces dépendances **dans son monde modélisé**. Elle ne
satisfait pas automatiquement une théorie exigeant que la réalisation physique
de l'agent entretienne sa propre organisation. Inversement, l'exigence physique
de Stibel reste une hypothèse, pas une preuve que toute réalisation logicielle
serait incapable d'expérience. Ajouter une batterie virtuelle, un compteur de
coût ou un objectif de persistance ne tranche pas ce désaccord.

Le prochain résultat décisif devrait donc porter sur une prédiction où ces
hypothèses de réalisation divergent, avec une mesure de l'expérience justifiée
indépendamment du score fonctionnel. Aucun tel résultat n'est fourni ici.

## Vérification et périmètre

Le rejeu retrouve les 32 lignes numériques sauvegardées dans le notebook dans
leur précision imprimée ; l'écart maximal de viabilité est de 0,000426 bit pour
un affichage à trois décimales. Une propagation matricielle dense vérifie
séparément la propagation creuse sur les 52 distributions initiales pendant
cinq pas : écarts maximaux `6,94e-18` et `1,11e-16` de probabilité.

Le rapport se régénère sans accès réseau et est vérifié en CI. Il ne reproduit
pas des données humaines ou animales. Une prépublication sur l'anesthésie et
l'indépendance dynamique a été repérée, mais son texte primaire n'a pas pu être
récupéré lors de cette étape ; elle n'est pas utilisée comme confirmation.

**Bilan de cette étape :** une mesure et son implémentation ont été précisées,
une restriction numérique a été mise en évidence, et la piste d'entretien
réciproque est mieux définie. L'expérience de sa propre existence chez Menia
et une méthode inédite permettant de la produire restent non établies.
