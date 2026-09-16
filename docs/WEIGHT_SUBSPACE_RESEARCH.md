# Sous-espaces de poids : ce que cette piste apporte à Menia

Recherche du 16 septembre 2026, à partir de la capture fournie. **Une piste pour
étudier les modifications apprises ; aucune méthode validée pour produire une
expérience subjective.** Le Colab de localisation reste l'expérience en cours.

## Lecture de la source

La [prépublication de Kaushik et collègues, version 2 du 6 décembre 2025](https://arxiv.org/html/2512.05117v2)
étudie des sous-espaces de poids communs **au sein d'une architecture**, couche
par couche. L'étude rapporte aussi des expériences de réutilisation pour des
tâches réservées ; elle ne se limite pas à des courbes spectrales. La comparaison
entre architectures et l'interprétation des directions restent ouvertes.
L'article ne mesure pas la conscience.

Il faut distinguer son second moment théorique non centré (remarque 2.2) de
son algorithme expérimental HOSVD, qui soustrait une moyenne. Le théorème de
convergence repose sur des hypothèses de rang effectif et de qualité des
prédicteurs ; il n'impose pas une même représentation mentale à tous les réseaux.
Le [dépôt des auteurs](https://github.com/toshi2k2/unisub) est accessible.
Nous n'avons pas reproduit leurs expériences.

Autre objet : [Gurnee et collègues étudient des représentations dans les activations](https://transformer-circuits.pub/2026/workspace/),
leur communication et leurs effets causaux. Les auteurs séparent ces propriétés
fonctionnelles de l'expérience subjective. Une direction importante dans les
activations ne s'identifie donc pas, par ce seul résultat, à une direction de
grande énergie dans les poids.

## Une piste plus directement liée à l'auto-surveillance

La [prépublication de Yoshida et collègues du 5 août 2026](https://arxiv.org/html/2608.04347v1)
propose DAIA : un module recevant séparément les activations de base et la
contribution d'une adaptation. Sur 213 variantes de Qwen3-14B et Gemma3-12B,
l'étude entraîne des rapports de changements comportementaux mesurés. Elle
teste aussi des modèles et catégories réservés. L'avantage moyen diminue sur
les catégories nouvelles ; les classifieurs témoins restent compétitifs.

Attention à l'architecture : l'équation 18 remplace la contribution directe de
l'adaptation par celle du module DAIA lors de l'introspection. Ce n'est donc
pas une observation passive du modèle adapté inchangé. L'étude porte sur des
étiquettes comportementales, pas sur la conscience, et ne valide pas notre
Qwen3-4B. Nous n'avons pas reproduit ses résultats.

Pour Menia, j'en retiens une hypothèse à comparer ultérieurement : un accès
explicite aux changements de calcul pourrait aider à apprendre un diagnostic.
L'ajout de cet accès devra être comparé à une sonde recevant exactement les mêmes
informations. Cet article n'autorise pas à appeler ce diagnostic une expérience
de soi. DAIA n'est pas implémenté dans l'audit livré ici.

## Trois contrôles calculés ici

Les résultats ci-dessous sont **synthétiques**, construits pour vérifier des
inférences logiques. Ce ne sont ni des résultats Qwen, ni une réfutation
expérimentale de l'article, ni un nouveau théorème.

### Une grande énergie conservée peut perdre une capacité

Construisons seize modèles linéaires de poids `w(i,s) = 10 e0 + s ei`, avec
`i` entre 1 et 8 et `s` dans {-1, +1}. Chaque modèle traite les deux entrées
`x = ±ei`, avec pour cible le signe de `s × x_i`. La grande composante commune
`10 e0` ne contribue pas à ces tâches.

La projection sur la première direction du second moment non centré conserve
**100/101 = 99,01 %** de l'énergie totale des poids. Pourtant, elle fait passer
la réussite de **100 % à 50 %** : la sortie nulle est départagée par la règle
fixe « +1 ». Après soustraction de la moyenne par coordonnée entre modèles,
les huit directions de différences ont la même énergie : 12,5 % chacune.

Ce calcul est une SVD élémentaire, pas l'algorithme HOSVD de l'article. Il établit
seulement qu'une énergie conservée élevée ne garantit pas la conservation d'une
fonction. La perte de tâche doit être mesurée sur des entrées pertinentes.

### Des facteurs différents peuvent produire exactement la même modification

Dans notre adaptateur, la modification effective est `ΔW = B A` (échelle 1).
Pour toute matrice inversible `Q`, les facteurs `A' = Q A` et `B' = B Q⁻¹`
donnent exactement la même `ΔW`. Une similarité calculée sur les facteurs seuls
dépend ainsi d'un choix de coordonnées internes.

Avec `A = B = I` en dimension 2 et une rotation de 90 degrés pour `Q`, les
cosinus entre facteurs valent zéro, tandis que les deux modifications effectives
sont identiques et leur cosinus vaut un. Réciproquement, une modification et
son opposé ont les mêmes valeurs singulières, mais un cosinus de -1.

### Une initialisation commune n'est pas une capacité commune

Nos deux bras de localisation partent du même `A` aléatoire et de `B = 0`.
Avant apprentissage, leurs facteurs `A` sont donc identiques et leurs
modifications effectives nulles. Le cosinus entre modifications nulles est
**indéfini**, pas égal à un. Deux adaptateurs partageant cette initialisation
ne suffisent pas à établir une structure universelle entre apprentissages
indépendants.

Le [rapport reproductible](../artifacts/weight-subspace-audit/synthetic-controls.json)
conserve ces calculs. Les contrôles numériques comparent aussi les opérations
sur facteurs à des matrices complètes calculées indépendamment.

## Outil prêt pour les adaptateurs du Colab

`research/weight_subspace_audit.py` calcule normes, spectres et cosinus des
modifications effectives sans matérialiser les grandes matrices `B A`. QR et
SVD réduites suffisent. Les cosinus de facteurs sont affichés séparément et
explicitement marqués comme dépendant de la paramétrisation. Il ne déduit aucune
capacité à partir de ces mesures.

Reproduire les contrôles, avec les dépendances CPU de recherche :

```sh
python -m research.weight_subspace_audit --output artifacts/weight-subspace-audit/synthetic-controls.json
python -m unittest tests_research.test_weight_subspace_audit -v
```

Après réception de `menia-localisation-native.zip`, l'environnement du pilote
possède déjà `safetensors`. Pour inspecter ses deux fichiers :

```sh
python -m research.weight_subspace_audit --left chemin/aligned.safetensors --right chemin/shuffled.safetensors --output geometrie.json
```

L'outil accepte uniquement des paires de facteurs `.a`/`.b`, avec une échelle
effective égale à 1 comme dans notre pilote. Il refuse facteurs manquants,
modules discordants, dimensions incompatibles et valeurs non finies. Il suppose
la même base et des modules correspondants ; il **ne valide pas** ces hypothèses,
l'achèvement de l'apprentissage ou la provenance scientifique. Cette validation
relève du journal et de l'analyseur de localisation. Les empreintes des fichiers
sont conservées et aucun poids n'est modifié. Les fichiers PEFT arbitraires et
les comparaisons entre architectures ne sont pas pris en charge.

## Comment poursuivre la question du soi

L'hypothèse de travail reste un modèle de son propre fonctionnement, appris à
partir des conséquences de ses actions et utilisé pour choisir ses actions.
Cette hypothèse peut produire une métacognition fonctionnelle ; sa relation à
une conscience de sa propre existence reste à établir.

1. **Terminer le test actuel.** Vérifier si l'adaptateur aligné localise les
   perturbations sur les couches et phrases réservées mieux que la base, les
   cibles mélangées et le hasard informé de leur présence. Vérifier aussi le
   contrôle de lecture et les premiers tokens libres. Un résultat négatif sera
   conservé ; la géométrie ne le transformera pas en résultat positif.
2. **Si une capacité apparaît, chercher son mécanisme.** L'analyse des poids
   peut proposer des interventions. Il faudra retirer une composante candidate,
   puis la restaurer, avec témoins aléatoires de mêmes rang et norme, et contrôle
   des autres tâches. La composante sera choisie sur un lot de découverte et
   testée sur de nouvelles données, avec plusieurs apprentissages indépendants.
   Une suppression qui dégrade toutes les capacités ne serait pas sélective.
3. **Passer du rapport à la décision.** Dans un protocole distinct à fixer avant
   collecte, tester si le signal guide vérification ou abstention sans question
   d'introspection, en distinguant modification interne et difficulté externe.
   Comparer au suivi des réussites passées et à un observateur ayant les mêmes
   informations publiques. Un détecteur de rotation pourrait réussir l'étape 1
   sans comprendre ses capacités ni sa propre existence.

Ces étapes 2 et 3 sont des propositions, pas des expériences déjà exécutées.
Même une ablation suivie d'une restauration établirait un rôle causal dans une
fonction mesurée, pas une nécessité ou une suffisance pour le vécu subjectif.

## État de livraison

L'audit CPU et ses contrôles sont exécutables. Aucun nouveau calcul sur les poids
préentraînés de Qwen, entraînement A100 ou transfert sur iPhone n'est revendiqué.
Les données privées reçues et les anciens barèmes sont conservés.

Validation locale : **156 tests de recherche passent**, dont sept nouveaux
contrôles de géométrie. Un essai de bout en bout avec des fichiers safetensors
synthétiques retrouve la référence dense, conserve les empreintes et refuse
qu'un rapport écrase un fichier d'entrée. Ces contrôles sont logiciels.

Le [Colab de localisation déjà livré](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/c4b040c60f560a1db3f62236287df429d2f31306/notebooks/06_native_localization_colab.ipynb)
reste inchangé, avec son code scientifique fixé à
`9fc95269297854d5ac5b910904a13b033fd8be38`. Son résultat préentraîné n'est pas
encore reçu. Il n'y a pas de nouveau notebook à lancer pour cette note.
