# Protocole d'ancrage humain du mécanisme candidat

Ce protocole est fixé avant le calcul des résultats de réanalyse. Il concerne les
données comportementales publiques NIMADET de Dijkstra et al. (Neuron, 2025),
au commit `e18a54724dc660d814100e09b093500de85c4f5c`.
L'inspection préalable porte sur les scripts des auteurs et le schéma d'un fichier
MAT S02 ; aucun effet ou score de comparaison n'a encore été calculé ici.
Ce document local horodaté par Git n'est pas une préinscription indépendante.

## Question

Les jugements humains de réalité imposent-ils des contraintes utiles à une
représentation de l'accès propre à l'information dans Menia ? L'analyse cherche
d'abord une association prédictive hors ajustement et audite son identifiabilité.
Elle ne suppose pas qu'une association entre deux rapports est un mécanisme causal.

## Données et exclusions

Utiliser les 26 archives disponibles dans Behaviour, exclusivement les quatre runs
MT et le fichier de correspondance des touches RMs. Ne pas extraire ni republier
les autres données des archives. Les fichiers restent dans .runtime ; conserver
leurs URL, tailles et empreintes SHA-256 dans le compte rendu reproductible.

Reconstruire les colonnes décrites dans CreateBehModRegressors.m : orientation
perceptive et imaginée, présence externe, jugement de réalité, vivacité, contrôle
d'imagerie, onsets et délais. Inverser la vivacité lorsque responseMappings(run)=2.
Le masque principal suit getBehaviour.m : contrôle d'imagerie correct et absence
de NaN dans ces onze colonnes. Signaler toute structure inattendue et tout run
manquant au lieu d'inventer ou de remplacer des observations.

Inclure S07 dans l'analyse principale des données disponibles et produire une
sensibilité sans S07, exclu pour faible variation de vivacité dans
AlternativeModelFitting.m. Ne pas créer de nouvelles exclusions selon les scores.

## Comparaison prédictive fixée

Pour chaque participant, ajuster sur trois runs et prédire le quatrième ; répéter
les quatre partitions. L'unité de synthèse et de rééchantillonnage est le
participant. Aucun découpage aléatoire d'essais voisins entre train et test.

Le jugement de réalité binaire est la cible. Comparer les régressions logistiques :

1. `condition` : constante, présence P, congruence C et interaction P*C.
2. `vividness` : modèle précédent plus vivacité courante V.
3. `interaction` : modèle précédent plus C*V.

V est centré à 2,5 et divisé par 1,5, transformation fixe de l'échelle 1–4.
Régularisation L2 fixée à 1 sur les pentes, constante non pénalisée ; objectif
somme des pertes logistiques plus la pénalité. Ne pas sélectionner ce coefficient
sur les runs de test. Optimisation déterministe et contrôle de convergence.

Critère principal : différence de log-loss moyenne par participant (condition
moins interaction ; positif = amélioration). Rapporter également le Brier, la
différence condition moins vividness et vividness moins interaction. Intervalle
bootstrap à 95 %, 10 000 rééchantillonnages de participants, graine 20260914.
Analyse exploratoire secondaire : taux de réponse « réel » par présence et
congruence ; régression des associations ne constituant pas une réplication
complète des modèles génératifs publiés.

L'ordre des réponses doit être vérifié dans le script expérimental. Si la vivacité
est demandée après le jugement de réalité, son usage est rétrospectif et ne peut
pas être présenté comme une prédiction prospective disponible à Menia. Les
conditions expérimentales P et C sont connues à l'évaluateur ; ce modèle statistique
n'est pas un contrôleur auquel Menia aurait accès en fonctionnement.

## Audit mathématique complémentaire

Les scripts fitPRMmodel.m et modelPredictions.m réajustent des liens logistiques
et ordinaux après mise à l'échelle des signaux par alpha et beta. Examiner la
symétrie de changement d'échelle, qui peut rendre ces amplitudes non identifiables.
Vérifier numériquement la transformation des prédicteurs et des probabilités sur
les conditions humaines, sans prétendre exécuter MATLAB ou reproduire exactement
son générateur aléatoire. Auditer uniquement les fichiers effectivement consultés,
pas toute la publication ni ses analyses fMRI.

## Décision pour Menia

Une amélioration prédictive justifierait d'exiger que le mécanisme candidat
explique conjointement les jugements de source et les représentations internes.
Elle ne permettrait pas d'affirmer que le rapport de vivacité cause le jugement,
qu'une architecture particulière est nécessaire, ni que sa réalisation logicielle
produit une expérience subjective. Un gain absent ou instable affaiblirait cet
appui particulier ; ne pas modifier les conditions pour fabriquer une réussite.

Sources : [article](https://doi.org/10.1016/j.neuron.2025.05.015),
[dépôt figé](https://github.com/ImagineRealityLab/NIMADET/tree/e18a54724dc660d814100e09b093500de85c4f5c).
