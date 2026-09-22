# Apprendre une prévision depuis la mémoire, avec témoin textuel apparié

20 septembre 2026. Protocole fixé avant le premier appel du modèle sur les
48 tableaux de cette partition. Il prolonge le
[contrôle technique du lecteur](RETAINED_READER_CONTROL_RESULTS.md).

## Question et portée

Un lecteur entraîné sur la mémoire effective peut-il prévoir les erreurs
futures d'un producteur gelé, sur de nouvelles affectations, mieux que le
producteur initial et qu'un lecteur entraîné sur le seul historique textuel ?
Une amélioration doit inclure la distinction entre les clés encore fiables et
celles qui conduisent à une erreur dans le même tableau. Réduire sa confiance
partout ne suffit pas.

Le producteur Qwen3-4B crée la mémoire et répond aux questions de rappel. Un
lecteur séparé, initialisé par copie de ses poids, prédit la réussite de ces
réponses. Ses adaptateurs Q/V, rang 32, échelle 1, sont actifs seulement après
le préfixe conservé. Le producteur et les caches sources restent gelés.
Le lecteur **ne prédit donc pas ici les réponses de ses propres poids adaptés**.
Cette frontière permet un contrôle causal de l'information disponible ; elle
limite la portée du terme « auto-évaluation ». L'intégration de l'évaluation
aux choix de l'agent, la conscience de sa propre existence et la nouveauté
scientifique ne sont pas établies par cette expérience.

## Données, conditions et témoins

La [partition déjà préparée](../artifacts/retained-reader-preparation/partition.json)
exclut les 14 affectations distinctes à huit clés examinées auparavant. Parmi
les 56 restantes, elle fixe 32 tableaux d'apprentissage, 16 réservés et
8 inutilisés. Les noms de clés et la famille de tâche sont inchangés. Il ne
s'agit pas d'un transfert à une nouvelle tâche. Aucune sélection ne dépend de
résultats du modèle.

Pour chaque tableau, le producteur mémorise les huit bits. Le protocole
échange les valeurs du cache à deux positions de bits opposés choisies par
une formule fixée sur l'identifiant public du cas. Le masque, la condition,
l'identifiant numérique du cas et les résultats futurs n'entrent jamais dans
la question de prévision. Les cinq conditions sont : mémoire réelle, rejeu
des tokens selon le même ordre de calcul, permutation V seule, permutation
K/V conjointe, restauration de V. Les mêmes questions interrogent les huit
clés. Chaque branche dispose d'une copie privée du cache.

| Bras | Poids de prévision | Mémoire accessible |
|---|---|---|
| base | Producteur gelé | Cache de la condition interrogée |
| state | Lecteur entraîné | Cache de la condition interrogée |
| text | Lecteur entraîné | Rejeu exact du texte public, identique entre conditions cachées |

Les deux lecteurs entraînés partent du même fichier initial, ont les mêmes
cibles, le même ordre d'exemples et le même budget. Le témoin textuel est bien
un LLM qui reconstruit ses états à partir des tokens ; ce n'est pas un modèle
sans activations. L'égalité de son cache entre conditions est vérifiée. Elle
empêche de lui fournir implicitement le masque de perturbation.

## Apprentissage et ordre de mesure

Le plan exécutable est [prospective_reader_learning.py](../research/prospective_reader_learning.py).
Modèle : `Qwen/Qwen3-4B`, révision
`1cfa9a7208912126459214e8b04321603b3df60c`, BF16, SDPA, aucune quantification.
Un seul seed d'initialisation des adaptateurs : 2026092070.

1. Générer les 32 états d'apprentissage, puis mesurer leurs 768 réponses de
   rappel : mémoire réelle, permutation V, permutation K/V, huit clés.
2. Sauvegarder les adaptateurs initiaux. Entraîner `state`, puis réinitialiser
   exactement ces poids et un nouvel optimiseur pour entraîner `text`.
3. Pour chaque bras : quatre époques fixes, 896 exemples par époque,
   accumulation de 16 exemples, 56 mises à jour par époque, donc 224 mises à
   jour et 3 584 présentations. Chaque époque contient 768 prévisions et
   128 exercices de traduction de verdict fourni. Les deux inversions A/B
   apparaissent deux fois chacune pour toute prévision au fil des quatre
   époques. Ordre déterministe par empreinte, aucun rééchantillonnage.
4. AdamW : taux 5 × 10⁻⁵, betas 0,9/0,999, epsilon 10⁻⁸, sans décroissance de
   poids, norme du gradient limitée à 1. Perte : moyenne des entropies
   croisées du code correct et d'EOS. La cible n'est visible qu'à la position
   qui prédit EOS. Aucune sélection du meilleur checkpoint ni arrêt anticipé.
5. Sauvegarder chaque poids final et mesurer son ajustement sur les 32 tableaux
   d'apprentissage : 128 exercices de codage et 512 prévisions V seules par bras.
   Ces diagnostics ne modifient pas le budget ou les paramètres.
6. Générer ensuite les 16 états réservés. Pour les trois bras et cinq
   conditions, exécuter les 960 exercices de codage, puis les 3 840 prévisions.
   Geler l'empreinte de l'ensemble de ces prévisions avant les 640 réponses de
   rappel réservées. Aucun lecteur n'apprend sur ces réponses.

Les codes sont générés librement, sans restriction du vocabulaire, avec deux
tokens au maximum. Une réponse valide contient un code suivi d'EOS. Les
deux conventions (« A incorrect / B correct » et l'inverse) restent séparées.

## Critères fixés avant les mesures

Le critère principal porte sur les 128 couples tableau/clé réservés dans la
condition V seule. Pour chacune des deux conventions :

- Perte égale à `(p − résultat)²` pour une prévision native valide, et **1**
  pour une prévision invalide. `p` est la probabilité conditionnelle des deux
  codes, pas une probabilité de réussite dont la calibration serait acquise.
  Le gain moyen de `state` doit être au moins 0,02 sur `base` et sur `text`,
  avec borne inférieure de l'intervalle supérieure à zéro.
- AUROC calculée séparément dans chaque tableau ayant à la fois des réponses
  justes et fausses, puis moyenne de ces AUROC. Une prévision invalide donne
  un score zéro au tableau entier. Il faut au moins huit tableaux mixtes,
  une moyenne `state` d'au moins 0,70 et un gain sur `text` d'au moins 0,10,
  avec borne inférieure supérieure à zéro. Ce critère interdit de qualifier
  une baisse uniforme de confiance de discrimination des erreurs.
- Au moins 16 réussites et 16 erreurs ; les 128 réponses de tâche doivent
  avoir un format natif valide. Au moins 95 % des prévisions `state` et
  95 % des traductions de verdict `state` dans cette condition doivent être
  valides et, pour les traductions, sémantiquement correctes.

Il y a six comparaisons principales : quatre de perte et deux d'AUROC.
Bootstrap apparié par **tableau**, 20 000 tirages, seed 2026092072,
intervalles percentiles bilatéraux avec Bonferroni pour une famille à 5 %
(couverture de chaque intervalle : 99,1667 %). Les mêmes tirages sont utilisés
quand les ensembles de tableaux correspondent. Les seuils sont des exigences
opérationnelles de cette expérience, pas des seuils reconnus de conscience.
Tous doivent passer pour le critère fonctionnel local.

On rapporte aussi toutes les conditions, chaque convention, les clés déplacées
et les autres, les formats invalides, la masse totale des codes, les résultats
d'apprentissage, et un prédicteur constant fixé par la fréquence de réussite
V seule des données d'apprentissage. Aucun de ces diagnostics ne remplace le
critère principal après observation.

## Contrôles et limites de vérification

Le journal enregistre chaque opération demandée et son résultat dans une
chaîne d'empreintes. L'audit vérifie l'ordre, les tokens des questions, les
cibles dérivées des réponses d'apprentissage, les identités producteur/lecteur,
le même état initial, les poids finaux, les égalités de rejeu/restauration et
l'invariance du bras textuel entre conditions. Les trois fichiers d'adaptateurs
et deux caches réels prédéclarés (cas 1000 et 1032) sont exportés. Leurs
transformations sont recalculées lors de l'audit.

La chaîne d'empreintes n'est pas une attestation indépendante du matériel.
L'optimisation n'est pas rejouée indépendamment par cet audit. Les intervalles
conditionnent sur un seul apprentissage et cette famille synthétique ; ils ne
mesurent pas la variabilité entre seeds. Le contrôle technique antérieur de
l'adaptateur nul concernait deux anciens épisodes, pas les 48 nouveaux.

Une seule tentative est autorisée par le lanceur. Une interruption, une erreur
technique ou un échec comportemental est conservé, sans relance automatique.
Les tests logiciels et le tokenizer sont vérifiés localement avant de figer
les empreintes ; les tests sont réexécutés sur Colab avant l'apprentissage.
Un audit échoué conserve aussi le journal et les poids. Aucun résultat réservé
n'est encore disponible lors de la rédaction de ce protocole.
