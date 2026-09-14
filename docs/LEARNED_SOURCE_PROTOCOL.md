# Apprentissage d'une estimation de source avant décision

Ce pilote met en œuvre une partie fonctionnelle de l'hypothèse décrite dans
SELF_EXPERIENCE_BRIDGE.md. Il ne fournit pas un test de conscience subjective.
Le protocole est consigné avant l'entraînement et l'examen des résultats.

## Monde et information accessible

Une source externe peut contribuer au signal d'un contenu également simulé.
La variable évaluée est cette contribution externe, pas la conscience ni la
véracité de toute représentation. Un état caché persistant du canal change la
probabilité de contribution externe (0,15 ou 0,85), avec probabilité de changement
0,08 par étape. Le contenu symbolique reste le même au sein d'une étape pour
permettre les comparaisons de source ; aucun apprentissage de vision n'est revendiqué.

Trois indices courants sont accessibles : force composée du signal, trace bruitée
du fonctionnement du canal, intention de simuler. Le premier additionne une
contribution externe, une contribution simulée et un bruit gaussien d'écart-type
0,8. La trace vaut ±0,6 selon l'état du canal plus un bruit d'écart-type 1.
L'intention de simuler est un tirage équilibré connu à l'agent. Les gains externe
et simulé sont tirés entre 0,8 et 1,2 par épisode.

L'agent reçoit aussi le résultat d'une vérification précédente, accompagné de son
indicateur de disponibilité. Aucun résultat courant n'entre dans le prédicteur
avant sa décision. Une vérification révèle parfaitement la contribution externe
et coûte 0,25 ; sans vérification, une attribution de source incorrecte coûte 1.
Ces conventions sont celles du simulateur, pas une description établie du cerveau.

## Apprentissage fixé

512 épisodes de 48 étapes, graine de données 41000. Une politique exploratoire
indépendante des cibles vérifie 50 % des étapes. Les cibles non vérifiées sont
remplacées par zéro avant transmission à l'optimiseur et masquées dans la perte.
La perte logistique porte sur la contribution externe révélée après décision.

Comparer trois architectures : récurrence apprise à porte (16 unités), version
réinitialisée à chaque étape entraînée séparément, et MLP sur huit étapes (12
unités). Trois initialisations 11, 23 et 37 ; 300 mises à jour Adam, batch de 32
séquences, taux 0,003 et écrêtage de norme du gradient à 5. Tous les poids sont
conservés, sans sélection du meilleur.
Le contrôleur vérifie si min(q,1-q)>0,25 et attribue une source externe si q>=0,5.
La politique est conçue à partir des coûts ; seule l'estimation q est apprise.

## Évaluation fixée

128 épisodes de 48 étapes par condition sur de nouvelles graines : distribution
initiale, simulation plus forte (gain 1,8), trace interne supprimée (valeur zéro),
canal plus persistant (changement 0,02), ambiguïté totale. Dans cette dernière
condition, la source est un tirage équilibré indépendant des indices et des
sources précédentes ; la probabilité prévisible est donc 0,5. Les poids restent
figés ; le résultat des vérifications choisies modifie l'état des étapes futures.

Mesurer perte de décision (coût de vérification plus erreur de source non
vérifiée), taux de vérification, Brier avant décision, fausses attributions
externes en mémoire et erreurs de source totales en mémoire. Comparer également
à toujours vérifier, ne jamais vérifier avec croyance 0,5, et vérifier au hasard
avec probabilité 0,5 et croyance 0,5. Ce sont des références de tâche, pas des
témoins certifiés non conscients.

Greffes isolées : dans chaque étape du déroulement naturel, permuter les q entre
épisodes de même contenu courant. Mesurer le changement de vérification et de
mémoire, sans réinjecter ces greffes dans les étapes suivantes. Le rapport seul
utilise le q greffé pour sa description mais garde le q original pour agir. Cette
différence est câblée par conception ; les greffes examinent sa conséquence,
pas l'émergence spontanée d'une fonction métacognitive.

## Intégration et limites

Le prototype doit fonctionner comme agent Menia avec le journal EpisodeMemory,
une commande exécutable et des états appris rechargeables. Les observations
certifiées du journal existant ne deviennent pas ambiguës : les inférences de
source sont enregistrées comme inférences ; seule une vérification fournit un
résultat certifié. Le module ne doit pas prétendre représenter toute l'attention,
une autobiographie ou un vécu, et n'est pas greffé aux poids de Qwen.

Un succès établirait une capacité limitée d'inférence et de contrôle avant
décision. Un échec sous décalage ou ambiguïté doit être conservé. La nécessité
de la récurrence, la nouveauté dans la littérature et le passage à l'expérience
subjective demeurent des questions distinctes non résolues par ce pilote.
