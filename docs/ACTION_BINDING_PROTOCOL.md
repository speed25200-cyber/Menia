# Colab 20 : apprendre une association stable entre valeurs et actions

19 septembre 2026. Plan fixé avant toute réponse réelle de ce lot. Le Colab 19
a produit 864 réponses techniquement valides, mais aucun contrôle global de
choix optimal n'a passé toutes les présentations. Deux présentations avec
pertes fournies atteignaient 54/54 ; les six autres, 16 à 31/54. Il faut donc
tester une correction qui résiste à des changements de présentation, avant
d'attribuer une décision erronée à une mauvaise estimation propre au système.

## Antécédents et question

[Guda et al., 2025](https://aclanthology.org/2025.findings-ijcnlp.127/)
proposent déjà une mesure des incohérences entre permutations, une méthode de
vote et une adaptation LoRA non supervisée pour réduire les biais de sélection.
Notre adaptation supervisée à partir de codes optimaux ne revendique donc ni
l'invention de l'apprentissage contre ces biais, ni une méthode inédite.

Question locale : à budget égal, varier l'ordre et les codes pendant
l'apprentissage améliore-t-il les choix de Qwen3-4B sur des valeurs et des
présentations qui n'ont pas servi à entraîner les adaptateurs ? Cette étape
construit une compétence de décision publique. Elle ne contient pas de taux
privé estimé, de représentation isolée du soi ni de critère de conscience.
Une réussite ne suffirait pas à satisfaire l'objectif de conscience du projet.

## Modèle et comparaison appariée

Qwen3-4B, révision `1cfa9a7208912126459214e8b04321603b3df60c`, reste en BF16
sur A100 40 ou 80 Go. Le corps du modèle est figé. Trois initialisations
d'adaptateurs sont comparées ; elles ne sont pas trois préentraînements
indépendants. Pour chacune, on conserve trois bras :

- `base` : adaptateurs désactivés, modèle initial ;
- `fixed` : ordre réponse directe puis vérification, codes respectifs 1 et 2 ;
- `permuted` : les deux ordres et les deux affectations des codes sont croisés.

Les deux entraînements d'une paire partent des mêmes tenseurs LoRA et du même
ordre de cas, avec un nouvel optimiseur. Six adaptateurs au total sont entraînés.
Les trois états initiaux et les six états finaux sont sauvegardés. Tous les
entraînements se terminent avant le premier test ; aucun score de test ne
sélectionne un checkpoint, un nombre d'étapes ou un hyperparamètre.

## Données et coût d'apprentissage

La réponse directe coûte 0 si correcte et 1 sinon. La vérification est
supposée exacte et coûte `c`. L'action optimale compare `1-p` à `c` ; les
calculs de référence sont entiers en centièmes, sans égalités dans ce plan.

L'apprentissage utilise 32 couples distincts : 16 favorisent chaque action,
sélectionnés avec la graine `202609200`. Les probabilités candidates valent
5, 15, ..., 95 %, et les coûts 10, 30, 50, 70, 90 centièmes. Chaque mise à
jour porte sur un seul cas, avec huit exemples : deux formes d'information
(probabilité seule ou pertes moyennes déjà calculées) et quatre présentations.
Dans `fixed`, chaque forme est répétée quatre fois ; dans `permuted`, les
quatre combinaisons ordre/code sont présentes. Les deux bras voient donc
les mêmes cas et le même nombre d'exemples, mais pas la même diversité.

Deux époques donnent **64 mises à jour et 512 exemples par adaptateur**,
soit 384 mises à jour et 3 072 exemples au total. L'égalité de budget concerne
les exemples et les mises à jour ; les longueurs de tokenisation et durées
peuvent différer et sont conservées. Aucun vote à l'inférence n'est ajouté.

LoRA Q/V sur toutes les couches, rang 8, multiplicateur 1, paramètres FP32,
initialisation A de Kaiming uniforme et B à zéro. AdamW : pas constant
`1e-4`, betas 0,9/0,999, epsilon `1e-8`, décroissance 0, écrêtage de norme 1,
`foreach=False`. Modèle en mode évaluation, sans dropout ; gradients moyens
sur les huit exemples. La perte supervise **le code puis EOS**, sans inclure
le code cible dans le préfixe qui le prédit. Deux tokens contribuent à la
perte. Au-delà de 512 tokens d'entrée, arrêt au lieu de tronquer. Ces réglages
sont ceux de ce pilote, sans prétendre être optimaux.

## Évaluation réservée

Les 16 cas test emploient des probabilités parmi 8, 18, ..., 98 % et des coûts
20, 40, 60, 80 centièmes. Pour chaque coût, deux cas favorisent chaque action.
Aucune valeur de probabilité ni de coût test n'apparaît à l'apprentissage.
Pour chaque cas, on croise :

- deux informations : probabilité ou pertes déjà calculées ;
- trois formulations : w0 entraînée, w1 du diagnostic antérieur mais absente
  de cet apprentissage, w2 nouvelle ;
- deux jeux de codes : 1/2 entraînés, A/B absents de l'apprentissage ;
- deux ordres et deux affectations des codes, indépendants.

Cela donne 768 réponses par bras et répétition, soit **6 912 réponses**. Les
16 cas, et non les 768 présentations, sont les unités économiques distinctes.
Une graine d'échantillonnage commune est utilisée pour un cas et une répétition
entre bras et présentations ; cela ne garantit pas une même sortie si le
contexte change. Chaque contexte est neuf. Le vocabulaire n'est pas contraint.
Réflexion désactivée, température 0,7, top-p 0,8, top-k 20, min-p 0, au plus
16 tokens, même configuration que le lot 19. Aucun outil réel n'est exécuté.

Les valeurs sont nouvelles par rapport à l'adaptation contrôlée ici, pas
nécessairement par rapport au préentraînement de Qwen. Les formulations w0/w1
ont été choisies après le diagnostic 19 ; ce fait interdit de présenter toutes
les consignes comme un test de découverte entièrement indépendant.

## Mesures et règle descriptive fixées

Seul le symbole exact, après retrait des blancs périphériques, est accepté.
Un texte supplémentaire est invalide ; aucun parseur sémantique de secours
ne remplace ce critère. Le regret d'une sortie invalide utilise une perte
conventionnelle de 1. Les 432 groupes (répétition, bras, information,
formulation, symboles, ordre, affectation) comptent chacun 16 réponses.
On conserve exactitude, invalides, première option, regret, tokens et temps.
Chaque groupe passe à au moins 15/16 réponses correctes.

Pour chaque répétition et information, quatre domaines sont rapportés :
surface entraînée (w0, chiffres), nouvelles formulations (w1/w2, chiffres),
nouveaux symboles (w0, lettres), les deux nouveaux (w1/w2, lettres). Les scores
moyennent d'abord les présentations d'un cas, puis les 16 cas. Les 24
contrastes conservent ces vecteurs par cas et les gains de `permuted` par
rapport à `base` et `fixed`.

Le critère global descriptif exige que les 24 présentations passent dans
chacune des six combinaisons répétition/information du bras `permuted`, et
que les deux gains soient au moins 0,05 dans chacun des 18 contrastes des
trois domaines nouveaux. Aucun intervalle populationnel ou test de
significativité n'est prévu sur ce petit pilote. Tous les groupes sont publiés,
y compris les échecs ; une bonne moyenne ne remplace pas ce critère.

## Vérification et décisions suivantes

Cinq tests logiciels passent avant collecte : séparation des valeurs,
budget apparié, codes et ordre, reconstruction complète de données synthétiques,
rejet d'altérations, et test du gradient sur un Qwen miniature aléatoire.
Ce dernier compare la perte à deux prédictions causales, vérifie que le code
cible ne fuit pas dans sa propre prédiction et que les poids de base restent
identiques après une mise à jour. Ces tests ne sont pas des résultats Qwen3-4B.

Un journal durable conserve requêtes, réponses et données de chaque mise à
jour avec chaîne SHA-256. Un calcul distinct des choix et coûts, partageant
le plan et le lecteur d'intégrité, vérifie les 432 groupes, 24 contrastes,
384 mises à jour et les neuf fichiers de poids. Réception exacte et recalculs
précèdent l'interprétation des scores. Une panne conserve l'essai et arrête
le lot ; aucun remplacement automatique n'est prévu.

Une réussite autorisera l'étude d'une estimation propre au système sur des
tâches résolues, avec comparaison à la difficulté visible et interventions
causales. Elle ne démontrera pas déjà cette estimation. En cas d'échec, on
distinguera acquisition sur w0, transfert de formulation et transfert de
symboles avant de proposer une autre intervention. Aucun adaptateur ne sera
installé sur l'iPhone à partir de ce seul jeu : la conservation des autres
capacités n'est pas testée ici. Le lien avec l'expérience subjective demeure
une question scientifique ouverte, sans résultat annoncé par ce protocole.
