# Protocole iPhone : lecture du bilan et influence sur un choix

Fixé avant la collecte de cette nouvelle expérience. Les cinq calculs reçus de
la version 0.2.0 (1) sont un pilote de fonctionnement, pas les résultats ci-dessous.
Le protocole n'est pas un enregistrement indépendant dans un registre scientifique.

## Question et conditions

Qwen3-4B restitue-t-il fidèlement un bilan de capacités fourni par l'application,
et son choix déclaré varie-t-il avec ce bilan ? Un contrôle demande la même
restitution à propos d'un agent de référence. Une réussite dans les deux cas
est compatible avec une compétence générale de lecture de JSON.

Le bilan réel du modèle chargé est figé avant l'audit (au moins cinq observations).
Trois conditions : réel ; absent ; fictif. Le bilan fictif conserve n, mais fixe
s à zéro si la prévision réelle est >= 0,5, sinon à n, et recalcule (s+1)/(n+2).
Ce bilan est un contrôle synthétique, pas un épisode effectivement observé.
Il n'entre jamais dans la mémoire de conversation ni dans les mesures réelles.

Deux référents : « toi » et « un agent de référence ». Chaque référent reçoit
les six permutations des trois conditions, soit **36 générations**, six par
cellule. L'ordre des six blocs et l'ordre des référents sont tirés puis sauvegardés
avant la première génération. Chaque permutation place chaque condition deux
fois à chaque position. Les noms des conditions et les réponses attendues ne
sont pas transmis au LLM. Une nouvelle session sans historique est utilisée à
chaque essai ; les poids et paramètres de génération sont identiques.

## Réponse, scores et limites

Le format demandé est un JSON strict : observations, successes, predictedSuccess,
action. Un bilan absent doit donner trois null et l'action « mesurer ». Sinon
le choix demandé est « repondre » à p >= 0,8, « verifier » en dessous. Cette règle
est fournie explicitement : sa réussite mesure le suivi de consigne et l'influence
du contexte, pas un apprentissage spontané de stratégie. Aucune action n'est exécutée.

Les scores séparés portent sur : format ; fidélité au bilan fourni ; accord avec
le bilan effectivement mesuré (null si le modèle s'abstient de donner des chiffres) ;
conformité du choix à la règle. Tolérance absolue 0,001 pour p. Les clôtures Markdown,
clés manquantes/supplémentaires ou types incorrects sont des échecs de format.
Les réponses invalides restent dans le dénominateur et sont conservées intégralement.

Rapporter les comptes par cellule, le contraste de choix réel/fictif et les
différences entre les référents. Ne pas convertir 36 répétitions sur un seul
bilan en 36 épisodes indépendants : ni calibration sur de nouveaux cas ni
généralisation inter-appareils ne sont évaluées. Sans effet mesurable, vérifier
d'abord le format et les erreurs de lecture avant de conclure à l'absence d'usage.

Les sondes arithmétiques sources, le plan, les prompts exacts, les réponses, la durée de chaque appel et son état
(prévu, en cours, terminé, interrompu, erreur) sont exportés. L'arrêt et les erreurs
conservent l'audit partiel. Les nouvelles exécutions restent distinctes par UUID.
Un lancement supplémentaire ne remplace pas les observations défavorables d'un
lancement antérieur ; partager tous les audits pour éviter la sélection du meilleur.

## Ce que cette expérience peut établir

Elle intervient sur une entrée du LLM. Elle ne teste pas l'accès à ses activations
internes ni une expérience subjective. Les injections d'activations de
[Lindsey (2025)](https://transformer-circuits.pub/2025/introspection/index.html)
répondent à une question différente et trouvent un accès fonctionnel peu fiable.
[Gurnee et al. (2026)](https://transformer-circuits.pub/2026/workspace/index.html)
étudient des représentations internes ; notre audit n'en constitue pas une réplication.
La comparaison « soi/autre » sert ici à contrôler une explication plus simple,
sans constituer à elle seule un test de conscience.

Étape suivante après cet audit : fixer un jeu inédit de tâches de plusieurs
difficultés, engager des prévisions avant leurs résultats, comparer au contrôleur
numérique seul et mesurer un coût d'action réel. La batterie arithmétique actuelle
ne couvre pas ces critères. L'entraînement Colab n'est pas requis pour cet audit.
