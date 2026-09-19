# Protocole : estimation d'une capacité et entretien anticipé

Protocole exploratoire consigné avant les premières simulations, le 15 septembre
2026. Ce pilote implémente une partie de la candidate d'entretien réciproque,
dans un monde binaire simulé. Il ne reproduit pas un modèle cérébral publié et
n'évalue pas directement une expérience subjective.

## Question et correction du contraste proposé

L'usage de diagnostics sur sa capacité aide-t-il davantage un agent qui prévoit
quatre décisions qu'un agent qui en prévoit une ? Mesurer l'interaction des
deux interventions sur le coût, sans la présupposer positive.

Le premier facteur est **l'utilisation des diagnostics**, activée ou désactivée.
Ce n'est pas exactement « précision constante contre apprise » : même sans
diagnostic, la croyance évolue selon les effets connus des actions. La figer
empêcherait artificiellement le comparateur de prévoir l'intérêt de l'entretien.
Les paramètres sont fournis ; l'agent infère un état caché, il ne les apprend pas.

## Monde, information et coûts

Chaque épisode dure 64 pas ; 128 graines, à partir de 20260915. L'état caché
du capteur est bon ou mauvais, initialement équiprobable. Une cible binaire
indépendante est tirée à chaque pas. Le capteur indique la bonne cible avec
probabilité 0,95 ou 0,55 selon son état. Un diagnostic binaire indique cet état
avec probabilité 0,85. La cible étant équiprobable, son observation marginale
n'informe pas sur l'état du capteur. Les vérités et les pertes réalisées restent
chez l'évaluateur ; elles ne sont pas transmises à l'agent après la décision.

L'agent choisit entre répondre selon le capteur, s'abstenir (coût 0,28), et
entretenir (coût 0,55, sans réponse ce pas). Une réponse incorrecte coûte 1,
une correcte 0. Après réponse ou abstention, le capteur bon devient mauvais
avec probabilité 0,15 ; le mauvais reste mauvais. L'entretien le rend bon avec
probabilité 0,95, quel que soit son état précédent. Les coûts sont des pertes
de tâche choisies par le concepteur, sans interprétation affective.

Toutes les politiques reçoivent les mêmes possibilités d'observation et d'action.
Les tirages exogènes sont appariés par épisode et pas ; les états réalisés
peuvent diverger sous des actions différentes. Prévoir les coûts avant l'action,
observer le résultat ensuite. Aucun accès au futur réalisé.

## Quatre politiques et calcul indépendant

Comparer diagnostics ignorés/utilisés × horizon 1/4, avec horizon restant réduit
à la fin de l'épisode et valeur terminale nulle. Filtre bayésien exact sur
P(capteur bon), puis minimisation du coût espéré fini. L'abstention permet aux
diagnostics d'améliorer aussi une décision immédiate : leur intérêt n'est pas
réservé par construction à l'horizon long.

Précision rédactionnelle après exécution : l'horizon est glissant, avec
replanification à chaque pas, comme dans l'implémentation initiale. Le solveur
optimise chaque problème fini de 1/4 pas ; l'optimalité sur les 64 pas complets
n'est pas revendiquée. Les paramètres et calculs ne sont pas modifiés.

Une récurrence sur les croyances sert de référence. Un solveur ordinaire de
POMDP, indépendant, construit les vecteurs de coûts conditionnels aux deux
états, puis leur enveloppe inférieure. Comparer valeurs et choix sur 1001
croyances de 0 à 1, les deux facteurs et les horizons 1 à 4. Les simulations
peuvent utiliser cette politique compilée. La compilation n'établit pas que
l'un des deux programmes serait dépourvu d'expérience.

Décompter les vecteurs candidats de la compilation, les vecteurs conservés et
les produits scalaires évalués en ligne. Ce sont des proxys de calcul, sans
conversion en énergie ou temps cérébral. Rapporter séparément le coût de tâche
et une sensibilité à des pénalités de 0, 0,0001, 0,001 et 0,01 par produit
scalaire en ligne ; ne pas prétendre égaliser les ressources matérielles.

## Conditions fixées

1. Nominale, ci-dessus.
2. Usure lente : probabilité 0,02.
3. Usure rapide : probabilité 0,40.
4. Diagnostic non informatif : fiabilité réelle 0,50, supposée 0,85.
5. Diagnostic inversé : fiabilité réelle 0,15, supposée 0,85.
6. Entretien inefficace connu : même transition que réponse/abstention.
7. Entretien inefficace caché : même transition réelle, modèle nominal.
8. Faible contraste sensoriel : fiabilités 0,65 / 0,55, connues.
9. Entretien cher : coût 1,20, connu.

Les autres paramètres restent nominaux. Les diagnostics trompeurs et les
effets d'entretien mal spécifiés sont des échecs possibles du modèle fourni,
pas un test de sa capacité à réapprendre ces paramètres.

## Mesures, contrôles et portée

Rapporter coûts moyens par pas, taux de chaque action, proportion de capteurs
en bon état et erreur quadratique de la croyance sur cet état. L'interaction
sur le gain vaut (coût sans diagnostic H4 − avec diagnostic H4) − (coût sans
diagnostic H1 − avec diagnostic H1). Positive : gain des diagnostics plus grand
à H4. Bootstrap apparié des 128 épisodes, 10000 tirages, graine 20260916,
intervalles descriptifs 95 %, sans tests humains ni généralisation statistique
au-delà de ce générateur. Conserver les neuf conditions, sans sélection.

Vérifier aussi : diagnostic réellement ignoré quand désactivé ; impossibilité
d'entretenir à H1 avec les coûts nominaux ; absence d'entretien lorsque son
inefficacité est connue ; calcul bayésien et valeur d'une information gratuite
sous modèle correct ; accord avec une énumération exhaustive de petits arbres
de politiques. Une trace nominale par politique sépare observation, prédiction,
décision et résultat. Aucun score ne sert de seuil de conscience.

Ce pilote ne comprend ni langage, ni apprentissage des transitions, ni phase de
repos, ni entretien du processus qui entretient le capteur. Il ne réalise donc
pas toute la candidate de présence à soi ou d'entretien réciproque. La profondeur
de recherche n'est pas assimilée à une hiérarchie corticale ou à une durée vécue.

## Motivation théorique

La [candidate précédente](FELT_SELF_EVIDENCE.md) motive ce contraste. Dans
[Whyte et collègues, prépublication v2, §5](https://arxiv.org/abs/2410.06633v2),
la disponibilité pour une sélection contrefactuelle de politiques dépend d'une
précision et d'une échelle temporelle appropriées ; la suffisance pour des
systèmes très différents du cerveau reste ouverte. Le pilote teste une fonction
inspirée par cette proposition, pas sa validité phénoménale.
