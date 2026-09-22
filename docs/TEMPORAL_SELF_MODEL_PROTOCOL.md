# Protocole : apprentissage des effets retardés

Pilote défini avant entraînement. L'objectif de conscience subjective et de
contribution inédite reste ouvert. Ce protocole ne fournit aucun critère de
conscience et le principe d'identification temporelle n'est pas revendiqué comme nouveau.

## Question et hypothèse

Le module précédent énumérait six causes connues. Ici les poids d'un petit réseau
apprennent à prédire deux canaux anonymes à partir de sept instants d'actions et
d'indices publics. Aucun délai ni canal contrôlé n'est fourni au prédicteur.
La fenêtre temporelle et les variables action/indice restent fournies par conception.
On étudie un modèle d'effets, pas encore une mémoire récurrente apprise ou une
politique d'exploration apprise.

Comparer un réseau temporel (14 entrées, 32 unités tanh, 2 sorties), un réseau
instantané de capacité proche (2 entrées, 100 unités tanh, 2 sorties) et une
régression ridge temporelle. Les deux réseaux voient les mêmes échantillons
et masques. Les prédictions sont évaluées avant toute mise à jour sur le test.

## Mondes

Quatre familles : effet affine retardé ; interaction multiplicative entre une
action passée et un indice public, avec un terme quadratique ; absence d'effet
de l'action ; délai de dix instants dépassant la fenêtre de sept instants.
Trois graines de monde par famille, 300–302. Le canal affecté, le signe du gain
et le délai interne 1–6 sont tirés pour chaque monde. Les paramètres réels restent
dans le simulateur et l'évaluateur. Un canal extérieur dépend uniquement d'indices
publics retardés. Un bruit gaussien d'écart-type 0,03 s'ajoute aux observations.

Chaque monde fournit 1 024 instants : soit A=U tout au long, soit A=U pendant
512 instants puis A et U indépendants. Les variables sont uniformes sur [-1,1].
Un seul canal est lu à chaque instant, et 20 % des lectures manquent. Les masques
empêchent l'apprentissage sur les cibles non reçues. Les matrices de cibles du
simulateur sont construites par l'évaluateur ; elles ne sont pas des observations
accessibles au modèle en dehors du masque.

L'entraînement est hors ligne sur ces journaux : 600 mises à jour Adam, batch 64,
taux 0,01, trois initialisations indépendantes 11, 23, 37. Aucun réglage fondé sur
les résultats de test ne sera présenté comme prévu à l'avance. Ridge utilise une
pénalisation 0,1 et les seules lectures reçues. Les trois modèles apprennent par
monde : il ne s'agit pas de généralisation zéro exemple vers des mondes inconnus.

## Évaluation et contrôles

512 nouveaux instants par monde, avec actions et indices indépendants. Même bruit
et familles que l'entraînement, nouveaux tirages. Le test mesure l'erreur quadratique
de prévision sur les deux canaux, sans mettre à jour les poids.

Pour 32 historiques supplémentaires, imposer successivement +1, -1 et 0 à chaque
action passée de délai 0–10, en gardant les indices et autres actions identiques.
Comparer les trois paires (+1,-1), (+1,0), (-1,0) : la première seule annulerait
un terme quadratique et manquerait certaines dépendances à l'action.
Le simulateur recalcule les deux observations sans bruit ; le modèle prédit les
deux. La différence évalue l'effet causal, et non une seule lésion de son entrée.
Un délai hors fenêtre ne modifie aucune entrée du modèle, qui ne peut donc pas
le détecter. On mesure MSE des contrastes et erreur sur les seuls contrastes
non nuls, séparément pour éviter qu'une majorité de zéros masque un échec.

Les fichiers conservent par monde, politique et initialisation : paramètres
appris, MSE de prévision, MSE causale totale et sur effets non nuls, plus norme
des faux effets dans les mondes sans contrôle. La variance entre initialisations
est distinguée de celle entre mondes. Les poids et empreintes des sources
permettent de refaire l'évaluation sans réentraîner.

Échecs informatifs attendus : corrélation A=U insuffisante pour identifier une
cause ; réseau instantané insuffisant pour les retards ; ridge insuffisante pour
certains effets non linéaires ; aucun modèle à fenêtre de sept instants capable
d'identifier un effet uniquement au délai dix. Les résultats effectifs, y compris
les exceptions, seront conservés. Un succès ne prouve ni une conscience ni la
nécessité de cette architecture.
