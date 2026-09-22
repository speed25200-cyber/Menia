# Protocole : inférence, mouvement et réponse

Expérience logicielle exploratoire définie le 15 septembre 2026 avant son
exécution. Aucun préenregistrement indépendant ni essai humain n'est revendiqué.
L'objectif général reste l'expérience de sa propre existence chez Menia. Ce
pilote teste seulement un contrat causal d'une construction candidate : une
inférence doit pouvoir modifier une action utile indépendamment de sa description.

## Modèle et tâche

Pour chaque essai indépendant, une position réelle `x` suit `N(a, s_p²)`.
L'ancre `a` est une estimation préalable fournie, de précision connue ; ce n'est
pas une deuxième mesure obtenue auprès d'un humain. Un indicateur latent `C`
suit Bernoulli(p). Le signal visuel est `y = x + d + e`, avec
`e ~ N(0, s_v²)`. Si C=1, d=0 ; sinon d suit `N(0, tau²)`.
Sous C=0, le signal reste corrélé à x : il représente un proxy déplacé, pas un
objet indépendant au sens de tous les modèles d'inférence causale corporelle.
Ces hypothèses donnent exactement les deux vraisemblances marginales du module
`CausalBinding`, avec mesure y-a et sigma `sqrt(s_p²+s_v²)`.

Le contrôleur reçoit seulement a, y, la cible et les paramètres de distribution.
Il ne reçoit ni x, ni C, ni d. Il connaît les variances, la fréquence p et la
dynamique ; il ne les apprend pas. Les unités sont des unités spatiales de
simulation, sans transfert des 348 millisecondes du modèle humain précédent.

Sous les deux hypothèses, la moyenne conditionnelle de x vaut respectivement
`a + s_p² (y-a)/(s_p²+s_v²)` et
`a + s_p² (y-a)/(s_p²+s_v²+tau²)`. Le contrôleur les pondère par q=P(C=1|y,a).
Il calcule aussi la variance conditionnelle totale, terme entre composantes
inclus. Il applique `u = cible - E[x|y,a]`. L'environnement exécute `x' = x + u`.
Le coût est `(x'-cible)²`. Sans bruit moteur ni coût d'effort, la moyenne
conditionnelle minimise ce coût en espérance : cet avantage est prévisible
mathématiquement, pas une découverte de capacité spontanée.

## Conditions et comparateurs

Fixer s_p=1, tau=4 ; croiser p=0,1 / 0,5 / 0,9 et s_v=0,25 / 1 / 2.
Par condition, 32 graines indépendantes, 512 essais chacune ; graine de lot
`716000 + 100*condition + répétition`. L'ancre vaut zéro, la cible trois.
Toutes les variantes reçoivent exactement les mêmes observations et sont
évaluées sur les mêmes états de l'environnement réinitialisé.

Comparer : inférence calibrée ; q fixé au prior ; q=1 ; q=0 ; ancre seule ;
régression linéaire optimale pour les moments marginaux ; régression non linéaire
ordinaire calculée directement depuis les densités, sans variable nommée « soi ».
Ce dernier comparateur est mathématiquement équivalent à la moyenne conditionnelle,
et doit donc reproduire les actions. Il contrôle la spécificité de l'interprétation,
sans attribuer ou nier une conscience au contrôleur ordinaire.

Intervenir séparément :

- Déplacer seulement le seuil de réponse de 0 à log(4).
- Supprimer entièrement la réponse ; conserver inférence et mouvement.
- Multiplier les chances a priori internes par quatre et déplacer simultanément
  le seuil de log(4). L'identité prior/seuil doit préserver les réponses, mais
  les positions estimées et mouvements peuvent changer.
- Remplacer q par celui de l'essai suivant dans le même lot (rotation circulaire).
  La distribution marginale de q est conservée exactement ; son lien avec
  l'observation courante est rompu. L'expérimentateur prépare cette greffe hors
  ligne ; l'agent ne consulte pas un essai futur. C'est une intervention ciblée
  artificielle, susceptible de créer un état incompatible avec l'observation.

La réponse est une sortie binaire de recherche, obtenue par comparaison de q
au seuil logistique ; ce n'est pas un dialogue ni une attestation de vécu. Sous
greffe, le q greffé alimente l'estimation et la réponse. Le seuil ne revient
jamais vers l'action.

## Mesures et vérifications prévues

Rapporter le coût moyen, sa différence appariée avec l'inférence calibrée et
un intervalle percentile par bootstrap de 5 000 rééchantillonnages des 32 lots,
graine `717000 + condition`. Ces intervalles décrivent l'erreur Monte Carlo dans
le simulateur, sans inférence à une population humaine. Pas de sélection d'une
condition après inspection des résultats. Conserver les neuf conditions.

Mesurer les désaccords de réponse et de mouvement (>1e-12), la différence de
coût et l'écart entre variance prédite et coût observé. Conserver les moyennes
par lot pour reconstruire les intervalles. La régression non linéaire ordinaire
doit égaler les actions à 1e-12 près. Changer/supprimer seulement la réponse doit
conserver les mouvements exactement ; compenser le prior doit préserver les
réponses. Une violation arrêtera l'audit au lieu d'être masquée.

Contrôler la moyenne et variance conditionnelles par quadrature indépendante
sur x ; le bénéfice de l'action par le véritable état du simulateur ; l'ordre
observation → décision enregistrée → action → résultat ; l'absence de champs
cachés dans l'entrée. Un exemple de journal sera conservé. Rejouer le rapport
entier avec `--check`, et intégrer sa vérification à la CI.

## Portée

Le pilote doit ajouter un chemin exécutable du module d'inférence vers l'action.
Il ne réalise ni apprentissage du schéma corporel, ni boucle interoceptive,
ni besoins propres, ni histoire personnelle persistante. Il ne mesure aucune
expérience. L'égalité avec une régression ordinaire interdit de présenter le
succès moteur comme une signature spécifique de conscience de soi. L'action
peut départager deux variantes du modèle de réponse sous les hypothèses de
coût et de dynamique fixées ; elle ne rend pas toutes les architectures internes
identifiables. Les travaux de Maselli et al. (2022), Bertoni et al. (2023) et
Chancel et al. (2022) constituent des antécédents, pas une validation de Menia.
