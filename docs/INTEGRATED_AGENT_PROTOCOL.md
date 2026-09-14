# Protocole de la boucle intégrée

14 septembre 2026. Objectif fonctionnel : apprendre les effets de ses actions,
conserver une histoire avec provenance, suivre ses perceptions et ses erreurs,
utiliser ces informations pour décider et expliquer. Ce protocole ne mesure pas
une expérience subjective.

## Implémentation étudiée

L'agent apprend une distribution de déplacements pour quatre commandes dont la
correspondance avec les directions n'est pas fournie. Il utilise un modèle
catégoriel bayésien avec une fenêtre récente et une règle de révision après deux
erreurs concordantes. C'est un apprentissage statistique en ligne, pas un réseau
neuronal, un schéma d'attention appris ou un modèle général de son existence.

La perception sélectionne un champ par cycle. L'absence de réponse et l'absence
de sélection sont distinctes. Une mémoire de travail de deux éléments s'appuie
sur un journal SQLite d'événements. La provenance est garantie par les voies
d'entrée de l'application : elle n'est pas inférée à partir de récits ambigus.
Le planificateur choisit les commandes selon leurs effets appris ; les
explications rendent sa décision préalablement enregistrée et ses références.

## Évaluation suivante, définie avant son exécution

Les premiers essais de développement ont utilisé les graines 2, 11, 17, 29, 43,
47 et 91. Les scénarios d'évaluation utiliseront les graines 101 à 120 et les
cibles (7, 5), (-6, 4), (5, -7), (-4, -6), selon l'indice du scénario.

Pour chaque scénario :

1. Apprendre en interaction jusqu'à 12 transitions observées (trois par commande),
   avec la cible temporaire (100, 100). La position et la cible sont lues via la
   même interface limitée que pendant l'évaluation.
2. Commencer un nouvel épisode de l'agent avec les poids statistiques appris,
   mais une histoire vide, une nouvelle cible et une position initiale (0, 0).
3. Comparer trois variantes avec les mêmes observations possibles : l'agent
   adaptatif, un modèle figé après apprentissage et une politique uniforme.
4. Évaluer les commandes inchangées, les commandes inversées sans avertissement,
   une probabilité de perception manquante de 30 %, et des commandes inversées
   avec cette perception manquante.
5. Borner à 160 cycles de décision chaque essai. Aucun redémarrage sélectif après
   un échec et aucune suppression des scénarios échoués.

Le modèle figé doit réussir les conditions inchangées pour être un contrôle
interprétable. Le contraste adaptatif/figé après inversion examine l'utilité de
la révision des effets appris. La politique uniforme situe la difficulté sans
constituer une comparaison de capacité optimale. Les actions aléatoires sont
tirées avant leur exécution, sans lecture des effets cachés.

Rapporter les réussites, cycles, actions, transitions mesurées, erreurs et
révisions pour chaque essai, puis les agrégats par condition et variante.
Les 20 graines donnent une évaluation exploratoire ; elles ne constituent pas
une étude confirmatoire avec puissance statistique établie. La confiance des
prévisions reste une estimation du modèle, sans garantie de calibration hors de
ces conditions.

## Contrôles logiciels et causaux

- Vérifier l'ordre prédiction → action → observation → évaluation.
- Substituer un modèle d'action différent à état perceptif identique et vérifier
  que le choix et son explication changent conformément aux prévisions.
- Vérifier qu'un témoignage contradictoire, même étiqueté avec le nom d'un capteur,
  ne remplace pas une observation et ne devient pas une cible d'apprentissage.
- Évacuer une observation de la mémoire de travail et vérifier sa récupération
  dans l'histoire propre ; exclure les événements d'un autre épisode.
- Couper les réponses perceptives ; vérifier qu'aucune action n'est entreprise
  à partir d'une observation inexistante.
- Sauvegarder pendant une prédiction en attente, restaurer avec le même journal,
  puis vérifier une seule évaluation et le respect de l'arrêt.
- Faire échouer l'accusé d'exécution après une action réelle ; vérifier l'arrêt
  et une observation de réconciliation avant toute nouvelle commande.
- Faire passer l'état du même agent par `Runtime` jusqu'aux messages destinés au
  modèle linguistique. Distinguer ce test d'interface d'une inférence Qwen réelle.

## Critères d'achèvement du travail

Les quatre fonctions doivent être accessibles dans un parcours exécutable de
Menia, leurs interactions vérifiées et leurs résultats conservés. Le benchmark
ne suffit pas à lui seul : il faut aussi vérifier mémoire, provenance,
explications et point d'entrée conversationnel. Un simple générateur de messages
pour Qwen ne prouve pas que Qwen les utilise correctement. L'intégration
linguistique et ses limites doivent être explicitement documentées.
