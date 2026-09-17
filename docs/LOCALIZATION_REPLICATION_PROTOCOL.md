# Reproduire le signal interne et contrôler les biais de position

**État : le [premier export est reçu et vérifié](LOCALIZATION_REPLICATION_RESULTS.md).
Le gain du Colab 08 ne se reproduit pas : localisation forte à 15/48, 28/48 et
15/48 contre 24/48 pour la base, et contrôle visible en échec sous numéros
inversés.** Le plan ci-dessous a été fixé avant cette collecte et n'est pas modifié.
Le [Colab 09](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/codex/recall-reliability/notebooks/09_localization_replication_colab.ipynb)
utilise un seul bloc, sans Drive ni import manuel.

## Question et point de départ

Le [diagnostic reçu](LEARNING_DIAGNOSTIC_RESULTS.md) réussit les contrôles visibles,
mais la localisation interne forte reste incomplète : 42/72 bonnes réponses avec
les absences, 23/48 sur les perturbations présentes, contre respectivement
24/72 et 24/48 pour la base. Le témoin mélangé obtient 9/48. Les interventions
changent le chiffre choisi dans 25/48 cas pour l'adaptateur fort, contre aucune
pour la base. Il existe donc une sensibilité partiellement informative dans
ce montage, sans gain sur le contraste principal contre la base.

**Cette sensibilité se reproduit-elle avec d'autres initialisations et données,
et suit-elle le numéro public de la phrase perturbée quand son contenu et sa
position changent ?** C'est la question de cette expérience. Les anciens jeux
de test ne servent pas à choisir un meilleur checkpoint. La couche, l'intensité,
le rang, le taux d'apprentissage et le nombre d'époques restent fixes.

La distinction est motivée par les critères fonctionnels de
[Lindsey](https://transformer-circuits.pub/2025/introspection/) : un compte rendu
doit être exact, dépendre de l'état interne et ne pas seulement suivre un biais
de sortie. Le travail souligne aussi la difficulté de démontrer une représentation
métacognitive distincte. Nos contrôles de numérotation examinent un raccourci
possible ; ils ne suffisent pas à établir cette représentation.
[Hahami, Sinha et Jain](https://arxiv.org/html/2607.14111v1) étudient l'entraînement
au rapport de perturbations, en laissant ouverte la question du transfert vers
des anomalies naturelles. Ici aussi, une détection de rotation artificielle
reste distincte d'une prédiction utile de ses erreurs et de la conscience.

## Trois répétitions complètes

Qwen3-4B, révision `1cfa9a7208912126459214e8b04321603b3df60c`, BF16, SDPA,
préfixe complet sans cache, sans échantillonnage ni chaîne de pensée. La base est
figée ; les petits adaptateurs Q/V portent seuls les gradients. Couche 17,
rotation de force 1, rang 8, échelle 1, AdamW à 0,0002, weight decay nul et
écrêtage de norme à 1. Les paramètres LoRA sont FP32, leurs sorties BF16.

La graine du plan est `202609170`. Les trois initialisations sont `202609171`,
`202609172` et `202609173`. Chaque répétition comporte huit paires de phrases
d'apprentissage et 24 paires de test, soit **192 phrases distinctes** au total.
Toutes sont exclues des Colab 06 et 08. Les 96 directions de bruit utilisent
des graines distinctes, exclues des deux expériences précédentes. La grammaire
et le vocabulaire restent communs ; ce n'est pas un transfert de domaine.

Chaque répétition entraîne trois adaptateurs à partir des mêmes poids initiaux
LoRA et sur les mêmes blocs :

| Bras | Signal pendant la tâche principale | Cibles d'apprentissage |
|---|---|---|
| Visible | `[SIGNAL]` explicite, aucune rotation | Numéro affiché, ou 0 si absent |
| Fort | Rotation interne de force 1 | Numéro de la phrase touchée, ou 0 |
| Mélangé | Même rotation forte | Permutation des trois cibles, fixée par bloc |

Le bras faible est écarté de cette nouvelle expérience après son échec
d'acquisition au Colab 08. Ce choix prospectif est explicite ; son ancien
résultat reste publié. Aucun ancien adaptateur n'est réutilisé.

Chaque adaptateur suit quatre époques fixes, soit **32 mises à jour**. Une mise
à jour accumule les trois cas principaux 0/1/2, chacun avec poids 1/6, puis deux
lectures sans rotation où le repère porte successivement sur chacune des deux
phrases, chacune avec poids 1/4. L'objectif utilise l'entropie croisée dans le
vocabulaire complet. AdamW intervient après ces cinq rétropropagations.

Les neuf entraînements se terminent avant toute évaluation. Aucun ajustement
ne dépend d'un résultat de test, aucune meilleure époque ou répétition n'est
sélectionnée. Changer à la fois données et initialisation entre répétitions
ne permet pas de séparer leurs contributions à la variabilité observée.

## Croisement contenu × numérotation

L'apprentissage conserve uniquement la présentation canonique. Le test croise
deux manipulations du texte, pour chaque bloc et chaque modèle :

| Présentation | Première ligne | Deuxième ligne |
|---|---|---|
| Canonique | PHRASE 1 : contenu A | PHRASE 2 : contenu B |
| Contenus inversés | PHRASE 1 : contenu B | PHRASE 2 : contenu A |
| Numéros inversés | PHRASE 2 : contenu A | PHRASE 1 : contenu B |
| Les deux inversés | PHRASE 2 : contenu B | PHRASE 1 : contenu A |

Le repère `[REPERE]` reste attaché au même **contenu**, pas au numéro ni à la
ligne. L'intervention est appliquée aux tokens du contenu de la première ou
de la deuxième **ligne physique**. La bonne réponse est le **numéro affiché**
sur cette ligne. Dans le contrôle visible, `[SIGNAL]` remplace la rotation et
suit la même règle. Ces distinctions sont testées explicitement dans le code.

Pour chaque présentation et tâche, les conditions sont : absence, première
ligne touchée, deuxième ligne touchée et copie témoin sans modification.
La direction de bruit reste la même dans les variantes d'un bloc ; elle est
projetée orthogonalement à chaque activation avant rotation. Les numéros et
les repères ne font pas partie des spans manipulés. Pour les bras cachés, le
texte est strictement identique entre les quatre interventions : ni la graine,
ni la cible, ni la position touchée n'y figurent.

Le modèle de base est évalué sous condition visible et forte, puis les trois
adaptateurs dans leur condition propre. Les exemples d'apprentissage sont
évalués séparément, sous présentation canonique seulement. Les 24 blocs de
test sont évalués dans les quatre présentations, pour localisation et lecture.

Budget fixé : **288 mises à jour, 1 440 passages avec rétropropagation et
12 480 évaluations**. Chaque répétition contient 4 160 évaluations. Les
**3 120 paires** copie témoin / absence doivent être exactement identiques sur
les trois logits, la masse des chiffres et le premier token.

## Mesures et lecture des résultats

Les 150 tableaux séparent répétition, apprentissage/test, modèle, présentation
et tâche. Chaque tableau de test contient 24 blocs et 72 conditions ordinaires ;
les 24 copies témoins sont exclues des scores. On rapporte exactitude complète,
localisation sur les 48 perturbations, reconnaissance des 24 absences, lecture,
premier token libre, masse des chiffres, Brier et matrice de confusion.

La sensibilité de présence est la proportion de perturbations pour lesquelles
le modèle répond 1 ou 2. La spécificité est la proportion d'absences où il répond
0. Leur moyenne évite de valoriser artificiellement la règle « toujours présent »
dans un plan qui contient deux perturbations pour une absence. La localisation
conserve les faux négatifs dans son dénominateur : on ne sélectionne pas les seuls
cas détectés pour revendiquer une capacité.

Les **neuf contrastes principaux** sont, pour chaque répétition et uniquement
dans la présentation canonique, l'exactitude de localisation forte moins celle
de la base, celle du témoin mélangé et la référence informée de la présence
à 50 %. Les intervalles descriptifs à 95 % utilisent 2 000 rééchantillonnages
appariés des 24 blocs, avec graine `202609170 + 900 + numéro de répétition`.
Il n'y a pas de correction de multiplicité ni de verdict automatique. Les
intervalles sont conditionnels à chaque adaptateur, pas des intervalles sur une
population d'entraînements.

Les autres présentations sont des contrôles secondaires fixés avant collecte,
et non des essais parmi lesquels retenir le meilleur score. Les moyennes et
minima/maxima entre les trois répétitions sont fournis, sans intervalle global
qui traiterait les milliers de variantes comme autant d'entraînements indépendants.
Tous les résultats de chaque répétition restent visibles, y compris ceux où
le contrôle visible échoue.

Un gain limité à la présentation canonique ne confirmerait pas une localisation
robuste. Si l'inversion des numéros fait aussi échouer le contrôle visible,
une difficulté de suivi de consigne reste plausible ; on ne l'attribuera pas
spécifiquement à l'accès interne. Si les gains ne se reproduisent pas, le signal
du Colab 08 restera un résultat isolé à expliquer. Même une réussite des trois
répétitions resterait compatible avec un détecteur appris de perturbations
artificielles ; les choix d'action, les erreurs naturelles, le maintien d'un
modèle de ses capacités et l'expérience subjective ne sont pas évalués ici.

## Exécution, conservation et validation

Le Colab fixe une révision scientifique immuable. Il contrôle l'A100 et les
dépendances, puis exécute les tests logiciels avant de charger les poids.
L'archive **`menia-replication-localisation.zip`** inclut les diagnostics dès
le début, le journal, les bilans et neuf fichiers safetensors, soit environ
106 Mo de poids avant compression. Aucun fichier préalable n'est demandé.

La reprise vérifie environnement, plan, sources et checkpoints terminés.
Un adaptateur incomplet repart de son initialisation fixe, avec un événement
de reprise. Les anciens pas restent comptés. Les requêtes terminées ne sont
pas réexécutées ; une requête interrompue est marquée avant sa reprise. Relancer
une tentative complète vérifie les empreintes sans recharger les poids Qwen.
Le stockage temporaire Colab disparaît avec l'environnement : conserver le ZIP.

La durée de ce nouveau plan sur A100 n'est pas encore mesurée. À partir des
169,93 s pour 128 mises à jour et 147,32 s pour 1 792 évaluations du pilote 08,
une extrapolation linéaire donne environ **24 minutes de calcul instrumenté**.
L'installation, le chargement, les exports et les différences de matériel ou
de charge s'y ajoutent ; prévoir plusieurs dizaines de minutes et du quota Colab.

Les contrôles locaux vérifient les cibles après permutation, l'absence de fuite
dans le texte, les jeux disjoints, une réponse constante et un oracle synthétique.
Un petit Qwen aléatoire vérifie réellement l'objectif pondéré, les gradients,
les poids de base figés, les initialisations, les spans et hooks, une interruption
d'entraînement après un checkpoint terminé, une interruption d'évaluation, la
reprise et le rejet d'un checkpoint corrompu. Ces résultats ne prédisent pas
ceux du Qwen préentraîné. Aucun adaptateur n'est installé sur iPhone.

À la livraison, les **177 tests de recherche**, les **trois nouveaux tests du
moteur sur petit Qwen** et les **27 tests d'amorçage/lanceur** passent sous Windows
Python 3.12. Les 23 tests de profils de lanceur passent aussi sous Ubuntu
Python 3.12. Le plan synthétique complet vérifie 12 480 résultats, 288 mises à
jour, 3 120 paires témoins et 150 tableaux. Le tokenizer réel de la révision Qwen
traite les 12 480 préfixes d'évaluation et les 576 lectures d'apprentissage
répétées : **90 à 115 tokens**, sous la limite de 512, avec chiffres distincts
15, 16 et 17. Ce contrôle ne charge pas les poids préentraînés.
