# Prévoir les conséquences d'une perturbation interne

Protocole distinct, fixé après le [premier résultat des états internes](ACTIVATION_MONITOR_RESULTS.md).
Le [premier export complet est désormais reçu et vérifié](PERTURBATION_MONITOR_RESULTS.md) :
**704 appels sans erreur technique**. La baisse du score strict sous rotation
forte est principalement liée au format de réponse ; aucun gain global du
moniteur interne n'est établi. Le protocole ci-dessous reste celui fixé avant
collecte, sans changement du barème après réception.

[Ouvrir le Colab à un seul bloc](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/codex/recall-reliability/notebooks/05_perturbation_monitor_colab.ipynb).
Choisir A100, puis « Tout exécuter ». Transmettre `menia-perturbations.zip`,
même en cas d'arrêt. **La version courante fonctionne sans Google Drive.**
Les résultats sont enregistrés dans `/content/menia-results/perturbation-monitor-v1`
et l'archive est téléchargée à la fin. Enregistrer le ZIP avant la fermeture ou
suppression de l'environnement Colab : ce stockage est temporaire. Si le
téléchargement automatique est bloqué, le ZIP est disponible dans le panneau
Fichiers, sous `/content`. La reprise est possible tant que le même environnement
existe ; ce mode ne lit ni ne modifie les anciens dossiers Drive.
Le notebook fixe le code scientifique à la révision
`8db4a3e98812b5680a84423283a996659e75da8d`.

### Incident reçu et correction du lancement

Le premier ZIP de ce protocole contient uniquement les diagnostics. L'installation,
le contrôle des dépendances, CUDA et les onze tests logiciels ont réussi sur
A100-SXM4-40GB / Python 3.13.15. Le programme s'est ensuite arrêté à
`drive.mount` avec `ValueError: mount failed`, avant le collecteur scientifique.
Le journal ne précise pas la cause sous-jacente du refus de montage. Le
[bilan du diagnostic](../artifacts/perturbation-launch-diagnostics/drive-mount-failure.json)
conserve ces faits ; ce premier ZIP ne contient aucun journal de l'étude.
Le ZIP suivant contient les 704 appels terminés, analysés dans le rapport ci-dessus.

Le lanceur propose désormais un stockage local utilisé par ce notebook. Il
omet le montage Drive et exporte les résultats et diagnostics depuis le dossier
local. Les essais précédents sont conservés, et le mode Drive des anciens
notebooks reste disponible. Le plan, les poids, les contrôles, les scores et
la révision scientifique ne changent pas. Ce correctif retire la dépendance
qui a bloqué le lancement ; il ne prétend pas réparer le service Google Drive.

## Question et portée

À texte identique, une lecture des activations permet-elle de prévoir les
conséquences d'une perturbation et d'améliorer les choix de vérification ?
Le précédent essai variait les questions avec des poids fixes et aucun
changement interne imposé. Ici, chaque question reçoit quatre conditions,
sans mention de la condition dans le texte. Les interventions sont définies
dans le collecteur, hors de l'entrée des moniteurs ordinaires.

Deux résultats de la littérature motivent les contrôles, sans garantir une
réussite. [Lindsey](https://transformer-circuits.pub/2025/introspection/index.html)
présente des injections de concepts, des essais témoins et des contrôles
destinés à distinguer une détection interne d'une inférence depuis les mots
déjà produits. Les perturbations trop fortes peuvent rendre la réponse
incohérente. Le rapport [OWMI de Ferrara](https://arxiv.org/abs/2608.20569)
distingue l'information décodable dans les activations des auto-rapports du
modèle et compare des interventions à des conditions témoins. Le présent
protocole reprend cette distinction méthodologique ; il ne réplique pas leurs
injections sémantiques ni leurs expériences d'auto-rapport.

Un moniteur ajouté lit les états ; une règle externe route les réponses. Les
poids de Qwen restent gelés. Même une réussite serait une mesure de suivi
fonctionnel dans cet assemblage, pas une preuve d'introspection native, de
conscience ou de nouveauté de la méthode générale.

## Intervention exacte

Qwen/Qwen3-4B reste à la révision `1cfa9a7208912126459214e8b04321603b3df60c`,
en BF16 sur A100 de 40 ou 80 Go. Les réglages restent ceux du premier pilote :
température 0,7, top-p 0,8, top-k 20, `enable_thinking=false`, maximum 256
nouveaux tokens, sans troncature des entrées ni quantification.

Le site est la sortie du bloc d'indice `nombre_de_blocs // 2 − 1` : indice 17
pour ce Qwen. **Seul le dernier token du préfixe, au premier passage, est
modifié.** Les étapes ultérieures de génération n'appliquent pas à nouveau
la perturbation. Aucun poids n'est édité. Tous les hooks sont retirés à la fin
de l'appel, y compris en cas d'erreur.

| Condition | Opération |
|---|---|
| `baseline` | Aucune modification ; lecture des états seulement |
| `sham` | Clone identique de la sortie, sans modification numérique |
| `rotateHalf` | Rotation de force 0,5 |
| `rotateOne` | Rotation de force 1 |

Pour un vecteur `v`, une direction gaussienne est projetée orthogonalement à
`v`, puis normalisée pour obtenir un vecteur unitaire `u`. La transformation
est `v' = (v + a × ||v|| × u) / sqrt(1 + a²)`. Elle préserve la norme en
arithmétique exacte, avec des angles d'environ 26,6° et 45°. Le passage float32
puis BF16 introduit un arrondi ; l'erreur relative de norme doit rester sous
1 %. Le déplacement effectif est journalisé et contrôlé.

La direction est propre à chaque question et partagée entre ses deux forces.
Elle utilise un générateur NumPy local distinct de l'aléa de génération de
tokens. La graine de génération du LLM est identique dans les quatre
conditions. Leur ordre est mélangé et fixé avant collecte. Une rotation de
norme préservée ne supprime pas toute signature artificielle de perturbation ;
c'est précisément pourquoi des témoins conditionnels restent nécessaires.

Les états centraux sont lus **après** cette transformation ; les états finaux
sont lus avant la tête de sortie et le premier token. Les embeddings d'entrée
doivent rester identiques pour les quatre conditions. Les paramètres de la
rotation sont conservés pour l'audit, sans devenir des caractéristiques des
moniteurs `inputOnly` et `internal`.

## Groupes, séparation et arrêt technique

Les tâches retenues sont le comptage de A dans huit lettres et les
soustractions de deux nombres de 10 à 99. Ce choix suit les difficultés
observées précédemment : elles laissent une marge pour mesurer une perte
de réussite. Les tâches presque toujours échouées du premier essai ne
permettaient guère de mesurer une dégradation supplémentaire. Cette sélection
est explicite ; elle ne permet pas de généraliser à tous les types de tâches.

Chaque question est nouvelle par rapport aux deux Colab précédents. Une
question appartient à un seul lot, avec toutes ses variantes. La graine du
plan est `202609152`.

| Lot | Questions par famille | Questions distinctes | Appels, quatre conditions |
|---|---:|---:|---:|
| Contrôle technique initial | 4 | 8 | 32 |
| Apprentissage | 48 | 96 | 384 |
| Validation | 12 | 24 | 96 |
| Test réservé | 24 | 48 | 192 |
| **Total** | **88** | **176** | **704** |

Les 32 premiers appels servent uniquement à vérifier l'identité exacte des
états et du texte entre `baseline` et `sham`, ainsi que celle des embeddings
d'entrée entre toutes les conditions. Une divergence arrête le programme
et conserve les journaux. Les mêmes contrôles continuent ensuite pour chaque
groupe complet. **Aucun taux de réussite ni effet favorable n'est utilisé pour
décider de continuer.** Les résultats du contrôle initial sont exclus de
l'apprentissage, de la validation et des scores du test.

Un statut technique `error` ou `interrupted` est conservé sans résultat inventé.
La reprise n'effectue pas de nouveau tirage pour remplacer un appel perdu.
Un contrôle initial incomplet interdit le passage à l'étude. En cas de
défaillance ultérieure, les scores partiels restent identifiés comme tels.
`complete` exige les 704 résultats `ok` et le contrôle technique réussi.

## Moniteurs et comparateurs

Les caractéristiques et projections restent celles du pilote précédent :
262 caractéristiques textuelles et de cellule, 128 coordonnées d'embeddings
d'entrée, 128 de la sortie centrale, 128 de la normalisation finale. Les deux
familles actives utilisent les mêmes positions dans le codage à six cellules.
Le moniteur d'entrée a 390 caractéristiques et le moniteur interne 646 :
leur capacité statistique n'est donc pas parfaitement appariée.

Les régressions ridge utilisent les seules données d'apprentissage pour la
normalisation et les coefficients. La validation choisit alpha parmi
0,001, 0,01, 0,1 et 1 selon le Brier, avec priorité à la première valeur en
cas d'égalité. Aucun réajustement ne suit la validation. Un événement `fit`,
avec coefficients et empreinte des données, précède toute requête de test.
Au moins 32 appels d'apprentissage et 8 de validation par famille/condition
sont nécessaires ; sinon l'ajustement est refusé.

| Prévision | Information disponible |
|---|---|
| `betaCell` | Réussites passées par famille/difficulté, toutes conditions réunies |
| `oracleCondition` | Réussites passées par famille **et condition réelle** |
| `inputOnly` | Texte et embeddings d'entrée, sans condition ni états contextualisés |
| `internal` | Même entrée, plus états centraux et finaux courants |
| `shuffledLabels` | Même modèle interne, étiquettes permutées dans chaque famille/condition |
| `donorState` | Modèle interne figé, états contextualisés remplacés par ceux d'un autre problème d'apprentissage de même famille/condition |

Les deux références numériques valent `(réussites + 1)/(appels + 2)`.
Ce sont des estimateurs lissés ; les appels d'une même question étant corrélés,
on n'interprète pas cette formule comme un postérieur fondé sur des essais
indépendants. Aucun intervalle Beta n'est utilisé.

`oracleCondition` reçoit volontairement une information privilégiée. Il indique
ce qu'explique la connaissance du mode expérimental ; ce n'est pas un observateur
équitablement privé de cette information. Le donneur bénéficie également de
la condition pour son appariement. Le contrôle à étiquettes mélangées conserve
les fréquences par mode et choisit sa régularisation sur les vraies étiquettes
de validation : il ne représente pas une distribution nulle complète.

Ces contrôles évitent de baptiser « prévision de ses erreurs » un simple
classement des conditions selon leur sévérité moyenne. Aucun moniteur n'a
accès à la réponse cible avant son engagement.

## Décisions exécutées et coûts

Sur chaque appel de test, les six probabilités et décisions sont synchronisées
sur disque **avant le premier token**. La règle fixée choisit `direct` si
`p >= 0,8`, sinon `verify`. Après la génération, les six routes sont réellement
exécutées sur le même candidat : conservation du texte, ou appel à un outil
déterministe qui analyse l'énoncé public et calcule la réponse. Chaque texte
final, action et durée de routage est enregistré.

Les routes sont six applications d'une règle à un candidat partagé ; ce ne
sont pas six épisodes autonomes ou six choix spontanés de Qwen. La génération
est poursuivie même lorsque la route choisit la vérification, afin de conserver
la cible brute nécessaire au Brier. **Aucun temps d'inférence LLM économisé
n'est revendiqué.** L'entraînement ne porte pas sur une politique apprise :
le seuil est fourni par le protocole.

La perte vaut `1 si réponse finale fausse + 0,2 si outil appelé`. Le coût 0,2
est une valeur expérimentale déclarée, pas une conversion du temps réel. Les
durées de routage sont rapportées séparément. L'outil exact rend les tâches
vérifiées correctes dans ce domaine restreint ; vérifier systématiquement
fournit une référence de perte 0,2. L'intérêt éventuel est de réduire les appels
à cet outil tout en maîtrisant les erreurs finales.

## Analyse fixée avant les nouvelles données

Le rapport conserve tous les Brier, AUC, réussites brutes et finales,
vérifications et pertes, globalement et dans chacune des huit
familles/conditions. Les réponses mal formées comptent comme échecs, les
erreurs techniques n'ont pas d'étiquette de réussite.

Pour chaque condition, les paires complètes indiquent les réponses perdues
ou gagnées par rapport au calcul normal. Il faut d'abord établir ce que la
rotation change effectivement ; sa présence ne garantit pas une dégradation.

Les écarts Brier et perte « comparateur moins interne » sont conservés pour
les cinq comparateurs. Positif favorise `internal`. Deux mille
rééchantillonnages prennent des **questions complètes avec leurs quatre
conditions**, stratifiés par famille. L'unité du test est donc **48 questions,
pas 192 observations indépendantes**. Les intervalles ne sont calculés que
si ces 48 groupes sont complets. Ils sont descriptifs, conditionnels à un seul
apprentissage et n'intègrent pas la variabilité de réentraînement.

Un signal intéressant nécessiterait un effet mesuré sur les capacités, un
gain prédictif et de décision sur les références simples, et une résistance
aux contrôles conditionnels. Il faudrait ensuite le répliquer sur de nouvelles
questions et d'autres interventions. Un avantage global dû uniquement au
mode de perturbation, un gain isolé dans une cellule ou un outil exact qui
corrige les réponses ne suffiraient pas. Il n'y a pas de règle de confirmation
de conscience ni de seuil de significativité confirmatoire caché.

## Validation et limites restantes

Huit tests de recherche vérifient un signal artificiel connu, un signal
expliqué uniquement par la condition, la séparation des questions,
l'exclusion du lot de test et du contrôle initial de l'ajustement, les routes
effectivement exécutées, les falsifications, le témoin identique et la reprise.
Trois tests sur un petit Qwen aux poids aléatoires vérifient l'ordre avant
logits, la norme et l'aléa privé, le retrait des hooks, la restauration des
réponses normales et l'absence de modification des poids. Cinq tests du
nouveau profil Colab vérifient la reprise, la séparation des fichiers,
l'export des erreurs précoces, l'autonomie du bloc, l'absence d'appel à Drive
en mode local et l'export des données partielles après échec. Les onze tests
des deux profils de lancement passent sous Windows/Python 3.13 et Ubuntu/Python 3.12.

Ces validations logicielles ne sont pas une exécution du modèle préentraîné
sur A100. La perturbation est artificielle, ciblée sur un seul site, et les
directions nouvelles appartiennent toujours à la même famille de rotations.
Le moniteur peut en détecter des signatures sans représenter un « soi ».
Les coefficients BF16 ne sont pas validés pour l'iPhone MLX en quatre bits.
Le plan et les réglages ne seront pas changés pour améliorer ce test après
réception des données. Relancer la tentative terminée ne produit pas une
réplication indépendante.

Recalcul :

```sh
python -m research.perturbation_monitor journal.jsonl --output bilan.json
```
