# Détecter une perturbation interne : critère sans seuil fixé avant collecte

**État : premier résultat Qwen3-4B reçu et vérifié le 19 septembre 2026.**
Le [rapport](PRESENCE_DETECTION_RESULTS.md) constate la réussite du critère
principal dans les trois répétitions, un transfert partiel et une dégradation
de la lecture inversée. Le protocole ci-dessous reste celui fixé avant collecte.
Le [Colab 10](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/46364f0cdb32fd2317881e95d27dc2da6728f2be/notebooks/10_presence_detection_colab.ipynb)
utilise un seul bloc, sans Drive ni import manuel.

## Question et point de départ

La [réplication reçue](LOCALIZATION_REPLICATION_RESULTS.md) est négative sur son
critère principal : la localisation entraînée ne dépasse ni la base ni le témoin
mélangé. Un diagnostic **ajouté après réception** indiquait cependant une
information de présence dans les logits de l'adaptateur fort, avec une AUROC de
0,82 à 0,87 dans les trois répétitions contre environ 0,5 pour la base. Cette
mesure a été choisie après avoir vu les données : son résultat est exploratoire,
pas une confirmation prospective.

**Un adaptateur entraîné à dire si une perturbation est présente sépare-t-il,
sur des phrases et des directions nouvelles, les calculs perturbés des calculs
intacts mieux que le témoin mélangé et que le hasard ? Cette séparation
se maintient-elle pour une intensité et des couches jamais vues à l'entraînement ?**
La mesure, les contrastes et la règle de lecture ci-dessous sont fixés avant
toute exécution sur le modèle préentraîné.

La question de la position est volontairement retirée. Le Colab 09 a montré que
la base non entraînée porte déjà un effet de position dans ses logits et que
le contrôle visible n'apprend pas la règle du numéro affiché : ces deux points
limitent l'interprétation du mécanisme et du transfert. Ils n'annulent pas le
résultat négatif sur le critère principal fixé. La détection binaire ne
dépend pas des numéros affichés.

Le cadre reste celui des critères fonctionnels de
[Lindsey](https://transformer-circuits.pub/2025/introspection/) : un compte
rendu doit être exact, dépendre causalement de l'état interne et ne pas suivre
un simple biais de sortie. Les autres sources figurent dans le
[protocole précédent](LOCALIZATION_REPLICATION_PROTOCOL.md). Cette expérience
adapte des travaux d'introspection entraînée ; elle ne revendique aucune
méthode originale et ne porte pas sur la conscience.

## Tâche

Deux phrases numérotées, l'une portant `[REPERE]`. La consigne cachée demande :
« Réponds 1 si une phrase est touchée, ou 0 si aucune phrase n'est touchée ».
La perturbation est la même rotation conservant la norme que précédemment,
appliquée aux tokens du contenu de la première ou de la deuxième ligne, dans une
direction aléatoire propre au bloc et projetée orthogonalement à chaque activation.
Pour les bras cachés, le texte est strictement identique entre absence,
première ligne, deuxième ligne et copie témoin : ni la graine, ni la condition,
ni la ligne touchée n'y figurent. Toute séparation vient donc de l'intervention.

La tâche de lecture « numéro de la phrase portant `[REPERE]` » est conservée à
l'entraînement et au test, pour vérifier que l'adaptateur ne détruit pas le suivi
de consigne comme au Colab 06.

## Trois répétitions complètes

Qwen3-4B, révision `1cfa9a7208912126459214e8b04321603b3df60c`, BF16, SDPA,
préfixe complet sans cache, sans échantillonnage ni chaîne de pensée. Base
figée ; adaptateurs Q/V de rang 8, échelle 1, paramètres FP32, AdamW à 0,0002,
weight decay nul, écrêtage de norme à 1.

La graine du plan est `202609180`. Les initialisations `202609181`, `202609182`
et `202609183` diffèrent de celles du Colab 09. Chaque répétition comporte
**16 paires de phrases d'apprentissage** et 24 paires de test, soit **240 phrases
distinctes** au total, toutes exclues des Colab 06, 08 et 09. Les 120 directions
de bruit utilisent des graines exclues des expériences précédentes. Grammaire et
vocabulaire restent communs : ce n'est pas un transfert de domaine.

| Bras | Signal pendant la tâche principale | Cibles d'apprentissage |
|---|---|---|
| Visible | `[SIGNAL]` explicite, aucune rotation | 1 si affiché, 0 sinon |
| Fort | Rotation, couche 17, force 1 | 1 si une phrase est touchée, 0 sinon |
| Mélangé | Même rotation | Permutation des trois cas du bloc, fixée par bloc |

Le passage de 8 à 16 blocs d'apprentissage est un choix prospectif : le bras
fort plafonnait à 19/24 sur ses propres exemples dans les trois répétitions du
Colab 09. Chaque adaptateur suit quatre époques fixes, soit **64 mises à jour**.
Une mise à jour accumule les trois cas principaux du bloc puis deux lectures
sans rotation. Un bloc contient une absence et deux présences : l'absence
reçoit le poids 1/4 et chaque présence 1/8, de sorte que les deux classes pèsent
autant. Le témoin mélangé conserve ces mêmes poids par étiquette. Chaque
lecture reçoit 1/4. L'objectif utilise l'entropie croisée dans le vocabulaire
complet.

Les neuf entraînements se terminent avant toute évaluation. Aucun réglage ne
dépend d'un résultat de test ; aucune époque ni répétition n'est sélectionnée.
Aucun adaptateur antérieur n'est réutilisé.

## Conditions de test

L'entraînement n'utilise que la présentation canonique et la condition
« entraînée ». Le test croise quatre conditions d'intervention et deux
présentations, pour chaque bloc :

| Condition | Couche | Force | Vue à l'entraînement |
|---|---:|---:|---|
| Entraînée | 17 | 1 | oui |
| Plus faible | 17 | 0,5 | non |
| Plus précoce | 11 | 1 | non |
| Plus tardive | 23 | 1 | non |

| Présentation | Première ligne | Deuxième ligne |
|---|---|---|
| Canonique | PHRASE 1 : contenu A | PHRASE 2 : contenu B |
| Les deux inversés | PHRASE 2 : contenu B | PHRASE 1 : contenu A |

La base est évaluée sous les familles visible et forte, puis les trois
adaptateurs dans leur famille propre. La famille visible n'a qu'une condition.
La lecture n'est évaluée que dans la condition entraînée. Sans perturbation,
aucun crochet n'est installé : le calcul est donc le même dans les quatre
conditions. Ces répétitions sont conservées et comparées ; tout écart est compté
comme un défaut de déterminisme, sans être utilisé comme donnée supplémentaire.

Budget fixé : **576 mises à jour, 2 880 passages avec rétropropagation et
12 864 évaluations**, dont 3 216 paires copie témoin / absence qui doivent
être exactement identiques sur les logits, la masse des chiffres et le
premier token.

## Mesure principale et règle de lecture fixées à l'avance

Le score de présence d'une ligne est **logit(« 1 ») − logit(« 0 »)**, lu sur
la tête de langage du modèle au premier token de réponse. Pour chaque tableau
de test, l'**AUROC de présence** compare les 48 lignes perturbées aux 24 lignes
intactes ; les égalités comptent pour un demi. Aucun seuil n'intervient.

Les **neuf contrastes principaux** sont, pour chaque répétition, dans la
présentation canonique et la condition entraînée : AUROC du fort moins celle
du mélangé, moins 0,5, et moins celle de la base. Les intervalles descriptifs
à 95 % utilisent 2 000 rééchantillonnages appariés des 24 blocs, l'AUROC étant
recalculée dans chaque tirage, avec la graine `202609180 + 900 + numéro de
répétition`. Ils sont conditionnels à chaque adaptateur.

**Règle de lecture.** Le signal de présence sera dit *reproduit dans ce montage*
seulement si, dans **chacune des trois répétitions**, les deux contrastes « fort
moins mélangé » et « fort moins 0,5 » sont positifs avec un intervalle excluant
zéro. Dans tout autre cas, il sera rapporté comme non reproduit, y compris si
deux répétitions sur trois réussissent. Le contraste contre la base est
rapporté sans entrer dans la règle, la base étant attendue proche de 0,5. La
conjonction de six intervalles est exigeante avec 24 blocs : un échec pourra
venir d'un manque de puissance et sera décrit comme tel, sans changer la règle.

Mesures secondaires fixées avant collecte, sans règle de décision :

- l'**ordre au sein du bloc** : proportion des lignes perturbées dont le score
  dépasse celui de la ligne intacte du même bloc, à texte identique. Une
  préférence constante propre au bloc s'y annule, contrairement à l'AUROC ;
- exactitude, sensibilité, spécificité et leur moyenne au seuil naturel de
  l'argmax entre « 0 » et « 1 », Brier à deux classes, premier token libre ;
- les mêmes quantités pour les trois conditions de transfert et la
  présentation inversée, soit 63 contrastes secondaires ;
- lecture du repère, et acquisition sur les exemples d'apprentissage.

Les moyennes et étendues entre répétitions sont fournies, sans intervalle global
traitant les variantes comme des entraînements indépendants. Changer à la fois
données et initialisation ne permet pas de séparer leurs contributions.

## Ce que les résultats pourront et ne pourront pas dire

Si le contrôle visible échoue, l'essai sera ininterprétable pour le bras caché.
Si la règle est satisfaite, on aura un détecteur entraîné qui sépare, dans ce
montage, des rotations artificielles injectées par l'expérimentateur. Un transfert
aux couches et à l'intensité non vues indiquerait que ce détecteur ne dépend pas
étroitement de la condition apprise ; son absence le restreindrait à celle-ci.

Dans tous les cas, l'expérience ne teste pas la prévision d'erreurs naturelles,
le choix d'actions, le maintien d'un modèle de ses capacités ni l'expérience
subjective. Un détecteur appris par gradient sur des étiquettes fournies reste
compatible avec une classification ordinaire d'activations anormales. Aucun
résultat de ce protocole ne pourra être présenté comme une conscience de Menia.

## Exécution, conservation et validation

Le Colab fixe une révision scientifique immuable, contrôle l'A100 et les
dépendances, puis exécute les tests logiciels avant de charger les poids.
L'archive **`menia-detection-presence.zip`** inclut les diagnostics dès le
début, le journal, les bilans et neuf fichiers safetensors, soit environ 106 Mo
de poids avant compression. Aucun fichier préalable n'est demandé.

La reprise vérifie environnement, plan, sources et checkpoints terminés. Un
adaptateur incomplet repart de son initialisation fixe, avec un événement de
reprise ; les requêtes terminées ne sont pas réexécutées. Le stockage
temporaire Colab disparaît avec l'environnement : conserver le ZIP.

La durée sur A100 n'est pas mesurée. À partir des 364,31 s pour 288 mises à
jour et des 982,36 s pour 12 480 évaluations du Colab 09, une extrapolation
linéaire donne environ **29 minutes de calcul instrumenté**, hors installation,
chargement et exports.

Les contrôles locaux vérifient le budget, les jeux disjoints, l'absence de
fuite de la condition dans le texte, l'équilibre des poids de classe, l'AUROC
contre une énumération directe, un oracle, une réponse constante et un biais
propre au bloc que seul l'ordre apparié traverse intact. Un petit Qwen
aléatoire vérifie réellement l'objectif pondéré, les gradients, la base figée,
les quatre conditions d'intervention, une interruption d'entraînement, une
interruption d'évaluation, l'égalité octet pour octet entre adaptateurs repris
et adaptateurs d'une tentative ininterrompue, et le rejet d'un checkpoint
corrompu. Ces résultats ne prédisent pas ceux du Qwen préentraîné.

À la livraison, les **183 tests de recherche**, les **25 tests du moteur**, dont
trois nouveaux sur petit Qwen, et les **30 tests d'amorçage/lanceur** passent
sous Windows Python 3.12. Le plan synthétique complet vérifie 12 864 résultats,
576 mises à jour, 3 216 paires témoins, 144 tableaux et 72 contrastes dont
neuf principaux. Le tokenizer réel de la révision Qwen traite les 12 864
préfixes d'évaluation et les 1 152 lectures d'apprentissage : **87 à 118
tokens**, sous la limite de 512, avec chiffres distincts 15, 16 et 17. Ce
contrôle ne charge pas les poids préentraînés. Aucun adaptateur n'est installé
sur iPhone.
