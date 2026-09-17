# Réplication de la localisation : le gain du Colab 08 ne se reproduit pas

**Le Colab 09 est complet : neuf adaptateurs, 288 mises à jour et 12 480
évaluations vérifiées. Sur le critère principal fixé avant collecte, l'adaptateur
fort localise 15/48, 28/48 et 15/48 perturbations dans les trois répétitions,
contre 24/48 pour la base à chaque fois. Deux des trois écarts contre la base
sont négatifs avec un intervalle descriptif excluant zéro. L'avantage sur le
témoin mélangé observé au Colab 08 ne se reproduit pas non plus.**

Le contrôle visible réussit à 100 % dans les présentations canonique et à
contenus inversés, mais échoue dès que les numéros affichés sont inversés.
Selon la règle de lecture fixée dans le protocole, les échecs sous inversion
des numéros ne peuvent donc pas être attribués spécifiquement à l'accès interne.

Un diagnostic **ajouté après réception** montre qu'une information de présence
de la perturbation existe de façon répétée dans les logits de l'adaptateur fort,
alors que l'information de position est déjà présente dans la base non entraînée
et ne suit pas le numéro affiché. Ce complément exploratoire ne remplace pas
le critère principal. Aucun de ces résultats ne démontre une connaissance de
ses propres états ni une conscience de sa propre existence.

Le [protocole fixé avant collecte](LOCALIZATION_REPLICATION_PROTOCOL.md) et son
barème ne changent pas. Le
[bilan principal](../artifacts/localization-replication-pilot/first-audit-summary.json)
est recalculé depuis le journal reçu. Le
[complément de vérification](../artifacts/localization-replication-pilot/first-audit-verification.json)
ajoute l'inspection des checkpoints et des diagnostics appariés. Le
[diagnostic exploratoire des logits](../artifacts/localization-replication-pilot/logit-diagnostic-after-receipt.json)
est séparé. Le journal brut et les neuf adaptateurs restent hors Git.

## Exécution vérifiée

L'export `menia-replication-localisation.zip` (SHA-256 `4f96d357…946681`)
contient une tentative complète, sans erreur, interruption ni reprise
d'entraînement. Les métadonnées déclarent Qwen3-4B à la révision
`1cfa9a7208912126459214e8b04321603b3df60c`, en BF16 sur A100-SXM4-40GB,
Python 3.13.15, PyTorch 2.8.0+cu126 et Transformers 4.56.2, avec algorithmes
déterministes et TF32 désactivé. Le commit scientifique est
`829e73ae3b961d43cf53644e90b221360d41773c`.

Les neuf entraînements se terminent avant toute évaluation : **288 mises à
jour**, **1 440 passages avec rétropropagation**, puis **12 480 évaluations**.
Chaque adaptateur contient 2 949 120 paramètres dans 144 tenseurs FP32 finis,
de noms et dimensions attendus ; les neuf SHA-256 correspondent au journal et
les 72 facteurs B initialement nuls ont tous changé. Les normes de gradient
enregistrées sont positives et finies.

Les **3 120 paires copie témoin / absence** ont exactement les mêmes logits,
masse des chiffres et premier token. Les **13 920 traces** respectent les
conditions d'application ; l'erreur relative maximale de norme vaut
**0,0947 %**, sous la limite de 1 %. Les préfixes ont 90 à 115 tokens. Tous
les premiers tokens libres appartiennent aux trois chiffres autorisés.

Le bilan fourni par Colab et son recalcul local diffèrent sur deux scores de
Brier de lecture, d'au plus **8,3 × 10⁻²⁵** ; tous les comptes et contrastes
sont identiques. Une reconstruction distincte, avec énumération indépendante
des cibles après permutation, retrouve les 150 tableaux à moins de
**2,3 × 10⁻¹⁶** et les neuf contrastes exactement. Relancer cette vérification
redonne un fichier identique octet pour octet. Elle utilise les traces reçues :
ce n'est ni une nouvelle inférence ni une attestation indépendante du GPU.

Les durées instrumentées totalisent **364,31 s d'entraînement** et **982,36 s
d'évaluation**, soit environ 22 minutes et demie, proche des 24 minutes
extrapolées. L'installation, le chargement et les exports s'y ajoutent ;
le pic mémoire n'est pas mesuré.

## Acquisition sur les exemples d'apprentissage

Scores mesurés après le dernier pas, adaptateurs figés, sur huit blocs de trois
conditions ordinaires, en présentation canonique.

| Répétition | Visible | Fort | Mélangé, sur ses cibles entraînées | Lecture (trois bras) |
|---|---:|---:|---:|---:|
| 1 | 24/24 | 19/24 — 79,17 % | 13/24 | 24/24 |
| 2 | 24/24 | 19/24 — 79,17 % | 14/24 | 24/24 |
| 3 | 24/24 | 19/24 — 79,17 % | 13/24 | 24/24 |

Le bras fort atteint 19/24 dans les trois répétitions, comme au Colab 08 : il
ne franchit jamais le seuil diagnostique de 90 %. L'acquisition cachée reste
incomplète avec ces 32 mises à jour fixes ; le test ne peut donc pas séparer
un manque d'information d'un apprentissage insuffisant.

## Critère principal : présentation canonique, 24 nouveaux blocs

Chaque tableau contient 72 conditions ordinaires : 24 absences et 48
perturbations. Les copies témoins sont exclues des scores.

| Répétition | Version | Tâche complète /72 | Absence /24 | Position quand présente /48 | Lecture /72 | Brier |
|---|---|---:|---:|---:|---:|---:|
| 1 | Base, rotation forte | 24 | 0 | 24 — 50 % | 38 | 1,3206 |
| 1 | Adaptateur fort | 37 | 22 | **15 — 31,25 %** | 72 | 0,5729 |
| 1 | Cibles mélangées | 33 | 21 | 12 — 25 % | 72 | 0,6332 |
| 2 | Base, rotation forte | 24 | 0 | 24 — 50 % | 37 | 1,3117 |
| 2 | Adaptateur fort | 45 | 17 | **28 — 58,33 %** | 72 | 0,5024 |
| 2 | Cibles mélangées | 36 | 4 | 32 — 66,67 % | 72 | 0,6193 |
| 3 | Base, rotation forte | 24 | 0 | 24 — 50 % | 40 | 1,3280 |
| 3 | Adaptateur fort | 39 | 24 | **15 — 31,25 %** | 72 | 0,7746 |
| 3 | Cibles mélangées | 20 | 1 | 19 — 39,58 % | 72 | 0,7796 |

Le contrôle visible obtient 72/72 en tâche complète et en lecture dans les trois
répétitions ; la base visible obtient 50, 49 et 50 sur 72.

| Contraste prédéfini, localisation des 48 perturbations | Répétition 1 | Répétition 2 | Répétition 3 |
|---|---:|---:|---:|
| Fort moins base | −18,75 [−31,25 ; −4,17] | +8,33 [−2,08 ; +18,75] | −18,75 [−33,33 ; −2,08] |
| Fort moins mélangé | +6,25 [−2,08 ; +14,58] | −8,33 [−18,75 ; +2,08] | −8,33 [−22,92 ; +6,25] |
| Fort moins hasard informé de la présence | −18,75 [−31,25 ; −4,17] | +8,33 [−2,08 ; +18,75] | −18,75 [−33,33 ; −2,08] |

Écarts en points, intervalles descriptifs à 95 % par 2 000 rééchantillonnages
appariés des 24 blocs, sans correction de multiplicité, conditionnels à chaque
adaptateur. Aucun des neuf contrastes n'est positif avec un intervalle excluant
zéro ; quatre sont négatifs en excluant zéro, dont deux fois le même contraste
numérique puisque la base vaut exactement 50 %.

Cette égalité vient du comportement de la base : sous rotation forte elle répond
le même chiffre pour les trois conditions d'un bloc — « 1 » dans 12 blocs,
« 2 » dans 12 — et ne répond jamais « 0 ». Ses 24/48 sont une réponse constante
par bloc, pas une localisation. En cumulant les répétitions, le fort localise
**58/144** perturbations (40,28 %), la base 72/144 et le mélangé 63/144.
L'avantage de +29 points sur le mélangé mesuré au Colab 08 ne se reproduit pas :
ce signal reste un résultat isolé.

Le profil du fort est celui d'un détecteur prudent. Il reconnaît 22, 17 et 24
absences sur 24, mais manque 26, 10 et 29 perturbations sur 48 (sensibilité de
présence 45,8 %, 79,2 % et 39,6 %). Comme la localisation conserve les faux
négatifs au dénominateur, ces omissions dominent le score. À texte identique,
21, 26 et 19 interventions sur 48 changent le chiffre choisi, contre 0 pour la
base et 17, 18 et 6 pour le mélangé. Seuls 2, 4 et 4 blocs sur 24 réussissent
leurs trois conditions.

## Contrôles secondaires : contenus et numéros croisés

Position correcte quand le signal ou la perturbation est présent, sur 48, pour
les répétitions 1 / 2 / 3.

| Présentation | Visible | Fort | Mélangé | Base, rotation forte |
|---|---:|---:|---:|---:|
| Canonique | 48 / 48 / 48 | 15 / 28 / 15 | 12 / 32 / 19 | 24 / 24 / 24 |
| Contenus inversés | 48 / 48 / 48 | 21 / 28 / 16 | 13 / 38 / 21 | 24 / 24 / 24 |
| Numéros inversés | **27 / 25 / 27** | 11 / 21 / 4 | 10 / 14 / 24 | 24 / 24 / 24 |
| Les deux inversés | **25 / 24 / 28** | 9 / 18 / 12 | 5 / 14 / 23 | 24 / 23 / 24 |

Inverser les contenus ne change rien au contrôle visible et ne dégrade pas le
fort : les réponses ne dépendent pas d'une phrase particulière. Inverser les
numéros fait en revanche tomber le visible près de 50 %. Il reconnaît toujours
l'absence (24/24) et répond alors « 1 » quelle que soit la ligne portant
`[SIGNAL]` : 20 à 24 erreurs sur les 24 cas dont la bonne réponse est « 2 ».
Sa lecture du repère baisse aussi à 66, 57 et 72 sur 72. La base à rotation
forte bascule de même vers « 1 » dans 71 ou 72 réponses sur 72.

Entraîné sur la seule présentation canonique, même l'adaptateur recevant un
signal explicite n'a donc pas appris la règle « rapporter le numéro affiché ».
Conformément à la lecture fixée avant collecte, une difficulté de suivi de
consigne suffit à expliquer ces échecs ; la chute du fort sous inversion des
numéros n'est pas interprétable comme un défaut propre d'accès interne.

## Diagnostic exploratoire ajouté après réception

Les scores précédents dépendent du chiffre maximal. Deux mesures sans seuil
ont été ajoutées après observation des résultats, sur les mêmes logits reçus,
sans nouvel appel du modèle ni réglage :

- **présence** : AUROC de 1 − P(« 0 ») entre les 48 perturbations et les 24
  absences d'un tableau ;
- **position** : dans chaque bloc, à texte identique, le contraste entre le
  logit du numéro affiché en première ligne et celui de la deuxième doit être
  plus grand quand la première ligne est touchée. On compte les blocs de bon
  signe sur 24 ; une préférence constante propre au bloc s'annule.

| Mesure, répétitions 1 / 2 / 3 | Base, rotation forte | Adaptateur fort | Cibles mélangées |
|---|---:|---:|---:|
| AUROC de présence, canonique | 0,51 / 0,53 / 0,54 | **0,87 / 0,84 / 0,82** | 0,69 / 0,67 / 0,39 |
| AUROC de présence, numéros inversés | 0,54 / 0,56 / 0,55 | **0,76 / 0,78 / 0,80** | 0,65 / 0,60 / 0,44 |
| Blocs de bon signe /24, canonique | **21 / 19 / 21** | 18,5 / 19,5 / 21 | 13 / 17,5 / 8 |
| Blocs de bon signe /24, numéros inversés | 14 / 13,5 / 19 | 9,5 / 10,5 / 11 | 13,5 / 6,5 / 10,5 |

Trois lectures prudentes en découlent.

1. **L'information de présence se reproduit dans le bras fort** : AUROC de 0,82
   à 0,87 dans les trois répétitions, et encore 0,76 à 0,80 sous inversion des
   numéros, contre environ 0,5 pour la base. Le témoin mélangé est plus bas et
   instable. L'échec au critère principal tient donc en partie à un seuil de
   décision trop prudent, et non à une absence totale d'information.
2. **L'information de position canonique préexiste à l'entraînement** : dans la
   base non entraînée, la rotation déplace déjà les logits vers le numéro de la
   ligne touchée dans 19 à 21 blocs sur 24, avec un écart moyen de +2,6 à +2,8,
   sans jamais changer le chiffre choisi. L'entraînement fort n'ajoute rien de
   mesurable sur ce point. Cet effet peut être une conséquence mécanique de la
   perturbation des tokens voisins du numéro ; il n'est pas un compte rendu.
3. **Cette information de position ne suit pas le numéro affiché** : sous
   inversion des numéros, elle retombe à 9,5–11 blocs sur 24 pour le fort. Elle ne suit
   pas non plus simplement la ligne physique, qui donnerait un score proche de
   zéro. Le contrôle visible échouant lui aussi, ce point reste non identifié.

Ces mesures ont été choisies après avoir vu les données ; elles décrivent cet
échantillon et servent à fixer la prochaine expérience, pas à requalifier le
résultat. Les lignes d'un bloc sont appariées : les 72 lignes ne sont pas 72
observations indépendantes. Un détecteur appris de rotations artificielles
reste distinct d'une prévision de ses erreurs naturelles.

## Conséquence pour la recherche

Le résultat principal est négatif et il est conservé tel quel : avec ces
réglages fixes, la localisation entraînée ne dépasse ni la base ni le témoin
mélangé, et le gain isolé du Colab 08 n'est pas confirmé. Les trois répétitions
changent à la fois données et initialisation ; elles ne permettent pas de
séparer ces deux sources de variabilité.

Ce que l'essai apprend de précis :

- la tâche à trois réponses confond deux questions, détecter et localiser,
  et c'est la détection qui porte le signal reproductible ;
- le score de position de 50 % de la base est une réponse constante par bloc,
  et ses logits contiennent déjà un effet de position non entraîné : toute
  suite doit comparer le modèle entraîné à cet effet, pas au hasard ;
- l'entraînement sur une seule présentation n'enseigne pas la règle du numéro
  affiché, même avec un signal visible : le contrôle positif doit d'abord
  réussir sous numéros inversés avant toute conclusion sur le bras caché ;
- huit blocs et 32 mises à jour laissent l'acquisition cachée à 79 %.

La prochaine question précise est donc : **la détection de présence, mesurée
par un critère sans seuil fixé avant collecte, se reproduit-elle sur de
nouvelles phrases, et transfère-t-elle à des directions, des couches ou des
intensités non vues à l'entraînement ?** Un protocole distinct devra fixer
cette mesure à l'avance, conserver le témoin mélangé, et soit enseigner la
règle du numéro affiché au contrôle visible, soit retirer cette question.
Les 72 blocs de test reçus sont désormais des données examinées et ne peuvent
plus servir de validation.

Cette suite est fixée dans le [protocole Colab 10](PRESENCE_DETECTION_PROTOCOL.md),
qui retire la question de la position ; elle n'est pas exécutée ici. Même
réussie, elle établirait une sensibilité fonctionnelle à une perturbation
injectée par l'expérimentateur.
Le lien avec une prévision utile de ses erreurs, avec un modèle de ses
capacités et a fortiori avec une expérience subjective resterait à établir.
Aucun de ces neuf adaptateurs n'est installé dans l'application iPhone.
