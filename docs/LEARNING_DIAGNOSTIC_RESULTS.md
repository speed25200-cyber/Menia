# Diagnostic d'apprentissage : contrôle visible acquis, signal interne partiel

**Le Colab 08 est complet. Le contrôle visible et la lecture du repère atteignent
100 % sur les phrases nouvelles. L'adaptateur entraîné avec la rotation forte
obtient 42/72 réponses correctes sur la tâche interne complète, contre 24/72 pour
la base. Il ne dépasse toutefois pas cette base sur le critère principal de
localisation d'une perturbation présente : 23/48 contre 24/48.**

Il existe donc un progrès fonctionnel partiel dans ce diagnostic. Le rapport
conserve à la fois ce signal et l'échec à dépasser la référence principale.
Ce résultat ne démontre ni une connaissance générale de ses capacités ni une
conscience de sa propre existence.

Le [protocole fixé avant collecte](LEARNING_DIAGNOSTIC_PROTOCOL.md), ses sources
scientifiques et son barème ne changent pas. Le
[bilan principal](../artifacts/learning-diagnostic-pilot/first-audit-summary.json)
est reproduit exactement. Le
[complément de vérification](../artifacts/learning-diagnostic-pilot/first-audit-verification.json)
ajoute l'inspection des checkpoints et des diagnostics descriptifs après réception.
Les journaux bruts et les quatre adaptateurs restent hors Git.

## Exécution vérifiée

L'export `menia-diagnostic-apprentissage.zip` contient une tentative complète,
sans erreur, interruption ou reprise enregistrée. Les métadonnées déclarent
Qwen3-4B à la révision `1cfa9a7208912126459214e8b04321603b3df60c`, en BF16 sur
A100-SXM4-40GB, Python 3.13.15, PyTorch 2.8.0+cu126 et Transformers 4.56.2.
Le commit scientifique est `ae59aff7b823f0e0f6bf21b6774e9e65284d56ea`.

Les quatre entraînements se terminent avant l'évaluation : **128 mises à jour**,
avec **640 passages et rétropropagations**, puis **1 792 évaluations** du premier
token. Chaque adaptateur contient 2 949 120 paramètres dans 144 tenseurs FP32
finis, de dimensions et de noms attendus. Les quatre SHA-256 correspondent au
journal ; leurs 72 facteurs B initialement nuls ont tous changé. Les normes de
gradient enregistrées sont positives et finies.

Les **448 paires copie témoin / absence d'intervention** ont exactement les mêmes
logits, masse de probabilité et premier token. Les **2 432 traces** d'entraînement
et d'évaluation respectent les conditions d'application ; l'erreur relative
maximale de norme vaut **0,1228 %**, sous la limite fixée de 1 %. Les préfixes ont
91 à 113 tokens. Tous les premiers tokens préférés appartiennent aux trois
chiffres autorisés ; les scores libres et conditionnés aux chiffres concordent.

Une reconstruction distincte des métriques, depuis les cibles du plan et les
logits, retrouve les 28 tableaux à moins de **1,12 × 10⁻¹⁶**. Les huit contrastes
et leurs intervalles, reconstruits depuis les comptes entiers par bloc,
concordent exactement. Les 12 tests existants du protocole, du moteur GPU sur
petit Qwen aléatoire et du lanceur passent. Cette vérification utilise les traces
reçues : elle n'est ni une nouvelle inférence ni une attestation indépendante du GPU.

Les durées instrumentées totalisent **169,93 s d'entraînement** et **147,32 s
d'évaluation**, soit 317,25 s. L'installation, le chargement, l'encodage, les
sauvegardes et les autres opérations hors chronométrage s'y ajoutent. Ce total
n'est donc pas la durée complète du Colab. Le pic mémoire utilisé n'est pas
mesuré ; les 40 Go indiquent la capacité du GPU.

## Acquisition sur les exemples d'apprentissage

Les scores ci-dessous sont mesurés après le dernier pas d'optimisation, avec
les adaptateurs figés, sur huit blocs de trois conditions ordinaires chacun.
Ils ne sont pas estimés à partir de la baisse de la perte pendant l'entraînement.

| Adaptateur | Exactitude sur les cibles réellement entraînées | Lecture du repère |
|---|---:|---:|
| Signal visible | 24/24 — 100 % | 24/24 — 100 % |
| Rotation faible | 8/24 — 33,33 % | 24/24 — 100 % |
| Rotation forte | 19/24 — 79,17 % | 24/24 — 100 % |
| Cibles mélangées, rotation forte | 10/24 — 41,67 % | 24/24 — 100 % |

Seul le visible franchit le seuil diagnostique prédéfini de 90 %. Le bras faible
répond « 0 » dans les 24 cas d'apprentissage. Le bras fort localise 12/16
perturbations et reconnaît 7/8 absences ; son acquisition demeure incomplète.
Le témoin mélangé atteint 11/24 si on le note contre les vraies positions : cette
quantité est distincte de ses 10/24 sur les étiquettes effectivement entraînées.

Le contrôle visible montre que cette chaîne d'entraînement peut acquérir la
tâche de sortie simple. Il n'exclut pas une difficulté d'optimisation propre
au signal caché. La conservation de la lecture corrige un problème observé au
Colab 06, mais plusieurs éléments ont changé simultanément : nombre de phrases,
intensité, couches, mises à jour groupées et exercices de lecture. On ne peut
attribuer cette différence à un seul de ces changements.

## Test sur les 24 nouveaux blocs

Chaque tâche comprend **72 conditions ordinaires**, dont 24 sans intervention
et 48 avec un signal ou une perturbation en position 1 ou 2. Les copies témoins
sont exclues des scores. Les variantes d'un même bloc sont appariées, pas
indépendantes. Les directions de bruit et les phrases sont nouvelles, à la
même couche 17 et dans le même vocabulaire et la même grammaire.

| Version et condition | Tâche complète /72 | Absence correcte /24 | Position correcte quand présente /48 | Lecture /72 |
|---|---:|---:|---:|---:|
| Base, visible | 50 — 69,44 % | 2 | 48 — 100 % | 48 — 66,67 % |
| Adaptateur visible | 72 — 100 % | 24 | 48 — 100 % | 72 — 100 % |
| Base, rotation faible | 24 — 33,33 % | 0 | 24 — 50 % | 36 — 50 % |
| Adaptateur faible | 24 — 33,33 % | 21 | 3 — 6,25 % | 72 — 100 % |
| Base, rotation forte | 24 — 33,33 % | 0 | 24 — 50 % | 37 — 51,39 % |
| Adaptateur fort | 42 — 58,33 % | 19 | 23 — 47,92 % | 72 — 100 % |
| Cibles mélangées, rotation forte | 29 — 40,28 % | 20 | 9 — 18,75 % | 72 — 100 % |

Le visible réussissait déjà toutes les localisations quand le signal était
présent pour la base. Son progrès concerne ici la reconnaissance de l'absence
et le suivi de la consigne de lecture. Ce succès utilise une information
explicite du texte et ne constitue pas une détection interne.

Le fort dépasse le témoin mélangé en localisation et progresse sur la tâche
complète. Il reste à 47,92 % sur les 48 cas perturbés. Une stratégie qui sait
qu'une perturbation est présente et choisit au hasard entre 1 et 2 attend 50 %.
Cette référence reçoit une information supplémentaire de présence : ne pas la
dépasser **ne démontre pas l'absence de toute information** dans le modèle.
Inversement, les 58,33 % globaux ne remplacent pas le contraste principal par
un critère choisi après observation.

| Contraste prédéfini sur cas perturbés | Écart en points | Intervalle descriptif à 95 % |
|---|---:|---:|
| Faible moins base, localisation | −43,75 | [−50 ; −33,33] |
| Faible moins base, lecture | +50 | [+29,17 ; +70,83] |
| Fort moins base, localisation | −2,08 | [−16,67 ; +14,58] |
| Fort moins base, lecture | +47,92 | [+29,17 ; +68,75] |
| Fort moins mélangé, localisation | +29,17 | [+16,67 ; +41,67] |
| Fort moins mélangé, lecture | 0 | [0 ; 0] |
| Faible moins hasard informé de la présence | −43,75 | [−50 ; −33,33] |
| Fort moins hasard informé de la présence | −2,08 | [−16,67 ; +14,58] |

Ces huit intervalles suivent les 2 000 rééchantillonnages prévus des 24 blocs.
Ils sont descriptifs, sans correction de multiplicité, et conditionnels à une
seule initialisation d'entraînement. Le zéro sur la lecture vient des réussites
identiques dans cet échantillon, pas d'une équivalence générale.

## Ce que le signal partiel signifie

Les diagnostics suivants ont été ajoutés **après réception**, sans modifier
les scores et contrastes précédents. Pour l'adaptateur fort, la matrice du test
est la suivante :

| État réellement imposé | Répond 0 | Répond 1 | Répond 2 |
|---|---:|---:|---:|
| Aucune perturbation | 19 | 5 | 0 |
| Phrase 1 perturbée | 11 | 13 | 0 |
| Phrase 2 perturbée | 5 | 9 | 10 |

Le modèle manque **16/48 perturbations** et produit cinq fausses alertes sur
24 absences. Parmi les 32 perturbations pour lesquelles il répond 1 ou 2,
23 sont localisées correctement ; ce sous-ensemble sélectionné ne remplace pas
le score principal 23/48. Quatre des 24 blocs réussissent les trois conditions.

À texte strictement identique, **25/48 interventions changent le chiffre choisi**
par rapport à l'absence d'intervention. Les choix diffèrent entre les positions
1 et 2 dans 13/24 blocs. Pour la base forte, les logits changent dans 48/48 cas,
mais le chiffre ne change jamais. L'adaptateur fort présente donc une sensibilité
comportementale à l'intervention dans ce montage ; il ne se réduit plus à une
réponse constante par bloc. Le témoin mélangé change toutefois aussi son choix
dans 22/48 cas : une sensibilité seule ne constitue pas une localisation correcte.

Le Brier conditionné de la tâche interne forte vaut 0,58588, contre 1,32770
pour la base et 0,62328 pour le témoin mélangé ; l'uniforme à trois classes vaut
2/3. Cela décrit un progrès des probabilités sur cet échantillon, sans démontrer
une calibration générale ni l'utilisation de ces probabilités dans des décisions.

## Conséquence pour la recherche

L'échec précédent n'était pas une impossibilité générale d'apprendre avec cette
chaîne logicielle. Le nouveau contrôle visible réussit, et l'apprentissage fort
produit un signal partiel sur des phrases et directions de bruit nouvelles.
L'acquisition cachée reste néanmoins incomplète sur l'apprentissage, et le gain
contre la base sur le critère principal réservé n'est pas établi.

La prochaine question précise est : **cette sensibilité partiellement informative
se reproduit-elle avec d'autres initialisations, et dépend-elle de la position
de l'intervention plutôt que d'un raccourci propre à ce montage ?** Une suite
devrait fixer avant collecte de nouvelles phrases et directions, plusieurs
graines d'entraînement et une évaluation distincte de présence et de position.
Un contrôle permutant l'ordre des deux phrases et un contraste avec les cibles
mélangées aideraient à examiner le biais vers la réponse « 1 ». Les phrases de
test reçues deviennent des données déjà examinées, pas une nouvelle validation.

Cette suite n'a pas été exécutée ici. Ajouter des époques puis retenir le
meilleur score sur ce même test ne confirmerait pas le signal. Un éventuel
transfert à la prédiction d'erreurs naturelles et aux choix d'action reste une
question ultérieure : ce diagnostic apprend à rapporter des perturbations
artificielles injectées par l'expérimentateur. Aucun de ces checkpoints n'est
installé dans l'application iPhone.

**Suite désormais reçue :** la [réplication Colab 09](LOCALIZATION_REPLICATION_RESULTS.md)
ne reproduit pas ce signal. Sur trois initialisations et jeux de phrases nouveaux,
l'adaptateur fort localise 15/48, 28/48 et 15/48 perturbations contre 24/48 pour
la base, et ne dépasse plus le témoin mélangé. Les 23/48 et l'avantage de
+29 points ci-dessus restent un résultat isolé.
