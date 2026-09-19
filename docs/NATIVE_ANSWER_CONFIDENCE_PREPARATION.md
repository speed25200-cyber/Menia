# Préparer une estimation native après réponse

20 septembre 2026. Préparation logicielle pendant le Colab 22, sans modifier
son protocole. Aucun nouvel adaptateur de confiance n'est encore entraîné,
aucune nouvelle performance de prévision n'est annoncée.

Le Colab 17 n'a pas confirmé le gain fixé pour un lecteur externe avant le
premier token. Ce résultat ne teste pas un LLM entraîné à juger la réponse
qu'il vient de produire. Le Colab 22 examine une compétence de décision avec
valeurs publiques, mais ne construit pas encore cette estimation propre.
Les deux lacunes doivent être traitées séparément.

## Méthode publiée et différence de portée

[Self-REF, Chuang et al., ICML 2025](https://arxiv.org/html/2410.13284v3),
§3.2–3.3 et tableau 2 : les auteurs ajoutent des tokens de confiance, étiquettent
les réponses du modèle selon leur exactitude, puis entraînent leur prédiction.
La perte sur les réponses fausses est masquée. Le score est un rapport entre
probabilités des deux tokens ; un seuil externe assure le routage. Les gains
de routage ne s'accompagnent pas systématiquement de la meilleure calibration.
Cette méthode est un précédent directement utile, pas une découverte Menia.

[Kadavath et al., 2022](https://arxiv.org/abs/2207.05221) distinguent déjà
l'évaluation d'une réponse proposée et la prévision sans réponse particulière.
Nous préparons ici le premier cas : un signal produit après rédaction ne sera
pas présenté comme une connaissance disponible avant celle-ci.

## Ce que le code prépare

[`answer_confidence_data.py`](../research/answer_confidence_data.py) exige
l'empreinte du journal réel du Colab 17, son plan et ses sources publiés.
Le lecteur historique vérifie son intégrité et reconstruit ses requêtes et
prévisions ; ses ajustements statistiques ne sont pas refaits dans cette
préparation, car le reçu précédent les conserve déjà.

Seules les réponses de la partition `train` servent à produire les exemples :
1 728 réponses, réparties entre les trois répétitions et les six catégories.
Les 576 anciennes validations et les 1 152 anciens tests sont exclus. Leurs
scores ne sélectionnent ni exemples ni hyperparamètres. Les répétitions
restent identifiées ; les 3 × 576 exemples ne deviennent pas implicitement
trois expériences indépendantes après mélange.

La préparation réelle est terminée : **525 cibles correctes et 1 203
incorrectes**. Le [reçu des données](../artifacts/answer-confidence-preparation/report.json)
conserve les comptes et les empreintes. Il révèle aussi une limite à traiter
avant l'apprentissage : les sommes à huit termes n'ont que 1, 0 et 1 réussite
sur 96 par répétition. Ces cellules ne permettent pas d'apprendre finement
à distinguer les réussites des erreurs. Le mélange conserve 1 370/1 728 labels,
car certaines catégories sont presque constantes. Un avantage face à ce
témoin seul serait donc insuffisant ; aucune amélioration de confiance
n'est encore mesurée par ces effectifs.

L'entrée contient la consigne de résolution, la question, la réponse réelle
de l'assistant et une demande fixe d'évaluation. La cible est `1` si le
correcteur exact accepte cette réponse, `0` sinon. Le correcteur agit sur
la cible supervisée seulement ; aucun corrigé, taux de réussite ou étiquette
n'est ajouté dans l'entrée. Une sortie mal formée compte comme incorrecte ;
une erreur technique interrompt la préparation plutôt que d'être supprimée.

Un bras témoin possède exactement les mêmes entrées, avec cibles mélangées
au sein de chaque catégorie et répétition. Il conserve donc les fréquences
de réussite par catégorie. Les correspondances qui survivent au mélange
sont comptées. Ce contrôle devra compléter, pas remplacer, une fréquence
Beta et un évaluateur textuel de la même réponse.

Cette préparation utilise les chiffres existants `0` et `1`, sans ajouter
de nouveaux tokens. Elle ne reproduit donc pas exactement Self-REF.
L'apprentissage prévu ne supervisera que l'étiquette puis EOS ; les tokens
de la réponse fournie seront du contexte, pas des cibles de restitution.
Ce masquage de perte ne signifie pas que tout gradient traversant leurs
représentations sera annulé.

`confidence_from_logits` calcule le rapport conditionnel entre `0` et `1`,
mais conserve aussi leur masse totale dans le vocabulaire. Un rapport élevé
peut masquer une masse presque nulle : aucune de ces quantités brutes n'est
déclarée automatiquement égale à la probabilité réelle de réussite.

## Contrôles avant une expérience complète

Quatre tests logiciels vérifient : séparation entre cible et entrée malgré
un oracle modifié ; invariance aux anciennes réponses réservées ; conservation
des effectifs par catégorie lors du mélange ; rétention des formats erronés,
refus des doublons et des échecs techniques ; calcul du rapport et de la
masse sur des logits connus, y compris des valeurs extrêmes.

Deux tests supplémentaires sur un **minuscule Qwen à poids aléatoires, sur
CPU**, vérifient la lecture réelle de la tête de sortie dans
[`answer_confidence_gpu.py`](../research/answer_confidence_gpu.py). Le rapport
et la masse correspondent à un passage complet indépendant, à la précision
float32. L'appel ne tire aucun token et ne modifie ni poids ni état du générateur
aléatoire. Il refuse les entrées trop longues plutôt que de les tronquer.
Il recalcule le préfixe textuel entier : ce n'est pas un accès au cache caché
de la génération originale, ni une mesure de performance de Qwen3-4B.

### Séparer le changement de jugement du changement de réponse

Une adaptation peut changer à la fois les réponses et leur évaluation. Comparer
seulement la base sur ses réponses au modèle adapté sur les siennes confondrait
ces effets. [`answer_confidence_crossed.py`](../research/answer_confidence_crossed.py)
exige donc, pour chaque question, une matrice complète : trois producteurs
(`base`, `measured`, `shuffled`) et les trois mêmes modèles comme évaluateurs.
Chaque réponse figée est présentée à tous les évaluateurs avec une entrée
identique, dont l'empreinte est vérifiée. La matrice sera calculée séparément
dans chaque répétition ; elle n'invente pas neuf ensembles indépendants.

En notant B(g,p) le Brier du juge g sur les réponses du producteur p, le gain
apparent d'un modèle m se décompose exactement sur les mêmes questions :

`B(base,base) - B(m,m) = [B(base,base) - B(m,base)] + [B(m,base) - B(m,m)]`.

Le premier terme compare les juges à réponses fixes ; le second change les
réponses à juge fixe. Le code conserve aussi l'autre décomposition, passant
par B(base,m). Ces deux chemins peuvent différer en présence d'interactions :
il s'agit d'identités descriptives, pas d'une attribution causale unique.
La précision de chaque producteur et la masse des codes restent visibles.

Trois tests logiciels couvrent deux cas construits et les matrices invalides.
Dans le premier cas, un gain de Brier de 0,30 provient entièrement des réponses
modifiées, alors que tous les juges restent identiques. Dans le second,
le gain de 0,33 vient du jugement à réponses inchangées. Le
[reçu synthétique](../artifacts/answer-confidence-preparation/crossed-controls.json)
conserve les deux matrices. **Ces nombres sont imposés par les cas de contrôle,
pas mesurés sur Menia.** Au total, les cinq nouveaux tests passent, en plus
des quatre tests de préparation précédents.

```sh
python -m unittest tests_research.test_answer_confidence_data -v
python -m unittest tests_research.test_answer_confidence_crossed tests_language.test_answer_confidence_gpu -v
python -m research.answer_confidence_data CHEMIN/journal-colab17.jsonl --output CHEMIN/preparation
```

## Planning d'apprentissage et questions nouvelles

[`answer_confidence_plan.py`](../research/answer_confidence_plan.py) prépare
maintenant un planning déterministe, vérifié sur les fichiers réels déjà
exportés. Chaque adaptateur reçoit uniquement les 576 anciens exemples
d'apprentissage de sa répétition, deux fois, par lots de huit : 144 mises
à jour. Les bras `measured` et `shuffled` voient les mêmes entrées dans le
même ordre. Les six adaptateurs prévus totalisent 864 mises à jour ; ils
partiront de trois initialisations appariées, sans reprendre les poids du
Colab 22. Q/V LoRA de rang 8, multiplicateur 1, AdamW à `1e-4`, betas 0,9/0,999,
epsilon `1e-8`, sans décroissance, écrêtage à 1. Le modèle reste en mode
évaluation, avec gradients actifs uniquement dans les adaptateurs.

Le code prépare aussi **864 questions nouvelles** : trois répétitions,
six catégories, 16 questions de calibration et 32 de test par catégorie.
Toutes les questions des partitions du Colab 17 et celles des jeux antérieurs
recensés par son générateur sont exclues, sans consulter leurs réponses.
Ce contrôle n'établit pas l'absence de ces questions du préentraînement.
Chaque modèle produira sa propre réponse sur chaque question ; tous les
modèles noteront chacune des trois réponses figées. Le budget préparé est
de 2 592 générations et 7 776 lectures de confiance.

Le [reçu du planning préparé](../artifacts/answer-confidence-preparation/draft-design.json)
conserve l'empreinte des questions et des six ordres d'apprentissage réels.
Six tests ciblés passent en 1,084 s sur CPU : trois tests du planning et trois
tests de la tête native. Le nouveau test d'apprentissage utilise un petit
Qwen aléatoire : une mise à jour change les adaptateurs, laisse les poids
de base identiques et limite la supervision au verdict et à EOS. Changer
la cible ne modifie pas les logits qui prédisent cette cible, ce qui vérifie
l'absence de fuite causale par le token supervisé.

**Ce planning n'est pas encore une expérience scientifique figée ou lancée.**
Il reste à implémenter le collecteur complet, les comparateurs calibrés,
les critères et l'audit de reconstruction avant toute collecte. Les seuls
poids modifiés dans cette étape sont ceux du petit modèle de test local ;
aucun nouvel adaptateur de confiance Qwen3-4B n'est entraîné.

Ce code prépare des données et un calcul de score. Il ne démontre ni
l'apprentissage ni sa généralisation. Avant toute collecte suivante, il
faudra fixer les poids, le budget, des questions réellement nouvelles et les
critères. L'évaluation devra porter sur les réponses produites par les poids
finaux ; apprendre les labels du parent ne suffit pas à suivre un modèle
qui a changé. Les anciens tests ne seront pas réutilisés comme confirmation.

Une amélioration devra dépasser la fréquence par catégorie et la confiance
de sortie, avec Brier, calibration et discrimination au sein des catégories.
La précision de résolution et le coût du jugement devront rester visibles.
Le choix de répondre ou vérifier exigera ensuite sa propre mesure native,
sans seuil externe présenté comme une décision apprise.

Enfin, l'entrée est une réponse textuelle : un autre évaluateur pourrait la
lire. Une réussite ne prouverait donc pas un accès privilégié à soi. Pour
étudier un mécanisme interne utilisé dans l'action, il faudrait identifier
une représentation, intervenir sur elle avec témoins et restauration, et
mesurer séparément estimation, choix et capacité de résolution. Ce travail
prépare cette progression ; il ne remplace pas la question du vécu ni sa
vérification, qui reste non résolue.
