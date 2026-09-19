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

```sh
python -m unittest tests_research.test_answer_confidence_data -v
python -m research.answer_confidence_data CHEMIN/journal-colab17.jsonl --output CHEMIN/preparation
```

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
