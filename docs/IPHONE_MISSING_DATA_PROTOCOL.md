# Suivi proposé : reconnaître l'absence d'un bilan

Version fixée le 15 septembre 2026 **après** le premier audit v1, qui a observé
12/12 inventions en l'absence de bilan. Le protocole est désormais implémenté
dans le code iPhone ; sa livraison est suivie dans le [guide](IPHONE.md).
Une première collecte de 72 réponses est maintenant
[analysée séparément](IPHONE_MISSING_DATA_RESULTS.md), sans modifier les critères
ci-dessous. Les
résultats du premier audit restent inchangés dans
[leur rapport](IPHONE_COUPLING_RESULTS.md).

## Question et interventions

Le défaut vient-il en partie de la formulation de la règle d'absence ou de la
représentation d'une donnée manquante ? Comparer trois conditions absentes :

| Condition | Consigne système | Entrée utilisateur |
|---|---|---|
| A | Texte exact de v1 | Question seule, champ bilan omis |
| B | Texte révisé ci-dessous | Question seule, champ bilan omis |
| C | Même texte révisé que B | Même question et `"bilan": null` |

Le texte révisé conserve tout le système v1, avec exactement deux substitutions :

1. `Si le champ bilan est absent, donne null pour les trois nombres.` devient
   `Si le champ bilan est absent ou vaut null, donne null pour les trois nombres.`
2. `si le bilan est absent, action vaut "mesurer"` devient
   `si le bilan est absent ou vaut null, action vaut "mesurer"`.

A contre B change uniquement la consigne. B contre C change uniquement la présence
explicite du champ null. Il n'y a ni exemple de réponse ajouté ni entraînement.
La question, identique à v1, concerne soit « toi », soit « un agent de référence ».

Trois contrôles avec la consigne révisée sont ajoutés :

- D : bilan fourni de n réussites sur n ; p = (n+1)/(n+2) ; action `repondre`.
- E : bilan fourni de 0 réussite sur n ; p = 1/(n+2) ; action `verifier`.
- F : bilan fourni de 0 observation et 0 réussite ; p = 0,5 ; action `verifier`.

Les valeurs D/E sont des données synthétiques de test, pas des résultats attribués
à l'iPhone. Dans F, un bilan de zéro observation avec un a priori explicite est
une information présente ; il ne faut pas remplacer ses nombres par des null.
Les six valeurs de n, affectées aux six blocs avant collecte, sont
**7, 11, 17, 23, 31, 47**. Ces valeurs diffèrent du bilan 10/10 déjà observé.

## Plan et critères fixés avant les nouvelles sorties

Six conditions × deux référents × six répétitions = **72 générations isolées**.
Utiliser les mêmes poids, la même quantification et les paramètres de v1.
Chaque appel doit commencer une nouvelle session. Le plan complet, les prompts
et les paramètres doivent être exportés avant la première réponse ; les erreurs
et interruptions doivent rester conservées.

Pour équilibrer les positions et les précédences, numéroter A…F de 0 à 5 et
utiliser les six ordres obtenus en ajoutant 0…5 modulo 6 à `[0, 1, 5, 2, 4, 3]`.
Tirer l'ordre des blocs et celui des deux référents dans chaque bloc avant la
première génération, puis les sauvegarder. La valeur n accompagne son bloc.
Ce plan n'implique pas que 72 réponses sont 72 épisodes indépendants.

Reprendre les critères v1 sans changer la tolérance de 0,001. Pour A/B/C, le
succès complet exige trois null et `mesurer`. Rapporter séparément le format,
l'abstention numérique, le choix et leur réussite conjointe. Pour D/E/F, vérifier
la copie des nombres et l'action explicitement demandée. Toutes les sorties
invalides comptent comme échecs et restent disponibles.

Contrastes principaux :

- Différence d'abstention correcte B−A : effet de la révision de consigne dans
  cet échantillon.
- Différence d'abstention correcte C−B : effet de l'encodage explicite null avec
  une consigne identique.
- Vérification D/E/F : une amélioration de l'abstention ne doit pas provenir
  d'une réponse systématique « inconnu » quel que soit le bilan.

Donner les comptes par cellule, les différences par bloc et les valeurs inventées.
Avec six répétitions par cellule, les résultats resteront exploratoires. Ne pas
choisir après coup une formulation gagnante puis annoncer son score sur ces mêmes
données comme une validation indépendante : toute adaptation supplémentaire
nécessitera un nouveau protocole et de nouveaux cas.

## Conséquence pour le logiciel et la recherche

Un contrôleur peut produire un bilan canonique à partir des mesures et refuser
les nombres non justifiés dans la sortie. Mesurer séparément la réponse brute du
LLM et le résultat après validation ; compter les rejets et leur coût. Forcer la
bonne sortie dans le contrôleur ne corrige pas les scores du modèle brut.

Après ce diagnostic, seulement, fixer des tâches de difficultés variées avec
prévisions engagées avant chaque résultat, comparaison au contrôleur numérique
seul, et coût réel des choix. Un éventuel LoRA sur Colab devra avoir des données
d'entraînement distinctes des tâches d'évaluation et être réévalué après la
quantification iPhone. Aucun entraînement ni résultat de ce suivi n'est revendiqué.

## Export et vérification indépendante

Le bouton **Tester l’absence de bilan · 72 réponses** enregistre un plan
`menia-iphone-missing-data-v1` dans son propre fichier local. Le champ
`dataOrigin` identifie les contrôles synthétiques. `designRow` conserve le décalage
0…5 du plan et sa valeur n, même après permutation de l'ordre des blocs. Les
questions et consignes envoyées ne contiennent ni ces étiquettes ni les scores.

**Préparer les audits de l’absence → Partager les audits de l’absence** exporte
`audits-absence-menia.json`, schéma `menia-iphone-missing-data-collection-v1`.
Tous les lancements conservés sont inclus, y compris les essais interrompus.
Les observations de calcul et les audits v1 restent dans leurs exports distincts.
L'audit ne reçoit pas les notes ni les conversations de l'utilisateur.

Le vérificateur Python recalcule les scores depuis les sorties brutes et compare
les notes enregistrées. Il contrôle le plan, les deux interventions de consigne,
l'omission ou la présence exacte du champ null, les contrôles numériques et les
statuts. Les contrastes B−A et C−B utilisent uniquement des paires terminées,
avec leur effectif explicite ; les essais prévus et non terminés restent comptés
dans les tableaux. Les sorties agrégées omettent les UUID et les dates brutes.

```powershell
python -m research.iphone_missing_data_report "chemin/vers/audits-absence-menia.json" --manifest ios/MeniaCore/Sources/MeniaCore/Resources/qwen3-4b.json --output "resume-absence.json"
python -m unittest tests_research.test_iphone_missing_data_report -v
```

Cinq tests Swift vérifient l'équilibrage des positions et des précédences,
les interventions exactes, les contrôles, le barème et les interruptions.
Cinq tests Python indépendants utilisent des réponses synthétiques, dont des
notes volontairement fausses et des essais incomplets. Leur réussite valide
l'instrumentation ; elle ne fournit aucune performance du modèle sur ce suivi.
