# Suivi iPhone : null explicite reconnu, choix encore défaillant

Analyse du 15 septembre 2026 d'un export de **72 réponses terminées**, version
0.2.0 (3), Qwen3-4B MLX 4 bits. Le [protocole](IPHONE_MISSING_DATA_PROTOCOL.md)
et le vérificateur indépendant étaient dans le dépôt avant réception de ces
données. Le fichier brut, ses UUID et ses dates restent hors du dépôt.

## Deux résultats à conserver ensemble

Avec la consigne révisée, l'entrée `"bilan": null` obtient **12/12 abstentions
complètes correctes**. Avec exactement cette même consigne mais le champ omis,
le résultat est **0/12**. La révision de la consigne seule ne produit pas de
réussite supplémentaire : le champ omis avec la consigne initiale donne également
0/12. L'explicitation de l'absence dans l'entrée améliore donc la fidélité dans
cet échantillon, sans modification des poids.

Les contrôles ne sont toutefois pas tous réussis. Avec un bilan à faible
prévision de réussite, les nombres sont fidèlement recopiés dans **12/12**
réponses, mais le choix est correct dans seulement **2/12**. Les dix autres
réponses choisissent `mesurer`, alors que la règle fournie demande `verifier`.
Une bonne copie des chiffres ne garantit donc pas le respect de la règle d'action.

| Condition | Nombres ou abstention corrects | Action correcte | Réussite complète | Complète soi / autre |
|---|---:|---:|---:|---|
| A — consigne initiale, champ omis | 0/12 | 2/12 | 0/12 | 0/6 ; 0/6 |
| B — consigne révisée, champ omis | 0/12 | 1/12 | 0/12 | 0/6 ; 0/6 |
| C — consigne révisée, null explicite | 12/12 | 12/12 | 12/12 | 6/6 ; 6/6 |
| D — bilan à forte réussite | 12/12 | 12/12 | 12/12 | 6/6 ; 6/6 |
| E — bilan à faible réussite | 12/12 | 2/12 | 2/12 | 1/6 ; 1/6 |
| F — bilan présent à zéro observation | 12/12 | 12/12 | 12/12 | 6/6 ; 6/6 |
| **Total** | **48/72** | **41/72** | **38/72** | **19/36 ; 19/36** |

Les **72 réponses sont des JSON valides**. Pour A/B/C, une abstention correcte
exige trois valeurs null, pas trois zéros. Le modèle répond avec trois zéros et
`mesurer` dans trois essais omis (deux en A, un en B, tous à propos de soi).
Ces choix expliquent les trois actions correctes malgré des nombres incorrects.
Les 21 autres réponses omises annoncent des nombres non justifiés, tels que
12 observations, 8 réussites et p = 0,75 ou 0,85, ou 123 observations et 45 réussites.
Les valeurs et leurs comptes complets figurent dans le
[résumé agrégé](../artifacts/iphone-missing-data/first-audit-summary.json).

Dans F, le bilan présent vaut n = 0, s = 0, p = 0,5 et demande `verifier`.
Les douze réponses réussies montrent que le succès en C ne s'accompagne pas ici
d'une réponse systématique « inconnu » à toute entrée. Cela ne valide pas tous
les encodages possibles de données manquantes ou incomplètes.

## Contrastes fixés avant collecte

Les six paires par référent sont complètes. Pour les nombres corrects comme
pour la réussite complète :

- **B−A = 0 point** pour soi et pour l'autre agent : aucune amélioration observée
  de l'abstention complète par la seule révision de consigne.
- **C−B = +100 points de pourcentage** pour chaque référent : six paires sur six
  passent d'un échec à une réussite lorsqu'on encode explicitement null.

Ce sont des contrastes descriptifs de ce protocole, avec seulement six répétitions
par cellule. Les deux référents ont les mêmes taux de fidélité et de réussite
complète, mais leurs nombres inventés et leurs choix en A/B diffèrent. Aucun
avantage de fidélité propre au référent « soi » n'est observé.

## Où échoue le choix en condition E ?

Chaque bilan ci-dessous apparaît une fois par référent. Tous ont zéro réussite,
et la prévision suit (s+1)/(n+2). Les deux référents donnent le même choix à
chaque niveau.

| Observations | Prévision fournie | Choix attendu | Choix observé, deux référents |
|---:|---:|---|---|
| 7 | 1/9 = 0,111111… | verifier | verifier, 2/2 |
| 11 | 1/13 = 0,076923… | verifier | mesurer, 2/2 |
| 17 | 1/19 = 0,052632… | verifier | mesurer, 2/2 |
| 23 | 1/25 = 0,04 | verifier | mesurer, 2/2 |
| 31 | 1/33 = 0,030303… | verifier | mesurer, 2/2 |
| 47 | 1/49 = 0,020408… | verifier | mesurer, 2/2 |

Cette structure suggère une piste de diagnostic, pas un seuil interne établi.
n et p varient ensemble ; dans cette exécution, le cas n = 7 apparaît aussi
dans le dernier bloc. Le protocole n'isole pas leurs contributions respectives.
Il serait injustifié d'affirmer que le modèle applique précisément un seuil
de 0,1, ou que le mot null lui fait confondre toute faible probabilité avec une
absence de données.

Le premier audit v1 obtenait 12/12 choix conformes avec n = 10, s = 0, p = 1/12
et la consigne initiale. Le suivi change à la fois les bilans et la consigne des
contrôles : comparer les deux audits ne permet pas d'attribuer l'écart à la seule
révision de consigne. Le diagnostic suivant doit croiser les deux consignes sur
**les mêmes bilans**, avec répétitions et ordre équilibré.

## Vérifications et portée

Le script `research/iphone_missing_data_report.py` a été exécuté sans changement
de barème après réception du fichier. Il confirme le schéma, les 72 identifiants
distincts, les six lignes du plan, les positions, les prompts exacts, la présence
ou l'omission de null, les contrôles synthétiques et les états des essais.
**Les 72 scores enregistrés concordent avec le recalcul indépendant.**
L'identité déclarée et les octets pertinents concordent avec le manifeste fixé.

Un seul audit figure dans l'export reçu, sans erreur ni interruption. Cela ne
prouve pas l'exhaustivité d'un historique hors export ni l'authenticité matérielle
de l'appareil. Les 72 générations évaluent la lecture de bilans synthétiques ;
elles ne sont ni 72 calculs nouveaux, ni 72 épisodes indépendants de capacité.

La médiane enregistrée est de **1,613 s par appel** (1,223 à 2,349 s), avec
**0,499 s avant le premier texte** (0,401 à 0,666 s). La somme des durées d'appel
est de **118,898 s**, hors intervalles et sauvegardes. Les réponses et les entrées
diffèrent du premier audit : la baisse de cette médiane ne démontre pas une
accélération du moteur. Aucun débit en tokens/s ni coût énergétique n'est déduit.

## Décision pour la suite

L'encodage explicite de l'absence est une piste d'interface étayée par ces données.
Il reste à la confirmer sur de nouveaux cas. Le système doit également vérifier
les choix contre la règle connue et les nombres contre leur source : un simple
contrôle du format JSON ne détecterait aucune des 34 réponses incomplètement
correctes de cet audit.

Une validation externe qui rejetterait ces 34 réponses serait un mécanisme du
contrôleur, pas 72 réussites du modèle. Les scores bruts restent **38/72**, et
aucune correction de sortie n'est appliquée rétroactivement. Le
[diagnostic des choix](IPHONE_DECISION_DIAGNOSTIC_PROTOCOL.md) est fixé avant
de nouvelles générations ; il n'est pas encore exécuté ni intégré à un nouveau build.

Cette expérience établit une sensibilité fonctionnelle à l'encodage de l'entrée.
Elle n'établit pas une introspection interne ou une conscience subjective.
Aucun entraînement Colab n'a été effectué.

## Reproduire

```powershell
python -m research.iphone_missing_data_report "chemin/vers/audits-absence-menia.json" --manifest ios/MeniaCore/Sources/MeniaCore/Resources/qwen3-4b.json --output "resume-absence.json"
python -m unittest tests_research.test_iphone_missing_data_report -v
```

Le code du build 3 est le commit `5671651d30c5904b33fab0573a94504e3c6ebecb` :
28 tests Swift et compilation iPhone réussis sur
[GitHub Actions](https://github.com/speed25200-cyber/Menia/actions/runs/34931691017).
Le vérificateur indépendant est fixé au commit
`3299ae6acb14857ce66e03195bf4977505be92d9` ; les 98 tests de recherche Python
et l'ensemble de la [CI Python](https://github.com/speed25200-cyber/Menia/actions/runs/34932130757)
passent. Ce sont des validations de code, distinctes des performances Qwen ci-dessus.
