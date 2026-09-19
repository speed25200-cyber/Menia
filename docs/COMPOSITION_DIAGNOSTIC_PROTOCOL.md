# Diagnostic des consignes et de la décision après le Colab 12

**Protocole prospectif du Colab 13, choisi après lecture du Colab 12.**
Il départage deux limites observées, sans réentraîner ni remplacer les critères
échoués du [test précédent](STATE_COMPOSITION_RESULTS.md). La nouveauté et la
conscience ne sont pas établies. Le diagnostic sert à choisir quelle limitation
corriger avant d'étudier un mécanisme d'auto-représentation plus fort.

## Deux questions séparées

1. La question publique reformulée sur PHRASE 2 est-elle déjà mal comprise
   par la base, ou sa compréhension se dégrade-t-elle avec les adaptateurs ?
2. Un seuil appris sur les anciens exemples d'apprentissage suffit-il à
   transformer le classement interne en décision sur de nouvelles phrases,
   et ce seuil se transfère-t-il à la formulation reformulée ?

Une réponse positive à la seconde question désignerait une information
exploitable par un lecteur numérique externe. Elle ne montrerait pas que
le modèle produit lui-même une réponse correcte, utilise cette information
pour agir, ni qu'il possède une expérience de soi.

## Données et calcul fixés avant nouvelle collecte

- Qwen3-4B à la révision déjà fixée
  `1cfa9a7208912126459214e8b04321603b3df60c`, BF16, SDPA, premier token sans
  échantillonnage ni chaîne de pensée ; couche 17, rotation de force 1.
- Trois répétitions, 24 blocs par répétition, soit 144 phrases distinctes.
  Les phrases et graines de rotation excluent les plans 06/08/09/10/11/12.
  Le vocabulaire et les deux formulations de consigne sont réutilisés : il
  s'agit de nouveaux exemples, pas de nouvelles familles linguistiques.
- Quatre bras : base, parent fort du Colab 10, composé et consignes du Colab 12.
  Les neuf adaptateurs nécessaires sont vérifiés par SHA-256 avant et après
  inférence. Aucune sélection de la meilleure répétition ; aucun entraînement.
- Deux formulations, canonique et reformulée ; deux correspondances entre
  oui/non et 0/1. Les logits 0/1/2/3 et le premier token libre sont conservés.
- Trois flux : présence cachée ; repère public sur PHRASE 1 visible ; repère
  public sur PHRASE 2 visible. Dans ces deux derniers flux, [SIGNAL] est un
  distracteur textuel, sans perturbation interne.
- Quatre positions par condition : absence, première phrase, seconde phrase,
  copie identique témoin. Le témoin doit reproduire exactement logits, masse
  des chiffres et premier token de l'absence.

Cela représente **13 824 inférences et 3 456 paires témoins**, zéro mise à jour.
Les contrôles publics sous perturbation et la présence visible ne sont pas
répétés ici ; ils figuraient dans le Colab 12. Cette restriction empêche de
conclure à leur préservation sur les nouveaux exemples du Colab 13.

## Seuils externes gelés

Le [manifeste](../artifacts/composition-diagnostic-pilot/frozen-thresholds.json)
contient les six valeurs, les poids sources, les 288 identifiants de requêtes
d'apprentissage retenus et les empreintes des scores. Il est publié avant la
collecte 13. Son journal source est celui du Colab 12 complet, mais la sélection
numérique prend exclusivement `train/composed/hidden/trained/monitor`, positions
0, 1 et 2 : 16 blocs, 48 scores par répétition et correspondance. Les scores de
test et les copies témoins ne participent pas à l'ajustement. Le choix de faire
ce diagnostic a toutefois été motivé par les résultats de test du Colab 12.

Le score est `logit(oui) − logit(non)`, selon la correspondance demandée.
L'exactitude équilibrée vaut la moyenne des sensibilités absence/présence :
poids 1/2 pour l'absence, 1/4 pour chaque position présente dans un bloc.
Les candidats sont les milieux des scores uniques consécutifs, zéro et les
deux bornes extérieures min−1/max+1. On maximise l'exactitude équilibrée ;
à égalité on choisit la plus petite valeur absolue, puis la plus petite valeur.
La décision externe est « présence » strictement lorsque le score dépasse
le seuil. Une égalité signifie absence.

| Répétition | Code normal | Code inversé |
|---|---:|---:|
| 1 | 10,1875 | 11,1875 |
| 2 | 10,5 | 11,6875 |
| 3 | 7,75 | 9,1875 |

Chaque seuil s'applique tel quel aux deux formulations nouvelles du composé.
Aucun seuil n'est réestimé sur le Colab 13. Deux références sont conservées :
le premier token effectivement produit et la décision numérique au seuil zéro.
Cette dernière peut différer du premier token en cas d'égalité des logits ou
de sortie hors code ; ces cas ne sont pas effacés des résultats natifs.

## Analyse annoncée

144 tableaux donnent l'exactitude équilibrée native, l'exactitude brute,
le respect du code, les masses de probabilité et le score de Brier à deux
classes (somme des deux termes). Les tableaux de présence donnent aussi
l'AUROC orientée, distincte d'une exactitude de réponse. Les douze tableaux
de présence du composé ajoutent les décisions aux seuils zéro et gelé.

108 contrastes descriptifs sont calculés :

- 48 changements d'exactitude publique reformulée moins canonique ;
- 36 différences de ces changements contre la base, pour chacun des trois
  adaptateurs ; une valeur négative indique une pénalité de formulation
  supplémentaire pour l'adaptateur ;
- 24 gains du seuil externe contre le premier token ou contre le seuil zéro.

Les intervalles à 95 % proviennent de 2 000 rééchantillonnages de blocs,
stratifiés par position du repère public (12 blocs de chaque classe).
Les tirages sont partagés entre les conditions d'une répétition. Les trois
positions non témoins d'un bloc restent ensemble ; aucune pseudo-réplication
par token ou position. Les intervalles restent descriptifs, sans correction
de multiplicité, conditionnels à trois familles de poids du même modèle.

Ce diagnostic n'a pas de verdict global « réussi ». Une pénalité supplémentaire
des adaptateurs orienterait vers la diversité des consignes et la préservation
des compétences ; un gain du seuil sur de nouveaux blocs orienterait vers
la calibration de décision. Un échec de transfert du seuil reformulé mettrait
en évidence un déplacement de score dépendant de la consigne. Aucun de ces
constats n'identifie à lui seul le mécanisme causal exact.

## Exécution et contrôle

Le moteur refuse tout événement d'entraînement, changement de source, de plan,
de poids, de seuil ou d'environnement à la reprise. Une requête interrompue
est enregistrée et rejouée ; une exécution complète n'est pas relancée.
Les tests incluent une reprise sur un petit Qwen réel, l'immutabilité des poids,
des oracles de classement sans bonne décision, l'exclusion des scores de test
de l'ajustement et le rejet des témoins altérés. Les données synthétiques des
tests ne sont pas des résultats de Menia.

Conserver `menia-diagnostic-composition.zip` et les journaux sources 10/12.
Le lanceur réutilise l'environnement A100 existant et ne réinstalle ni
ne réentraîne les parents. Aucun adaptateur n'est déployé sur iPhone par ce test.

**Après collecte :** le [résultat complet](COMPOSITION_DIAGNOSTIC_RESULTS.md)
est reçu et audité le 19 septembre 2026. Les règles ci-dessus restent celles
du protocole fixé avant les inférences ; aucun seuil n'est réajusté sur le test.
