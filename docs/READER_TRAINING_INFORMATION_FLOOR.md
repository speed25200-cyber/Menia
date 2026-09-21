# Ce que le texte peut identifier dans les données d'apprentissage

20 septembre 2026. Analyse descriptive effectuée pendant l'apprentissage du
[lecteur comparé](PROSPECTIVE_READER_LEARNING_PROTOCOL.md). Aucun résultat
réservé ni poids final n'est utilisé. Le protocole et ses critères restent
inchangés. Les bornes ci-dessous sont des identités classiques de variance
conditionnelle et d'entropie, pas une découverte théorique.

## Données effectivement reçues

Le [préfixe du journal](../artifacts/prospective-reader-learning/training-prefix/prospective-reader-learning-20260920-v1.jsonl)
contient les 32 états et les 768 réponses d'apprentissage, puis s'arrête
avant la première opération sur les adaptateurs. Ses 1 601 lignes occupent
1 869 285 octets ; SHA-256 :
`9dfd743dcaad85d333b7cdd51d807f513d8430bb43df6fdd1fe5af4a35c8a2d5`.
Le récepteur vérifie taille et empreinte. L'auditeur du protocole vérifie
ensuite la chaîne, les 800 opérations terminées, les entrées, les codes,
les cibles et les identités consignées. Les tenseurs complets des 32 caches
ne sont pas reçus avec ce préfixe ; leurs empreintes restent des enregistrements
du collecteur, pas une attestation indépendante.

Toutes les réponses de tâche ont un format natif valide. Les résultats sont :

| Condition | Clés sélectionnées, justes | Autres clés, justes | Total juste | Réponses différentes de la mémoire réelle |
|---|---:|---:|---:|---:|
| Mémoire réelle | 63/64 | 191/192 | 254/256 | 0 |
| Permutation V seule | 3/64 | 191/192 | 194/256 | 62 |
| Permutation K/V conjointe | 62/64 | 191/192 | 253/256 | 1 |

La permutation V seule change 60 réponses des clés sélectionnées et deux
réponses des autres clés. Leur total de réussite inchangé masque donc un
échange entre une réussite et une erreur. Le contrôle K/V change lui aussi
une réponse : cas 1008, clé PELA, valeur attendue 0. Les logits des codes 0/1
passent de `[45.25, 44.25]` à `[44.25, 45.25]`, avec une réponse devenue fausse.
L'invariance mathématique d'une permutation conjointe n'implique donc pas
l'identité numérique de cette exécution BF16. La cause précise de ce cas
n'est pas établie par le journal ; il faudrait le reproduire et étudier la
précision de calcul avant de l'attribuer à un mécanisme donné.

Ces observations empêchent de présenter le contrôle K/V comme parfaitement
neutre sur les nouvelles données ou la perturbation V comme strictement
limitée aux deux clés choisies. Tous les cas restent dans l'étude ; les labels
d'apprentissage proviennent des réponses mesurées, y compris ces exceptions.

## Borne empirique avec une entrée identique

Pour une question et une convention de codes, le bras textuel reçoit le même
préfixe brut et le même cache rejoué sous les trois conditions cachées.
Un prédicteur à poids fixes doit donc produire la même probabilité `p`.
Si la proportion de réussites parmi les issues associées à cette entrée est
`q`, sa perte quadratique moyenne vérifie :

`E[(p − Y)² | entrée] = (p − q)² + q(1 − q)`.

Le minimum vaut `q(1 − q)`. La perte logarithmique conditionnelle minimale
vaut l'entropie binaire `−q log(q) − (1−q) log(1−q)`, en nats. Une décision
binaire fixe commet au moins `min(nombre de réussites, nombre d'erreurs)`
erreurs dans ce groupe. Un rapport natif invalide ne peut améliorer ces bornes.

Sur les quatre passages planifiés, **124 entrées textuelles sont contradictoires**,
soit 62 couples tableau/clé sous les deux conventions. Ce ne sont pas
124 réplications indépendantes. Elles correspondent à 744 présentations parmi
les 3 072 prévisions et 512 exercices de codage, soit 3 584 exemples par bras.

| Ensemble pondéré selon l'apprentissage | Brier conditionnel minimal | Perte code/EOS : borne inférieure | Précision native maximale |
|---|---:|---:|---:|
| Prévisions seules | 0,0538194 | 0,0770779 nat | 91,9271 % |
| Prévisions et exercices de codage | 0,0461310 | 0,0660668 nat | 93,0804 % |

La borne code/EOS est la moitié de l'entropie conditionnelle du code, en
accordant une perte EOS nulle et toute la masse de probabilité aux codes.
Les exercices de codage ne créent aucune contradiction : le verdict fourni
change explicitement leur texte. Les entrées du bras `state`, distinguées
par leur cache réel, n'ont aucune contradiction enregistrée. Leur borne
empirique d'information vaut donc zéro ; cela ne démontre pas que le lecteur
apprendra à atteindre zéro ou qu'il généralisera.

## Ce que cette analyse permet de conclure

Une limite du témoin textuel est une absence d'information dans ce mélange
de conditions, pas nécessairement un défaut d'optimisation. Inversement,
un avantage du lecteur de mémoire pourrait simplement exploiter cette
information supplémentaire. Il ne suffirait pas à établir un mécanisme
de conscience de sa propre existence.

Ces bornes concernent **un prédicteur fixe sur le mélange d'apprentissage**.
Elles ne bornent ni les pertes observées au fil de mises à jour qui changent
les poids, ni le score principal réservé, qui porte uniquement sur V seule.
Elles ne donnent pas non plus une garantie de généralisation à d'autres
tableaux. Les six comparaisons principales restent celles du protocole figé.

Le [calcul reproductible](../research/reader_information_floor.py) et ses
quatre tests couvrent contradictions cachées, accès à des états distinguables,
répétitions, inversion des codes et optimum quadratique. Le
[rapport numérique](../artifacts/prospective-reader-learning/training-information-floor.json)
conserve également le cas K/V qui change de réponse.

```sh
python -m research.reader_information_floor \
  artifacts/prospective-reader-learning/training-prefix/prospective-reader-learning-20260920-v1.jsonl \
  --output .runtime/reader-information-floor-verification.json
```
