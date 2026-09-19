# Vérification des déplacements des poids du Colab 14

19 septembre 2026. **Analyse auxiliaire ajoutée après l'entraînement et calculée
sur les poids sauvegardés, avant lecture des résultats comportementaux complets.**
Elle ne modifie ni les 18 contrastes primaires ni les 306 contrastes du
[protocole fixé](OPTIMIZER_MEMORY_PROTOCOL.md).

Les trois branches de poursuite partent des mêmes paramètres et états AdamW.
Le contrôle à gradients nuls déplace réellement les poids. Son déplacement est
presque parallèle à celui de la poursuite originale dans les trois répétitions.
La branche qui efface uniquement le premier moment se déplace beaucoup moins.
Ces mesures décrivent les paramètres ; leur effet sur les réponses reste à lire
dans l'évaluation séparée.

| Répétition | Norme du déplacement original | Norme avec gradients nuls | Norme après effacement du moment | Cosinus original / gradients nuls |
|---|---:|---:|---:|---:|
| 1 | 0,367001 | 0,366864 | 0,000855 | 0,999999 |
| 2 | 0,483120 | 0,486696 | 0,003396 | 0,999804 |
| 3 | 0,402383 | 0,402103 | 0,001910 | 0,999989 |

Le [rapport numérique](../artifacts/optimizer-memory-pilot/weight-displacement.json)
porte sur les 2 949 120 paramètres FP32 des adaptateurs de chaque répétition.
Les différences sont prises contre le checkpoint commun après 48 mises à jour.
La norme est euclidienne dans ces coordonnées. Elle dépend de la paramétrisation
des adaptateurs : elle n'est pas une distance entre comportements, une quantité
d'information ni une mesure de conscience.

## Contrôle indépendant de la branche sans nouveaux gradients

Pour un pas k après le point commun t=48, avec premier moment m et second moment v :

```text
m_k = beta1^k * m
v_k = beta2^k * v
delta_k = -lr * [m_k / (1 - beta1^(t+k))]
                 / [sqrt(v_k / (1 - beta2^(t+k))) + eps]
```

Le calcul additionne les 16 déplacements en float64, sans appeler
`torch.optim`. Les paramètres enregistrés sont `lr=0,0002`, `beta1=0,9`,
`beta2=0,999`, `eps=1e-8`, sans décroissance des poids. La comparaison avec
les checkpoints réellement produits en FP32 donne une erreur maximale par
coordonnée entre **9,28 × 10⁻⁹ et 9,90 × 10⁻⁹**. Le résidu relatif en norme
est entre **3,15 × 10⁻⁶ et 4,19 × 10⁻⁶**. Ce calcul idéal ne reproduit pas
chaque arrondi de l'exécution GPU ; il n'est pas présenté comme une égalité
octet pour octet.

Les douze fichiers de poids et trois états d'optimiseur sont recontrôlés par
leurs empreintes contre le [constat d'entraînement](../artifacts/optimizer-memory-pilot/training-freeze.json).
L'ordre des paramètres est obtenu à partir de l'architecture Qwen et de
l'installation des adaptateurs sur des tenseurs meta, sans charger les poids
du modèle de base. L'ordre lexicographique des noms de fichiers ne conviendrait
pas, car il place la couche 10 avant la couche 2. Les formes et compteurs sont
également contrôlés.

Un test numérique compare la formule fermée à une récurrence indépendante
sur trois coordonnées, dont une avec moments nuls ; il vérifie aussi le cas
sans pas et le rejet d'un second moment négatif. Le calcul complet sur les
trois états sauvegardés réussit.

```sh
python -m unittest tests_research.test_optimizer_displacement -v
python -m research.audit_optimizer_displacement DOSSIER_DES_POIDS --freeze artifacts/optimizer-memory-pilot/training-freeze.json --output rapport-deplacements.json
```

## Ce que l'on peut en déduire

La proximité des trajectoires est compatible avec un rôle important du moment
hérité dans le mouvement des paramètres pendant cette phase. Les nouveaux
gradients peuvent cependant modifier quelques directions décisives pour une
réponse. Le contraste d'effacement change aussi ces gradients futurs. Ces
normes et cosinus ne décomposent donc pas causalement le comportement en une
« part du moment » et une « part des gradients ».

La prochaine lecture reste celle annoncée : réponses natives, pertes, absence
et présence séparées, reformulation, vocabulaire réservé et tâches publiques.
Un déplacement plus faible n'est pas automatiquement une meilleure rétention.
Ce mécanisme classique d'optimisation ne constitue pas une invention inédite.
