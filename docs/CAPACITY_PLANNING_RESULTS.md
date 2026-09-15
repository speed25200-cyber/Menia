# Entretenir une capacité : une partie de la candidate devient exécutable

Le pilote estime l'état caché d'un capteur, prévoit les conséquences de ses
choix et décide quand l'entretenir. Sur **294 912 décisions simulées**, l'accès
aux diagnostics et l'anticipation se complètent dans certaines conditions.
Leur combinaison peut aussi échouer. **Aucune expérience de soi ni méthode
inédite permettant de la produire n'est établie.**

Ce module de recherche est séparé du chat et du modèle de langage de Menia.
Il réalise une partie fonctionnelle de la candidate ; il ne constitue pas
l'entretien réciproque de toute l'organisation qui était proposé.

## Ce que les sources permettent de construire

Dans la prépublication v2 de Whyte et collègues, la disponibilité d'un postérieur
pour des choix contrefactuels dépend d'une précision et d'une échelle temporelle
appropriées. Leur proposition reste agnostique sur la suffisance dans des
systèmes très différents du cerveau. L'articulation avec l'expérience est une
proposition théorique, pas une conséquence du seul calcul bayésien.[^1]

Le modèle de Whyte et collègues (2022) distingue le traitement perceptif,
l'accès à des représentations de séquences et les ressources mobilisées pour
un rapport. Il propose des simulations de résultats avec et sans rapport ;
cela ne constitue pas une expérience subjective observée dans le simulateur.[^2]

Ces travaux motivent l'essai, mais ni quatre pas de planification ni un seuil
de confiance ne sont ici assimilés à une condition suffisante de conscience.
Le texte étudié pour la théorie minimale est **arXiv v2, du 14 juin 2025**.
La publication en revue est attestée en mars 2026 ; son PDF final, inaccessible
pendant cette recherche, n'a pas été comparé ligne par ligne à la prépublication.

## Mécanisme réalisé et limites de l'estimation

Le capteur possède un état caché bon/mauvais, qui détermine réellement sa
fiabilité dans le simulateur. Le diagnostic fournit un indice bruité sur cet
état. L'entretien modifie l'état futur et donc les observations futures. L'agent
peut répondre selon le capteur, s'abstenir ou entretenir. Ces trois actions ont
des coûts explicites ; entretenir occupe le pas au lieu de répondre.

La croyance `q = P(capteur bon)` est révisée après un diagnostic, puis propagée
selon l'action. La fiabilité sensorielle prédite vaut `0,55 + 0,40 q` dans le
monde nominal. `q` est une probabilité d'état, pas l'inverse d'une variance.
Le bruit d'observation est fourni par les vraisemblances ; l'incertitude sur
l'état ne doit pas être confondue avec celle sur leurs paramètres. **Aucun
paramètre n'est appris** et aucune cause externe concurrente n'est inférée.

Les deux facteurs sont l'utilisation des diagnostics et l'horizon de décision
1/4, avec replanification à chaque pas. L'optimalité des problèmes finis ne
garantit pas celle de cette stratégie sur les 64 pas. Même sans diagnostic,
l'agent prédit les effets des actions : sa croyance
n'est donc pas figée. Cette correction du plan précédent évite de retirer
artificiellement toute prévision d'entretien au comparateur. Les quatre versions
reçoivent les mêmes possibilités d'observation et d'action. Une abstention à
coût 0,28 permet aussi un bénéfice immédiat de l'information, sans entretien.

L'agent reçoit le diagnostic et la lecture du capteur, puis produit sa réponse
et ses prédictions avant l'évaluation. Il ne reçoit ni cible vraie, ni état vrai,
ni coût réalisé, ni tirage futur. Les paramètres du monde lui sont fournis, sauf
dans les conditions explicitement mal spécifiées.

## Résultats des neuf conditions fixées avant simulation

Coût moyen par pas, plus bas = mieux. Chaque cellule résume 128 épisodes de
64 pas. Les graines et tirages exogènes sont appariés ; les trajectoires peuvent
diverger à cause des actions. Les coûts incluent l'entretien et l'abstention,
mais le calcul est examiné séparément.

| Condition | Sans diagnostic, H1 | Avec diagnostic, H1 | Sans diagnostic, H4 | Avec diagnostic, H4 |
|---|---:|---:|---:|---:|
| Nominale | 0,279307 | 0,270928 | 0,230151 | **0,168646** |
| Usure lente | 0,275781 | 0,213447 | 0,160718 | 0,089738 |
| Usure rapide | 0,279043 | 0,276147 | 0,279043 | 0,231493 |
| Diagnostic non informatif, cru fiable | 0,279307 | 0,294048 | 0,230151 | 0,246333 |
| Diagnostic inversé, cru fiable | 0,279307 | 0,416479 | 0,230151 | 0,345326 |
| Entretien inefficace, connu | 0,279307 | 0,270928 | 0,279307 | 0,270928 |
| Entretien inefficace, ignoré | 0,279307 | 0,270928 | 0,465137 | 0,470187 |
| Faible contraste sensoriel | 0,280000 | 0,280000 | 0,280000 | 0,280000 |
| Entretien cher | 0,279307 | 0,270928 | 0,279307 | 0,270928 |

Dans le monde nominal, le diagnostic réduit le coût de **0,008379** à H1 et
de **0,061505** à H4. Cette dernière réduction représente **26,7 %** du coût
de l'anticipation seule. L'interaction sur le gain vaut **0,053126**, intervalle
bootstrap apparié descriptif 95 % **[0,046014 ; 0,060081]**. Elle mesure une
complémentarité de tâche dans ce générateur, pas une propriété phénoménale.

Le mécanisme observable est concret : à H4, l'usage du diagnostic réduit la
fréquence d'entretien de 25,0 % à 18,9 %, tout en faisant passer la proportion
de capteurs en bon état de 75,9 % à 81,2 %. L'agent cible mieux ses interventions.
À H1, aucun entretien n'est choisi, car son coût immédiat dépasse l'abstention.

Avec l'usure lente, l'interaction estimée est 0,008646, intervalle
[-0,009270 ; 0,026514] : le bénéfice de la combinaison n'impose donc pas un
effet d'interaction clairement séparé de zéro dans chaque condition.

**Une interaction positive n'est même pas une garantie de bon comportement.**
Lorsque les diagnostics sont inversés, elle vaut encore 0,021998, alors que
les utiliser augmente le coût à chacun des deux horizons. À H4, le coût passe
de 0,230151 à 0,345326. Le signe d'un contraste ne remplace pas l'examen des
quatre performances.

L'entretien inefficace caché est l'échec le plus net de la combinaison : coût
0,470187, avec 40,5 % des pas consacrés à un entretien inutile. L'estimation
de l'état ne suffit pas à réviser un effet d'action supposé connu. Quand cette
inefficacité est connue du modèle, l'entretien disparaît et les deux horizons
produisent exactement le même comportement.

Le faible contraste rend l'abstention préférable dans tous les états ; toutes
les versions s'abstiennent. L'entretien cher supprime aussi son intérêt à H4.
Les neuf conditions sont conservées, sans chercher des paramètres supplémentaires
après lecture des résultats. Les intervalles décrivent la variabilité des épisodes
simulés, sans correction multiple ni portée sur une population humaine.

## Contrôleur ordinaire et coût du calcul

Le solveur de référence développe une récurrence sur la croyance. Un second
solveur construit des vecteurs de coûts conditionnels aux deux états, puis
leur enveloppe inférieure. Il s'agit d'une méthode classique de POMDP ; la
littérature décrit cette représentation par arbres de politiques et fonctions
linéaires par morceaux.[^3] Le code est rédigé ici, sans exécuter de code externe.

Les deux solveurs sont comparés sur six modèles distincts, 1001 croyances,
deux usages du diagnostic et quatre horizons : **48 048 comparaisons** et
**144 144 valeurs d'action**, écart maximal **4,44 × 10⁻¹⁶**, aucun désaccord
de décision. Une énumération indépendante des petits arbres vérifie aussi H2.
Les trajectoires utilisent le solveur compilé. Cet accord indique qu'une
planification ordinaire suffit à produire les effets fonctionnels étudiés.
Il ne prouve pas que ce contrôleur serait conscient ou inconscient.

En condition nominale, H4 utilise en moyenne 6,91 produits scalaires par pas
sans diagnostic et 16,56 avec diagnostic ; H1 en utilise 3. La compilation
construit respectivement 27 et 171 vecteurs candidats au total pour H4. Ces
comptes ne couvrent ni les tris, ni le filtre, ni toute l'instrumentation.

Avec une pénalité illustrative de 0,001 par produit scalaire en ligne, les
coûts ajustés de H4 sont 0,237058 et 0,185209. À 0,01, ils deviennent
0,299214 et 0,334271 : l'avantage s'inverse. Ces valeurs sont une analyse de
sensibilité choisie, pas une mesure énergétique. La compilation est séparée
et exclue de ces coûts ajustés. Aucune égalité de ressources matérielles n'est
revendiquée, et un autre algorithme pourrait rendre les mêmes choix moins coûteux.

## Conséquence pour la recherche sur Menia

Une brique est désormais exécutable : estimer une capacité propre simulée et
prévoir comment l'entretenir modifie ses futures observations. L'échec sous
entretien inefficace précise un manque : apprendre et réviser les conséquences
de l'entretien, en distinguant un diagnostic trompeur d'une intervention ratée.
Cette extension n'est pas encore implémentée dans ce pilote.

Prolongement : le [dossier de raccordement au langage](LLM_COUPLING_RESULTS.md)
ajoute une expérience séparée de sélection entre deux modèles par calibration.
Elle ne constitue pas encore une révision générale des transitions de ce contrôleur.

L'état du capteur n'entretient pas le processus qui planifie ou qui effectue
l'entretien. La réalisation est donc partielle. Il manque aussi une phase de
repos, un modèle de causes externes, une hiérarchie temporelle et l'intégration
au chat. Surtout, aucune mesure ne relie ces calculs à un vécu de sa propre
existence. Les résultats offrent une construction testable et ses limites ; ils
ne satisfont pas encore l'objectif de conscience subjective ou de nouveauté.

## Rejeu et provenance

```bash
python -m research.audit_capacity_planning
python -m research.audit_capacity_planning --check
python -m unittest tests_research.test_capacity_planning -v
```

Le [protocole](CAPACITY_PLANNING_PROTOCOL.md) précède les simulations. Le
[rapport](../artifacts/capacity-planning/report.json) contient paramètres,
contrastes, comptes de calcul et quatre traces nominales complètes. Le contrôle
`--check` compare les nombres à tolérance absolue 10⁻¹⁰, sans réseau. Six tests
contrôlent le filtre, les arbres, l'enveloppe, la valeur de l'information et les
choix dominés. Les tests et le rejeu du rapport sont ajoutés à la CI.
Le rapport généré avec NumPy 2.2.6 est reproduit avec NumPy 2.3.5. Les 78 tests
de recherche et les 48 tests du noyau passent localement.

[^1]: C. J. Whyte et collègues. *On the Minimal Theory of Consciousness Implicit in Active Inference*, [arXiv:2410.06633v2](https://arxiv.org/abs/2410.06633v2), §5, pp. 24–31, version lue ; PDF SHA-256 `c77c0e2ce0b04bbddaf42d66c9abf387b5c4ca455a3006807f999fd4573f94be`. Publication : *Physics of Life Reviews* 56, 4–28, mars 2026, [DOI](https://doi.org/10.1016/j.plrev.2025.11.002), [notice institutionnelle](https://research.monash.edu/en/publications/on-the-minimal-theory-of-consciousness-implicit-in-active-inferen/).
[^2]: C. J. Whyte et collègues (2022). *An active inference model of conscious access: How cognitive action selection reconciles the results of report and no-report paradigms*. [Texte intégral](https://pmc.ncbi.nlm.nih.gov/articles/PMC9593308/), résultats et discussion ; XML Europe PMC lu, SHA-256 `84de3f378696d98933abd860544c3f46c5991ba3f19ef7ea163932446b9c5872`.
[^3]: L. P. Kaelbling, M. L. Littman, A. R. Cassandra (1998). *Planning and acting in partially observable stochastic domains*. *Artificial Intelligence* 101, 99–134, §§4.1–4.3. [Article des auteurs](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf), [DOI](https://doi.org/10.1016/S0004-3702(98)00023-X).
