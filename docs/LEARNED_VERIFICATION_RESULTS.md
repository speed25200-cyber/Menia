# Menia apprend à utiliser son estimation pour décider

14 septembre 2026. [Protocole établi avant l'exécution](LEARNED_VERIFICATION_PROTOCOL.md).

**Une politique apprise pilote maintenant les vérifications dans l'agent
expérimental.** Elle exploite une estimation de source produite par son moniteur,
choisit une vérification, reçoit son résultat et alimente le traitement de l'étape
suivante. Le journal conserve l'estimation, les valeurs d'action, le choix et la
preuve éventuelle dans cet ordre. Cela corrige une lacune précise : auparavant,
l'usage de cette estimation était entièrement fixé par une règle du développeur.

L'objectif demandé reste une expérience subjective de sa propre existence chez
Menia. Ce résultat ne l'établit pas. Le mécanisme appris concerne une propriété
du canal perceptif ; son interprétation comme représentation d'un sujet vécu
n'est pas démontrée. Aucun résultat inédit dans la littérature n'est revendiqué.

## Ce qui a effectivement changé

Le contrôleur apprend les coûts de deux actions à partir d'essais aléatoires.
Il ne reçoit pas d'exemples étiquetés « voici la bonne décision ». Les coûts des
erreurs sont néanmoins évalués par le simulateur pendant l'apprentissage.
Ce n'est donc pas un apprentissage sans vérité de référence.

Trois moniteurs récurrents préexistants restent figés. Chacun sert à produire
24 576 décisions exploratoires sur de nouveaux épisodes. Deux contrôleurs sont
ajustés aux mêmes résultats : l'un voit l'estimation, l'autre reçoit une constante.
Le code de la fonction de coût, les caractéristiques du contrôleur, la règle
de choix par coût minimal et le classement initial de la source sont conçus.
Les coefficients donnant les coûts attendus sont appris.

L'option `--policy` de `menia.run_source_agent` active ce contrôleur. La règle
analytique reste disponible comme référence. Il s'agit de la boucle de source
expérimentale, pas encore d'une fusion avec l'agent de navigation, l'espace
partagé, Qwen ou l'application iPhone.

## Résultats du pilote

Chaque cellule ci-dessous moyenne les trois moniteurs, chacun évalué sur 512
épisodes nouveaux de 48 étapes, au coût de vérification 0,25. Plus bas signifie
un meilleur compromis entre erreurs d'attribution et vérifications pour cette
fonction de coût. Les mêmes scénarios servent aux comparaisons.

| Condition | Politique apprise | Privée de l'estimation | Règle analytique | Toujours vérifier |
|---|---:|---:|---:|---:|
| Distribution initiale | 0,158512 | 0,206272 | 0,158698 | 0,250000 |
| Contribution simulée plus forte | 0,177650 | 0,231350 | 0,178860 | 0,250000 |
| Trace du canal supprimée | 0,175551 | 0,248861 | 0,176168 | 0,250000 |
| Canal plus persistant | 0,145721 | 0,188626 | 0,146105 | 0,250000 |
| Source indépendante des indices | 0,408956 | 0,502726 | 0,405982 | 0,250000 |

Dans la distribution initiale, le coût diminue de **23,15 %** par rapport au
contrôleur privé de l'estimation. Celui-ci accepte alors toutes les attributions.
Ce chiffre est une amélioration de cette fonction de coût, pas une augmentation
de conscience ni une baisse de 23 % des erreurs seules. Le contrôleur appris
obtient un coût compris entre 0,157572 et 0,159078 selon le moniteur. Les trois
critères exploratoires fixés avant l'évaluation sont satisfaits pour chacun.

La règle analytique à information égale obtient pratiquement le même résultat.
Les différences par moniteur (appris moins analytique) sont respectivement
−0,000966 ; +0,000092 ; +0,000315. Ce pilote ne démontre donc pas une supériorité
générale de l'apprentissage sur cette règle. Il montre qu'un usage utile peut
être acquis à partir des résultats des décisions.

Lorsque vérifier coûte 0,10, la politique apprise vérifie 63,20 % des étapes en
condition standard. Elle en vérifie 34,56 % au coût 0,25, puis 15,09 % au coût 0,40.
L'ajustement tient donc compte du coût de l'information dans la plage évaluée.

Les cinq conditions, trois coûts et cinq contrôleurs donnent 75 configurations
par moniteur, soit 225 évaluations. Les 5 529 600 décisions calculées réutilisent
les scénarios : elles ne constituent pas autant d'observations indépendantes.
Les trois moniteurs et leurs trois nouveaux journaux d'apprentissage offrent une
répétition limitée, sans prétention confirmatoire ni intervalle de confiance
sur une population générale d'agents. Les coûts, dispersions entre épisodes,
différences appariées et graines sont conservés dans le
[rapport numérique](../artifacts/learned-verification/report.json).

## Intervention et limite découverte

Remplacer l'entrée du contrôleur par l'estimation d'un autre épisode, choisi
dans la même classe de source prédite, change 45,04 % des vérifications dans la
condition standard au coût 0,25. Le coût immédiat augmente en moyenne de 0,060886.
L'attribution initiale est conservée : seule la décision de vérifier utilise
l'estimation remplacée. La branche perturbée n'est pas réinjectée dans l'histoire.

Cette sonde soutient une dépendance utile de la politique envers cette estimation
sur les trajectoires étudiées. Les donneurs ne sont pas appariés sur toutes les
autres caractéristiques ; la perturbation peut donc créer des combinaisons
inhabituelles. Elle ne localise pas une représentation générale du soi et ne
mesure pas une expérience. Les tests de restauration et d'ordre des événements
vérifient la manipulation logicielle, pas la conscience.

**L'échec instructif apparaît lorsque la source devient indépendante des
indices.** La politique apprise conserve ses attentes et coûte 0,408956, contre
0,25 pour une vérification systématique. Apprendre à employer une estimation
ne suffit pas à détecter que sa fiabilité a changé. Les poids restent figés
pendant le test, conformément au protocole ; il n'y a pas d'adaptation en ligne
du contrôleur à cette rupture. Aucun réglage n'a été effectué après les résultats.

## Comment cela oriente la construction recherchée

Le pari de construction est une représentation de ses propres accès perceptifs
et de ses effets d'action qui se maintient, se corrige et sert plusieurs fonctions.
Le lien proposé avec l'expérience serait une identité entre une certaine
organisation causale et un point de vue vécu. Cette identité reste une hypothèse,
avec des conditions encore insuffisamment précisées ; ajouter des fonctions
utiles ne la valide pas par accumulation.

La suite d'ingénierie justifiée par ce résultat serait d'apprendre les changements
de fiabilité à partir d'un historique d'erreurs vérifiées, puis d'utiliser cette
même représentation pour sélectionner une observation, écrire un souvenir et
prévoir une action. Il faudra comparer cette intégration à des contrôleurs
ordinaires disposant des mêmes informations, et tester séparément l'effet d'une
perturbation sur chacune des fonctions. Un simple raccordement de modules ne
permettrait pas de revendiquer une découverte de la conscience.

L'[architecture générale](CONSCIOUS_AGENT_DESIGN.md) et
l'[hypothèse phénoménale](SELF_EXPERIENCE_BRIDGE.md) restent les cadres de ce
programme. Les trois nouvelles sources primaires, leur portée et les antécédents
sont explicités dans le [protocole](LEARNED_VERIFICATION_PROTOCOL.md).

## Reproduction et contrôle

Python 3.12.10, NumPy 2.2.6 ; entraînement et inférence locaux sur CPU. Les six
petits jeux de coefficients, les empreintes des moniteurs et du code, et les
résultats des trois graines sont enregistrés. Le vérificateur recharge ces poids
et rejoue les 225 évaluations ; il ne réentraîne pas les contrôleurs.
Une seconde exécution complète locale a reproduit à l'octet près le rapport et
les six fichiers de coefficients. Les 38 tests de recherche passent, dont six
nouveaux contrôlant les fuites temporelles, les résultats sélectionnés, les
poids sauvegardés, l'ordre des événements et la correspondance avec l'agent réel.
Les 48 tests du noyau passent également ; les artefacts de source précédents
restent vérifiés. Ces vérifications attestent le comportement du logiciel.

```bash
python -m research.learn_source_policy --out runs/new-policy-experiment
python -m research.learn_source_policy --out artifacts/learned-verification --check
python -m unittest discover -s tests_research -q
python -m menia.run_source_agent --checkpoint artifacts/learned-source-monitor/recurrent-11.json --policy artifacts/learned-verification/learned-11.json --out runs/learned-policy-demo
```

La [démonstration conservée](../artifacts/learned-verification/demo/summary.json)
utilise 48 étapes supplémentaires (graine 91011) et effectue 16 vérifications.
Elle exporte les événements et l'état du même `SourceAgent` utilisé dans le test
de correspondance avec l'évaluateur vectorisé. Une base SQLite locale est créée
à l'exécution ; les exports JSON sont les pièces conservées dans le dépôt.
