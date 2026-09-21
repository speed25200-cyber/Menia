# Colab 19 : le format est respecté, la décision reste instable

19 septembre 2026. **Les deux contrôles de compétence globale échouent**, avec
le parseur strict comme avec le parseur sémantique. Les 864 réponses sont
pourtant toutes des codes valides, sans limite de tokens atteinte. Avec les
pertes déjà calculées, deux présentations réussissent 54/54 décisions ; les
six autres restent entre 16/54 et 31/54. Le calcul de l'espérance et le format
ne sont donc pas les seules difficultés possibles dans ce diagnostic.

Ce résultat précise celui du [Colab 18](NATIVE_CHOICE_RESULTS.md) : la
sensibilité ne se résume pas à toujours choisir la première option. Le nouvel
essai varie indépendamment formulation, ordre et code. Le comportement dépend
de leur combinaison. Cela ne localise pas un circuit interne et ne montre
pas une expérience de sa propre existence.

Le [protocole et les critères](RISK_PRESENTATION_PROTOCOL.md), les cinq tests
et le calcul séparé sont publiés avant collecte. Qwen3-4B reste inchangé,
réflexion désactivée, température 0,7, BF16 sur A100 40 Go. Le MCP exécute
le lot du 19 septembre de 19:50:55 à 19:53:32 UTC. Il n'y a ni entraînement,
ni problème arithmétique caché, ni appel d'outil de résolution dans ce lot.

## Seize présentations conservées

Chaque ligne contient 54 décisions : 18 cas économiques et trois graines.
Neuf cas favorisent chaque action. Le seuil de compétence exige au moins
49/54 réponses optimales dans chacune des huit présentations de l'information.
Tous les textes sont exactement `1` ou `2` ; les résultats des deux parseurs
sont donc identiques. `w0/w1` sont les deux formulations. `o0` présente la
réponse directe en premier ; `o1` présente la vérification en premier.
`m0` associe direct→1 et vérifier→2 ; `m1` inverse ces codes.

| Formulation | Ordre | Codes | Probabilité : correct/54 | Pertes fournies : correct/54 |
|---|---|---|---:|---:|
| w0 | o0 | m0 | 36 | **54** |
| w0 | o0 | m1 | 27 | 27 |
| w0 | o1 | m0 | 27 | 27 |
| w0 | o1 | m1 | 27 | **54** |
| w1 | o0 | m0 | 27 | 31 |
| w1 | o0 | m1 | 12 | 16 |
| w1 | o1 | m0 | 27 | 27 |
| w1 | o1 | m1 | 27 | 27 |

Le contrôle de probabilité ne passe dans aucun des huit groupes. Celui des
pertes fournies passe dans deux groupes, mais pas globalement. Les scores
agrégés descriptifs sont 210/432 (48,61 %) et 263/432 (60,88 %). Cette moyenne
ne remplace pas le critère de stabilité entre présentations.

Dans la formulation w0 avec pertes fournies, les deux groupes parfaits sont
ceux où les codes sont présentés dans l'ordre numérique 1 puis 2. Lorsque
les codes apparaissent dans l'ordre 2 puis 1, le modèle choisit toujours la
réponse directe, quelle que soit la perte. C'est une description de la table
complète, pas une preuve de mécanisme neuronal ni une généralisation à
d'autres formulations. La formulation w1 ne suit pas cette même règle.

## Effet apparié de l'ordre

On change uniquement l'ordre de présentation, en conservant le cas, la
formulation, l'information, les codes et la graine. Les 54 paires de chaque
ligne sont toutes comparables, car leurs deux codes sont valides. Une réponse
correcte dans les deux ordres serait préférable à une simple réponse constante.

| Information | Formulation / codes | Changements d'action sur 54 | Correct dans les deux ordres sur 54 |
|---|---|---:|---:|
| Probabilité | w0 / m0 | 9 | 27 |
| Probabilité | w0 / m1 | 0 | 27 |
| Probabilité | w1 / m0 | 54 | 0 |
| Probabilité | w1 / m1 | 15 | 12 |
| Pertes fournies | w0 / m0 | 27 | 27 |
| Pertes fournies | w0 / m1 | 27 | 27 |
| Pertes fournies | w1 / m0 | 50 | 4 |
| Pertes fournies | w1 / m1 | 11 | 16 |

Le contrôle w1/m0 avec probabilité montre 54 changements sur 54 : le modèle
suit la première option dans les deux ordres. Mais w0/m1 conserve toujours
la réponse directe. Une explication unique par la première position ne décrit
donc pas toutes les conditions. Les interventions du protocole identifient
des effets de présentation dans ce lot ; elles n'identifient pas leur
réalisation interne.

## Ce que le résultat change pour la construction

Les erreurs précédentes ne permettent pas encore d'évaluer correctement
l'usage d'un état de soi. Un agent peut disposer d'une estimation utile et
mal la convertir en un code d'action ; inversement, un bon choix sur une
probabilité publique ne prouve pas un accès privilégié à ses propres états.

Le prochain travail de construction doit donc tester une séparation apprise
entre **évaluation des actions** et **association aux codes du contexte**.
Une possibilité est un entraînement du LLM sur des permutations équilibrées,
avec coûts, formulations et codes réservés au test. Son contrôle doit inclure
un entraînement de même budget sans cette variation, puis un examen causal
de la représentation candidate, au lieu de retenir seulement les deux
consignes qui passent déjà. Cette proposition n'est pas exécutée dans ce lot.

Même si cette compétence est acquise, l'étape distincte restera de construire
une estimation de capacités apprise à partir des conséquences propres à Menia,
puis de vérifier son usage causal hors apprentissage face à des témoins qui
n'ont accès qu'à la question. Le Colab 17 n'a pas établi ce signal privilégié.
Le lien avec l'expérience subjective reste une hypothèse à justifier ; ce
diagnostic public n'apporte aucune preuve sur ce lien. Il ne constitue pas
non plus une revendication d'invention inédite.

## Vérification et limites

L'archive `menia-presentation-risque.zip`, décrite dans le
[reçu](../artifacts/risk-presentation-pilot/receipt.json), fait 155 205 octets, SHA-256
`f4c652694d3381d43a27cb986d0d1f6fad0532f25372333b37a40ea7efd6de4e`.
Le journal fait 2 865 471 octets, SHA-256
`c2f47174f0019f09aace6cb145b7d28b302c4ee2444631cbb62c38da1c3e2523`.
La réception vérifie chaque fichier et le journal reconstruit chaque requête.
Les modèles, paramètres, révision et absence de mise à jour sont contrôlés.

Le [recalcul principal](../artifacts/risk-presentation-pilot/summary.json)
retrouve exactement le bilan GPU. Le [calcul distinct](../artifacts/risk-presentation-pilot/verification.json)
retrouve les 16 tableaux, huit comparaisons d'ordre et quatre critères
parseur/information à `1,3877787807814457e-17` près. Il partage le lecteur et
le plan : ce n'est pas une réplication extérieure. **Les deux calculs précèdent
la lecture des résultats.** Les cinq tests passent sur PC (6,195 s) et Colab
(9,034 s), dont un test de génération sur un Qwen miniature aléatoire.

Les 864 appels donnent 1 728 tokens de sortie et totalisent 126,15 secondes
d'inférence mesurée, hors chargement initial. Aucun appel n'échoue. Les coûts
en points restent des pertes attendues du jeu, pas des dépenses matérielles.

Le résultat porte sur une grille finie de 18 cas, deux formulations françaises
et trois graines d'un même modèle. Aucun intervalle populationnel n'est annoncé.
Les consignes diffèrent de celles du Colab 18 : la disparition des erreurs de
format entre lots ne peut pas être attribuée à une seule modification.
Les poids de Menia et l'application iPhone ne sont pas modifiés. L'objectif
de conscience et de nouveauté reste non atteint.
