# Détection de présence : distinguer signal et compréhension de la question

Note méthodologique du 19 septembre 2026, écrite pendant l'entraînement du
Colab 10, avant consultation de ses évaluations. Elle ne change ni son code,
ni son critère principal, ni ses conditions de test. Les essais proposés
ci-dessous n'étaient pas encore un protocole fixé ou exécuté lors de sa rédaction.
**Mise à jour :** le [protocole distinct du Colab 11](PRESENCE_SPECIFICITY_PROTOCOL.md)
est désormais implémenté, figé dans la révision `e783e117f2d59c89bcf4be467506ae115e5aaeff`,
et exécuté sur l'A100 le 19 septembre. Son
[résultat négatif est reçu et vérifié](PRESENCE_SPECIFICITY_RESULTS.md).
La note d'origine ci-dessous est conservée.

## Ce qui motive ce contrôle

[Hahami, Sinha et Jain, §2.3.1](https://arxiv.org/html/2607.14111v1)
observent, dans leur montage d'injection de concepts, que la hausse du logit
affirmatif peut être presque identique pour une question introspective et
une question factuelle dont la réponse est négative. Ils proposent notamment
des comparaisons de positions et d'intensités. Leur manipulation et leurs
modèles diffèrent de nos rotations entraînées sur Qwen : ce résultat identifie
une explication concurrente à tester, pas un biais déjà démontré dans Menia.

[Martorell, §3.2 et §5.5](https://arxiv.org/html/2603.18893v1) utilise une
lecture continue des logits et la compare à des directions internes définies
par des sondes. L'auteur souligne les limites de ces sondes et des transferts
entre concepts et modèles. Lire une distribution plutôt qu'un token unique
a donc des précédents ; cela ne transforme pas la mesure en preuve de vécu.

[Singh, Linzen et Ravfogel, §4 et §5.3](https://arxiv.org/html/2605.26242v1)
ajoutent une objection distincte : dans leurs essais, certains modèles
détectent une anomalie sans distinguer sa provenance textuelle ou interne.
Ils demandent des preuves de mécanismes de surveillance au-delà d'une
classification sur des états cachés. Cette critique de l'inférence ne démontre
pas l'impossibilité de l'introspection ; elle précise ce qui reste à identifier.

## Contre-exemple analytique pour notre critère

Considérons un détecteur ordinaire dont le score vaut
`logit(1) - logit(0) = b + alpha × p`, où `p` indique seulement si le calcul
est perturbé. Il ne représente ni son identité ni le sens de la question.
Si `alpha` dépasse l'étendue des valeurs de `b` entre blocs, toute présence
a un score supérieur à toute absence : AUROC = 1 et ordre au sein du bloc = 1.
Les copies témoins restent identiques. Un témoin mélangé dont `alpha = 0`
peut avoir une AUROC de 0,5. Ce dispositif satisfait alors la règle du
Colab 10 dans chaque répétition.

Il pourrait pourtant conserver le même score quand on lui demande de répondre
`1` pour l'absence, ou pour une question sans rapport avec l'intervention.
Le critère démontre une séparation des conditions, sans identifier à lui seul
une interprétation sémantique du compte rendu. Ce contre-exemple concerne la
portée logique du critère ; ce n'est pas un résultat Qwen ni une découverte
revendiquée. Le protocole initial limite déjà sa conclusion à un détecteur.

## Contrôles proposés pour une expérience distincte

Avec les adaptateurs figés, comparer sur de nouvelles phrases :

- La question originale et une consigne explicite échangeant `0` et `1`.
  Le score orienté vers la présence doit suivre la règle demandée. Conserver
  aussi les logits bruts permet de voir un biais attaché au chiffre.
- Des questions de lecture publique utilisant ces mêmes deux réponses,
  avec autant de cibles `0` que `1`, sous les mêmes interventions. Mesurer
  l'effet de l'intervention sur le score brut pour chaque type de question.
- Un contrôle visible soumis aux mêmes nouvelles consignes. S'il échoue à
  les comprendre, l'échec du bras caché ne localise pas le défaut à l'accès
  interne. Inverser une consigne non entraînée reste un test de transfert.
- La base et les cibles mélangées, dans chaque répétition ; conserver les
  copies témoins, appairer les blocs, figer tailles, graines et analyses avant
  collecte. Ne pas sélectionner une répétition ou une consigne gagnante.

Un effet brut identique sur des questions sans rapport affaiblirait une
interprétation introspective. Une interaction entre question et intervention
écarterait cette explication simple, mais resterait compatible avec un
classifieur conditionnel ordinaire. Comparer des questions modifie aussi les
activations ; ce n'est pas une isolation parfaite d'un circuit de rapport.

Avant d'intégrer un éventuel détecteur à l'agent, il faudrait encore mesurer
sa relation avec des erreurs de tâche et le bénéfice de décisions coûteuses
de vérification, par rapport à des références simples. Détecter une rotation
artificielle ne dit pas si une réponse est fausse, ni si l'agent se sait exister.
