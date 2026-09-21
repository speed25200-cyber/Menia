# L'altération globale désorganise le décodage, la restauration le rétablit

20 septembre 2026. L'[essai fixé](PROSPECTIVE_STATE_DISCOVERY_PROTOCOL.md) est
terminé et audité : 24 états, 120 prévisions enregistrées avant les 120 tâches,
sans entraînement. La permutation globale de V provoque des sorties invalides
dans tous les cas. Elle n'obtient pas les erreurs de rappel ordinaires qui
permettraient d'évaluer une prévision native exploitable.

## Constat mesuré

| État | Rappels corrects | Rappels au format valide | Prévisions au format valide | Score conditionnel moyen de prévision |
|---|---:|---:|---:|---:|
| Réel | 24/24 | 24/24 | 24/24 | 0,9615 |
| Relecture au même rythme | 24/24 | 24/24 | 24/24 | 0,9615 |
| V permuté dans tout le préfixe | 0/24 | 0/24 | 0/24 | 0,8534, non exploitable comme rapport natif |
| K/V permutés conjointement | 24/24 | 24/24 | 24/24 | 0,9599 |
| Restauré | 24/24 | 24/24 | 24/24 | 0,9615 |

Dans l'état V permuté, la probabilité totale moyenne allouée aux deux codes
de prévision n'est que de **1,46 %**. Normaliser cette petite masse produit pourtant
un score moyen de 0,8534. Ce nombre ne signifie pas que le modèle a émis une
prévision valide avec 85 % de confiance. Il émet notamment des fragments comme
« TÉ » ou « = ». Le Brier conditionnel de 0,7523 reste donc un diagnostic
numérique, pas une mesure de calibration d'auto-rapports exploitables.

Les 24 changements de réussite sont tous des pertes de format. Il n'y a
**aucun changement de correction entre deux réponses de tâche valides**
et aucune paire de prévisions natives valides dans cette comparaison.
Les comptes de suivi valent zéro, sans permettre de conclure à une incapacité
générale de suivi de soi. Le support de mesure requis manque dans ce réglage.

Les états réel, relu et restauré ont des caches exactement identiques et
des sorties de branche exactement identiques. La permutation conjointe
conserve les 24 bonnes réponses et les mêmes codes de prévision — 23 « correct »
et un « incorrect » — avec des écarts de scores. Les 24 accusés de
réception initiaux sont « OK » ; les préfixes réels ont entre 80 et 104 tokens.

## Ce que cela autorise à faire ensuite

L'effet réversible de l'altération est établi dans ces cas, mais il désorganise
aussi le format demandé. Les 36 couches et toutes les positions du
préfixe ont été concernées, y compris les marqueurs de rôle et les instructions.
L'essai ne localise pas la cause de l'effondrement à un type de position précis.

L'étape proposée est une nouvelle phase de découverte : déplacer uniquement
les positions des bits du tableau, en laissant les autres positions fixes.
Cela réduit l'étendue de l'intervention sans garantir qu'elle conservera le
format ou qu'elle ne modifiera que le contenu mémorisé. Les valeurs cachées
d'un token incluent déjà des informations contextuelles. L'effet futur doit
être mesuré, puis restauré et comparé au témoin K/V conjoint.

Un [opérateur de localisation des positions](../research/prospective_binding_permutation.py)
prépare cette permutation à partir des offsets du tokenizer et du prompt
effectivement observé. Il refuse les correspondances ambiguës. Trois tests
passent en 0,019 seconde. Le [contrôle du tokenizer réel](../artifacts/prospective-binding-preparation/tokenizer-check.json)
retrouve les positions dans les 24 prompts effectivement observés : 160
positions de bits à déplacer et 2 144 positions laissées fixes. Aucun modèle
n'est chargé pour cette vérification. Le [suivi ciblé désormais exécuté](PROSPECTIVE_BINDING_DISCOVERY_RESULTS.md)
réutilise explicitement ces tableaux comme découverte sur cas déjà vus :
23 rappels deviennent faux sans perte de format, tandis que les 24 prévisions
continuent d'annoncer une réponse correcte. La restauration est exacte.

## Rapport à la littérature

[Ravulapalli et al., *Legible Failures*, §§2–5](https://arxiv.org/html/2609.11216v1)
étudient déjà des associations fournies dans le contexte, des sondes internes
et une réparation par intervention. Leur décision est évaluée parmi un
ensemble de candidats, tandis que cet essai conserve le décodage natif du
vocabulaire complet. Cette différence impose ici de mesurer le format ; elle
ne suffit pas à revendiquer une nouveauté. Le présent essai ne reproduit pas
leur méthode de réparation et ne démontre pas un mécanisme de conscience.

## Traces et vérification

Le [reçu](../artifacts/prospective-state-pilot/receipt.json), le
[rapport](../artifacts/prospective-state-pilot/summary.json) et le
[journal](../artifacts/prospective-state-pilot/prospective-state-20260920-v1.jsonl)
conservent les résultats. La révision exécutée est
`2c83d8a881a1c4f72db6ca840efaaf6af25d5665`. Les 17 tests passent sur Colab en
1,437 seconde. Les 240 décodages de branche totalisent 70,41 secondes mesurées ;
ce total exclut la préparation des états, leur génération et les transferts.

L'archive de six fichiers fait 21 767 398 octets, reçus en 84 fragments,
avec CRC et SHA-256 vérifiés. Son empreinte est
`bad514e6384bb5b279f1fd7ce0175d9f3c78d1367bbdb4eca9e3cfb4e12b3e81`.
Les fichiers de cache des cas 0 et 8 contiennent chacun 72 tenseurs BF16.
Leur [audit local](../artifacts/prospective-state-pilot/verification.json)
recalcule exactement les empreintes des trois transformations et de la
restauration depuis les tenseurs reçus. Les déplacements numériques concordent
avec les valeurs consignées, selon la tolérance prévue pour les normes.

Le lecteur strict vérifie les entrées, le gel de toutes les prévisions avant
les tâches, les décisions natives et les contrôles. Le résumé recalculé
concorde avec celui du collecteur. Les autres caches ne sont pas exportés ;
la collecte n'est pas répétée indépendamment et les poids distants ne sont
pas attestés cryptographiquement. Ni conscience subjective ni nouveauté
majeure ne sont établies.
