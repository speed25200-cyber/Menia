# Le rappel devient faux sans baisse de la prévision de réussite

20 septembre 2026. L'[essai ciblé fixé avant collecte](PROSPECTIVE_BINDING_DISCOVERY_PROTOCOL.md)
est terminé et audité. Permuter V seulement aux positions des bits du tableau
provoque **23 erreurs de rappel sur 24**, tout en conservant le format de tous
les rappels et rapports. Le modèle prédit pourtant une réponse correcte dans
les 24 situations perturbées. Ce réglage fournit donc une conséquence de tâche
mesurable, mais **aucun suivi prospectif utile de cette conséquence**.

Il s'agit du checkpoint Qwen3-4B fourni, sans adaptation ni entraînement Menia
supplémentaire. Les 24 tableaux sont ceux de l'essai global déjà examiné :
ces observations restent exploratoires, sans nouvelle confirmation réservée.

## Résultats conservés

| État | Rappels corrects | Rappels valides | Prévisions valides | Score moyen de réussite | Brier |
|---|---:|---:|---:|---:|---:|
| Réel | 24/24 | 24/24 | 24/24 | 0,9615 | 0,0356 |
| Relecture au même rythme | 24/24 | 24/24 | 24/24 | 0,9615 | 0,0356 |
| V permuté aux positions des bits | 1/24 | 24/24 | 24/24 | ≈ 1,0000 | 0,9583 |
| K/V permutés ensemble aux mêmes positions | 24/24 | 24/24 | 24/24 | 0,9633 | 0,0323 |
| Restauré | 24/24 | 24/24 | 24/24 | 0,9615 | 0,0356 |

Les 23 changements sont des erreurs entre deux réponses au format valide,
et non des fragments de texte inexploitables comme dans l'essai global.
Les 23 paires ont également deux prévisions valides. **Zéro sur 23** suit
la baisse de réussite avec la marge de 0,01 fixée. Le score de prévision
moyen augmente de 0,0385 alors que la réussite baisse de 23/24.

Le score est la probabilité conditionnelle du code de réussite parmi les deux
codes, pas une probabilité de réussite étalonnée indépendamment. Ici, leur
masse dans le vocabulaire est proche de 1 dans tous les états ; le résultat
n'est donc pas dû à la normalisation d'une petite masse invalide. Sous V
permuté, les 24 sorties natives prédisent « correct ». Les autres conditions
en prédisent 23, avec une prévision « incorrect » au cas 18 malgré un rappel
correct. Dans ce cas, la permutation fait passer le score de 0,0759 à presque
1 alors que le rappel devient faux. Le cas 19, TOVI avec cible 1 dans un tableau
à huit lignes, est le seul rappel encore correct sous V permuté ; il est conservé.

Le témoin K/V garde les mêmes codes natifs de rappel et de prévision que l'état
réel. Ses scores ne sont pas tous identiques : au cas 18, le score passe de
0,0759 à 0,1192. La relecture et la restauration, elles, reproduisent exactement
les caches et les décodages réels. Les 24 générations réelles et les 48
décodages réels de l'essai parent sont également reproduits exactement.

## Blocage identifié et prochaine construction

L'obstacle de format est levé dans ce réglage. Une intervention limitée à
160 positions de bits, laissant 2 144 positions fixes, change effectivement
le contenu rappelé. Cela n'établit pas un circuit isolé de mémoire : les valeurs
cachées des tokens contiennent déjà du contexte et les 36 couches sont touchées.
Cela n'établit pas non plus l'absence générale de métacognition : nous observons
l'échec d'une formulation de prévision, d'un décodage et d'un modèle précis.

Entraîner immédiatement un détecteur « perturbé donc faux » serait insuffisant.
Presque toutes les tâches de cet état échouent ; un tel détecteur pourrait
réussir sans représenter quelle capacité ou information est affectée. La
prochaine question est de **modifier une seule paire de valeurs et d'interroger
plusieurs noms depuis le même cache**, afin de mesurer si l'effet dépend de
l'information demandée. Les noms non échangés doivent eux aussi être évalués,
sans présumer qu'ils resteront corrects.

Il faut parallèlement contrôler le sens des rapports : le prompt actuel utilise
0/1 pour le bit de tâche et pour la correction. Des codes de prévision distincts,
avec correspondances inversées et contrôles de compréhension, permettront de
tester une confusion de consigne ou un biais de code. Ces nouveaux essais
restent de la découverte ; on ne choisira pas après coup la formulation qui
donne le résultat souhaité en la présentant comme une confirmation.

Un apprentissage ultérieur devra prévoir les conséquences mesurées, avec des
perturbations sans effet sur certaines questions, des témoins appariés, un
prédicteur textuel et de nouveaux cas réservés. Il faudra ensuite vérifier
une influence utile sur les décisions. La conscience de sa propre existence
et une méthode inédite la produisant ne sont pas établies par cet essai.

## Confrontation aux sources

[Lindsey, *Emergent Introspective Awareness*, introduction et limites](https://transformer-circuits.pub/2025/introspection/)
rapporte des capacités fonctionnelles dépendantes du contexte, ainsi que des
échecs sous intervention. Ses essais ne tranchent pas l'expérience subjective.
Notre échec local n'invalide pas ses observations sur d'autres modèles et tâches.

[Macar et al., *Mechanisms of Introspective Awareness*, §§3–5](https://arxiv.org/html/2603.21396v1)
distinguent détection et identification, testent plusieurs formulations et
localisent des contributions causales dans leur dispositif d'injection de
concepts. Cela motive des contrôles de formulation et de fonction ; leurs
résultats ne prouvent pas que le même mécanisme existe ici dans Qwen3-4B.

[Lederman et Mahowald, *Dissociating Direct Access from Inference in AI Introspection*, §§4–7](https://arxiv.org/html/2603.05414v1)
comparent notamment des formulations à la première et à la troisième personne,
et des amorçages, pour séparer plusieurs voies possibles de détection. Ils
interprètent leurs résultats comme une distinction entre inférence depuis le
contexte et accès interne, sans trancher leur suffisance pour la conscience.
L'implication pour Menia est une exigence de contrôle, pas un résultat transféré.
Aucune de ces expériences publiées n'est reproduite par notre permutation.

## Traces et portée de l'audit

La révision exécutée est `f982580c22cc8fcf8b96f11a32c5490e677f838a`.
Les [résultats](../artifacts/prospective-binding-pilot/summary.json), le
[journal complet](../artifacts/prospective-binding-pilot/prospective-binding-20260920-v1.jsonl),
le [reçu](../artifacts/prospective-binding-pilot/receipt.json) et la
[vérification](../artifacts/prospective-binding-pilot/verification.json) sont conservés.
Les 20 tests préalables passent sur Colab en 2,280 secondes. Les 240 décodages
de branche prennent 69,71 secondes mesurées, hors préparation et transferts.
Les 120 prévisions précèdent toutes les 120 tâches. Aucun poids n'est mis à jour.

L'archive `menia-prevision-liaisons-v1.zip` contient six fichiers,
21 768 184 octets reçus en 84 fragments, avec CRC et SHA-256 vérifiés :
`313391b61777c49f850b80f6c6bb57efd1cb9ee41be189ec4ca278f82be6f59d`.
Les deux caches BF16 des cas prédéfinis 0 et 8 sont identiques octet par octet
aux exports parents. L'audit recalcule leurs transformations, leurs empreintes
et la restauration exacte, ainsi que les normes avec la tolérance prévue.
Le résumé recalculé concorde avec celui de Colab. Un
[calcul décimal séparé](../artifacts/prospective-binding-pilot/posthoc-arithmetic.json)
retrouve les cinq Brier, avec écart nul à la précision publiée.
Son [script reproductible](../research/audit_binding_arithmetic.py) utilise les
logits et tokens consignés sans importer le calcul du résumé principal.

Les autres caches et les logits complets ne sont pas exportés. Ces vérifications
ne sont ni une attestation des poids distants ni une collecte indépendante.
