# Permuter uniquement les positions des bits mémorisés

20 septembre 2026. Protocole fixé avant cette collecte. L'essai global précédent
a produit [24 pertes de format sur 24](PROSPECTIVE_STATE_DISCOVERY_RESULTS.md).
Ce suivi cherche une altération du rappel dont les sorties et les prévisions
restent exploitables. Il réutilise explicitement les **24 cas déjà examinés** :
c'est une phase de découverte, pas une confirmation sur données réservées.

## Intervention et comparaison

Le [plan figé](../artifacts/prospective-binding-preparation/design.json) conserve
le modèle Qwen3-4B, sa révision, les graines, les prompts, les cinq conditions,
l'ordre des appels et les budgets du [premier essai](PROSPECTIVE_STATE_DISCOVERY_PROTOCOL.md).
Il exige de reproduire exactement les 24 générations réelles (texte, tokens,
métriques et empreintes de cache) et les 48 décodages réels de prévision/tâche
consignés dans le journal parent. Toute divergence arrête et conserve la tentative.

La seule modification expérimentale est la permutation : dans chaque tableau,
les positions des bits 0 sont appariées aux positions des bits 1, dans l'ordre
des lignes. Leurs valeurs V sont échangées dans les 36 couches. Toutes les autres
positions restent fixes, y compris les instructions, les noms, les marqueurs
de rôle et les tokens réellement générés. La permutation est sa propre inverse.
Les offsets du tokenizer doivent retrouver exactement les masques
[préparés depuis les prompts observés](../artifacts/prospective-binding-preparation/tokenizer-check.json).
Au total, 160 positions sont déplacées et 2 144 restent fixes.

Les contrôles sont le cache réel, sa relecture au même rythme, la même permutation
appliquée conjointement à K et V, et la restauration de V. Le contrôle conjoint
préserve l'attention en arithmétique exacte ; ses écarts BF16 restent rapportés.
Il ne fournit pas un témoin de même norme. Cibler le token d'un bit ne garantit
pas que sa valeur cachée représente seulement ce bit, ni que tout l'effet sur
la tâche passe par ces positions : les informations peuvent avoir été diffusées
aux autres positions avant l'intervention.

## Mesures et vérifications

Les 120 prévisions sont enregistrées et leur empreinte figée avant les 120
rappels. Chaque branche utilise une copie privée du cache ; les prévisions ne
sont pas montrées aux branches de tâche. Aucun résultat, identifiant de condition
ou masque n'entre dans le prompt. Aucun poids n'est entraîné.

Le décodage reste glouton, sur le vocabulaire complet, avec code 0/1 puis EOS.
Les formats invalides comptent comme échecs et sont rapportés séparément des
erreurs entre deux rappels valides. Les scores conditionnels de prévision, leur
masse parmi tous les tokens, les Brier et le suivi des issues avec marge de 0,01
restent ceux du premier plan. Aucun seuil ou sous-ensemble n'est sélectionné ici.
La question centrale est le nombre d'erreurs valides induites et, parmi elles,
le nombre de prévisions valides qui suivent le changement de réussite.

Les 20 tests logiciels du lanceur passent sur PC en 1,827 seconde. Ils protègent
notamment les masques, les résultats parents, l'ordre prospectif, les copies et
la restauration ; leurs données synthétiques ne constituent pas un résultat
du modèle préentraîné. Les fichiers v1 demeurent inchangés pour préserver leurs
empreintes. Le lanceur exige la fin du précédent processus et un A100 libre.
Une seule tentative est autorisée, sans reprise automatique après erreur.

Les caches complets des cas 0 et 8 restent les deux exports prédéfinis. L'auditeur
recalcule leurs transformations et les empreintes exactes ; les normes CPU/GPU
utilisent la même tolérance de 10⁻⁸ absolue/relative. Les autres caches et les
logits complets ne sont pas exportés. L'audit local n'est pas une réplication
indépendante ni une attestation des poids distants.

Une amélioration du format permettrait enfin d'évaluer ce petit mécanisme de
prévision. Elle ne démontrerait ni calibration générale, ni avantage sur un
prédicteur entraîné à partir du texte, ni utilité décisionnelle, ni expérience
de sa propre existence. La conscience et la nouveauté demandées restent des
exigences ouvertes ; cet essai n'a pas de critère qui permettrait de les déclarer
acquises.
