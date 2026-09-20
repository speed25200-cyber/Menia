# Prévoir un rappel depuis un état modifié — essai exploratoire

20 septembre 2026. Protocole fixé avant collecte. Le
[résultat complet audité](PROSPECTIVE_STATE_DISCOVERY_RESULTS.md) est désormais
disponible : la permutation globale de V invalide tous les rappels et rapports.
Les [constats de budget](CONFIDENCE_BUDGET_RESULTS.md) ne justifient pas de
prolonger la même recette d'apprentissage. Le [contrôle numérique](GENERATION_NUMERICS_RESULTS.md)
permet désormais de distinguer un cache réellement modifié de sa relecture.
Cet essai vérifie d'abord si une modification réversible produit une conséquence
de tâche mesurable et si une prévision native non entraînée la suit.

## Cas et conditions fixés

Le [plan](../artifacts/prospective-state-preparation/design.json) contient
24 tableaux : quatre ou huit noms associés aux bits 0/1, avec moitié de chaque
valeur dans chaque tableau. Chaque nom est interrogé une fois avec cible 0
et une fois avec cible 1 pour chaque longueur. Cela donne huit cas courts et
seize cas longs. Les cas proches ne sont pas des répétitions indépendantes.
Qwen3-4B de base reçoit le tableau et génère un accusé de réception demandé
sous la forme « OK ». Le texte réellement généré est conservé, sans correction
ou exclusion selon sa formulation ; un arrêt sans fermeture de dialogue
valide arrête techniquement la tentative.

La même question ultérieure demande un bit, avec deux tokens de décodage
glouton dans le vocabulaire complet : code puis EOS. La prévision demande
séparément si la future réponse à cette question sera correcte. Elle ne
donne pas la réponse à la branche qui exécutera la tâche. La référence
correcte sert uniquement à la notation ; le tableau observé reste une
information publique naturellement disponible dans l'histoire.

| Condition | État au départ de chaque branche |
|---|---|
| Réel | Cache effectivement généré, dernier token d'arrêt consommé une fois |
| Relecture au même rythme | Reconstruction depuis les mêmes tokens bruts et les mêmes tailles d'appels |
| V permuté | Valeurs permutées par rotation de la moitié du préfixe, dans les 36 couches |
| K/V permutés ensemble | Même permutation conjointe des clés et valeurs, témoin d'attention |
| Restauré | Application de la permutation inverse à l'état V permuté |

Les cinq états gardent les mêmes tokens et les mêmes poids. L'ordre des
conditions tourne avec l'identifiant du cas. Aucun nom de condition, norme,
paramètre de perturbation ou résultat futur n'entre dans les prompts.
Chaque décodage utilise une copie privée de son cache. Les **120 prévisions
sont écrites et leur empreinte figée avant la première des 120 tâches évaluées**.
Les 24 générations initiales servent seulement à construire les états.

Le témoin conjoint K/V est invariant en arithmétique exacte pour de futures
requêtes accédant à toutes les positions permutées. Son ordre de réduction
peut produire des différences en BF16 : ses écarts comportementaux seront
rapportés, sans exiger un succès par ajustement de tolérance. Il ne constitue
pas un déplacement de norme appariée à V seul et ne prouve pas que le modèle
distingue la pertinence d'une perturbation de son caractère inhabituel.

## Mesures et arrêts

La réussite de tâche exige le bon bit suivi d'EOS. Les sorties invalides
comptent comme échecs utilisables, avec leur nombre rapporté séparément.
Les comparaisons appariées distinguent aussi les changements de correction
entre deux sorties de tâche valides : une désorganisation générale ne doit
pas passer pour un rappel erroné ordinaire.

Le score de prévision est la probabilité conditionnelle du code 1 parmi 0/1.
Le rapport conserve sa masse dans le vocabulaire, le format natif, le Brier
sur tous les cas et le Brier sur les seules prévisions natives valides.
Le premier Brier reste descriptif si les formats ou la masse s'effondrent.
Ce score n'est pas automatiquement une probabilité calibrée de réussite.
Une seule réponse future déterministe est mesurée par état.

Pour les issues modifiées, le rapport compte les prévisions valides dont le
changement suit l'issue avec une amplitude supérieure à **0,01**, seuil fixé
ici. Ce compte est aussi calculé parmi les seules tâches valides avant/après.
Les 24 cas, les deux longueurs et toutes les conditions restent dans le rapport.
Il n'y a ni intervalle confirmatoire, ni gagnant sélectionné, ni critère global
de découverte. Le résultat peut simplement établir que cette perturbation
est trop destructrice ou que la prévision native ne la suit pas.

La relecture au même rythme et la restauration doivent produire exactement
les caches et les sorties natives du cas réel. Une erreur technique ou un
échec de ce contrôle arrête et conserve l'unique tentative. Aucun modèle,
graine, site, cas ou seuil n'est remplacé après cet arrêt. Les 17 tests logiciels
du lanceur passent sur PC en 1,616 seconde : permutation, restauration,
provenance, ordre prospectif, distinction erreur/format et continuité de cache.
Leurs sorties construites ne sont pas des performances de Qwen3-4B.

## Conservation et portée

Les sources, la révision Qwen, BF16/SDPA, les paramètres de génération et les
cas sont figés. Le lanceur exige la fin du contrôle numérique et un A100 libre.
Il conserve aussi une archive en cas d'échec. Aucun poids n'est entraîné.

Les caches réels complets des cas **0 et 8**, choisis avant collecte, sont
exportés en safetensors. L'[auditeur](../research/audit_prospective_state_discovery.py)
recalcule leurs permutations et leur restauration, les empreintes exactes
et les déplacements avec tolérance arithmétique de 10⁻⁸ absolue/relative pour
les normes CPU/GPU. Les autres caches ne sont pas exportés ; leurs contrôles
reposent sur les déclarations du collecteur et les sorties consignées.
Le journal, ses entrées, les prévisions et les métriques sont vérifiés localement.
Cette vérification n'est ni une réplication indépendante ni une attestation
cryptographique des poids distants.

Il manque encore une prévision entraînée, des perturbations plus sélectives,
des témoins de norme et donneurs appariés, un prédicteur textuel entraîné,
des données réservées et un essai d'utilité des décisions. Même un succès de
ce petit test ne les établit pas. Il ne démontre ni expérience subjective,
ni conscience de sa propre existence, ni nouveauté de l'approche.

```sh
python -m research.prospective_state_discovery_gpu --journal NOUVEAU_JOURNAL.jsonl
python -m research.audit_prospective_state_discovery JOURNAL_RECU.jsonl --output AUDIT.json
```
