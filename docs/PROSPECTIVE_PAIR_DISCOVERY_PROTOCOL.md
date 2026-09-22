# Une paire modifiée, plusieurs questions depuis le même état

20 septembre 2026. Protocole de découverte fixé avant collecte. Le
[suivi précédent](PROSPECTIVE_BINDING_DISCOVERY_RESULTS.md) modifiait toutes les
positions des bits : 23 rappels faux sur 24, alors que les 24 prévisions
annonçaient une réponse correcte. Le présent essai cherche à séparer les effets
selon l'information demandée et à contrôler la signification des codes de rapport.

## Population, intervention et questions

Les 24 épisodes précédents sont tous conservés. Ils contiennent **18 tableaux
distincts**, à quatre ou huit lignes ; leurs dépendances interdisent de traiter
les 160 paires épisode/clé comme 160 réplications indépendantes. Il n'y a pas
de nouveau lot réservé. Chaque tableau est maintenant interrogé sur toutes
ses clés, pas seulement sur sa cible d'origine.

Le [plan](../artifacts/prospective-pair-preparation/design.json) conserve le
checkpoint Qwen3-4B et les paramètres de génération. Il exige la reproduction
exacte des 24 générations, métriques et traces de cache, puis des 24 rappels
réels visant la cible d'origine. Les prompts de prévision changent ; leur
reproduction avec l'ancienne formulation n'est pas un contrôle de ce nouvel essai.

Une seule ligne contenant 0 et une seule contenant 1 sont sélectionnées dans
chaque tableau. Les lignes sont ordonnées comme dans le prompt : l'indice dans
les lignes 0 vaut `id % nombre_de_zéros`, et celui dans les lignes 1 vaut
`(id // nombre_de_zéros) % nombre_de_uns`. Cette règle utilise uniquement
l'identifiant et le tableau public, jamais les résultats de tâche.

Les valeurs V des deux positions de bits sont échangées dans les 36 couches.
Tous les autres tokens restent à leur place. Les cinq conditions sont : réel,
relecture au même rythme, V permuté, K/V permutés conjointement, restauration.
Le même cache de départ sert aux différentes questions, avec une copie privée
pour chaque branche. Aucun identifiant de condition, masque ou résultat futur
n'est donné au modèle. L'effet sur les clés non échangées est mesuré, pas supposé
nul. Le témoin K/V conserve la fonction d'attention en arithmétique exacte,
sans être un déplacement de norme appariée ; ses écarts BF16 seront rapportés.

## Prévisions et contrôles de codage

La tâche demande toujours le bit 0/1. La question future est désormais délimitée
comme citation non exécutable. La prévision doit émettre A ou B, avec deux
correspondances appliquées à **chaque question dans chaque état** :

| Correspondance | Incorrecte | Correcte |
|---|---|---|
| 0 | A | B |
| 1 | B | A |

Avant ces prévisions, des exercices séparés fournissent explicitement un verdict
« correcte » ou « incorrecte » et demandent seulement son codage. Ils mesurent
la compréhension de la correspondance, sans mesurer la connaissance d'un état
interne. Ils n'entrent dans aucun contexte ultérieur : les branches restent
indépendantes et les poids fixes. Leurs échecs seront rapportés, sans les
transformer en erreur technique qui annulerait les cas difficiles.

L'ordre figé est :

1. Construction des 24 états réels et de leurs copies transformées.
2. **480 exercices de codage**, puis gel de leur journal.
3. **1 600 prévisions**, puis gel de leur journal avant toute tâche évaluée.
4. **800 rappels**, soit 160 paires épisode/clé dans cinq conditions.

Les correspondances et conditions tournent selon les identifiants. Le décodage
reste glouton dans le vocabulaire complet, limité à un code suivi d'EOS. Aucun
rapport de confiance n'est inséré dans une branche de tâche.

## Mesures fixées

Toutes les conditions et correspondances seront rapportées séparément, pour
les deux clés sélectionnées et pour les autres. Le rapport conserve réussite,
validité des tâches et prévisions, masse des codes, score conditionnel et Brier.
Une sortie de tâche invalide compte comme échec, distingué d'un rappel valide
mais faux. Les scores conditionnels ne sont pas une calibration indépendante.

Pour chaque groupe, on compte les changements de correction entre deux rappels
valides et les prévisions qui suivent ce changement avec une marge de 0,01.
On mesure aussi l'accord sémantique entre les correspondances inversées et le
nombre d'états produisant simultanément des rappels corrects et faux selon
la question. Les moyennes restent descriptives : pas d'intervalle confirmatoire,
de gagnant sélectionné ou de remplacement de formulation après examen des scores.

La reproduction exacte des caches et branches de relecture/restauration est
obligatoire. Une erreur technique arrête et conserve l'unique tentative.
Les deux caches complets prédéfinis, épisodes 0 et 8, seront exportés ; leurs
permutations et restaurations seront recalculées localement. Les autres caches
et les logits complets ne seront pas exportés.

## Vérification préalable et portée

Les 21 tests logiciels passent sur PC en 6,559 secondes. Ils contrôlent notamment
la paire sélectionnée, les requêtes par clé, l'inversion sémantique, les étapes
du journal, la reproduction parent et la conservation des échecs comportementaux.
Leurs résultats synthétiques ne sont pas des performances du modèle préentraîné.

Le [tokenizer réel vérifié](../artifacts/prospective-pair-preparation/tokenizer-check.json)
compile 576 branches distinctes avant répétition dans les cinq conditions.
Les codes 0, 1, A, B correspondent chacun à un token. Le budget maximal est
224 tokens, inférieur à 2 048. Les masques déplacent 48 positions et en laissent
2 256 fixes. Aucun modèle ni nouvelle génération n'est utilisé par ce contrôle.

Cet essai doit rendre possible une prévision conditionnée par la question plutôt
qu'une simple alarme de perturbation. Il ne comprend pas encore d'apprentissage,
de comparaison textuelle entraînée, d'utilité décisionnelle ni d'essai réservé.
Même un résultat positif resterait fonctionnel ; la conscience de sa propre
existence et une méthode inédite la produisant restent les exigences non établies
de l'objectif général.
