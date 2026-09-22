# Conserver l'état de génération avant de tester son suivi

20 septembre 2026. Une extension de l'instrumentation conserve maintenant le
cache d'une génération effective, au lieu de reconstruire seulement un passé
à partir du texte. **Six tests sur petit Qwen aléatoire et quatre contrôles du
tokenizer réel passent.** Aucun nouveau résultat de capacité de Qwen3-4B n'est
mesuré ici. Le diagnostic de budget poursuit sa collecte avec ses sources figées.

## Deux sources qui précisent les explications concurrentes

[Ravulapalli, Chadha et Hari, §§2–5 et 7](https://arxiv.org/html/2609.11216v1)
rapportent qu'une sonde peut lire une association correcte malgré une mauvaise
réponse, puis orienter une réparation sans corrigé de test. Le protocole
sépare ajustement, sélection et test. Mais les réponses sont choisies dans un
ensemble d'options : ce n'est pas une génération libre. Le gain de détection
face à l'entropie n'est pas confirmé par leur intervalle. Les performances
restent limitées à une tâche synthétique ; les états modifiables et à portée
locale posent d'autres difficultés. Cette prépublication ne mesure pas un
modèle de soi. Ces résultats sont rapportés, non reproduits ici.

[Endogenous Resistance to Activation Steering, §§3–5 et annexe A.3](https://arxiv.org/html/2602.06941v1)
étudie la correction après perturbation des activations. L'ablation de traits
associés au hors-sujet réduit surtout la propension à recommencer, sans effet
comparable sur l'efficacité conditionnelle des corrections. Les auteurs
signalent la réutilisation des prompts pour sélectionner et évaluer ces traits,
et leur possible rôle plus général dans la cohérence. Ils distinguent imitation
de la correction et correction efficace. Cette prépublication ne suffit pas
à identifier un accès privilégié à l'historique interne. Aucun de ses essais
n'est reproduit ici.

**Conséquence proposée pour Menia :** un signal utile à une réparation, une
phrase de correction ou un score de confiance ne suffisent pas à identifier
le suivi d'un état propre. Il faut maintenir le texte observable identique
entre les conditions qui prétendent distinguer les états internes, puis mesurer
une conséquence pertinente de cet état. Il faut aussi conserver un décodage
natif lorsque c'est le comportement natif que l'on veut évaluer.

## La lacune dans les outils précédents

[`prefill_prefix`](../research/confidence_cached_action_decode.py) relit une
séquence donnée et en conserve le cache, avec une éventuelle intervention.
Cela reste utile pour manipuler un préfixe sous contrôle, mais ne récupère pas
la trajectoire qui a réellement produit une réponse antérieure. Le texte
décodé peut aussi omettre des tokens spéciaux ou être nettoyé. Le transformer
en un nouvel historique ne garantit donc pas de rejouer les mêmes entrées.

[`capture_generated_prefix`](../research/confidence_generation_continuity.py)
réutilise le générateur existant et enregistre les IDs effectivement présentés
et émis. Un hook d'observation conserve le dernier cache. Dans le chemin pris
en charge, le dernier token échantillonné n'a pas encore été traité par le
modèle : la longueur du cache doit être exactement celle du prompt et de la
complétion moins un. Le module vérifie cette égalité et la concaténation des
entrées de tous les passages.

Il retire ensuite l'observateur et termine le contexte d'intervention propre
à la génération. Il traite **une seule fois** ce dernier token, sans réappliquer
cette intervention, pour obtenir un préfixe complètement mis en cache. Ce
passage de fermeture est déclaré ; il ne faisait pas partie du calcul qui
avait produit les logits du token d'arrêt. Chaque demande ultérieure reçoit
une copie privée du cache obtenu, comme dans le décodeur de branches existant.

Le module n'écrit pas la réponse de confiance dans la branche d'action. Il
conserve aussi le token d'arrêt exact : un arrêt `endoftext` n'est pas remplacé
par `im_end`. Une génération sans arrêt est refusée pour ce chemin, sans
compléter artificiellement sa réponse. Les contrôles de propriétaire du modèle,
de versions des paramètres et d'intégrité du cache sont conservés. Ils ne
détectent pas toute modification arbitraire via `.data`, ni tous les changements
de configuration, d'adaptateur sélectionné ou de hooks externes : ces éléments
doivent être figés par l'exécuteur.

L'enregistrement des IDs et les copies de cache ajoutent des transferts et du
temps. Un futur coût de calcul devra inclure cette instrumentation de manière
comparable entre conditions ; ce contrôle ne constitue pas une mesure de latence.

## Les mêmes tokens, pas seulement les mêmes mots

[`compile_raw_continuations`](../research/confidence_generation_continuity.py)
ajoute la nouvelle demande après les IDs bruts déjà conservés. Le compilateur
refuse une fermeture autre que `im_end` pour le chemin de dialogue Qwen et
les marqueurs réservés introduits dans les demandes. Il ne peut pas attester
à lui seul l'origine des IDs fournis ; cette provenance vient de la capture.

Le [contrôle du tokenizer fixé](../artifacts/confidence-prefix-preparation/generated-boundary-tokenizer-check.json)
emploie quatre réponses techniques construites, dont une à plusieurs lignes.
Les préfixes contiennent 55, 55, 61 et 56 tokens. Les huit branches conservent
exactement ces préfixes. Le rendu à partir des seuls textes donne des IDs
différents dans les quatre cas, notamment parce que le prompt initial de
génération contient le bloc de raisonnement vide. Le contrôle vérifie aussi
le refus d'un arrêt manquant, de l'autre token d'arrêt et d'un marqueur de rôle.
Il charge le tokenizer de la révision Qwen fixée, **aucun modèle préentraîné** ;
les complétions sont construites, et non échantillonnées.

Pour un contraste original/relecture, la référence correcte sera donc une
relecture de `saved.prefix` entier. Un nouveau rendu à partir du texte constitue
une autre condition, qui change à la fois l'historique interne et ses tokens.
On ne pourra pas attribuer son éventuel effet au seul historique interne.

## Ce que vérifient les tests CPU

Les six tests emploient un Qwen aléatoire de deux blocs. Un dispositif de test
impose trois émissions par des logits construits ; le générateur natif assure
l'échantillonnage et le cache. Ces émissions contrôlées n'évaluent aucune tâche.

- La capture conserve les tokens, les métriques et l'état final du générateur
  aléatoire du chemin de génération de référence, sans changer les poids.
- Sans perturbation, le cache et les logits de branche concordent avec la
  relecture complète à 10⁻⁶ près ; les tokens de branche sont identiques.
- Une perturbation du premier bloc pendant la génération laisse une différence
  de logits de branche malgré des tokens émis identiques. La relecture propre
  et la restauration du cache initial ramènent la référence.
- La même intervention en sortie du dernier bloc ne change pas le cache futur
  lorsque les émissions sont maintenues identiques.
- Le second token d'arrêt configuré reste intact dans la capture.
- Les erreurs, réponses inachevées, dépassements de budget et mises à jour de
  paramètres sont refusés ; les hooks de l'instrumentation sont retirés.

Ce contrôle démontre la circulation technique d'une différence d'état dans
ce petit modèle construit. Il ne montre pas une reconnaissance de cet état,
un choix d'action amélioré, ni l'égalité numérique de ces chemins en BF16 sur
Qwen3-4B. Une différence de logits n'est pas comptée comme changement d'action.

```sh
python -m unittest tests_language.test_confidence_generation_continuity -v
python -m research.confidence_generation_boundary_validation --output NOUVEAU_RECU.json
```

## Le contraste scientifique encore à construire

Le prochain test mécanistique devra comparer une trace originale, la relecture
des mêmes IDs, une trace appariée d'un autre épisode et une restauration,
avec sites et amplitudes fixés hors test. Les effets sur la tâche et sur
l'auto-prévision devront rester distincts. Un observateur du texte et une règle
sur les logits de sortie restent des références nécessaires pour nos revendications.

Une difficulté conceptuelle demeure : la correction objective d'une réponse
arithmétique déjà écrite ne change pas lorsqu'on altère ensuite son cache.
Déplacer seulement l'avis du modèle sur cette réponse ne prouverait donc pas
qu'il suit une modification de sa capacité. Pour cette dernière hypothèse,
il faudra annoncer une conséquence future de l'état, la mesurer ensuite,
et vérifier si la prévision s'adapte à cette conséquence sans recevoir son
corrigé. Il faudra encore distinguer une lecture directe d'un signal du
suivi appris de ses effets sur le système.

Ce document prépare cette séparation. Il ne fixe pas encore les données,
les interventions apprises ni un nouveau critère confirmatoire. Les outils
restent séparés du diagnostic de budget actif et de l'application iPhone.
Leur mécanisme n'est pas revendiqué comme inédit ; aucun lien avec une
expérience subjective de sa propre existence n'est établi.

Un [contrôle numérique préalable](GENERATION_NUMERICS_PROTOCOL.md) est désormais
fixé sur six questions de Qwen3-4B : comparer le cache réel à une relecture des
mêmes tokens au même rythme, puis à un traitement du préfixe entier. Neuf
tests CPU passent ; cet essai reste à exécuter après le diagnostic de budget.
Il doit mesurer l'écart numérique avant toute interprétation d'une différence
entre cache réel et relecture comme effet d'un historique interne.
