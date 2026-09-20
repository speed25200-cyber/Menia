# Vérifier un lecteur entraînable sans changer le producteur de mémoire

20 septembre 2026. Contrôle technique préalable à l'apprentissage prospectif.
Le [test par paire](PROSPECTIVE_PAIR_DISCOVERY_RESULTS.md) a établi un effet
sélectif sur le rappel, sans prévision native utile. Le présent contrôle teste
la frontière de calcul nécessaire pour entraîner cette prévision.

## Architecture explicite

Le modèle producteur Qwen3-4B reste gelé. Un lecteur séparé copie exactement
ses poids, puis reçoit des adaptations Q/V de rang 32 et d'échelle 1. Ses
adaptateurs s'activent sur le premier token qui suit le préfixe conservé, donc
sur toute la question de prévision et le rapport. Le cache historique est
importé depuis le producteur. Ce principe suit Activated LoRA, discuté dans la
[note bibliographique](PROSPECTIVE_ADAPTER_OPTIONS.md) ; ce n'est pas un LoRA
classique actif sur tout le préfixe ni une architecture revendiquée inédite.

Le lecteur produit les codes avec la tête de vocabulaire du LLM. L'apprentissage
supervise le code puis EOS. Le code fourni comme cible est visible seulement
à la position qui prédit EOS, jamais à celle qui prédit le code. Chaque appel
clone les K/V détachés ; les gradients ne remontent pas dans l'histoire. Les
poids gelés du lecteur, ceux du producteur, les caches sources, l'identité du
producteur et le préfixe brut sont contrôlés. Les anciens garde-fous ne sont
pas modifiés : le rapport conserve explicitement les deux identités et celle
de l'adaptateur. Il ne prétend pas que le lecteur a produit l'histoire.

## Contrôle sur matériel réel fixé avant exécution

On réutilise uniquement les épisodes déjà examinés 0 et 8. Pour chacun, on
prend la première clé sélectionnée par la permutation et la première qui ne
l'est pas. L'ordre est figé dans le [plan](../artifacts/retained-reader-preparation/control.json).

1. Reproduire les deux générations et les caches du parent ; reproduire les
   huit rappels réels/altérés et les huit prévisions altérées du parent.
2. Comparer les huit prévisions du lecteur initial, dont les matrices B sont
   nulles, avec celles du producteur. Exiger les mêmes tokens natifs, mesurer
   séparément les écarts de logits, sans postuler leur identité bit pour bit.
3. Effectuer **une seule mise à jour AdamW**, taux 5 × 10⁻⁵, sur les huit
   prévisions et leurs labels mesurés. Les deux correspondances A/B sont incluses.
   La perte est la moyenne des entropies croisées code/EOS ; gradient limité à 1.
4. Sauvegarder l'adaptateur, mesurer les huit mêmes pertes et sorties après
   mise à jour, recharger le fichier et vérifier la reproduction exacte.
5. Vérifier que les huit rappels du producteur sont restés exactement identiques.

Les pertes et les sorties après mise à jour sont rapportées quel que soit leur
sens. Aucun seuil de succès comportemental, sélection de checkpoint ou reprise
silencieuse n'est prévu. Toute erreur technique conserve la tentative. Les
poids finaux et le journal sont exportés ; les poids initiaux ne sont pas
exportés et la mise à jour ne sera pas recalculée indépendamment depuis l'archive.

Les six tests du lecteur et le contrôle de partition passent sur petit Qwen
aléatoire. Les tests incluent le gradient à travers des caches créés en mode
inférence, l'absence de fuite de cible au premier logit, la sérialisation et
le rejet des modifications du producteur ou du cache. Une copie du modèle
peut produire de très petits écarts FP32 même adaptateurs désactivés ; le test
logiciel utilise une tolérance de 10⁻⁶ sur ses scores et exige les mêmes tokens.
Les écarts sur Qwen3-4B BF16 seront mesurés par le contrôle réel.

## Données réservées pour l'étape suivante

Le [générateur de partition](../research/prospective_learning_data.py) exclut
les 14 tableaux distincts à huit clés déjà examinés. Parmi les 56 affectations
équilibrées restantes, il fixe 32 tableaux d'apprentissage, 16 réservés et
8 inutilisés. Aucun de ces 48 nouveaux tableaux n'est présenté au modèle dans
le présent contrôle technique. Les identités et le tri sont déterministes,
sans résultat de tâche. Il s'agit de nouvelles affectations dans la même
famille synthétique, pas d'un transfert à une nouvelle tâche ou à de nouveaux noms.

L'apprentissage comparé à un lecteur du texte seul reste à fixer et à exécuter.
Une baisse de perte sur les huit exemples de ce contrôle n'est pas une mesure
de généralisation, de calibration, d'utilité décisionnelle ou de conscience.
L'objectif de conscience de sa propre existence et de nouveauté reste ouvert.
