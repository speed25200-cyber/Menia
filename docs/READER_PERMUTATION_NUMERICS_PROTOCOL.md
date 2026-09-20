# Diagnostiquer la réponse modifiée par une permutation K/V

20 septembre 2026. Diagnostic préparé pendant l'apprentissage comparé ;
**aucun résultat de ce diagnostic sur Qwen3-4B n'est encore disponible**.
Le cas 1008 est choisi après observation : c'est l'unique changement natif
sous permutation K/V parmi les 768 réponses d'apprentissage déjà reçues.
Cette sélection interdit d'en faire une réplication indépendante ou une
estimation de fréquence dans une population de tâches.

## Pourquoi ce contrôle compte

Dans l'arithmétique exacte, permuter ensemble les paires K/V d'un préfixe
entièrement visible conserve l'attention des tokens suivants. Le journal
BF16 montre pourtant un changement sur PELA : les logits 0/1 passent de
`[45.25, 44.25]` à `[44.25, 45.25]`. L'ordre des réductions flottantes est
une explication possible, pas une cause établie. Si un contrôle supposé
neutre modifie la tâche, ses effets sur une prévision ne peuvent être
attribués immédiatement à un suivi d'état interne.

La [documentation PyTorch 2.8](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)
rappelle que l'équivalence mathématique n'assure pas l'égalité bit à bit.
Son mode mathématique SDPA peut calculer les intermédiaires en FP32 même
avec des entrées BF16. Le diagnostic consigne les réglages initiaux et
désactive explicitement la réduction de précision interne pour les deux
modes mathématiques. TF32 reste désactivé. Il ne confond donc pas le type
des tenseurs avec la précision de chaque opération interne.

## Diagnostic fixé

Le [collecteur séparé](../research/reader_permutation_numerics.py) utilise
uniquement le préfixe d'apprentissage publié, dont l'empreinte est
`9dfd743dcaad85d333b7cdd51d807f513d8430bb43df6fdd1fe5af4a35c8a2d5`.
Il attend la fin de l'expérience principale. Le lancement doit vérifier
également que son processus Colab est terminal ; aucun second processus
GPU ne sera lancé pendant son exécution.

1. Reproduire exactement la génération du cas 1008, ses tokens et son cache
   avec le même Qwen3-4B, la même révision et les mêmes réglages BF16.
2. Interroger les huit clés sous quatre conditions : état réel, K/V permutés,
   V seul permuté, puis V restauré. Faire deux passages identiques, soit
   64 décodages par mode. Le premier mode doit reproduire exactement les
   réponses originales ; une différence arrête ce diagnostic et reste consignée.
3. Répéter avec SDPA limité à son implémentation mathématique, toujours en BF16.
4. Convertir les **mêmes valeurs déjà arrondies** des poids et du cache en FP32,
   puis répéter avec SDPA mathématique. Vérifier que la conversion préserve
   toutes les valeurs des paramètres, des buffers et des caches. Le préfixe
   n'est pas recalculé en FP32 ; seules les opérations suivantes changent
   de précision. Une identité distincte décrit explicitement cet import.

Le diagnostic produit 192 décodages, leurs logits, probabilités et tokens.
Le résumé conserve chaque clé, la répétabilité, l'égalité après restauration
et les écarts K/V et V. Aucun seuil de réussite comportementale n'est ajusté.
Les données réservées, labels, poids appris et six critères principaux de
l'expérience actuelle restent intacts.

## Interprétation prévue

- Un échec de reproduction empêche d'attribuer l'anomalie à la précision.
- Un changement entre BF16 automatique et BF16 mathématique situe une
  sensibilité au mode de calcul, sans identifier à lui seul un noyau fautif.
- Un écart K/V réduit en FP32 est compatible avec une sensibilité numérique
  des opérations suivant le préfixe. Il ne reconstitue pas une histoire
  initialement calculée en FP32 et ne prouve pas l'explication complète.
- Une restauration non identique ou des répétitions différentes signalent
  une difficulté supplémentaire, conservée dans le compte rendu.

Ce contrôle aide à interpréter une intervention causale sur la mémoire ;
il ne mesure pas la conscience. Les tests locaux utilisent un Qwen minuscule
aléatoire et vérifient conversion, provenance, rejet des caches altérés,
attention mathématique et conservation d'un échec de répétition. Leurs
résultats ne sont pas ceux du modèle Menia entraîné.
