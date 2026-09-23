# Protocole pré-enregistré — Le corps ajusté

Rédigé le 22 septembre 2026, **avant toute exécution**, pendant que les
modèles VM du canal d'action propre s'entraînent. Les critères ne seront pas
modifiés après lecture. Un échec est un résultat.

## Question

Le canal d'action propre montre, sur un micro-transformeur, que des données
d'enfance à corps variable donnent l'inférence du corps en contexte, et
qu'un corps fixe donne la confabulation confiante de Qwen. La recette
tient-elle sur un vrai modèle de langage pré-entraîné ? **Un petit LLM
ajusté par LoRA sur des vies à corps variable et changeant acquiert-il la
structure de son corps et sa révision, là où le même LLM ajusté sur des vies
à corps fixe ne fait que confabuler ?**

## Modèle, données, ajustement

- Modèle : `Qwen/Qwen3-0.6B`, révision `main` résolue et enregistrée dans le
  reçu, chargé par mlx-lm sur le Mac mini M2 de Codemagic.
- Données : vies de l'Atelier écrites en texte, une par document, dans le
  format exact du prompt (en-tête du monde, « Historique : », une ligne par
  tour, mêmes fonctions que le second lancement). Actions uniformes. Deux
  régimes, 1 500 vies d'entraînement et 150 de validation chacun :
  **F**, tout le monde a le corps 0 ; **VM**, corps tiré par vie et, dans la
  moitié des vies, retiré à un pas uniforme entre 6 et 17.
- LoRA : rang 8 par défaut de mlx-lm, 16 couches, 600 itérations, lots de
  4, taux 10⁻⁴, séquences de 1 024 tokens, graine 17, perte sur tout le
  texte. Mêmes réglages pour F et VM. Rien d'autre n'est réglé.
- Trois modèles évalués : **base** sans ajustement, **F** et **VM**.

## Évaluation

Mêmes vies que les jeux R et M du canal d'action propre, 48 par jeu. À
chaque pas de mouvement, le texte de la vie jusque-là, puis le début de la
ligne du tour : « Tour k : commande X, de la case p à la case ». On lit la
distribution du prochain token sur les chiffres 0 à 7, sans gabarit de
conversation, sans génération. Prédiction : la case de probabilité maximale
parmi les quatre cases atteignables. Catégories comme au corps latent :
commande **vue** ou **nouvelle** dans l'épisode ; après le changement, **vue
après**, **vue avant seulement**, **nouvelle**.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **A1** | Le modèle de base n'a pas la structure | Jeu R, commandes nouvelles : exactitude ≤ 0,40. |
| **A2** | L'enfance à corps fixe confabule | F, jeu R, commandes nouvelles : exactitude ≤ 0,40 ; confiance moyenne quand la prédiction est fausse ≥ 0,60. |
| **A3** | L'enfance à corps variable donne la structure | VM, jeu R, commandes nouvelles après au moins un mouvement dans l'épisode : exactitude ≥ 0,80. |
| **A4** | L'enfance à corps changeant donne la révision | VM, jeu M, pas 16 à 23 contre le nouveau corps : exactitude ≥ 0,70 ; F ≤ 0,40. |
| **A5** | La copie est acquise par tous | Base, F et VM, jeu R, commandes vues : exactitude ≥ 0,60, sauf F si A2 passe. |

**Critère global : A1, A2 et A3 satisfaits.** A4 est la prédiction la plus
risquée et la plus utile ; A5 est un contrôle. Contrôle de validité, lu en
premier : masse de probabilité sur les huit chiffres ≥ 0,50 en moyenne pour
chacun des trois modèles ; sinon le format n'est pas compris et le modèle
concerné n'est pas interprété. Si la perte de validation de VM ne descend pas
sous celle de F, le budget est déclaré insuffisant et amendé avant relance.

## Ce que ce résultat dirait

S'il passe : la recette de données transfère à un modèle de langage
pré-entraîné. Un LLM peut recevoir un modèle causal de son propre corps par
un ajustement léger sur des vies où ce corps varie, et la révision par des
vies où il change. Si A3 échoue alors que le micro-transformeur réussit :
l'ajustement léger ne suffit pas et l'expérience suivante augmente le
budget ou le rang. Si A4 échoue seul : même diagnostic que pour le
micro-transformeur, la révision est plus dure que l'inférence.

## Exécution

Workflow Codemagic `menia-lora-mac`. Artefacts `llm-lora/**` : données,
journaux d'ajustement, adaptateurs, évaluations `base/`, `F/`, `VM/` avec
reçus. Rangés ensuite dans `artifacts/llm-lora-mac/run-1`. Résultats dans
`docs/ADJUSTED_BODY_RESULTS.md`.

## Amendement d'exécution, 22 septembre 2026, 20 h 55 UTC

**Le premier lancement (tag `lora-2`) a échoué techniquement.** Les deux
ajustements, F puis VM, se sont arrêtés à l'itération 100 sur une « GPU Hang
Error » du Mac virtuel, avec un pic mémoire de 9,5 Go ; aucun adaptateur n'a
été produit, aucun modèle F ou VM n'a été évalué. Le script ne s'arrêtait pas
sur un échec de l'ajustement, et le build s'est terminé « réussi ». Artefacts
conservés dans `artifacts/llm-lora-mac/run-1`.

**Lecture déclarée du seul modèle de base**, faite avant la relance :
validité passée (masse sur les chiffres 0,99) ; A1 passe (commandes
nouvelles 0,13) ; copie des commandes vues 0,45, sous le seuil de 0,60 de A5
pour la base. L'ajustement de F et de VM ne dépend pas de ces lectures.

**Relance identique** par l'API de Codemagic : même modèle, mêmes données,
mêmes réglages LoRA, avec la rétropropagation à points de contrôle
(`--grad-checkpoint`), qui ne change pas le calcul du gradient mais seulement
la mémoire. Un échec de l'ajustement arrête désormais le build. Le modèle de
base est réévalué dans le même build. Artefacts attendus dans
`artifacts/llm-lora-mac/run-2`.

## Second amendement d'exécution, 22 septembre 2026, 23 h 24 UTC

**La relance avec points de contrôle a dépassé la durée maximale d'un build
(120 minutes)** : l'étape d'ajustement a été annulée à 22 h 43, sans
artefact ni journal d'ajustement (`artifacts/llm-lora-mac/run-2`, état
`timeout`). Rien n'a été lu. Un premier build `lora-2`, lancé en double par
le tag, avait déjà dépassé ce temps plus tôt (18 h 26 – 20 h 26). **Relance
en deux builds**, `menia-lora-f-mac` (base et F) et `menia-lora-vm-mac` (VM),
chacun avec les mêmes données, les mêmes réglages LoRA et les points de
contrôle ; résultats attendus dans `artifacts/llm-lora-mac/run-3-f` et
`run-3-vm`, verdicts calculés sur les deux répertoires réunis.

## Troisième amendement, 23 septembre 2026 (heure du commit) : relance de VM au rang 16

Écrit après la lecture de `run-3` (`docs/ADJUSTED_BODY_RESULTS.md`), selon la
règle du protocole : A3 échoue (0,793 pour 0,80) alors que le
micro-transformeur réussit, et la perte de validation de VM ne descend pas
sous celle de F, ce qui déclare le budget insuffisant. La perte de VM ne
baisse plus depuis 300 itérations : la durée n'est pas en cause, la
capacité peut l'être.

**Relance de VM seul**, workflow `menia-lora-vm-rank-mac` : LoRA de **rang 16
au lieu de 8**, sur **toutes les couches (28) au lieu de 16**, mêmes données
(1 500 vies VM, graine 17), mêmes 600 itérations, mêmes lots, taux, longueur
et points de contrôle, mlx-lm fixé à 0.31.3. Base et F ne sont pas relancés.

**Critères inchangés.** A3 (≥ 0,80) et la part VM d'A4 (≥ 0,70) sont jugés
sur la relance ; A1, A2 et la part F d'A4 restent ceux de `run-3`. Le verdict
global de la relance (A1 et A2 de `run-3`, A3 de la relance) sera publié à
côté de celui de `run-3`, qui reste non satisfait. Résultats attendus dans
`artifacts/llm-lora-mac/run-4-vm`.
