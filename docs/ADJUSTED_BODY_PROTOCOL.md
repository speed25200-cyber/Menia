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
