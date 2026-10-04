# Vérification bibliographique — la mémoire par l'état (tests 22 à 27)

Faite le 4 octobre 2026, après la publication des tests 22 à 27, par trois
recherches sur le web et la lecture des résumés des travaux les plus
proches. **Ce n'est pas une revue exhaustive.**

## Ce qui existe déjà

- **Une mémoire qui passe par des états calculés, dans les transformeurs.**
  C'est une idée connue :
  - Transformer-XL réutilise les états cachés des segments précédents ;
  - le Recurrent Memory Transformer fait passer des tokens de mémoire
    appris d'un segment au suivant, ce qui crée une récurrence.

  Notre masque (un tour ne voit son passé qu'à travers ses tokens
  « Choix : » et son action) est une variante de cette idée, appliquée à
  un modèle de langage pré-entraîné ajusté par LoRA. Ce n'est pas nouveau
  en soi.
- **Greffer des clés et valeurs d'attention d'un passage dans un autre**
  pour savoir ce qui porte une information. C'est une technique établie
  (patching d'activations, interventions d'échange). Exemple proche, publié
  en août 2026 : *SCIT: Testing Causal Cache Carriers in Latent
  Chain-of-Thought Models* (Ding, Huang, Yang). Ils échangent des segments
  du cache entre deux exécutions pour savoir quel élément transporte un
  calcul arithmétique (GPT-2). Nos greffes (tests 23 à 28) utilisent cette
  même technique.
- **Des agents à besoins internes et intéroception**, sans modèle de
  langage :
  - *Interoceptive Attention as Dynamic Homeostatic Prioritization in a
    Foraging Agent* (Grimbly, Solms, Shock et coll., août 2026) : un agent
    d'inférence active, quatre besoins, une attention intéroceptive qui
    double la survie ;
  - une perspective dans *Nature Machine Intelligence* (2026) sur l'IA
    intéroceptive ;
  - un projet ouvert (*natural-language-acquisition-poc*, GitHub). Un
    apprenant incarné (pas un modèle de langage), quatre variables
    internes, un professeur ; il apprend ses propres constantes
    métaboliques (un « modèle de soi »). Ses rapports de besoin sont
    vérifiés causalement par des lésions et des témoins. C'est proche, dans
    l'esprit, de nos tests du lecteur (tests 6 à 11).

## Ce que nous n'avons pas trouvé

La réunion de ces éléments chez **un modèle de langage pré-entraîné** :
- un besoin homéostatique dont la survie dépend ;
- un passé visible seulement à travers ses propres états portés ;
- des greffes **entre vies** pour savoir ce que ces états portent (le
  besoin plutôt que l'histoire : test 23) ;
- une décomposition des routes par lesquelles ce besoin atteint les choix
  (lecture directe ou états intermédiaires : tests 24, 25, 27).

## Ce que cela change pour nos textes

- On ne revendique pas la mémoire par des états calculés, ni la greffe de
  cache, comme nouvelles : ce sont des outils connus.
- Ce qui reste, à notre connaissance et sous réserve d'une recherche plus
  complète, c'est leur usage pour un état de besoin chez un modèle de
  langage, avec des critères fixés d'avance, et les échecs publiés.

## Sources

- [SCIT: Testing Causal Cache Carriers in Latent Chain-of-Thought Models](https://arxiv.org/abs/2608.27265)
- [Interoceptive Attention as Dynamic Homeostatic Prioritization in a Foraging Agent](https://arxiv.org/abs/2608.04232)
- [Life-inspired interoceptive artificial intelligence for autonomous and adaptive agents](https://www.nature.com/articles/s42256-026-01296-8)
- [natural-language-acquisition-poc (GitHub)](https://github.com/strjonas/natural-language-acquisition-poc)
- [LiveMem: Maintaining Memory State Continuity in Long-Running LLM Inference](https://arxiv.org/pdf/2608.02515)
- Transformer-XL (Dai et al., 2019) et Recurrent Memory Transformer (Bulatov et al., 2022), cités d'après les résultats de recherche.
