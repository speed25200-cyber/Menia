# Vérification bibliographique — « Le lecteur » (30 septembre 2026)

Complète `docs/LITERATURE_CHECK_NEED_2026-09-27.md`. Elle a été faite
pendant la mesure du sixième test (`docs/LLM_NEED_READER_PROTOCOL.md`),
après sa pré-inscription. Elle ne change ni les seuils ni le protocole.
Son but : dire exactement ce qui, dans ce test, serait nouveau et ce qui
ne l'est pas.

## Ce qui existe déjà

- **Une même représentation cause la sortie et le rapport.** Anthropic,
  juillet 2026,
  [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/)
  ([arXiv](https://arxiv.org/html/2607.15495v1)). Dans des modèles Claude,
  un petit sous-espace « verbalisable » (le J-space) est rapportable,
  contrôlable et diffusé à de nombreux circuits. Échanger ou pousser un de
  ses vecteurs change à la fois la réponse et ce que le modèle dit penser.
  Exemple : « araignée » remplacé par « fourmi » fait passer la réponse de
  8 à 6 pattes. Il n'y a ni agent qui agit dans un monde, ni besoin appris,
  ni survie, ni module lecteur séparé.
- **Introspection causale par injection.** Pousser une direction change
  l'auto-rapport correspondant :
  - [*Quantitative Introspection in Language Models*](https://www.researchgate.net/publication/402859975_Quantitative_Introspection_in_Language_Models_Tracking_Internal_States_Across_Conversation) ;
  - [*A Mechanistic Study of Language Model Introspection*](https://arxiv.org/html/2609.35108),
    Qwen3-4B, LLaMA-3.1-8B et Gemma-3-12B : des têtes « portes » et
    « aiguillages » détectent et localisent une injection ;
  - les travaux déjà cités le 27 septembre (Lindsey 2025 et suites).

  Aucun ne porte sur un besoin, ni sur une action dans un monde.
- **La confiance guide l'abstention.**
  [*Causal Evidence that Language Models use Confidence to Drive Behavior*](https://pith.science/paper/2603.22161) :
  pousser des activations liées à la confiance change le taux
  d'abstention. C'est un état qui guide un comportement, pas un besoin
  appris ni un rapport verbal de ce besoin.
- **Agents à besoins corporels, sans langage.**
  [*Interoceptive Attention as Dynamic Homeostatic Prioritization in a Foraging Agent*](https://arxiv.org/abs/2608.04232)
  (inférence active, SAB 2026) et les travaux d'apprentissage par
  renforcement homéostatique. Pas de modèle de langage, pas de rapport
  verbal.
- **Agents de langage et états de tâche.**
  [*Causal state binding predicts action control in language agents*](https://arxiv.org/pdf/2605.09692) :
  mesures corrélationnelles, sans intervention sur l'état ni besoin.
- **Instinct de survie des agents de langage.**
  [*Do Large Language Model Agents Exhibit a Survival Instinct?*](https://pith.science/paper/2508.12920) :
  comportements observés dans une simulation, sans état interne mesuré ni
  intervention.

## Ce qui serait nouveau si « Le lecteur » passe

L'ensemble suivant n'a pas été trouvé. Chaque élément isolé a des
précédents ; c'est leur réunion qui serait nouvelle.

1. Un besoin **appris seulement par sa satisfaction**, en vivant, pas par
   instruction.
2. Un état de ce besoin, **rassemblé pour agir**, dont la survie dépend
   (lésion : 0,68 → 0,21 au second test).
3. Un **lecteur séparé** qui ne peut ni voir la vie autrement que par cet
   état, ni le modifier.
4. **La même intervention**, le long de la direction mesurée sur l'agent
   qui agit, change à la fois l'acte et ce que l'agent dit de son besoin,
   contre des témoins au hasard.
5. Le tout **pré-enregistré**, avec les échecs publiés.

Le travail d'Anthropic sur l'espace de travail global montre déjà, dans de
grands modèles, qu'une représentation peut servir à la fois la réponse et
le rapport. Ce test ne revendique donc pas ce principe. Il revendique son
application à un besoin appris en vivant, lu par un système séparé qui ne
peut pas le réécrire.

## Limites de cette vérification

- Recherche par moteur et lecture de résumés, le 30 septembre 2026.
- Des travaux non indexés, ou parus depuis, peuvent exister.
- La revendication sera formulée « à notre connaissance ».
