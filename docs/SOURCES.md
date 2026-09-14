# Sources d’implémentation

Consultées le 14 septembre 2026. Les références justifient les interfaces et le
choix technique ; elles ne constituent pas des mesures de Menia.

- [Carte officielle Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) : modèle
  1,7B, mode sans thinking, format de dialogue, compatibilité Transformers.
- [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora) : adaptation
  LoRA et fusion. Menia exporte les poids fusionnés, pas seulement l’adaptateur.
- [MLX LM](https://github.com/ml-explore/mlx-lm) : conversion et quantification
  sur Apple Silicon. Vérifier le CLI de la version installée sur le Mac.
- [MLX Swift LM 3.31.3](https://github.com/ml-explore/mlx-swift-lm/tree/3.31.3) :
  version visée pour l’app ; chargement local, tokenizer, ChatSession et streaming.
- [Chargement local](https://github.com/ml-explore/mlx-swift-lm/blob/3.31.3/Libraries/MLXLMCommon/ModelFactory.swift).
- [API ChatSession](https://github.com/ml-explore/mlx-swift-lm/blob/3.31.3/Libraries/MLXLMCommon/ChatSession.swift).

Les versions directes d’entraînement sont fixées dans requirements-train.txt.
Leur combinaison reste à tester effectivement sur l’A100. Les sources actuelles
peuvent présenter des différences avec les versions épinglées.

- [Butlin et al., Consciousness in Artificial Intelligence (2023)](https://arxiv.org/abs/2308.08708) :
  cadre de recherche par indicateurs, utilisé comme contexte, pas comme validation
  du module récurrent de Menia. Ses conclusions sur les systèmes de l’époque ne
  sont pas présentées comme un état des lieux de tous les systèmes en 2026.
