# Vérification bibliographique — un besoin qui compte dans un modèle de langage

Faite le 27 septembre 2026, avant le protocole `docs/LLM_NEED_PROTOCOL.md`,
par recherche web (résumés et pages des articles ; les textes complets
n'ont pas tous été lus). Elle peut avoir manqué un travail ; « nous
n'avons rien trouvé » ne veut pas dire « cela n'existe pas ».

## Ce qui existe

- **Injection de concepts et introspection.** Lindsey (Anthropic, 2025),
  [*Emergent Introspective Awareness in Large Language Models*](https://transformer-circuits.pub/2025/introspection/index.html) :
  un vecteur de concept injecté dans le flux résiduel est parfois détecté
  et nommé (environ 20 % des essais au mieux). Suites :
  [*Mechanisms of Introspective Awareness*](https://arxiv.org/abs/2603.21396)
  (le mécanisme naît de l'entraînement par préférences) ;
  [*Steering Awareness*](https://arxiv.org/abs/2511.21399) (Rivera et
  Africa : un modèle ajusté détecte 95,5 % des injections, concepts
  tenus à l'écart compris) ;
  [*Introspection Fine-Tuning*](https://arxiv.org/abs/2607.14111) (petits
  modèles entraînés à localiser une perturbation, transfert à une tâche
  voisine) ; critiques :
  [*Can LLMs Introspect? A Reality Check*](https://arxiv.org/abs/2605.26242),
  [*Detecting the Disturbance*](https://arxiv.org/abs/2512.12411) (la
  détection serait souvent aveugle au contenu). **Aucun de ces travaux
  n'étudie un agent qui agit, ni un besoin** : les concepts injectés sont
  arbitraires, et le rapport est la seule sortie.
- **Apprentissage homéostatique.** Keramati et Gutkin (2014) ; agents
  simulés et robots dont la récompense est la réduction d'une pulsion
  interne ([Yoshida et al., 2024](https://www.sciencedirect.com/science/article/pii/S0893608024003034) ;
  [Horibe et al.](https://arxiv.org/abs/2411.12304)). **Pas de modèle de
  langage**, pas de rapport verbal.
- **Propositions.** [*Life-inspired interoceptive AI*](https://arxiv.org/abs/2309.05999)
  (Lee, Friston, Woo et al. ; perspective) et
  [*Embodiment in multimodal large language models*](https://arxiv.org/abs/2510.13845)
  (*Neuron*, 2026) : proposent de doter les modèles de variables
  internes (énergie, température) et de les tester dans des environnements
  où elles fluctuent. **Ce sont des propositions, sans expérience.**
- **Pilotage d'agents par injection.** [*Entropic Activation Steering*](https://arxiv.org/abs/2406.00244) :
  on change le comportement d'exploration d'un agent de langage par
  injection ; pas de besoin, pas de rapport.

## Ce que nous n'avons pas trouvé

Une expérience où, dans un modèle de langage : (1) un état de besoin est
**calculé par le modèle** (il ne le voit pas écrit) ; (2) il a été appris
**par la seule satisfaction du besoin**, sans professeur ; (3) **la même
direction interne**, injectée, change à la fois **l'action** et **le
rapport** sur ce besoin, de façon spécifique ; (4) sa **lésion** fait
mourir l'agent. C'est ce que teste `docs/LLM_NEED_PROTOCOL.md`. Un résultat positif serait, à notre
connaissance, nouveau ; il resterait à le faire relire.
