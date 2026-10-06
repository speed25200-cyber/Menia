# Vérification bibliographique — la boucle qui porte ses besoins (6 octobre 2026)

Faite pendant le test 33, **avant la fin de ses apprentissages et sans
lecture de ses greffes**, pour savoir ce qui est déjà connu. Recherche web
(moteur de recherche, puis lecture des articles accessibles). Elle n'est
pas exhaustive.

## La question

Un petit réseau dont seul un état passe d'un tour au suivant, appris par la
seule survie, porte-t-il dans cet état des besoins **qu'il ne voit jamais**
(ils ne sont jamais donnés en entrée), et cet état, **greffé** d'une vie à
une autre, fait-il choisir selon les besoins de la vie d'origine ?

## Ce qui existe

- **Apprentissage par renforcement homéostatique avec agents
  récurrents.** Yoshida et al., « Emergence of integrated behaviors through
  direct optimization for homeostasis », *Neural Networks*, 2024
  (https://www.sciencedirect.com/science/article/pii/S0893608024003034 ;
  page non lisible ici, résumé d'après la recherche). L'agent a un état
  interne qu'il **observe** (intéroception) ; des comportements intégrés
  émergent, et l'analyse des agents montre des représentations internes.
  Voir aussi « Meta-Reinforcement Learning in Homeostatic Regulation »
  (Yoshida, CCN 2025), avec une couche récurrente (LSTM).
- **Agents récurrents et à impulsions, énergie interne, observabilité
  partielle.** Hayes, « Predictive Allostatic Organization in Recurrent and
  Spiking Agents Under Partial Observability », arXiv 2608.11506, 2026. Les
  agents reçoivent des observations partielles et bruitées, **dont l'état
  d'énergie** ; le niveau d'énergie bas reste fortement décodable même
  quand on retire les variables liées à l'énergie ; des perturbations au
  moment de l'évaluation changent le comportement.
- **État caché qui suit une réserve, et intervention causale.** Chaturvedi,
  El-Gazzar et van Gerven, « Emergence of Internal State-Modulated Swarming
  in Multi-Agent Patch Foraging System », arXiv 2510.18886, 2026. Réseau
  récurrent à temps continu, appris par stratégie évolutionnaire
  (CMA-ES) ; la réserve est **donnée en entrée** ; des unités cachées la
  suivent, et **les forcer** vers l'état « épuisé » avance le comportement
  d'approche.
- **Mémoire apprise par la récompense sous observabilité partielle** :
  agents récurrents classiques (par exemple DRQN, 2015) ; transformeurs à
  mémoire transmise (Feedback Transformer, 2020 ; Recurrent Memory
  Transformer, 2022) ; états à porte (GRU, LSTM).
- **Greffe d'activations** (« activation patching », intervention
  d'échange) : méthode courante d'interprétabilité, y compris sur des
  agents.

## Ce que cela veut dire pour le test 33

- Qu'un agent récurrent appris par renforcement suive dans ses états un
  niveau interne, et que forcer ces états change son comportement, **est
  déjà montré** (Chaturvedi et al. ; Hayes), mais avec le niveau **donné
  en entrée** (ou observé avec du bruit).
- Ce qui reste propre au test 33, si ses critères passent : les besoins ne
  sont **jamais observés** ; l'agent doit les reconstruire à partir des
  événements et de ses actions ; la greffe se fait **entre deux vies** qui
  ont pris la même action, contre un témoin sans mémoire ; elle sépare le
  besoin de l'histoire (LOOP4) ; le tout est pré-enregistré, sur dix
  graines. C'est une contribution **étroite**, de méthode et de contrôle,
  pas une découverte de principe.
- Aucun de ces travaux, ni le nôtre, ne dit rien d'un ressenti.

## Sources

- https://www.sciencedirect.com/science/article/pii/S0893608024003034
- https://2025.ccneuro.org/abstract_pdf/Yoshida_2025_Meta-Reinforcement_Learning_Homeostatic_Regulation.pdf
- https://arxiv.org/abs/2608.11506
- https://arxiv.org/abs/2510.18886
- https://arxiv.org/abs/2404.15255 (comment utiliser et interpréter la greffe d'activations)
