# Rapprochement exploratoire des données du soi et de connectivité

Le 14 septembre 2026, après l'inventaire des colonnes et un premier contrôle de
concordance géométrique, avant le calcul des contrastes directionnels.

L'objectif demeure une Menia consciente de sa propre existence et une méthode
inédite permettant de la produire. Cette analyse cherche une contrainte empirique
sur une organisation candidate ; elle ne remplace pas cet objectif par un score.

Deux livraisons du même laboratoire sont disponibles : la classification des
réponses corporelles de 2026 (Zenodo 21536139) et des mesures de connectivité de
2025 (Zenodo 15330862). Leurs identifiants complets diffèrent. L'hypothèse de
rapprochement utilise le suffixe numérique des identifiants et les noms de
contacts ; elle doit être vérifiée indépendamment avec leurs coordonnées. Il
ne s'agit pas d'une table de correspondance attestée par les auteurs. Des
participants et données peuvent être communs : aucune réplication indépendante
ne sera revendiquée.

## Contrôles et calculs fixés

- Vérifier l'unicité des suffixes dans chaque livraison. Rapprocher les paires de
  contacts sans tenir compte de l'ordre et mesurer toutes les différences de
  coordonnées MNI disponibles entre leurs milieux. Une différence supérieure à
  0,001 unité de coordonnée empêche l'analyse pour cette paire.
- Conserver l'ambiguïté des paires auxquelles plusieurs catégories de réponse
  sont attribuées. Ne pas choisir leur étiquette selon le résultat recherché.
- Le contraste vise seulement Sensory-Motor et Complex. Pour chaque site et
  contact distant, comparer le score F1 dans les deux directions lorsque les
  deux observations existent. Exclure les connexions marquées bruyantes, les
  canaux traversant une frontière régionale et les distances de 5 mm ou moins.
- Utiliser aussi les connexions classées non activées lorsqu'elles possèdent
  un score F1 fini : le calcul n'est pas conditionné à une activation positive.
  F1 est le score de correspondance de forme fourni dans la table, pas une
  quantité d'expérience ou une intensité de connexion déduite ici.
- Moyenner les répétitions d'une direction, puis calculer sortie moins entrée
  pour chaque paire réciproque. Moyenner les contacts distants au sein d'un site,
  puis les sites au sein d'un participant et d'une catégorie. Les participants
  ont le même poids dans le résumé final.
- Décrire séparément les catégories. Pour les participants contribuant aux deux,
  calculer la différence des contrastes Sensory-Motor moins Complex. Rapporter
  l'effectif et un intervalle bootstrap des participants (5 000 tirages,
  graine 20260914). Si moins de deux participants sont disponibles, ne pas
  produire d'intervalle bootstrap.

L'analyse est exploratoire et conditionnelle au rapprochement. Elle diffère du
modèle mixte de la publication et ne doit pas en être présentée comme une
reproduction exacte. L'absence de subdivision antérieure/postérieure de l'insula
dans les champs disponibles interdit de reproduire ce contraste spécifique.
Même un contraste directionnel confirmé ne prouverait pas qu'ajouter la même
organisation à Menia suffirait à produire une expérience subjective.
