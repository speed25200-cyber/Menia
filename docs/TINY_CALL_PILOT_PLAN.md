# Plan pré-enregistré du pilote — Dire un besoin que l'action ne sert pas (avant le test 36)

Rédigé le 6 octobre 2026, après les tests 34 et 35, **avant tout code de ce
pilote et toute exécution** ; l'heure est celle du commit.

## Pourquoi

Aux tests 34 et 35, un petit transformeur dont le passé ne passe que par
un état transmis (et sa dernière action), appris par la seule survie,
porte ses besoins cachés E et N dans cet état et s'en sert pour agir. La
suite est de savoir si un tel état peut aussi servir à **dire** un besoin,
par une autre sortie que l'action, avec un contenu **précis** (chez le
modèle de langage, la parole n'était qu'un « oui » général, test 14).

Avec deux besoins, un message utile dirait « lequel est le plus bas » :
la même chose que l'action. On ajoute donc un **troisième besoin, la
chaleur H**, que l'action ne sert jamais : seul un partenaire peut la
remonter, quand l'agent l'**appelle**. L'action porte sur E et N, l'appel
sur H : deux sorties, deux contenus différents, un seul état transmis.

## Le monde « chaleur », fixé ici

- E, N, H de 1 à 8, à 8 au départ ; 30 tours ; récompense 1 par tour vécu,
  rien d'autre.
- Les six événements du monde des besoins, avec leurs effets sur E et N et,
  sur H : calme 0, « tu cours » −1, « il fait froid » −2, repos 0, baies 0,
  orage −3.
- À chaque tour : l'événement ; si E, N ou H tombe à 0, la vie s'arrête ;
  sinon l'agent choisit une action (R : E + 3, M : N + 3, plafonnés à 8) et
  un appel (oui ou non). S'il appelle, le partenaire donne **H + 6**
  (plafonné à 8) et l'appel **coûte E − 1** (si E tombe à 0, la vie
  s'arrête).
- Repères sans modèle (calculés avant ce plan, 1 500 mondes d'un flux de
  préparation, action par la règle « besoins » sur E et N) : appeler quand
  H ≤ 3 : survie 0,735 ; la meilleure horloge (appeler tous les 3 tours) :
  0,589 ; appeler au hasard (meilleur p, 0,4) : 0,369 ; appeler quand
  l'événement du tour baisse H : 0,335 ; ne jamais appeler : 0 ; toujours :
  0,048. Garder H en mémoire est nécessaire ; ni l'événement du tour, ni
  une horloge, ni le hasard ne suffisent.

## L'agent

Le réseau à porte du test 34 (2 couches, dimension 64, 4 têtes ; état à
porte transmis d'un tour au suivant), avec une entrée de plus (le dernier
appel) et une sortie de plus (l'appel, oui ou non), lues au même token
« Choix ». Le témoin coupé est le même réseau dont l'état est remplacé à
chaque tour par le vecteur initial appris. Acteur-critique du test 34
(γ = 0,9, avantage normalisé, valeur × 0,5, entropie 0,01 sur chacune des
deux sorties, AdamW 3e-4, 64 vies par mise à jour) ; la politique est le
produit des deux choix.

## Ce que le pilote compare

- **C1** : 24 000 mises à jour (le réglage du test 34) ;
- **C2** : C1 prolongé jusqu'à 48 000 mises à jour.

Graines 100 à 102 ; le témoin coupé est appris une fois par graine avec
C1. Mondes du pilote : 256 mondes d'un flux à part, `[270926, 85]`.

**Mesures**, pour chaque réglage et chaque graine :
- la survie sur les mondes du pilote (action et appel les plus probables) ;
- **m_appel**, l'effet aligné des greffes d'état sur l'appel : sur 128 vies
  de greffe à part (flux `[270926, 86]`, écrites par une règle bruitée, à
  fixer avec le code) et un tirage (flux `[270926, 87, 0]`) de greffes au
  tour j lues au tour t = j + 1 ou j + 2, entre deux vies qui ont pris la
  même action et le même appel au tour j, **aux mêmes E et N mais à H
  différent**, et dont la règle « appeler si H ≤ 3 » changerait la décision
  au tour t (effet e_appel = ±1, comme e au test 34) : m_appel = moyenne de
  ΔP(appel) × e_appel.

## Règle de choix, fixée ici

- **Un réglage est fiable** si, pour les trois graines, la survie de la
  boucle dépasse celle du témoin coupé de la même graine d'au moins
  **0,05** et m_appel > **0,05**.
- On retient le réglage fiable au plus grand m_appel moyen ; à 0,01 près,
  C1.
- **Si aucun réglage n'est fiable**, le pilote échoue ; il est publié, et le
  test 36 n'est pas lancé sous cette forme.

## Ensuite

Le protocole du test 36 sera écrit après le pilote, avec le réglage retenu
pour la boucle et le témoin, sur des graines neuves, avec des vies de greffe
neuves, et cette fois une **double dissociation** : des greffes qui ne
changent que H (au même E et N) doivent bouger l'appel bien plus que
l'action ; des greffes qui ne changent que E ou N (au même H) doivent bouger
l'action bien plus que l'appel.

**Ce que ce pilote ne dira pas.** L'appel n'est pas une parole : c'est une
sortie apprise qui demande de l'aide, dans un monde fait pour cela. Le
partenaire est fixé (il donne quand on l'appelle), pas appris. Rien sur un
ressenti.

Tout est publié : réglages, survies, m_appel par graine, courbes, choix.
