# Protocole pré-enregistré — Le code du besoin existe-t-il avant l'apprentissage ?

Rédigé le 2 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat. Sans apprentissage : torch sur le processeur local.

## D'où vient ce test (dit tel quel)

- **Test 17** (publié). La direction du besoin d'un agent pousse aussi
  l'autre agent : +0,138 chez le premier (90 % de sa propre poussée),
  +0,018 chez le second (53 %, sous le seuil de 0,02). Critère global non
  satisfait ; la lecture pré-enregistrée en cas d'échec est « chaque agent
  s'est fait son propre code ». Mais les données suggèrent un code en
  partie commun, que nous avons attribué, sans le tester, au modèle de
  langage partagé.
- **Ce test vérifie cette attribution.** Si le code vient du modèle de
  langage, la même poussée doit déjà agir sur le modèle **sans aucun
  adaptateur**, qui n'a jamais appris ce monde.

**Question.** Pousser −4·d_E au bloc 12, sur « Choix : », fait-il choisir
« R » (se recharger) au modèle de base Qwen3-0.6B ?

## Mesures

- **Modèle** : Qwen3-0.6B, révision `c1899de`, **sans adaptateur**, en
  fp32, comme dans tous les tests en torch.
- **Contextes** : exactement ceux du test 17 pour le premier agent : ses
  vies de test du sixième test, les **600 premières** décisions où les
  deux besoins valent 6 ou plus. Le modèle de base lit le texte de ces
  vies jusqu'à « Choix : » (il ne les a pas vécues).
- **Conditions** (en un lot, sur les trois tokens « Cho », « ix », « : »
  au bloc 12) :
  1. aucune ;
  2. −4·d_E du premier agent ;
  3. −4·d_E du second agent, ramené par token à la norme de celle du
     premier ;
  4. à 6. trois poussées au hasard de cette norme (flux `[270926, 38, i]`,
     dimension massive à zéro).
- **Lecture** : P(R) = P(« R ») / (P(« R ») + P(« M »)) à la fin de
  « Choix : », et la masse P(« R ») + P(« M »).
- **Contrôle d'exécution** : la lecture en lot avec le cache égale la
  lecture du texte entier à 1e-4 près (4 contextes).

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages). Effet : P(R) avec la
poussée moins P(R) sans, par contexte.

| | Prédiction | Critère |
|---|---|---|
| **BASE1** | La direction du premier agent agit déjà sur le modèle de base | −4·d_E du premier augmente P(R) d'au moins **0,05** (borne basse > 0) ; poussées au hasard : au plus le tiers (moyenne des \|Δ\|). |
| **BASE2** | La direction du second aussi | même critère pour −4·d_E du second. |

**Critère global** : BASE1 et BASE2.

**Publié sans seuil** :
- P(R) moyen du modèle de base sans poussée, et sa masse sur « R » et
  « M » ;
- le rapport entre ces effets et ceux mesurés chez le premier agent au
  test 17 (+0,153 et +0,138).

**Validité** : contrôle d'exécution ; au moins 300 contextes ; masse sur
« R » et « M » ≥ **0,2** en moyenne sans poussée (le modèle de base n'a pas
appris à répondre « R » ou « M » ; en dessous, P(R) n'a pas de sens).

## Ce que le résultat dira

**Si le critère passe.** La direction qui fait agir les agents agit déjà
sur le modèle de langage seul : « énergie basse → se recharger » est un
code qui existait avant l'apprentissage. Les agents ont appris à **y
écrire** leur besoin, pas à l'inventer. Cela expliquerait le transfert du
test 17.

**S'il échoue.** Le code n'agit pas sans l'apprentissage : il a été
construit, ou rendu efficace, par l'apprentissage de chaque agent ; le
transfert du test 17 viendrait alors de ce que des apprentissages
semblables trouvent des solutions semblables.

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti.

## Précautions

Aucune vie nouvelle n'est vécue ; des textes déjà enregistrés sont relus.
Aucun manque nouveau n'est créé.

## Exécution

- **Mesures** : `research/need_base_code.py`, torch sur le processeur
  local. Sorties : `artifacts/llm-need/base-code`.
- **Verdicts** : numpy seulement, vérifiés en CI.
