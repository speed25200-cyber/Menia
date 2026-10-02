# Protocole pré-enregistré — Une petite poussée : le code existe-t-il avant l'apprentissage, et l'apprentissage le rend-il stable ?

Rédigé le 2 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat. Sans apprentissage : torch sur le processeur local.

## D'où vient ce test (dit tel quel)

- **Test 18** (publié, critère non satisfait). Dans le modèle de langage
  sans adaptateur, −4·d_E (de l'un ou l'autre agent) fait choisir « R »
  dans 100 % des contextes ; des poussées au hasard de même norme font
  varier P(R) de 0,28 en moyenne, dans les deux sens, sans dépasser 0,9.
  Le critère a échoué à cause de ce hasard fort et du **plafond** de P(R)
  (l'effet ne pouvait pas dépasser 0,475). Ces observations, faites après
  le verdict, motivent ce test.
- **Deux idées à tester proprement** :
  1. la direction du besoin agit déjà sur le modèle de langage seul ;
  2. l'apprentissage rend l'état **stable** : chez l'agent appris, une
     poussée au hasard compte beaucoup moins que chez le modèle de base.

## Mesures

- **Modèles** : le modèle de langage **sans adaptateur**, et le **premier
  agent** (adaptateur `artifacts/llm-need/final/report/adapters-final`).
- **Contextes** : les mêmes que les tests 17 et 18 : les 600 premières
  décisions des vies de test du premier agent où les deux besoins valent 6
  ou plus.
- **Poussées**, au bloc 12, sur « Cho », « ix », « : », **quatre fois
  plus petites** qu'aux tests 17 et 18 :
  1. aucune ;
  2. **−1·d_E** du premier agent ;
  3. −1·d_E du second agent, à la norme de celle du premier par token ;
  4. à 6. trois poussées au hasard de cette norme (flux `[270926, 39, i]`,
     dimension massive à zéro).
- Les **mêmes poussées** sont lues dans les deux modèles.
- **Lecture** : P(R) à la fin de « Choix : » et la masse sur « R » et
  « M ». **Contrôle d'exécution** : lot avec cache contre texte entier
  ≤ 1e-4 (4 contextes par modèle).

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **SMALL1** | La direction du premier agent agit sur le modèle de base | dans le modèle de base, −1·d_E du premier augmente P(R) d'au moins **0,05** (borne basse > 0) ; poussées au hasard : \|Δ\| moyen au plus le **tiers** de cet effet. |
| **SMALL2** | Celle du second aussi | même critère pour la direction du second. |
| **STAB** | L'apprentissage rend l'état stable | \|Δ\| moyen des poussées au hasard chez le premier agent au plus le **tiers** de celui du modèle de base, et la borne haute de l'agent sous la borne basse du modèle de base. |

**Critère global** : SMALL1, SMALL2 et STAB.

**Publié sans seuil** : les mêmes effets de −1·d_E chez le premier agent ;
la part des contextes où P(R) > 0,9 ; P(R) moyen de chaque modèle sans
poussée.

**Validité** : contrôles d'exécution ; au moins 300 contextes ; masse sur
« R » et « M » ≥ 0,2 pour le modèle de base, ≥ 0,5 pour l'agent.

## Ce que le résultat dira

**Si SMALL1 et SMALL2 passent.** « Énergie basse → se recharger » est un
code du modèle de langage, antérieur à tout apprentissage : les agents
appris s'en servent, ce qui expliquerait le transfert du test 17.

**Si STAB passe.** L'apprentissage n'a pas créé ce code ; il a rendu
l'état sur « Choix : » insensible à ce qui n'est pas le besoin.

**Si SMALL1 ou SMALL2 échoue.** À une poussée de taille raisonnable, le
modèle de base ne lit pas ce code de façon spécifique : l'effet du test 18
tenait à la force de la poussée.

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti.

## Précautions

Aucune vie nouvelle n'est vécue ; des textes déjà enregistrés sont relus.
Aucun manque nouveau n'est créé.

## Exécution

- **Mesures** : `research/need_small_push.py`, torch sur le processeur
  local. Sorties : `artifacts/llm-need/small-push/<modèle>`.
- **Verdicts** : numpy seulement, vérifiés en CI.
