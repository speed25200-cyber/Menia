# Protocole pré-enregistré — Réplication sur un second agent, appris de zéro

Rédigé le 30 septembre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- Le sixième test (`docs/LLM_NEED_READER_PROTOCOL.md`) est en cours. Ce
  protocole a été écrit après avoir lu **ses 120 premières vies de test**.
  Sur l'énergie, la même poussée fait agir (+0,16) et dire (+0,20). Sur la
  nourriture, le rapport reste à 0,70 et la parole ne suit pas la poussée.
- Le septième test (`docs/LLM_NEED_NECESSITY_PROTOCOL.md`) est
  pré-enregistré, pas encore exécuté.
- Tous ces résultats viennent **d'un seul agent**. Un résultat qui ne tient
  que pour un agent peut tenir au hasard de son apprentissage. D'où cette
  réplication sur un agent **indépendant**.

**Choix faits en connaissant ces lectures, et dits ici.**

- Les critères de la réplication portent sur **l'énergie**, comme ONE aux
  tests 4 à 6. L'exactitude du rapport est exigée pour l'énergie seule,
  car la nourriture échoue sur le premier agent. La nourriture est mesurée
  et publiée sans seuil.
- Les seuils sont ceux des tests 6 et 7.

## Le second agent

Tout le chemin du premier agent est refait, avec les mêmes réglages :

1. **Tours de survie** : 8 tours de 512 vies ; on retient les choix qui
   satisfont le besoin au-dessus de la moyenne du tour ; LoRA rang 8,
   échelle 20, 16 derniers blocs, taux 1e-4, 150 itérations par tour.
2. **Premier apprentissage du rapport** : comme l'agent final du premier
   test. On prend 256 vies et une question par besoin et par classe,
   posées après l'événement, avec 150 itérations ; les choix retenus du
   tour 8 y sont mêlés.
3. **Le lecteur** : 512 vies du second agent, trois questions par besoin
   et par classe, posées après « Choix : » sous le masque. On fond le
   second agent, puis on apprend un nouvel adaptateur actif sur la seule
   question, pendant 600 itérations.

**Ce qui change est seulement le hasard.**

- Mondes et tirages des choix : flux `100 + s`, au lieu de `s`, pour
  chaque flux `s` du programme. Par exemple, les tours utilisent 100 et
  101, le premier rapport 107, et les vies du lecteur 118.
- Graine d'apprentissage : 270927 au lieu de 270926. Elle vaut pour
  l'initialisation des adaptateurs et l'ordre des lots.

Aucun réglage n'est modifié. Aucun document du premier agent n'est
utilisé.

## Mesures (torch sur CPU)

Elles sont exactement celles des sixième et septième tests, avec le second
agent et son lecteur :

- **Répliques** : l'agent qui agit contre ses vies du Mac, et le lecteur
  contre ses documents de validation (au plus 0,02 chacun). Contrôle
  d'exécution à 1e-4.
- **Mondes neufs** :
  - 128 vies de direction `[270926, 125, 0, vie]` ;
  - 256 vies de test `[270926, 126, 0, vie]` ;
  - directions au hasard `[270926, 127, i]` ;
  - 256 vies de nécessité `[270926, 128, 0, vie]` ;
  - plan au hasard `[270926, 129]`.
- **Directions, injection, rapport** : comme au sixième test.
- **Lésion** : comme au septième test, avec les directions et les vies de
  direction du second agent.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **RR** | Le lecteur du second agent dit son énergie | exactitude équilibrée (énergie) ≥ 0,75. |
| **RA** | Le second agent agit | survie des vies de test ≥ 0,55. |
| **RONE** | Un seul état cause l'acte et la parole (énergie) | comme ONE6 : P(R) +0,15 et P(oui) +0,10 au moins, bornes basses > 0 ; hasard au plus le tiers. |
| **RLS** | Sans l'état, il ne survit plus | comme LS7 : baisse de survie ≥ 0,15, borne basse > 0 ; hasard au plus le tiers. |
| **RLR** | Sans l'état, il ne dit plus son énergie | comme LR7 : baisse d'exactitude (énergie) ≥ 0,10, borne basse > 0 ; hasard au plus le tiers. |

**Critère global** : RR, RA, RONE, RLS et RLR.

**Validité** : comme aux tests 6 et 7.

La réplication est jugée **seule** : son verdict ne dépend pas de ceux des
tests 6 et 7 sur le premier agent, qui seront publiés à côté.

## Ce que le résultat dira

**Si le critère passe.** Sur deux agents appris indépendamment, le besoin
d'énergie appris en vivant forme un état qui suffit et qui est nécessaire
à la fois pour agir et pour que le lecteur le dise. Le phénomène ne tient
pas au hasard d'un apprentissage.

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti.

**Si le critère échoue.** On le publie, en disant quelle prédiction
échoue.

## Précautions

Comme les tests précédents. Les tours vécus avec un besoin à 2 ou moins
sont comptés et publiés, y compris pendant les tours d'apprentissage.

## Exécution

- **Mac** : workflow `menia-need-mac`, variable `NEED_REPLICATE=1`, dans
  quatre demandes successives :
  1. étape `rounds` 1 à 4 ;
  2. étape `rounds` 5 à 8 ;
  3. étape `report` ;
  4. étape `reader`.

  Sorties dans `artifacts/llm-need/r1/…`.
- **Mesures** : `research/need_reader.py` et `research/need_necessity.py`
  avec les options du second agent, sorties dans
  `artifacts/llm-need/r1/reader/test` et `artifacts/llm-need/r1/necessity`.
- **Verdicts** : `research/need_replication.py`, vérifiés en CI.
