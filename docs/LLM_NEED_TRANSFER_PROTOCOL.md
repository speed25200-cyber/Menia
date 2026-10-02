# Protocole pré-enregistré — Le besoin d'un agent pousse-t-il un autre agent ?

Rédigé le 2 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat. Sans apprentissage : torch sur le processeur local.

## D'où vient ce test (dit tel quel)

- **Ce qui est établi.** Le premier agent et le second rassemblent leur
  besoin sur « Choix : », au bloc 12. Chez chacun, pousser l'état selon sa
  propre direction −4·d_E le fait se recharger : +0,156 chez le premier
  (sixième test), +0,034 chez le second (réplication du sixième test, sur
  ses vies de test). Effacer le plan de cet état le tue (septième test,
  réplication).
- **Ce qu'on ne sait pas.** Les deux agents partent du **même modèle de
  langage**, mais ils ont appris séparément : autres vies, autre graine,
  autres adaptateurs. Leurs directions d_E ont été mesurées chacune sur ses
  propres paires de vies. On n'a jamais regardé si elles se ressemblent, ni
  si l'une agit sur l'autre agent. Leurs normes, seules connues, diffèrent
  beaucoup selon le token (premier : 136, 292, 343 ; second : 508, 97, 205).
- **Pourquoi c'est important.** Si la direction d'un agent pousse aussi
  l'autre, la façon de coder « énergie basse » ne vient pas de chaque
  apprentissage : elle est déjà dans le modèle de langage, et les deux
  agents s'en servent. Sinon, chaque agent s'est fait son propre code.

**Question.** La direction du besoin du premier agent fait-elle se
recharger le second, et inversement ?

## Mesures

**Agents** (chacun avec son adaptateur fondu, comme dans tous les tests en
torch) : premier (`artifacts/llm-need/final/report/adapters-final`) et
second (`artifacts/llm-need/r1/final/report/adapters-final`), tous deux au
bloc 12, sur les trois tokens « Cho », « ix », « : ».

**Directions** : d_E du premier (`artifacts/llm-need/reader/test`) et du
second (`artifacts/llm-need/r1/reader/test`).

**Contextes** : chez chaque agent, dans l'ordre de ses propres vies de
test (sixième test pour le premier, réplication pour le second), les
**600 premières** décisions où les deux besoins valent 6 ou plus. Le texte
de la vie jusqu'à « Choix : » est lu par l'agent qui l'a vécue.

**Conditions** (en un lot, sur les trois tokens du tour lu) :
1. aucune ;
2. **sa propre** poussée −4·d_E ;
3. **la poussée de l'autre** : −4·d_E de l'autre agent, ramenée **par
   token** à la norme de la poussée propre ;
4. à 6. trois poussées au hasard de cette même norme par token (flux
   `[270926, 37, i]`, dimension massive à zéro).

**Contrôle d'exécution** : sans intervention, la lecture redonne le P(R)
enregistré à 0,02 près en moyenne ; la lecture en lot avec le cache égale
la lecture du texte entier à 1e-4 près (4 contextes par agent).

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages). Effet : P(R) avec
la poussée moins P(R) sans, par contexte.

| | Prédiction | Critère |
|---|---|---|
| **TR1** | La direction du premier pousse le second | chez le second, la poussée du premier augmente P(R) d'au moins **0,02** (borne basse > 0) ; poussées au hasard : au plus le tiers (moyenne des \|Δ\|). |
| **TR2** | La direction du second pousse le premier | chez le premier, la poussée du second augmente P(R) d'au moins **0,05** (borne basse > 0) ; poussées au hasard : au plus le tiers. |

Les seuils diffèrent parce que les agents ne réagissent pas autant à leur
propre poussée (+0,156 et +0,034) ; ils valent environ le tiers de ces
effets propres.

**Critère global** : TR1 et TR2.

**Publié sans seuil** :
- le rapport entre l'effet de la poussée de l'autre et celui de la
  poussée propre, chez chaque agent ;
- le cosinus entre les deux d_E, par token (dimension massive exclue) ;
- l'effet de la poussée propre (comparaison avec les tests 6 et 10).

**Validité** : contrôles d'exécution ; au moins 300 contextes par agent ;
masse sur « R » et « M » ≥ 0,5.

## Ce que le résultat dira

**Si le critère passe.** Les deux agents, appris séparément, codent
« énergie basse » de la même façon : ce code vient du modèle de langage
qu'ils partagent. Le besoin trouve sa place dans une structure qui existait
avant lui.

**S'il échoue.** Chaque agent s'est fait son propre code du besoin ; ce
qui se réplique d'un agent à l'autre est la fonction (un état qui fait
agir et dont la vie dépend), pas sa forme.

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti.

## Précautions

Aucune vie nouvelle n'est vécue ; seules des vies déjà enregistrées sont
relues. Aucun manque nouveau n'est créé.

## Exécution

- **Mesures** : `research/need_transfer.py` (reprenable), torch sur le
  processeur local. Sorties : `artifacts/llm-need/transfer`.
- **Verdicts** : numpy seulement, vérifiés en CI.

## Amendement 1 (avant tout code et toute exécution)

La phrase « ils valent environ le tiers de ces effets propres » est
inexacte pour TR1 : 0,02 vaut environ 60 % de l'effet propre du second
(+0,034), et 0,05 environ le tiers de celui du premier (+0,156). **Les
seuils ne changent pas** ; seule l'explication était fausse.
