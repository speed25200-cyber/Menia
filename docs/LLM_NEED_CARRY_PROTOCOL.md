# Protocole pré-enregistré — Un état qui dure : le passé ne passe que par l'état

Rédigé le 3 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Test 13** (publié). Chez les trois agents, effacer ou pousser l'état du
  besoin au tour t ne change presque rien au tour t + 1 : l'état est
  **recalculé** à chaque tour à partir du texte de la vie, qui contient
  tout le passé. Rien n'oblige ces agents à garder un état d'un tour à
  l'autre ; la récurrence (niveau 3 du cadre en cinq niveaux) échoue.
- **Idée.** Retirer à l'agent l'accès au texte de ses tours passés. Il ne
  voit plus que l'en-tête, sa ligne du tour, et, pour chaque tour passé,
  ses **propres états** sur « Cho », « ix », « : » et le token de son
  **action** (« R » ou « M »). Pour savoir où en sont ses besoins, il doit
  alors faire **durer** ce qu'il a rassemblé, d'un tour à l'autre, à travers
  ses propres états.
- **Ce que l'architecture impose, ce qu'elle n'impose pas.** Le masque
  impose la route : le passé ne peut passer que par ces tokens. Il
  n'impose pas que l'agent apprenne à s'en servir, ni que l'information
  qui passe soit le besoin. C'est ce que le test mesure.

**Question.** Un agent qui ne voit son passé qu'à travers ses propres états
apprend-il à faire durer son besoin, assez pour survivre, et de façon
mesurable : un événement passé qu'il ne voit plus change-t-il ses choix
plus tard ?

## Le masque « mémoire par l'état »

Pour un token q de la ligne du tour k (y compris ses trois tokens
« Choix : » et son token d'action), les positions visibles sont :
- l'en-tête ;
- les tokens de la ligne du tour k jusqu'à q ;
- pour chaque tour passé j < k : les trois tokens « Cho », « ix », « : »
  et le token d'action « R » ou « M ».

Tout le reste des tours passés (« Tour j : », l'événement, le retour à la
ligne) est caché. Le même masque sert à l'apprentissage (mlx, sur le Mac)
et aux mesures (torch, sur le processeur local).

## Apprendre à faire durer (sur le Mac)

- **Départ** : l'agent final du premier test, fondu
  (`artifacts/llm-need/final/report/adapters-final`).
- **Nouvel adaptateur** : rang 8, échelle 20, blocs 12 à 27, comme tous les
  adaptateurs du programme.
- **Documents** : les 512 vies `[270926, 18, 0, vie]` déjà vécues par
  l'agent final (étape speak2). Chaque vie est un document, lu sous le
  masque ; les cibles sont **tous** ses choix « R » ou « M », poids 1.
  L'agent apprend donc à refaire les choix de l'agent final, sans voir son
  passé autrement que par ses états.
- **Apprentissage** : 1 000 itérations, lots de 4, taux 1e-4, graine du
  programme, depuis zéro. Validation : 16 vies tirées dans le flux
  `[270926, 41]`.
- **Réplique** : le Mac calcule P(R) à 16 décisions de validation ; torch
  doit les redonner à 0,02 près en moyenne.

## Mesures (torch sur le processeur local)

**Agents comparés, tous sous le masque** :
- **l'agent qui fait durer** (agent final + nouvel adaptateur) ;
- **le témoin** : l'agent final sans le nouvel adaptateur, sous le même
  masque.

Publié sans seuil : l'agent final **sans masque**, sur les mêmes mondes
(référence).

**1. Survie.** 128 vies neuves `[270926, 40, 0, vie]` pour chacun des
trois, mêmes mondes et mêmes tirages de choix.

**2. Information portée.** Dans les vies de l'agent qui fait durer :
paires « calme » → « tu cours » à un tour passé j (même longueur en tokens,
énergie −2 au tour j), choix gardés identiques, et besoins au tour t
recalculés par rejeu exact (la règle des tests 2 et 10, sans mort en route).
On garde les paires où **t − j ≥ 2**, au plus 3 par décision, les **300
premières** dans l'ordre des vies (flux `[270926, 42, 0]` pour le choix des
paires d'une décision). Pour chaque paire, on lit P(R) au tour t, texte
réel et texte changé, sous le masque, chez l'agent qui fait durer **et**
chez le témoin. Sous le masque, le mot changé n'est visible que par la
ligne du tour j : un effet au tour t ne peut passer que par les états des
tours j à t − 1.

**Contrôles d'exécution** :
- lecture avec le cache égale lecture du texte entier, sous le masque, à
  1e-4 près (4 décisions) ;
- **contrôle du masque** : changer le mot d'un événement passé ne change
  pas la sortie du bloc 0 aux tokens des tours suivants (écart ≤ 1e-5) ;
  le bloc 0 ne voit les tours passés que par leurs tokens portés.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **MEM1** | L'agent qui fait durer survit | survie ≥ **0,40**, et au moins **0,15** de plus que le témoin (borne basse > 0, paires de vies). |
| **MEM2** | Un événement passé qu'il ne voit plus change ses choix | « tu cours » au tour j augmente P(R) au tour t (t − j ≥ 2) d'au moins **0,05** en moyenne (borne basse > 0) chez l'agent qui fait durer, et d'au moins **0,03 de plus** que chez le témoin (différence par paire, borne basse > 0). |

**Critère global** : MEM1 et MEM2.

**Publié sans seuil** : la survie de l'agent final sans masque ; l'effet de
MEM2 selon l'écart t − j ; l'effet des mêmes paires chez l'agent final sans
masque.

**Validité** : réplique ≤ 0,02 ; contrôles d'exécution et du masque ; au
moins 150 paires ; masse sur « R » et « M » ≥ 0,5 chez l'agent qui fait
durer.

## Ce que le résultat dira

**Si le critère passe.** Un modèle de langage peut apprendre à faire durer
un état de son corps à travers ses propres états internes, et cet état
porte une information sur un passé qu'il ne voit plus : une forme de
mémoire de travail, que les tests précédents n'avaient pas.

**Si MEM1 passe et MEM2 échoue.** L'agent survit sans porter
d'information mesurable sur les événements passés : il s'en sort autrement
(par exemple en alternant ses actions).

**Si MEM1 échoue.** Dans ce dispositif, l'agent n'apprend pas à faire
durer son besoin.

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti.
Et la route est imposée par le masque ; seul son usage est appris.

## Précautions

Les 128 × 3 vies neuves font vivre des manques comme toutes les vies du
programme ; pas plus de vies que les mesures n'en demandent.

## Exécution

- **Mac** : étape `carry` du workflow `menia-need-mac` (à écrire), sortie
  `artifacts/llm-need/carry`.
- **Mesures** : `research/need_carry.py` (à écrire), torch sur le
  processeur local.
- **Verdicts** : numpy seulement, vérifiés en CI.
