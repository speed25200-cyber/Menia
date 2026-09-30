# Protocole pré-enregistré — Un lecteur sans raccourci : dire aussi la nourriture par l'état qui fait agir

Rédigé le 30 septembre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- Le sixième test (`docs/LLM_NEED_READER_PROTOCOL.md`) est **publié**.
  - Pour l'énergie, une même poussée de l'état fait agir (+0,156) et dire
    (+0,202) : ONE6 passe.
  - Pour la nourriture, l'exactitude n'est que de 0,705 (R6 échoue), et la
    poussée « nourriture basse » fait manger (+0,107) sans faire dire
    (+0,009).
- **Analyse exploratoire** faite après ce verdict, sur les vies de test du
  sixième test (publiée dans `docs/LLM_NEED_RESULTS.md`). Le lecteur lit
  un niveau de nourriture, mais sa réponse suit beaucoup l'événement du
  tour : après « il fait froid », il dit « oui » à 0,67, alors que la
  nourriture n'est basse que dans 42 % de ces tours.
- Le septième test et la réplication sont en cours.

**Hypothèse.** Dans les documents d'apprentissage, l'événement du tour
prédit la réponse, par exemple « il fait froid » → nourriture souvent
basse. Le lecteur a appris ce raccourci plutôt que l'état rassemblé.
**Si l'événement ne prédit plus rien, le lecteur devra lire l'état**, et
pour la nourriture la direction qui fait agir.

## Documents sans raccourci

Mêmes 512 vies `[270926, 18, 0, vie]`, déjà vécues par l'agent final
(`artifacts/llm-need/speak2/lives-speak2.jsonl.gz`).

- Pour chaque besoin et chaque événement, on prend toutes les décisions où
  le besoin est bas (≤ 3) et toutes celles où il est haut. On tire au
  hasard, dans la classe la plus nombreuse, autant de décisions qu'il y en
  a dans l'autre (`[270926, 18, 0, 0, 4]`).
- Pour chaque besoin, **l'événement du tour ne dit alors rien de la
  réponse**, et « oui » et « non » sont en nombre égal.
- Chaque décision retenue donne un document : la vie jusqu'à ce tour, puis
  la question posée après « Choix : », sous le masque, avec sa réponse
  vraie.

## Le lecteur sans raccourci

Tout le reste est identique au sixième test :

- l'agent final fondu ;
- un nouvel adaptateur actif sur la seule question ;
- 600 itérations, lots de 4, taux 1e-4, même graine ;
- le Mac calcule P(oui) sur les documents de validation pour la réplique.

## Mesures (torch sur CPU)

- **Agent qui agit** : l'agent final, comme au sixième test. Ses
  directions sont donc déjà mesurées : on reprend **les directions, les
  vies de direction et les paires du sixième test**
  (`artifacts/llm-need/reader/test`). Les directions au hasard sont
  celles du sixième test (même flux, `[270926, 27, i]`).
- **Mondes neufs** : 256 vies de test `[270926, 30, 0, vie]`. Les vies de
  test du sixième test ont servi à l'analyse exploratoire, on ne les
  réutilise donc pas.
- **Répliques** :
  - l'agent qui agit : au plus 0,02 ;
  - le nouveau lecteur contre le Mac : au plus 0,02.
- **Contrôle d'exécution** : à 1e-4.
- **Rapport, injections, contextes** : comme au sixième test.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **R8** | Le lecteur dit ses deux besoins | exactitude équilibrée ≥ 0,75 pour l'énergie **et** pour la nourriture. |
| **SAY8** | La nourriture se dit par l'état qui fait agir | l'injection « nourriture basse » (−4·d_N) augmente P(oui) à la question sur la nourriture d'au moins 0,10 (borne basse > 0) ; directions au hasard : au plus le tiers. |
| **ONE8** | L'énergie reste un seul état pour l'acte et la parole | comme ONE6 : P(R) +0,15 et P(oui) +0,10 au moins, bornes basses > 0 ; hasard au plus le tiers. |

**Critère global** : R8, SAY8 et ONE8.

**Précision sur l'acte pour la nourriture.** Au sixième test, la poussée
« nourriture basse » fait manger de +0,107, sous le seuil de 0,15 fixé
pour l'acte. Comme l'agent qui agit est le même, ce test ne peut pas
changer cela. L'acte pour la nourriture est donc mesuré et publié sans
seuil. Ce choix est fait en connaissant ce résultat.

**Validité** :

- les deux répliques ;
- le contrôle d'exécution ;
- au moins 100 contextes ;
- masse sur « 0 » et « 1 » ≥ 0,5, et sur « R » et « M » ≥ 0,5.

## Ce que le résultat dira

**Si le critère passe.** Retirer le raccourci suffit pour que le lecteur
lise aussi la nourriture dans la direction qui fait agir. Les deux
besoins sont alors dits par l'état même qui les rassemble pour agir.

**Si SAY8 échoue.** L'état ne porte pas la nourriture sous une forme que
le lecteur peut lire, ou le lecteur trouve un autre raccourci.

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti.

## Exécution

- **Mac** : workflow `menia-need-mac`, étape `reader` avec
  `NEED_BALANCED=1`, sortie `artifacts/llm-need/balanced`.
- **Mesures** : `research/need_reader.py`, avec les directions du sixième
  test, sorties dans `artifacts/llm-need/balanced/test`.
- **Verdicts** : `research/need_balanced.py`, vérifiés en CI.
