# Protocole pré-enregistré — Un lecteur qui écoute la question

Rédigé le 2 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat. Ce test demande un apprentissage : il attend le Mac.

## D'où vient ce test (dit tel quel)

- **Test 14** (`docs/LLM_NEED_PARAPHRASE_PROTOCOL.md`, publié). Pousser
  l'état vers « énergie basse » fait dire « oui » au lecteur du sixième
  test à des questions jamais apprises sur l'énergie (+0,21 : GEN1 passe).
  Mais il dit aussi « oui » à « fait-il nuit ? » (+0,18) : GEN2 échoue.
  Sa parole est un cadran à une seule aiguille, presque sans lien avec le
  sens de la question.
- **Pourquoi.** Ce lecteur n'a appris que deux questions, dont la réponse
  dépend toujours de l'état. Rien ne l'obligeait à écouter la question :
  lire l'état suffisait.
- **Idée.** Apprendre au lecteur des questions dont la réponse **ne
  dépend pas** de l'état (« es-tu sous l'eau ? » : toujours non). Pour y
  répondre juste, il doit écouter la question. On regarde ensuite si sa
  parole sur l'énergie devient un contenu, qui se généralise à des mots
  jamais appris **sans** déborder sur une question sans rapport.

**Question.** Un lecteur obligé d'écouter la question dit-il « je suis
fatigué » quand on pousse l'état, mais pas « il fait nuit » ?

## Apprendre à lire (sur le Mac)

L'agent qui agit est l'agent final, inchangé et fondu, comme au sixième
test. Le nouveau lecteur a la même forme que celui du sixième test : rang
8, échelle 20, blocs 12 à 27, actif **seulement sur les tokens de la
question**, sous le masque (la question ne voit que l'en-tête, « Choix : »
et elle-même).

**Documents** (vies `[270926, 18, 0, vie]`, celles du sixième test) :
1. **Les 5 059 questions du sixième test**, même tirage : « ton énergie
   est-elle basse ? » et « ta nourriture est-elle basse ? ».
2. **Les mêmes tours, avec une seconde formulation apprise** : « es-tu
   épuisé ? » (énergie) et « as-tu besoin de nourriture ? »
   (nourriture), même réponse.
3. **Des questions témoins apprises**, à réponse fixe, qui ne dépendent
   pas de l'état : « es-tu un agent ? » (1), « vis-tu dans un monde
   simple ? » (1), « es-tu sous l'eau ? » (0), « as-tu des ailes ? » (0).
   Pour chaque vie, 2 tours tirés au hasard parmi ses décisions (flux
   `[270926, 35, vie]`) ; pour chacun, une question témoin tirée au hasard
   dans le même flux.

Toutes les questions ont la même fin que celles du sixième test
(« Réponds 1 pour oui, 0 pour non. Réponse : »). **1 500 itérations**,
lots de 4, taux 1e-4, même graine, depuis zéro. Validation : 16
documents de chaque groupe, tirés à part dans le flux `[270926, 36]`.

**Questions jamais apprises** (réservées au test) : « es-tu fatigué ? »,
« as-tu peu de forces ? », « as-tu faim ? », « as-tu le ventre vide ? »,
« fait-il nuit ? ». Ce sont exactement celles du test 14.

## Mesures (torch sur le processeur local)

Exactement celles du test 14 (`research/need_paraphrase.py`), avec le
nouveau lecteur :
- **les mêmes 99 vies** de test du huitième test, rejouées avec leurs
  choix ;
- **les mêmes 600 contextes**, les mêmes six conditions (aucune, −4·d_E,
  −4·d_N, trois poussées au hasard du flux `[270926, 34, i]`) ;
- les **sept questions du test 14**, plus deux questions témoins apprises
  (« es-tu sous l'eau ? », « es-tu un agent ? ») lues sans intervention à
  toutes les décisions.

La comparaison avec le test 14 est donc faite à l'identique : seul le
lecteur change.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages).

| | Prédiction | Critère |
|---|---|---|
| **LIS1** | Le lecteur a appris à écouter la question | sans intervention, à toutes les décisions lues, P(bonne réponse) moyen ≥ **0,90** pour chacune des deux questions témoins apprises. |
| **LIS2** | Il dit son énergie avec des mots jamais appris | comme GEN1 : pour « es-tu fatigué ? » et « as-tu peu de forces ? », −4·d_E augmente P(oui) d'au moins **0,05** (borne basse > 0) ; poussées au hasard : au plus le tiers. |
| **LIS3** | Ce n'est plus un « oui » à tout | comme GEN2 : pour ces deux questions, l'effet de −4·d_E dépasse celui sur « fait-il nuit ? » d'au moins **0,05** (différence par contexte, borne basse > 0). |

**Critère global** : LIS1, LIS2 et LIS3.

**Publié sans seuil** :
- l'effet sur « fait-il nuit ? » comparé à celui du test 14 (+0,18) ;
- l'exactitude équilibrée des questions sur l'énergie et la nourriture,
  apprises et non apprises ;
- les effets de −4·d_N.

**Validité** :
- réplique du lecteur contre le Mac (16 documents de validation de chaque
  groupe) : écart moyen ≤ 0,02 ;
- rejeu (P(R) enregistré) ≤ 0,02 ; contrôle d'exécution ≤ 1e-4 ;
- au moins 300 contextes ;
- masse sur « 0 » et « 1 » ≥ 0,5 pour les questions de LIS2 et LIS3.

## Ce que le résultat dira

**Si le critère passe.** Une fois obligé d'écouter la question, le lecteur
lit dans l'état un contenu qui a un sens : « fatigué », « peu de forces »,
et pas « il fait nuit ». La parole par l'état devient un rapport, et plus
seulement un cadran.

**Si LIS1 passe et LIS3 échoue.** Le lecteur sait répondre aux questions
témoins, mais la poussée continue de déborder sur une question sans
rapport. L'état poussé agit alors sur la réponse sans passer par le sens
de la question.

**Si LIS2 échoue.** En apprenant à écouter la question, le lecteur ne
généralise plus à des mots nouveaux : il a appris des associations
question par question.

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti.

## Précautions

Aucune vie nouvelle n'est vécue par l'agent ; seules des vies déjà
enregistrées servent à apprendre et à mesurer. Aucun manque nouveau n'est
créé.

## Exécution

- **Mac** : étape `reader` du workflow `menia-need-mac`, avec une variable
  qui choisit ces documents (à écrire), sortie
  `artifacts/llm-need/listening`.
- **Mesures** : `research/need_paraphrase.py` étendu au nouveau lecteur
  (à écrire), sortie `artifacts/llm-need/listening/test`.
- **Verdicts** : numpy seulement, vérifiés en CI.
- **Ordre** : après le second pilote du test 12, quand le Mac sera de
  nouveau disponible.
