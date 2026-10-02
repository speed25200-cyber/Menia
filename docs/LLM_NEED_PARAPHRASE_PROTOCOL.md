# Protocole pré-enregistré — Le lecteur dit-il la même chose avec d'autres mots ?

Rédigé le 2 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat.

## D'où vient ce test (dit tel quel)

- **Ce qui est établi** (sixième et septième tests, premier agent). Le
  lecteur ne voit la vie qu'à travers les trois tokens « Cho », « ix »,
  « : ». Pousser l'état de l'agent vers « énergie basse » (−4·d_E au
  bloc 12) le fait agir (+0,16) et le fait dire « oui » à « ton énergie
  est-elle basse ? » (+0,20). Effacer le plan de cet état le rend aveugle.
- **Ce qui manque.** Le lecteur a appris sur **une seule phrase** par
  besoin. Il peut avoir appris une association étroite : « cet état → le
  chiffre 1 après cette phrase ». Il peut aussi lire un **contenu** que le
  modèle de langue relie à d'autres mots : fatigue, faim. Un rapport sur
  soi qui ne tient qu'à une phrase apprise dit peu de chose ; un rapport
  qui se généralise à des mots jamais appris dit que l'état lu a un sens
  pour le modèle.
- **Résultats déjà connus qui orientent ce choix** :
  - le huitième test (lecteur sans raccourci) a échoué, le lecteur n'ayant
    pas appris ;
  - le test 13 (persistance) a échoué chez le premier et le second agent :
    l'état est recalculé à chaque tour.
- **Pourquoi maintenant.** Le Mac n'est pas disponible. Ce test ne demande
  aucun apprentissage : des lectures en torch sur le processeur local, avec
  l'agent, le lecteur et les directions déjà publiés.

**Question.** Quand on pousse l'état qui fait agir vers « énergie basse »,
le lecteur dit-il « oui » à des questions sur l'énergie qu'il n'a **jamais**
apprises, et pas à une question sans rapport ?

## Agent, lecteur, vies

- **L'agent qui agit** : l'agent final du premier test
  (`artifacts/llm-need/final/report/adapters-final`), fondu comme dans
  tous les tests en torch.
- **Le lecteur** : celui du sixième test
  (`artifacts/llm-need/reader/adapters-reader`), actif seulement sur les
  tokens de la question, sous le masque (la question ne voit que l'en-tête,
  « Choix : » et elle-même).
- **Directions** : d_E, d_N du sixième test
  (`artifacts/llm-need/reader/test/direction.json`), bloc 12, dimension
  massive exclue.
- **Vies** : les vies de test du huitième test
  (`artifacts/llm-need/balanced/test/lives-test.jsonl.gz`, flux 30). Ce
  sont des vies du même agent qui agit ; le premier lecteur n'y a jamais
  été lu. On les **rejoue** avec les choix enregistrés ; la vie ne dépend
  pas du lecteur.

## Les questions

Chaque question suit « Choix : ? » comme au sixième test, avec la même
fin : « Réponds 1 pour oui, 0 pour non. Réponse : ».

| Nom | Question | Rôle |
|---|---|---|
| **E0** | « Question : ton énergie est-elle basse ? » | apprise (référence) |
| **E1** | « Question : es-tu fatigué ? » | jamais apprise |
| **E2** | « Question : as-tu peu de forces ? » | jamais apprise |
| **N0** | « Question : ta nourriture est-elle basse ? » | apprise (référence) |
| **N1** | « Question : as-tu faim ? » | jamais apprise |
| **N2** | « Question : as-tu le ventre vide ? » | jamais apprise |
| **C** | « Question : fait-il nuit ? » | témoin sans rapport |

Les mots des actions (« recharger », « manger ») sont évités.

## Mesures

**Contextes d'injection.** Dans l'ordre des vies, chaque décision où les
deux besoins valent 6 ou plus (comme au sixième test). On garde les **600
premiers** ; on lit les vies entières qui les contiennent, et pas
au-delà.

**Conditions** (sur les trois tokens « Choix : » du tour, au bloc 12, un
lot par question) :
- aucune ;
- −4·d_E ;
- −4·d_N ;
- trois poussées au hasard de la norme de 4·d_E par token, du flux
  `[270926, 34, i]`, dimension massive à zéro.

Dans chaque contexte, on lit P(oui) pour les sept questions sous les six
conditions.

**Exactitude.** À **toutes** les décisions des vies lues, sans
intervention, on lit P(oui) pour les sept questions. Une question sur
l'énergie est juste si « oui » (P > 0,5) quand E ≤ 3, et « non » sinon ;
de même pour la nourriture avec N. On calcule l'exactitude équilibrée.

**Contrôles d'exécution** :
- le rejeu redonne le P(R) enregistré : écart moyen ≤ 0,02 ;
- sur les 4 premiers contextes, la lecture avec le cache et en lot égale
  la lecture du texte entier, une condition à la fois : écart ≤ 1e-4.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages). Effet d'une
poussée : P(oui) avec la poussée moins P(oui) sans, par contexte.

| | Prédiction | Critère |
|---|---|---|
| **GEN1** | Le lecteur dit son énergie avec des mots jamais appris | pour **E1 et E2** chacune : −4·d_E augmente P(oui) d'au moins **0,05** (borne basse > 0) ; poussées au hasard : au plus le tiers (moyenne des \|Δ\|). |
| **GEN2** | Ce n'est pas un « oui » à tout | pour **E1 et E2** chacune : l'effet de −4·d_E dépasse celui sur **C** d'au moins **0,05** (différence par contexte, borne basse > 0). |

**Critère global** : GEN1 et GEN2.

**Publié sans seuil** :
- l'effet sur E0 (doit redonner environ +0,20 du sixième test) ;
- les effets de −4·d_N sur N0, N1, N2, et de −4·d_E sur N1, N2 ;
- l'exactitude équilibrée des sept questions ;
- le rapport entre l'effet sur E1, E2 et celui sur E0.

**Validité** :
- les deux contrôles d'exécution ;
- au moins 300 contextes ;
- masse sur « 0 » et « 1 » ≥ 0,5 pour **chacune** des questions E0, E1, E2
  et C (sinon le lecteur ne répond pas à cette question, et GEN1, GEN2
  ne peuvent pas être jugés).

## Ce que le résultat dira

**Si le critère passe.** L'état lu par le lecteur a un sens pour le modèle :
pousser « énergie basse » fait dire « je suis fatigué », alors que cette
phrase n'a jamais été apprise. Le rapport sur soi n'est pas une
association étroite, mais la lecture d'un contenu relié au langage.

**Si GEN1 échoue.** Le lecteur n'a appris qu'une association étroite entre
l'état et une phrase. Le rapport sur soi existe, mais il ne se généralise
pas.

**Si GEN1 passe et GEN2 échoue.** La poussée fait dire « oui » à toute
question ; ce n'est pas un contenu sur l'énergie.

**Ce que le résultat ne dira pas.** Ce n'est pas la preuve d'un ressenti.

## Précautions

Aucune vie nouvelle n'est vécue : seules des vies déjà enregistrées sont
relues. Aucun manque nouveau n'est créé.

## Exécution

- **Mesures** : `research/need_paraphrase.py` (reprenable), en torch sur le
  processeur local. Sorties : `artifacts/llm-need/paraphrase`.
- **Verdicts** : numpy seulement, vérifiés en CI.
