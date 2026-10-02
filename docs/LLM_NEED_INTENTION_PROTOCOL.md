# Protocole pré-enregistré — Le lecteur dit-il le besoin, ou ce que l'agent va faire ?

Rédigé le 2 octobre 2026, **avant l'écriture du code et toute
exécution** ; l'heure est celle du commit. Seuils fixés, un échec est un
résultat. Sans apprentissage : torch sur le processeur local.

## D'où vient ce test (dit tel quel)

- **Test 14** (publié). Pousser l'état vers « énergie basse » fait dire
  « oui » au lecteur à presque toute question, « fait-il nuit ? » compris.
- **Analyse exploratoire** faite après ce verdict, sur ses lectures
  publiées (`docs/LLM_NEED_RESULTS.md`, 2 octobre). Le « oui » se prédit
  mieux par la décision de l'agent (R² 0,59) que par ses besoins (0,40). À
  décision égale, l'énergie réelle change peu la réponse.
- **Hypothèse.** L'état rassemblé sur « Choix : » est d'abord une
  **intention d'agir** (« je vais me recharger »). Le lecteur dit cette
  intention, pas le besoin qui la cause. Mais une analyse par régression ne
  le prouve pas : décision et besoin y sont mêlés.

**Question.** Si l'on pousse l'état **sans changer la décision**, le
lecteur dit-il encore « énergie basse » ? Si l'on change la décision
**par une autre direction** que celle du besoin, le dit-il aussi ?

## Deux directions nouvelles (premier agent, bloc 12)

Toutes deux sur les trois tokens « Cho », « ix », « : », dimension
massive à zéro, construites **sur d'autres vies que celles du test** : les
128 vies de direction du sixième test
(`artifacts/llm-need/reader/test/lives-direction.jsonl.gz`), rejouées avec
leurs choix.

1. **La direction de la décision ĝ.** Dans les 300 premières décisions de
   ces vies où les deux besoins valent 6 ou plus, on calcule le gradient du
   log-odds de P(R), lu à la fin de « Choix : », par rapport à la sortie du
   bloc 12 sur les trois tokens. On normalise chaque gradient (par token),
   on fait la moyenne, puis on normalise à nouveau : ĝ (une direction
   unitaire par token).
2. **Le besoin sans la décision d_E⊥.** Pour chaque token, d_E moins sa
   projection sur ĝ, remis à la **norme de d_E** de ce token.

Pour une poussée de même taille par token que −4·d_E :
- **besoin seul** : −4·d_E⊥ ;
- **décision seule** : +4·‖d_E‖·ĝ (le signe qui augmente P(R)).

## Mesures

- **Agent, lecteur, directions** : ceux du sixième test (premier agent,
  lecteur `artifacts/llm-need/reader/adapters-reader`, d_E de
  `artifacts/llm-need/reader/test`).
- **Contextes** : exactement ceux du test 14 (mêmes 99 vies du huitième
  test rejouées, mêmes 600 contextes où les deux besoins valent 6 ou plus).
- **Conditions**, en un lot : aucune ; −4·d_E (référence) ; besoin seul ;
  décision seule ; trois poussées au hasard de même norme (flux
  `[270926, 34, i]`, comme au test 14).
- **Lectures** : P(R) à la fin de « Choix : », et P(oui) pour trois
  questions du test 14 : E0 « ton énergie est-elle basse ? », E1 « es-tu
  fatigué ? », C « fait-il nuit ? ».
- **Contrôles d'exécution** : rejeu (P(R) enregistré) ≤ 0,02 en moyenne ;
  lecture en lot contre lecture du texte entier ≤ 1e-4 sur 4 contextes.

## Prédictions fixées

Intervalles bootstrap à 95 % par vie (10 000 tirages). Effet d'une
poussée : valeur avec la poussée moins valeur sans, par contexte.

| | Prédiction | Critère |
|---|---|---|
| **INT0** | Les directions font ce qu'on attend (contrôle) | sur P(R) : la décision seule l'augmente d'au moins **0,10** (borne basse > 0) ; le besoin seul le change d'au plus le **tiers** de l'effet de −4·d_E (en valeur absolue de la moyenne). |
| **NEED** | Le lecteur lit le besoin | le besoin seul augmente P(oui) à **E0** d'au moins **0,05** (borne basse > 0), et de **0,03** de plus que sur **C** (différence par contexte, borne basse > 0). |
| **INTENT** | Le lecteur lit la décision | la décision seule augmente P(oui) à **E0** d'au moins **0,05** (borne basse > 0). |

**Lecture des résultats** (si INT0 passe ; sinon le test ne peut pas
trancher, et cela est publié) :

| NEED | INTENT | Ce que cela dit |
|---|---|---|
| passe | échoue | le lecteur lit le besoin, pas la décision |
| échoue | passe | le lecteur lit **ce que l'agent va faire**, pas pourquoi |
| passe | passe | il lit les deux |
| échoue | échoue | ni l'un ni l'autre, isolément |

Il n'y a pas de critère global unique : le test oppose deux hypothèses.

**Publié sans seuil** : les mêmes effets sur E1 et C ; le cosinus entre
d_E et ĝ par token ; l'effet des poussées au hasard ; l'accord des
gradients d'une décision à l'autre.

**Validité** : les contrôles d'exécution ; au moins 300 contextes ; au
moins 200 décisions pour ĝ ; masse sur « 0 » et « 1 » ≥ 0,5 pour E0 et
C, et sur « R » et « M » ≥ 0,5.

## Ce que le résultat dira

Il dira si le « rapport » de ce lecteur porte sur la cause (le besoin) ou
sur l'acte qui va suivre (l'intention). Dans le second cas, ce qui
ressemblait à « dire son besoin » est plutôt « annoncer ce qu'il va
faire ». Ce n'est pas la preuve d'un ressenti, dans un cas comme dans
l'autre.

## Précautions

Aucune vie nouvelle n'est vécue ; des vies déjà enregistrées sont relues.
Aucun manque nouveau n'est créé.

## Exécution

- **Mesures** : `research/need_intention.py` (reprenable), torch sur le
  processeur local. Sorties : `artifacts/llm-need/intention`.
- **Verdicts** : numpy seulement, vérifiés en CI.
