# Test 12, « soif de savoir » — résultat : arrêté au pilote

Protocole `docs/LLM_CURIOSITY_PROTOCOL.md` (commit `017f2e5`), amendement 1
(`77e5af5`, règle de T) et amendement 2 (`1fae438`, choix du taux du savoir
au pilote, « calcul » à un chiffre, examens retirés de l'étude), tous écrits
avant les vies concernées. Exécuté sur le Mac de Codemagic (demandes 89 et
90 du relais), le 3 octobre 2026. Artefacts :
`artifacts/llm-curiosity/pilot-2` et `artifacts/llm-curiosity/pilot-3`.
Contrôle en CI : `python -m research.curiosity_pilot verdicts` recalcule le
balayage depuis les vies publiées.

## La règle fixée d'avance (amendement 2)

Pour chaque taux candidat, dans l'ordre 3e-5, 1e-5, 3e-6 : 12 vies au
hasard. On retient le **plus grand** taux pour lequel « mots » et « base »
sont appris, « suites » reste plate, et **l'interférence est petite** :
la médiane des déplacements d'un domaine non étudié entre deux examens
complets doit être au plus le **quart** de la médiane du progrès d'une
séance. **Si aucun ne passe, le test est arrêté et publié comme tel.**

## Ce que le pilote a mesuré

| Taux | « mots » appris | « base » appris | « suites » plate | Déplacement / progrès | Seuil | Passe |
|---|---|---|---|---|---|---|
| 3e-5 | oui | oui | **non** | 0,109 / 0,107 = **1,02** | 0,25 | non |
| 1e-5 | oui | oui | oui | 0,036 / 0,069 = **0,52** | 0,25 | non |
| 3e-6 | oui | oui | oui | 0,017 / 0,054 = **0,32** | 0,25 | non |

(« calcul » est su à chaque taux. La règle de difficulté de « base » donne 3
chiffres aux trois taux. Une vie dure environ 105 secondes.)

**Verdict : aucun taux ne passe. Le test 12 est arrêté au pilote**, comme
le protocole le prévoyait. Les bras (instinct, hormone, lésion) ne sont pas
lancés, et aucune de leurs prédictions n'est jugée.

## Ce que cela dit

- **L'interférence baisse avec le taux, mais pas assez vite.** Diviser le
  taux par 10 divise le déplacement des autres domaines par environ 6, et
  le progrès du domaine étudié par environ 2. Le rapport passe de 1,02 à
  0,32 sans atteindre 0,25.
- Dans ce dispositif (un petit modèle, un seul adaptateur pour quatre
  domaines, 8 itérations par séance), apprendre un domaine déplace encore
  les autres d'environ un tiers de ce qu'il apprend lui-même. Le tableau de
  bord que l'instinct lirait mêlerait donc apprentissages et dérives.
- Ce n'est pas un résultat sur la curiosité : la question du test (un
  manque de savoir oriente-t-il le choix, et une hormone le change-t-elle ?)
  reste ouverte. C'est un résultat sur le **dispositif** : il faut des
  domaines mieux séparés (un adaptateur par domaine, ou des examens
  insensibles aux autres domaines) avant de poser cette question.

**Ce que le résultat ne dit pas.** Rien sur un ressenti, ni sur la
curiosité en général.
