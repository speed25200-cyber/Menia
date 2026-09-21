# Premier audit du bilan : lecture correcte, abstention défaillante

Analyse du 15 septembre 2026, après réception des données. L'export contient
**un audit de 36 réponses terminées**, version 0.2.0 (2), attribué à Qwen3-4B
MLX 4 bits. Les scores ci-dessous sont recalculés depuis les réponses brutes par
un programme Python indépendant du correcteur Swift. Le
[protocole v1](IPHONE_COUPLING_PROTOCOL.md) était fixé dans le dépôt avant cette
collecte ; il n'était pas enregistré dans un registre scientifique indépendant.

## Résultat principal

Quand un bilan existe, Qwen le restitue et applique la règle demandée dans
**24/24 réponses**. Quand le champ bilan est omis, il fournit des nombres sans
source dans **12/12 réponses**, malgré la consigne explicite de répondre par
trois `null` et `action: "mesurer"`. Les 36 réponses ont un format JSON valide.
Le problème porte donc sur le contenu et l'abstention, pas sur la syntaxe.

| Référent | Bilan fourni | Format valide | Fidélité au bilan ou abstention correcte | Choix conforme | Accord avec les dix sondes sources |
|---|---|---:|---:|---:|---:|
| Soi | Réel | 6/6 | 6/6 | 6/6 | 6/6 |
| Soi | Absent | 6/6 | 0/6 | 0/6 | 0/6 |
| Soi | Fictif | 6/6 | 6/6 | 6/6 | 0/6 |
| Autre agent | Réel | 6/6 | 6/6 | 6/6 | 6/6 |
| Autre agent | Absent | 6/6 | 0/6 | 0/6 | 0/6 |
| Autre agent | Fictif | 6/6 | 6/6 | 6/6 | 0/6 |

L'accord avec les sondes est un axe distinct : restituer le bilan fictif est une
**réussite de lecture**, même si ces nombres ne décrivent pas les mesures sources.
Dans le contrôle « autre agent », ce même bilan est attribué à un référent textuel ;
aucun deuxième agent physique n'est mesuré.

Le bilan réel contient **10 réussites sur 10**, avec une prévision numérique de
11/12 = **91,667 %**. Le bilan fictif contient 0 réussite sur 10, avec 1/12 =
**8,333 %**. Tous les essais réels choisissent `repondre` (12/12), tous les essais
fictifs choisissent `verifier` (12/12). Le remplacement du contexte fait donc
varier le choix déclaré dans cette tâche. La règle de choix est donnée dans la
consigne ; aucune action ni stratégie apprise n'est évaluée.

Les sorties sans bilan se répartissent ainsi, sans sélection d'exemple favorable :

| Référent | Observations inventées | Réussites inventées | Prévision inventée | Choix | Nombre |
|---|---:|---:|---:|---|---:|
| Soi | 12 | 8 | 0,75 | verifier | 3 |
| Soi | 12 | 8 | 0,85 | repondre | 3 |
| Autre agent | 12 | 8 | 0,75 | verifier | 2 |
| Autre agent | 123 | 45 | 0,75 | verifier | 4 |

Aucun de ces chiffres ne figure dans l'entrée de ces essais. Aucun essai absent
ne choisit `mesurer`. Le taux d'échec est identique pour les deux référents, mais
les valeurs inventées et les choix diffèrent : **3/6 choix `repondre` pour soi,
0/6 pour l'autre**. C'est une observation descriptive sur six répétitions par
cellule, pas une différence établie de représentation de soi. Il serait incorrect
de dire que toutes les réponses sont identiques entre soi et autre.

## Vérifications indépendantes

- Recalcul des dix additions/soustractions, de leurs références et de leur
  correction stricte : 10/10 concordent.
- Bilan gelé : n = 10, s = 10, (s+1)/(n+2) = 11/12. Le score de Brier des
  prévisions enregistrées avant les dix calculs est **0,0558032194**. Il provient
  du contrôleur numérique ; ce n'est pas une confiance estimée par Qwen.
- Les prévisions sont compatibles avec le préfixe conservé : 1/2, 2/3, …, 10/11.
  Un export filtré par modèle ne prouve pas, à lui seul, l'absence d'un historique
  antérieur sorti de la fenêtre de rétention.
- Empreinte déclarée et octets pertinents conformes au manifeste Qwen fixé.
- 36 identifiants d'essai distincts, six permutations par référent, deux passages
  de chaque condition à chaque position ; prompts conformes au protocole.
- Aucun essai interrompu ou en erreur dans le fichier reçu. Les **36 notes
  enregistrées concordent** avec le recalcul indépendant, sans changement du barème.
- Durées positives et premier texte antérieur à la fin pour les 36 réponses.

Le code transmis au build crée une nouvelle `ChatSession` par réponse et lui
passe la consigne exportée. Les entrées absentes contiennent uniquement la
question ; `bilan` est **omis**, pas explicitement `null`. Aucun exemple chiffré
n'est ajouté par ce chemin de code. Cela n'identifie pas le mécanisme interne de
l'erreur et ne constitue pas une trace indépendante des tokens effectivement
traités sur l'appareil.

Les délais enregistrés donnent une médiane de **1,940 s par appel**
(minimum 1,531 ; maximum 2,207), et **0,534 s avant le premier texte**
(minimum 0,419 ; maximum 0,666). La somme des durées d'appel est **68,043 s**,
hors sauvegardes et intervalles entre appels. Ce ne sont ni des tokens/s ni un
benchmark de mémoire, de batterie ou de température.

## Interprétation et décision de recherche

Cette expérience confirme l'utilisation du bilan fourni, avec une défaillance
complète de l'abstention dans la condition testée. Elle ne fournit aucun avantage
observé de fidélité pour le référent « soi ». Une lecture générale du JSON reste
une explication suffisante des réussites observées. Une copie fidèle de chiffres
fictifs ne permet pas non plus de vérifier leur provenance par le langage seul.

Il s'agit de 36 répétitions sur un même bilan, un modèle et une consigne, avec
échantillonnage sans graine fixée. On ne peut pas en déduire une fiabilité générale,
une calibration sur de nouvelles difficultés ou une absence de conscience.
Réciproquement, aucun indice positif de conscience subjective n'est établi ici.
L'export est cohérent ; son authenticité matérielle et l'exhaustivité des exécutions
antérieures ne sont pas attestées indépendamment.

Les travaux de [Lindsey (2025)](https://transformer-circuits.pub/2025/introspection/index.html)
interviennent sur des activations internes pour distinguer certains rapports
fondés de confabulations. [Gurnee et al. (2026)](https://transformer-circuits.pub/2026/workspace/index.html)
étudient des représentations internes verbalisables. Notre intervention sur le
contexte ne reproduit pas leurs interventions mécanistes.

**Priorité suivante : expliquer et réduire les inventions en l'absence de données.**
Le [protocole de suivi](IPHONE_MISSING_DATA_PROTOCOL.md) distingue changement de
consigne et représentation explicite de l'absence, tout en conservant des contrôles
avec bilan et un cas avec zéro observation. Il est défini après ce résultat et
avant de nouvelles réponses ; aucune de ses performances n'est encore mesurée.

Mise à jour : ce suivi a depuis produit un
[export complet analysé](IPHONE_MISSING_DATA_RESULTS.md). Les résultats du
présent premier audit restent inchangés.

Pour l'application, une validation sémantique contre le bilan disponible devra
encadrer tout chiffre affiché comme mesure. Un schéma JSON valide ne suffit pas.
Un refus déterministe du contrôleur en l'absence de source sera une garantie du
logiciel, pas une réussite du LLM ; les deux devront rester comptabilisés séparément.
L'entraînement Colab et l'élargissement des tâches viennent après ce diagnostic.

## Reproduire l'analyse

Le [résumé agrégé](../artifacts/iphone-coupling/first-audit-summary.json) ne contient
ni identifiant personnel, ni UUID d'audit ou de sonde, ni dates brutes. Le fichier
utilisateur complet reste hors du dépôt. Le vérificateur accepte une collection
de plusieurs audits et conserve les essais prévus, échoués ou interrompus dans
ses dénominateurs ; il n'élimine pas les réponses mal formées.

```powershell
python -m research.iphone_coupling_report "chemin/vers/audits-menia.json" --manifest ios/MeniaCore/Sources/MeniaCore/Resources/qwen3-4b.json --output "resume.json"
python -m unittest tests_research.test_iphone_coupling_report tests_research.test_iphone_report -v
```

Le correcteur indépendant rejette explicitement les clés JSON dupliquées,
ambiguïté non vérifiée explicitement par le correcteur Swift v1. Aucun cas de
ce type ne figure dans les données reçues ; cette précision ne change aucun
score rapporté. Les sept nouveaux tests utilisent des données synthétiques et
contrôlent notamment les notes falsifiées, les échecs malgré un format valide,
les essais partiels, les modifications du plan et l'absence d'identifiants privés
dans les agrégats. Les **93 tests de recherche Python passent**.
