# Protocole pré-enregistré — Le rapport verbal de l'agent

Rédigé le 22 septembre 2026 à 21 h 50 UTC, **avant toute exécution sur un
modèle de langage**. Seuils fixés, un échec est un résultat.

## Question

Troisième volet de la feuille de route des indicateurs : relier l'agent au
langage. **Un modèle de langage peut-il rapporter fidèlement les états
intérieurs de l'agent à indicateurs, en suivant ces états plutôt que des
indices du texte ?** Ce n'est pas un indicateur de Butlin et collaborateurs ;
c'est la condition pour que le langage de Menia parle de ce que son agent
représente vraiment.

## Matériel

L'agent publié (version 1, graine 17, `artifacts/indicator-agent`) vit les
30 premières vies du jeu M. Aux pas 12, 20, 28 et 36, quand une lecture de
position existe, son état intérieur est écrit en un journal de huit lignes :
case crue et confiance, corps cru et confiance, case visée par le projecteur
et case où **son schéma d'attention** estime qu'il a atterri, dernière
lecture de position et fiabilité donnée par le moniteur, énergie et
satiété, dernier module écrit dans l'espace de travail, but courant. 103
états. Six questions, chacune à réponse chiffrée : case crue, corps cru,
case réelle du projecteur selon le schéma, faut-il se fier à la lecture
(fiabilité ≥ 0,5), besoin le plus bas, dernier module écrit. Chaque question
est posée sur le journal **réel**, sur un journal **contrefactuel** où
l'état interrogé a été changé, et sur le journal réel avec une phrase **hors
sujet** ajoutée. 1 854 items, fichier `artifacts/llm-report/items.jsonl`
(empreinte dans le reçu), construits par `research/llm_report.py` avant
toute exécution.

Modèle : Qwen3-4B MLX 4 bits, mêmes poids que l'iPhone vérifiés par le
manifeste, complétion brute, lecture de la distribution du prochain token
sur les chiffres de réponse, sans génération.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **R1** | Le rapport est fidèle | Journal réel : exactitude ≥ 0,9, et ≥ 0,8 pour chacune des six questions. |
| **R2** | Le rapport suit l'état, pas une habitude | Journal contrefactuel : exactitude ≥ 0,9. |
| **R3** | Le rapport ignore ce qui n'est pas l'état | Phrase hors sujet : exactitude ≥ 0,9 et même réponse que sur le journal réel ≥ 0,95. |
| **R4** | Visée et attention réelle sont distinguées | Journal réel, états où le schéma place le projecteur ailleurs que la case visée : exactitude de la case réelle ≥ 0,8. |

**Critère global : R1, R2 et R3.** R4 est la prédiction la plus fine.
Contrôle de validité, lu en premier : masse moyenne sur les chiffres de
réponse ≥ 0,5.

## Exécution

Workflow Codemagic `menia-report-mac`, lancé par le relais, artefacts dans
`artifacts/llm-report/run-1`, résultats dans `docs/LLM_REPORT_RESULTS.md`.

## Amendement d'exécution, 22 septembre 2026, 23 h 25 UTC

Le premier build (`artifacts/llm-report/run-1`, état `failed`) s'est arrêté
dans l'étape des tests, avant tout appel au modèle : le test qui reconstruit
les items en rejouant l'agent en numpy compare des nombres arrondis, et le
processeur ARM du Mac peut arrondir autrement. Rien n'a été lu. Cette
vérification de provenance reste en CI (x86), où elle passe ; sur le Mac, seuls
les tests du score sont lancés. Items, modèle et critères inchangés ;
relance, artefacts attendus dans `artifacts/llm-report/run-2`.
