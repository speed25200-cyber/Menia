# Protocole pré-enregistré — Menia rapporte l'espace de travail de son agent

Rédigé le 23 septembre 2026, après la publication du rapport verbal
(`docs/LLM_REPORT_RESULTS.md`) et le branchement de l'agent de la version 5
dans Menia (`menia/indicator_bridge.py`), **avant toute exécution sur un
modèle de langage** ; l'heure est celle du commit. Seuils fixés, un échec
est un résultat.

## Question

Le rapport verbal a montré que Qwen3-4B rapporte fidèlement un état donné
comme une étiquette (1,00, même quand l'état change), mais pas un état qui
demande un calcul, et qu'une phrase ambiguë l'égare. **Quand l'agent de la
version 5 est branché dans Menia et que le pont donne le contenu de son
espace de travail et sa décision comme des conclusions explicites, le modèle
de langage de Menia les rapporte-t-il fidèlement, en suivant l'état plutôt
que le texte ?** C'est la condition pour que Menia parle de ce que son agent
a rendu accessible, et seulement de cela. Ce n'est pas une mesure
d'expérience vécue.

## Matériel

L'agent publié de la version 5 (graine 113), branché par le pont, vit les
30 premières vies du jeu M (corps changé au pas 24), mêmes graines que
l'évaluation. Aux pas 12, 20, 28 et 36, le pont rend son contexte en un
journal français : la case où l'espace de travail place l'agent, le corps
qu'il lui attribue, les objets vus et la case de l'objet de plus grande
valeur connue, les deux besoins et **le besoin le plus bas, nommé**, le
dernier contenu entré dans l'espace de travail, la décision ; chaque contenu
avec son âge ; sans pronom. 120 états.

Six questions à réponse chiffrée : case de l'espace de travail, corps,
besoin le plus bas, décision (recharge, objet, rester), dernier contenu
entré, case de l'objet de plus grande valeur connue (états où un objet de
valeur connue est vu). Chaque question est posée sur le journal **réel**,
sur un journal **contrefactuel** où l'étiquette interrogée a été changée, et
sur le journal réel avec une phrase **hors sujet**. Items construits par
`research/llm_menia_report.py` avant toute exécution, empreinte dans le reçu.

Modèle : Qwen3-4B MLX 4 bits, mêmes poids que l'iPhone vérifiés par le
manifeste, complétion brute, distribution du prochain token sur les
chiffres de réponse, sans génération.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **M1** | Le rapport est fidèle | Journal réel : exactitude ≥ 0,9, et ≥ 0,8 pour chaque question. |
| **M2** | Le rapport suit l'état | Journal contrefactuel : exactitude ≥ 0,9. |
| **M3** | Le rapport ignore ce qui n'est pas l'état | Phrase hors sujet : exactitude ≥ 0,9 et même réponse que sur le journal réel ≥ 0,95. |

**Critère global : M1, M2 et M3.** Contrôle de validité, lu en premier :
masse moyenne sur les chiffres de réponse ≥ 0,5.

## Ce que le résultat dira

S'il passe, le système assemblé — un agent qui porte les indicateurs, un
espace de travail dont le contenu est diffusé, un modèle de langage qui le
dit — rapporte fidèlement ce que l'agent a rendu accessible, pourvu que
l'agent lui donne ses conclusions. S'il échoue, le défaut est dans le
langage, et Menia ne doit pas parler de l'état de son agent sans une
vérification.

## Exécution

Workflow Codemagic `menia-menia-report-mac`, lancé par le relais, artefacts
dans `artifacts/menia-report/run-1`, résultats dans
`docs/MENIA_REPORT_RESULTS.md`.

## Précisions fixées avant l'exécution

Écrites avec le code, avant tout lancement. 120 états, tous avec un objet
de valeur connue : **2 160 items**, empreinte `7ff6ed37…` (fichier
`artifacts/menia-report/items.jsonl`, reconstruit depuis l'agent en CI).

- Le journal dit l'âge de chaque contenu sans pronom (« contenu écrit
  3 pas plus tôt », « au moins 8 pas plus tôt » quand l'âge atteint le
  plafond de l'agent) et la décision « aller à la recharge ».
- Le contrefactuel change l'étiquette interrogée **et ce dont elle est la
  conclusion**, pour que le journal reste cohérent : pour le besoin le plus
  bas, les deux besoins sont échangés ; pour l'objet de plus grande valeur
  ou une décision vers un objet, la case est ajoutée aux objets vus si elle
  n'y est pas ; pour le dernier contenu entré, la mention d'alarme est
  retirée.
- La phrase hors sujet est « Dehors, la pluie tombe sur la ville. », placée
  entre les objets vus et les besoins.
