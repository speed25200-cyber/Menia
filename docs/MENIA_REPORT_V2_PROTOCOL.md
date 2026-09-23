# Protocole pré-enregistré — Menia rapporte l'espace de travail de son agent, deuxième version

Rédigé le 23 septembre 2026, après la publication du premier rapport
(`docs/MENIA_REPORT_RESULTS.md`), **avant la construction des items et
toute exécution de cette version** ; l'heure est celle du commit. Seuils
inchangés, un échec est un résultat.

## Ce qui a été vu

Le premier rapport a passé M2 et M3 et échoué M1 sur une question : quand
le dernier contenu entré dans l'espace de travail est « les besoins »,
Qwen3-4B répond « la position » (3 réponses justes sur 45, 0 sur 26 dans le
contrefactuel), alors que la position, le corps et la vision sont rapportés
sans erreur. Et les 11 erreurs sur la case de l'agent donnent toutes la
case de l'objet de plus grande valeur connue, dont la ligne dit aussi « sur
la case N ». Dans le rapport verbal (`docs/LLM_REPORT_RESULTS.md`), la même
information donnée par le nom du module (« Dernier module écrit dans
l'espace de travail : Intéro », options « 1 pour Pos, 2 pour Corps, 3 pour
Vision, 4 pour Intéro ») était rapportée à 1,00 dans les trois variantes.

## Ce qui change, et rien d'autre

1. **Le dernier contenu entré est nommé par son module** : « Dernier module
   entré dans l'espace de travail : Intéro. » (avec « (par une alarme) »
   comme avant), et la question devient « quel module est entré en dernier
   dans l'espace de travail ? Réponds 1 pour Pos, 2 pour Corps, 3 pour
   Vision, 4 pour Intéro. »
2. **La position de l'agent a sa propre étiquette** : « Position de l'agent
   selon l'espace de travail : case 7 (contenu écrit 1 pas plus tôt). », et
   la question devient « quelle est la position de l'agent selon l'espace
   de travail ? Réponse : case ».
3. **L'agent est celui de la version 6** (graine 151,
   `docs/INDICATOR_AGENT_V6_RESULTS.md`), que Menia branche désormais par
   défaut ; il vit les 30 premières vies du jeu M avec les graines de
   l'évaluation, et le pont rend son contexte aux pas 12, 20, 28 et 36 :
   120 états nouveaux.

Les autres lignes du journal, les quatre autres questions, les trois
variantes (réel, contrefactuel, hors sujet), la construction des
contrefactuels, le modèle (Qwen3-4B MLX 4 bits, poids de l'iPhone vérifiés
par le manifeste, complétion brute, distribution sur les chiffres de
réponse) et les critères sont **identiques au premier rapport** :

| | Prédiction | Critère |
|---|---|---|
| **M1** | Le rapport est fidèle | Journal réel : exactitude ≥ 0,9, et ≥ 0,8 pour chaque question. |
| **M2** | Le rapport suit l'état | Journal contrefactuel : exactitude ≥ 0,9. |
| **M3** | Le rapport ignore ce qui n'est pas l'état | Phrase hors sujet : exactitude ≥ 0,9 et même réponse que sur le journal réel ≥ 0,95. |

**Critère global : M1, M2 et M3.** Contrôle de validité, lu en premier :
masse moyenne sur les chiffres de réponse ≥ 0,5. **Prédiction : les trois
passent**, et chacune des six questions atteint 0,9 sur le journal réel.

## Ce que le résultat dira

S'il passe, Menia dit fidèlement ce que l'espace de travail de son agent
contient, pourvu que chaque ligne porte l'étiquette que la question
emploie : l'échec du premier rapport tenait à l'écriture, pas au modèle. S'il
échoue encore, le défaut n'est pas d'écriture, et Menia doit vérifier chaque
réponse sur le journal. Aucune troisième version n'est prévue. Ce test ne
mesure pas une expérience vécue.

## Exécution

Items construits par `python -m research.llm_menia_report build --journal 2`
avant toute exécution (`artifacts/menia-report/items-v2.jsonl`, empreinte
dans le reçu) ; workflow Codemagic `menia-menia-report-mac` avec ces items,
lancé par le relais, artefacts dans `artifacts/menia-report/run-2`,
résultats ajoutés à `docs/MENIA_REPORT_RESULTS.md`.

## Précisions fixées avant l'exécution

Écrites avec le code, avant tout lancement. 120 états, dont 118 avec un
objet de valeur connue : **2 154 items**, empreinte `d26bfb4c…` (fichier
`artifacts/menia-report/items-v2.jsonl`, reconstruit depuis l'agent de la
version 6 en CI). Les items de la première version sont inchangés. Le
workflow lit le fichier d'items dans la variable `MENIA_REPORT_ITEMS`, que
le relais fixe à ce fichier.
