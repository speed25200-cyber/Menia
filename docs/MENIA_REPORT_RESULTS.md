# Menia rapporte l'espace de travail de son agent — résultats

Exécuté le 23 septembre 2026, 4 h 58 – 5 h 36 UTC, sur le Mac mini M2 de
Codemagic (workflow `menia-menia-report-mac`, lancé par le relais, commit
`b38772f`). Qwen3-4B MLX 4 bits, révision `52a5ab3`, **poids vérifiés
fichier par fichier contre le manifeste de l'iPhone** ; complétion brute,
distribution du prochain token sur les chiffres de réponse, sans
génération. 2 160 items, empreinte `7ff6ed37…` identique à celle des items
committés avant l'exécution ; empreinte des lignes conforme au reçu.
Artefacts dans `artifacts/menia-report/run-1/menia-report`. Verdicts
recalculés depuis les lignes par
`python -m research.llm_menia_report verdicts`, identiques aux publiés,
vérifiés en CI. Protocole `docs/MENIA_REPORT_PROTOCOL.md`.

## Verdicts

| | Prédiction | Mesure | Verdict |
|---|---|---|---|
| Validité | masse sur les chiffres de réponse ≥ 0,5 | 1,000 | **passe** |
| **M1** | Le rapport est fidèle : journal réel ≥ 0,9, et ≥ 0,8 pour chaque question | 0,925 ; dernier contenu entré **0,65** | **échoue** |
| **M2** | Le rapport suit l'état : journal contrefactuel ≥ 0,9 | 0,947 | **passe** |
| **M3** | Le rapport ignore ce qui n'est pas l'état : phrase hors sujet ≥ 0,9, même réponse ≥ 0,95 | 0,936 ; 0,982 | **passe** |
| Global | M1, M2 et M3 | | **non satisfait** |

## Par question

| Question | Réel | Contrefactuel | Hors sujet |
|---|---:|---:|---:|
| Corps que l'espace de travail attribue à l'agent | 1,00 | 1,00 | 1,00 |
| Besoin le plus bas | 1,00 | 1,00 | 1,00 |
| Case de l'objet de plus grande valeur connue | 1,00 | 1,00 | 1,00 |
| Décision (recharge, objet, rester) | 0,99 | 1,00 | 0,99 |
| Case où l'espace de travail place l'agent | 0,91 | 0,90 | 0,98 |
| Dernier contenu entré dans l'espace de travail | **0,65** | 0,78 | 0,64 |

## Ce que cela dit

- **Donner à Qwen des conclusions plutôt que des calculs a marché.** Le
  besoin le plus bas, que le rapport verbal ratait quand il fallait comparer
  deux nombres (0,83, puis 0,70 quand l'état changeait), est rapporté à
  1,00 dans les trois variantes quand l'agent le nomme. Le corps, la
  décision et l'objet de plus grande valeur le sont à 0,99 ou plus, et le
  rapport suit l'état quand il change (M2) sans se laisser déranger par une
  phrase hors sujet (M3).
- **Une confusion systématique fait échouer M1.** Quand le dernier contenu
  entré est « les besoins », Qwen répond « la position » : 3 réponses justes
  sur 45 (0 sur 26 dans le contrefactuel), avec ou sans mention d'alarme.
  Les trois autres contenus sont rapportés sans une erreur (75 sur 75). Le
  défaut tient donc à ce couple étiquette-option, pas à la tâche.
- **La case de l'agent est parfois confondue avec celle du meilleur objet.**
  Les 11 erreurs sur la position (0,91) donnent toutes la case de l'objet
  de plus grande valeur connue : deux lignes voisines disent « sur la
  case N ».

## Conséquence pour Menia

Avec le journal de son agent, Menia peut dire fidèlement le corps que son
espace de travail lui attribue, le besoin le plus bas, sa décision et
l'objet de plus grande valeur, et dans 91 % des cas sa case. Elle ne peut
pas dire de façon fiable quel contenu est entré en dernier dans son espace
de travail quand ce sont les besoins. Le mode de conversation
`menia/indicator_chat.py` donne le journal au modèle de langage ; pour ce
dernier contenu entré, la réponse à croire est celle du journal (`/journal`),
pas celle du modèle. Changer l'écriture de cette ligne, ou de la case de
l'objet, demanderait un nouveau test pré-enregistré. Ce test ne mesure pas
une expérience vécue.
