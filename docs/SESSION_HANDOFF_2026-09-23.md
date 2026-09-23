# Passation de session — 23 septembre 2026

Suite de `docs/SESSION_HANDOFF_2026-09-22b.md`. Branche :
`claude/causal-self-model-llm-results-mauabk`. Mêmes règles : protocole
committé avant exécution, aucun seuil retouché après lecture, lectures de
développement déclarées, audit indépendant en CI, pas de PR sans demande,
réponses courtes en français au propriétaire.

## Accès au Mac de Codemagic

Le relais `.github/workflows/codemagic-fetch.yml` se déclenche à chaque
modification de `.github/codemagic-fetch.json`. Entrées : `launch`
(démarre un build par l'API), `tag`, `build` (identifiant d'un build).
Il attend la fin de tous les builds (`wait_minutes`, 345 au plus), rapatrie
les artefacts dans `dest` et pousse sur la branche. Il ne retire que les
préfixes `llm-latent`, `llm-lora` et `llm-atelier` : les autres artefacts
arrivent sous `dest/<nom du dossier>`. Les builds passent un par un.
**La prochaine demande porte le numéro 14.**

## Où en est le programme

- **Agent à indicateurs** (`docs/CONSCIOUSNESS_PROGRAM_BILAN.md`) :
  version 5, 10 propriétés sur 14 (`docs/INDICATOR_AGENT_V5_RESULTS.md`) ;
  sur de nouvelles graines, **9 en lecture principale, 11 en seconde
  lecture** (`docs/INDICATOR_AGENT_SECOND_READING_RESULTS.md`). Neuf
  propriétés robustes sur six graines ; AE-1 à la limite (recharge 0,78
  en moyenne pour un seuil de 0,8) ; HOT-3 et HOT-4 jamais démontrées.
  Analyses exploratoires et erratum de HOT-4 :
  `docs/INDICATOR_AGENT_V5_EXPLORATIONS.md`.
- **Corps ajusté** (`docs/ADJUSTED_BODY_RESULTS.md`) : A1, A2 et A4
  passent ; A3 échoue d'un cas (0,793 pour 0,80) ; budget déclaré
  insuffisant ; **relance VM au rang 16** en cours.
- **Rapport verbal** (`docs/LLM_REPORT_RESULTS.md`) : Qwen3-4B rapporte
  fidèlement les états donnés comme étiquettes, pas ceux qui demandent un
  calcul ; R4 passe, global non satisfait.
- **Menia branchée sur l'agent** : `menia/indicator_bridge.py` donne au
  modèle de langage l'espace de travail et la décision, en conclusions
  explicites sans pronom (`journal`). Rapport pré-enregistré
  (`docs/MENIA_REPORT_PROTOCOL.md`), 2 160 items committés
  (`artifacts/menia-report/items.jsonl`, empreinte `7ff6ed37…`).

## Builds en cours (lancés par le relais)

| Demande | Workflow | Destination | Suite |
|---|---|---|---|
| 11 | `menia-inquiry-mac` | `artifacts/llm-inquiry/run-1` | verdicts I1–I3 par `research.llm_inquiry`, puis `docs/LLM_INQUIRY_RESULTS.md` et un contrôle CI |
| 12 | `menia-lora-vm-rank-mac` | `artifacts/llm-lora-mac/run-4-vm` | A3 et la part VM de A4 ; verdict global avec A1 et A2 de run-3 ; ajouter à `docs/ADJUSTED_BODY_RESULTS.md` ; si A3 passe, pré-enregistrer la boucle P-soi |
| 13 | `menia-menia-report-mac` | `artifacts/menia-report/run-1/menia-report` | `python -m research.llm_menia_report verdicts --rows …/rows.jsonl --check …/summary.json`, puis `docs/MENIA_REPORT_RESULTS.md` et un contrôle CI |

## Pièges

- **Python local 3.11, CI en 3.12** : `sum()` des flottants a changé en
  3.12, si bien que `scripts/check_agent_artifacts.py` ne se reproduit
  qu'avec `/usr/bin/python3.12` (sans numpy). Toute modification de
  `menia/conversation.py`, `menia/core.py` ou `menia/agent.py` oblige à
  rejouer l'agent intégré sous 3.12 et à mettre à jour son empreinte.
- Les tests qui rejouent l'agent en numpy (classes `…ProvenanceTests`)
  tournent en CI, pas sur le Mac (arrondis ARM).
- Les audits complets de l'agent rejouent 27 000 vies (10 à 20 minutes).
- Ne jamais écrire l'URL du webhook Codemagic ni un jeton dans le dépôt
  (dépôt public).
- Rappel constant : aucun de ces résultats n'établit une expérience vécue.
