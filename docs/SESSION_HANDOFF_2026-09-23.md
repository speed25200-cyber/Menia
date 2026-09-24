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
**La prochaine demande porte le numéro 34.** La 32 a lancé l'étape 1 des graines 23 et 29, mais un défaut du relais (deux lancements du même workflow partageaient une clé) a copié le build de la graine 29 dans les deux dossiers ; corrigé (`keys_of` dans `scripts/codemagic_fetch.py`, test de non-régression), et la 33 récupère le build de la graine 23 par son identifiant. Codemagic fige le commit à la
création du build : un build lancé par une demande utilise le commit de
cette demande, même s'il démarre plus tard.

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
| 11 | `menia-inquiry-mac` | `artifacts/llm-inquiry/run-1` | **fait** : invalide (masse sur les symboles 0,014 et 0,007), `docs/LLM_INQUIRY_RESULTS.md` |
| 12 | `menia-lora-vm-rank-mac` | `artifacts/llm-lora-mac/run-4-vm` | **fait** : A3 0,793 (échoue encore), A4 passe, global non satisfait ; pas de boucle P-soi |
| 13 | `menia-menia-report-mac` | `artifacts/menia-report/run-1/menia-report` | **fait** : M2 et M3 passent, M1 échoue (« les besoins » rapportés comme « la position »), `docs/MENIA_REPORT_RESULTS.md` |
| 15 | `menia-menia-report-mac`, `MENIA_REPORT_ITEMS=items-v2.jsonl` | `artifacts/menia-report/run-2/menia-report` | **fait** : M1, M2 et M3 passent (0,997 ; 0,999 ; 0,999), global satisfait |
| 31 | `menia-lora-action-mac`, `CODE_ADAPTER=…/adapters-VML-5100` | `artifacts/llm-lora-mac/run-20-action-full` | **fait** : U1 (+5,19), U2 (+5,38), enquête 48/48 (gain 0,50) ; confiance calibrée (0,842 pour 0,848) |
| 30 | `menia-lora-stage-mac`, `STAGE_REGIME=VML` depuis l'étape C | `artifacts/llm-lora-mac/run-19-consolidation` | **fait** : Q1, Q2, Q3 passent ; code entier lu (0,82 à 0,86) |
| 29 | `menia-lora-action-mac`, `CODE_ADAPTER=…/adapters-VMLABCC-4350` | `artifacts/llm-lora-mac/run-18-action` | **fait** : U1 (+5,23) et U2 (+5,35) passent ; enquête 48/48 ; modèle de soi utile par ablation |
| 28 | `menia-lora-stage-mac`, `STAGE_REGIME=VMLABCDDD` | `artifacts/llm-lora-mac/run-17-rehearsal-d` | **fait** : K-D échoue (A, B, C effacées ; D 0,447 lue à tous les lieux) ; plan arrêté |
| 27 | `menia-lora-stage-mac`, `STAGE_REGIME=VMLABCC` | `artifacts/llm-lora-mac/run-16-rehearsal-c` | **fait** : K-C passe (A 0,881, B 0,866, C 0,869) ; D naît (0,28) |
| 26 | `menia-lora-stage-mac`, `STAGE_REGIME=VMLAB` | `artifacts/llm-lora-mac/run-15-rehearsal` | **fait** : R1 passe (A 0,829, B 0,825) ; C commence (0,30) |
| 25 | `menia-lora-stage-mac`, `STAGE_REGIME=VMLB` | `artifacts/llm-lora-mac/run-14-stage-b` | **fait** : E1 passe (B 0,789), E2 échoue (A 0,252) ; étape échouée, plan arrêté |
| 23 | `menia-lora-full-mac` (`docs/LLM_MARK_FULL_PROTOCOL.md`) | `artifacts/llm-lora-mac/run-12-full` | **fait** : C1 échoue (P1(BCD) 0,262), C2 passe (P1(A) 0,893) ; global non satisfait |
| 24 | `menia-lora-seek-mac`, `SEEK_ADAPTER=…/adapters-FULL-2100` | `artifacts/llm-lora-mac/run-13-full-seek` | **fait** : C3 passe (48 vies sur 48, gain 0,163), C4 passe |
| 22 | `menia-lora-seek-mac` (`docs/LLM_MARK_SEEK_PROTOCOL.md`) | `artifacts/llm-lora-mac/run-11-seek` | **fait** : S1 passe (lieu 1 préféré dans 48 vies sur 48, gain 0,123), S2 passe ; global satisfait |
| 21 | `menia-lora-hand-long-mac` (`docs/LLM_MARK_HAND_LONG_PROTOCOL.md`) | `artifacts/llm-lora-mac/run-10-vmla-long` | **fait** : G1 et G2 passent, P1(A) 0,312 → 0,737 → 0,811 ; global satisfait ; la marque est lue pour A, pas pour B à D |
| 20 | `menia-lora-hand-mac` (`docs/LLM_MARK_HAND_PROTOCOL.md`) | `artifacts/llm-lora-mac/run-9-vmla` | **fait** : H1 échoue (P1(A) 0,312), H3 échoue (0,56), H4 passe ; global non satisfait |
| 19 | `menia-lora-reading-mac` (`docs/LLM_MARK_READING_PROTOCOL.md`) | `artifacts/llm-lora-mac/run-8-vml` | **fait** : L0 passe (aucun adaptateur publié ne lit la marque), L1 et L2 échouent (VML 0,251), L3 passe ; global non satisfait |
| 18 | `menia-lora-weighted-mac` (amendement 1 de `docs/LLM_MOTOR_OBJECTIVE_PROTOCOL.md`) | `artifacts/llm-lora-mac/run-7-vmw` | **fait** : valide ; K1 échoue (0,44), K2 passe ; global non satisfait ; structure 0,93 ; la marque n'est pas lue (`research/llm_mark_use.py`) |
| 17 | `menia-lora-motor-mac` (`docs/LLM_MOTOR_OBJECTIVE_PROTOCOL.md`) | `artifacts/llm-lora-mac/run-6-vmm` | **fait** : invalide (symboles oubliés), erreur de conception |
| 16 | `menia-lora-vmi-mac` (`docs/LLM_INQUIRY_DATA_PROTOCOL.md`) | `artifacts/llm-lora-mac/run-5-vmi` | **fait** : valide ; J1 échoue (0,29), J2 passe, J3 +0,032 ; global non satisfait |
| 14 | `menia-inquiry-mac` (amendement 2) | `artifacts/llm-inquiry/run-2/llm-inquiry` | **fait** : valide ; I1 échoue (0,40), I2 passe, I3 échoue ; `docs/LLM_INQUIRY_RESULTS.md` |

## Version 6 de l'agent

Protocole `docs/INDICATOR_AGENT_V6_PROTOCOL.md` (commit `3c33994`), code
`research/indicator_agent_v6.py` (commit `87bfce8`). Graine de
développement 23 (au plus trois essais, déclarés dans le protocole), puis
`python -m research.indicator_experiment --version 6 --jobs 3` sur 151,
157 et 163, audit complet (`verification.json`), seconde lecture,
contrôles CI, `docs/INDICATOR_AGENT_V6_RESULTS.md`.

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
