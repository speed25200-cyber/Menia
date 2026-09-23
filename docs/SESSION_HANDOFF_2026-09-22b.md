# Passation de session — 22 septembre 2026, soir

Suite de `docs/SESSION_HANDOFF_2026-09-22.md`. Branche de travail de cette
session : `claude/causal-self-model-llm-results-mauabk`, qui contient toute
la branche `claude/codex-repo-analysis-ycbook`. Mêmes règles : protocole
committé avant exécution, aucun seuil retouché après lecture, lectures de
développement déclarées, audit indépendant en CI, réponses courtes en
français au propriétaire.

## Accès au Mac de Codemagic depuis la session

Le proxy de la session refuse `api.codemagic.io`. Le relais passe par
GitHub Actions :

- `.github/workflows/codemagic-fetch.yml`, déclenché par toute modification
  de `.github/codemagic-fetch.json` sur une branche. Il utilise le secret
  `CODEMAGIC_API_TOKEN` (ajouté par le propriétaire). Une entrée `tag`
  récupère un build existant ; une entrée `launch` (`workflow`, `branch`,
  `variables`) **démarre un build par l'API**, sans webhook. Le relais
  attend jusqu'à `wait_minutes`, rapatrie artefacts, statut et fin des
  journaux dans `dest`, puis pousse sur la branche. Incrémenter `request` à
  chaque nouvelle demande.
- Les builds Codemagic passent un par un : un build lancé attend la fin des
  précédents.

## Ce qui a été fait

1. **Corps latent** (`docs/LATENT_BODY_RESULTS.md`) : en conversation,
   validité échouée (0,43) ; **relu en complétion brute** (validité 0,99) :
   L1 échoue de peu (copie 0,64), L2 passe (0,16 sur les commandes
   nouvelles), L3 échoue (0,53), L4 et L5 passent ; global non satisfait.
2. **Corps ajusté** (`docs/ADJUSTED_BODY_RESULTS.md`, historique) : trois
   échecs techniques (deux dépassements de 120 minutes, une GPU Hang Error) ;
   base seule lue (validité 0,99, A1 passe). **Relance en deux builds**
   `menia-lora-f-mac` et `menia-lora-vm-mac`, résultats attendus dans
   `artifacts/llm-lora-mac/run-3-f` et `run-3-vm`, verdicts avec
   `python -m research.llm_body_verdicts --lora run-3-f run-3-vm`. Ensuite,
   lancer `menia-inquiry-mac` (protocole `docs/LLM_INQUIRY_PROTOCOL.md`,
   pré-enregistré) dès que les adaptateurs existent.
3. **Programme des indicateurs de conscience**
   (`docs/CONSCIOUSNESS_INDICATORS_ROADMAP.md`) : les quatorze propriétés de
   Butlin, Long et collaborateurs, réunies par construction dans un agent de
   l'Atelier des sens ; cinq versions pré-enregistrées, quinze graines
   confirmatoires, audits complets sans écart :
   - v1 à v3 : 6, 7 et 6 propriétés sur 14 ;
   - v4 (`docs/INDICATOR_AGENT_V4_RESULTS.md`) : 6/14 ; alarmes prioritaires
     et valeurs contre la pulsion, l'arbitrage n'est pas appris ;
   - **v5 (`docs/INDICATOR_AGENT_V5_RESULTS.md`) : 10/14.** L'agent apprend
     un modèle de ses besoins et les simule sur 16 pas avant de choisir ; la
     recharge se déplace après usage (amendement déclaré). AE-1 et AE-2
     passent pour la première fois, avec GWT-1, GWT-2, GWT-3 et les cinq
     robustes (RPT-2, HOT-1, HOT-2, AST-1, PP-1).
   Restent : RPT-1, GWT-4, HOT-3, HOT-4 (diagnostics dans les résultats v5).
4. **Rapport verbal** (`docs/LLM_REPORT_PROTOCOL.md`) : Qwen3-4B lit le
   journal intérieur de l'agent ; 1 854 items ; le premier build s'est
   arrêté dans les tests (arrondi ARM), relancé ; résultats attendus dans
   `artifacts/llm-report/run-2`.
5. **Analyses exploratoires**, déclarées : le goulot atténue l'effet des
   lésions sans l'expliquer entièrement (`masking.json`) ; une règle fixe
   des besoins passe la moitié « objet » d'AE-1 mais pas la recharge, parce
   que l'énergie est lue dans un espace de travail limité
   (`rule-arbitration.json`). Bilan simple : `docs/CONSCIOUSNESS_PROGRAM_BILAN.md`.

## À faire ensuite

- **Corps ajusté** : l'ajustement F est fini (23 h 30 – 0 h 51), le VM
  tourne depuis 0 h 51 ; le relais (demande 7) pousse `run-3-f` et
  `run-3-vm` ensemble. Verdicts :
  `python -m research.llm_body_verdicts --lora artifacts/llm-lora-mac/run-3-f artifacts/llm-lora-mac/run-3-vm`,
  puis `docs/ADJUSTED_BODY_RESULTS.md` et une vérification en CI.
- **Test d'enquête** : son amendement (chemins des adaptateurs) est écrit ;
  le lancer par le relais (`menia-inquiry-mac`, destination
  `artifacts/llm-inquiry/run-1`) dès que les adaptateurs sont sur la branche.
- **Rapport verbal** : build en file après le VM ; résultats dans
  `artifacts/llm-report/run-2`, puis `docs/LLM_REPORT_RESULTS.md`.
- Si le corps ajusté passe A3 et A4 : pré-enregistrer la boucle P-soi sur
  le LLM ajusté, puis la recette sur Qwen3-4B (A100 de Colab nécessaire).
- **Indicateurs** : une version 6 pour RPT-1, GWT-4, HOT-3 et HOT-4, en ne
  changeant le monde que là où il empêche le test de mesurer (plus d'objets
  pour que la mémoire compte, pannes plus fréquentes pour que le moniteur
  compte), mêmes critères, nouvelles graines.
- La demande suivante du relais porte le numéro 10.
- Rappel constant : aucun de ces résultats n'établit une expérience vécue.

## Pièges

- Pas de torch ni de HuggingFace dans la session : tout le calcul local est
  en numpy, les LLM tournent sur le Mac.
- Le Mac virtuel a 10 à 15 Go : un ajustement LoRA à lots de 4 sur 1 024
  tokens y a planté ; garder `--grad-checkpoint`.
- Les audits complets de l'agent à indicateurs rejouent 27 000 vies :
  8 à 15 minutes ; la CI ne rejoue que 2 vies par jeu et recalcule les
  verdicts.
