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
   Butlin, Long et collaborateurs. Un agent dans l'Atelier des sens les
   réunit toutes par construction ; trois versions pré-enregistrées, neuf
   graines confirmatoires, audits complets sans écart :
   - v1 (`docs/INDICATOR_AGENT_RESULTS.md`) : 6/14 ;
   - v2 (`docs/INDICATOR_AGENT_V2_RESULTS.md`) : 7/14, GWT-1 s'ajoute ;
   - v3 (`docs/INDICATOR_AGENT_V3_RESULTS.md`) : 6/14, prédiction réfutée.
   **Robustes dans les trois** : RPT-2, GWT-2, HOT-1, HOT-2, AST-1, PP-1.
   **Point faible** : l'arbitrage des buts appris (GWT-1, GWT-4, AE-1).
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

- Récupérer les trois builds (le relais les pousse seul), écrire
  `docs/ADJUSTED_BODY_RESULTS.md`, compléter `docs/LATENT_BODY_RESULTS.md`,
  écrire `docs/LLM_REPORT_RESULTS.md`, avec les verdicts pré-enregistrés.
- Si le corps ajusté passe A3 et A4 : pré-enregistrer la boucle P-soi sur le
  LLM ajusté, puis la recette sur Qwen3-4B.
- Indicateurs : une version 4 donnerait aux alarmes intéroceptives un accès
  prioritaire à l'espace de travail (théoriquement fondé, et c'est ce que
  l'analyse de la règle fixe désigne), jugée aux mêmes seuils sur de
  nouvelles graines. Le volet « apprentissage » d'AE-1 resterait difficile :
  la politique de départ est déjà bonne.
- La recette sur Qwen3-4B ne tient pas dans un build de 120 minutes sur ce
  Mac ; il faudrait l'A100 de Colab.
- Rappel constant : aucun de ces résultats n'établit une expérience vécue.

## Pièges

- Pas de torch ni de HuggingFace dans la session : tout le calcul local est
  en numpy, les LLM tournent sur le Mac.
- Le Mac virtuel a 10 à 15 Go : un ajustement LoRA à lots de 4 sur 1 024
  tokens y a planté ; garder `--grad-checkpoint`.
- Les audits complets de l'agent à indicateurs rejouent 27 000 vies :
  8 à 15 minutes ; la CI ne rejoue que 2 vies par jeu et recalcule les
  verdicts.
