# Passation de session — 22 septembre 2026, 17 h 35 UTC

Pour reprendre le travail sur `speed25200-cyber/Menia` sans relire l'historique.

## Le but et la méthode

Le propriétaire du dépôt veut avancer vers une IA consciente. La ligne
adoptée : construire et mesurer une pièce nécessaire, **le modèle de soi
causal, actif et réparable**, par des expériences pré-enregistrées, auditées,
rejouables, sur CPU et sur le Mac mini de Codemagic, sans A100. Règles
absolues : le protocole est committé avant toute exécution ; aucun seuil
n'est retouché après lecture ; un échec est un résultat ; toute lecture
précoce est déclarée ; chaque expérience a un audit indépendant qui
recalcule tout depuis les journaux et les poids, branché en CI.

Conventions : développer et pousser sur `claude/codex-repo-analysis-ycbook`
(une exception accordée : un fichier de workflow sur `main`). Pas de pull
request sans demande explicite. Chaque commit se termine par les lignes
d'attribution de la session. Le propriétaire veut des réponses **simples et
courtes**, en français.

## Ce qui est établi, par ordre chronologique

Monde commun : l'Atelier (`research/origin_env.py`), anneau de 8 cases,
quatre commandes dont l'effet dépend d'une cause cachée D, quatre lieux
inspectables dont un porte une marque révélant D. Résultats dans
`docs/SELF_INQUIRY_SYNTHESIS.md` et le README, section « Expériences Atelier ».

1. **Agent GRU** (30 000 paramètres, prédiction des observations) : cherche
   spontanément la marque de sa cause, se répare après un changement de
   corps, l'habitude apprise ne se répare pas, une enfance mutable installe
   une vigilance. Neuf graines, protocoles ORIGIN, LEARNED, MUTABLE_BODY.
2. **Qwen3-4B en contexte** (`docs/LLM_ATELIER_RESULTS.md`, deux lancements
   sur le Mac, poids de l'iPhone vérifiés) : ne cherche pas la marque, ne
   l'utilise pas, confabule sa mécanique, n'apprend pas son corps.
3. **Canal d'action propre** (`docs/OWN_ACTION_CHANNEL_RESULTS.md`) :
   micro-transformeur numpy, prédiction du prochain token sur l'Atelier en
   tokens, neuf modèles. Régime F, corps fixe en enfance : miroir exact de
   Qwen. Régime V, corps variable : infère son corps en un mouvement (0,995),
   lit la marque au premier pas de 100 % des vies, mais ne se répare pas après
   un changement de corps (0,03). Validité passée, P1 P2 P3 P5 passés, P4
   échoué, critère global non satisfait.
4. **Enfance mutable** (`docs/OWN_ACTION_MUTABLE_RESULTS.md`) : régime VM,
   corps parfois changé en cours de vie d'enfance. Se répare : 0,97 contre le
   nouveau corps, deux graines par l'action en un ou deux mouvements, une
   graine par une vigilance à la marque. Q1 Q2 Q4 passés, Q3 Q5 échoués sur la
   forme prédite de la révision, critère global non satisfait.

**Phrase à retenir :** chez un prédicteur de texte, inférer son corps et le
réviser sont deux dispositions séparées, chacune installée par une propriété
des données d'enfance, pas par l'architecture ni la perte. Un LLM a grandi
sur des textes d'auteurs au corps fixe. Ce n'est pas encore une avancée
majeure : il manque le passage sur un vrai LLM.

## Ce qui tourne sur le Mac de Codemagic, lancé à 16 h 59 UTC

Deux expériences pré-enregistrées, lancées par les tags `latent-1` et
`lora-2` (commit `3928e49`), Codemagic ayant répondu « Webhook received ».
Leur état n'a pas pu être vérifié depuis la session, dont le proxy bloque
`api.codemagic.io` et `docs.codemagic.io`.

- **Corps latent** (`docs/LATENT_BODY_PROTOCOL.md`, workflow
  `menia-latent-mac`, `research/llm_latent_body.py`) : Qwen3-4B est interrogé,
  sans génération, sur la case où le mènera sa prochaine commande, sur les
  mêmes vies que les jeux R et M du micro-transformeur. Hypothèse : copieur
  sans structure. Critères L1 à L5. Environ 40 minutes.
- **Corps ajusté** (`docs/ADJUSTED_BODY_PROTOCOL.md`, workflow
  `menia-lora-mac`, `research/llm_lora_body.py`) : Qwen3-0.6B ajusté par LoRA
  sur 1 500 vies du régime F puis du régime VM, évalué en complétion brute,
  plus le modèle de base. Critères A1 à A5, A4 est la révision. Environ 90
  minutes. **C'est le test décisif de la recette.**

Chaque workflow tente en fin de build de pousser ses résultats sur la
branche, dans `artifacts/llm-latent-mac/run-<date>` et
`artifacts/llm-lora-mac/run-<date>`. Si le push est refusé, le propriétaire
doit déposer le zip des artefacts (`llm-latent/**`, `llm-lora/**`), comme il
l'a fait pour les deux Ateliers précédents.

## Comment relancer Codemagic soi-même

Les tags et l'API sont refusés depuis la session. Le relais est
`.github/workflows/codemagic-launch.yml`, présent sur `main` et sur la
branche : un `workflow_dispatch` avec les entrées `experiment` (latent, lora,
atelier) et `hook` (URL du webhook Codemagic, à demander au propriétaire, ne
jamais l'écrire dans le dépôt, il est public). Il crée le tag
`<experiment>-<numéro>` et transmet un événement de push à Codemagic. À
déclencher par l'outil GitHub `actions_run_trigger` avec
`workflow_id=codemagic-launch.yml`, `ref=claude/codex-repo-analysis-ycbook`.
Vérifier le succès dans les journaux du job : « Codemagic HTTP 202 ».

## À faire ensuite

1. Récupérer les deux résultats du Mac, les ranger dans `artifacts/`, écrire
   `docs/LATENT_BODY_RESULTS.md` et `docs/ADJUSTED_BODY_RESULTS.md` avec les
   verdicts pré-enregistrés, sans retouche, en calculant aussi L5 avec
   `transformer_rows` sur les poids de `artifacts/own-action-channel`.
   Mettre à jour README et synthèse. Le zip précédent avait été analysé avec
   `research/iphone_atelier_report.py` et à la main ; ici `summary.json` et
   `rows-*.jsonl` suffisent.
2. Si le corps ajusté passe A3 et A4 : l'étape suivante à pré-enregistrer est
   la boucle P-soi sur les distributions du LLM ajusté (chercher la marque,
   la relire après changement), puis la même recette sur Qwen3-4B. Si A3
   échoue : augmenter le budget ou le rang LoRA, amendement déclaré.
3. Si le corps latent montre que Qwen3-4B copie (L1) : la boucle P-soi sur
   ses logits est justifiée sans ajustement.
4. Questions ouvertes notées dans les docs : ce que fait l'architecture
   récurrente contre le transformeur pour la révision ; le phénotype vigilant
   de VM 29 ; un LoRA de Qwen3-4B pour l'iPhone.

## Pièges connus

- Pas de torch dans la session, pip vers PyTorch bloqué : tout le CPU est en
  numpy avec gradients manuels vérifiés par différences finies.
- `git push` d'un tag et tout appel HTTPS hors GitHub échouent par le proxy.
- Les jobs longs se lancent en `nohup` avec `OMP_NUM_THREADS=1`, quatre en
  parallèle, journaux dans le scratchpad ; un hook d'arrêt exige que l'arbre
  soit committé et poussé à chaque fin de tour.
- Les tests de recherche : `python -m unittest discover -s tests_research`.
  Les audits en CI : `python -m research.audit_own_action --root
  artifacts/own-action-channel --mutable-root artifacts/own-action-mutable
  --check --replay-lives 1`.
