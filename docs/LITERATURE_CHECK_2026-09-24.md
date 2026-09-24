# Ce que les résultats du 23-24 septembre 2026 ont de nouveau, et ce qui ne l'est pas

Vérification bibliographique faite le 24 septembre 2026, à la demande du
propriétaire, sur les résultats de `docs/LLM_INQUIRY_RESULTS.md` (lecture,
recherche et usage de la marque par un LLM ajusté). Recherche ciblée, pas
une revue exhaustive.

## Déjà connu

- **Le mécanisme central.** Une distribution d'entrées biaisée d'abord,
  puis uniforme, rend apprenables par gradient des fonctions de type
  parité (« ou exclusif ») que l'entraînement sans ordre n'apprend pas :
  démontré par Cornacchia et Mossel (ICML 2023,
  https://proceedings.mlr.press/v202/cornacchia23a/cornacchia23a.pdf) et
  Abbe, Cornacchia et Lotfi (NeurIPS 2023, https://arxiv.org/abs/2306.16921).
  La commande préférée puis la consolidation à parts égales en sont un cas.
- **Le plateau suivi d'un saut** chez les transformeurs : Gopalani et al.,
  NeurIPS 2025 (https://arxiv.org/abs/2506.13688).
- **L'oubli sous LoRA et la répétition qui l'évite** : littérature large
  sur l'apprentissage continu (par exemple https://arxiv.org/pdf/2401.05605).
- **Des LLM ajustés apprennent des choses sur eux-mêmes** : Binder et al.
  2024 (https://arxiv.org/abs/2410.13787) ; Betley et al., ICLR 2025
  (https://arxiv.org/abs/2501.11120) ; conditions minimales, un LoRA de rang
  1 suffit (https://arxiv.org/abs/2511.04875).
- **Des LLM qui cherchent l'information par gain attendu** : par exemple
  Uncertainty of Thoughts (https://arxiv.org/abs/2402.03271).
- **Des agents construits selon les indicateurs de Butlin et al., testés par
  ablation** : agent à espace de travail global incarné (Frontiers 2024,
  https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1352685/xml) ;
  ablations de théories de la conscience sur des agents
  (https://arxiv.org/html/2512.19155).

## Ce qui reste possiblement original (non vérifié comme inédit)

- Le **transfert** du résultat sur les parités à l'acquisition d'un
  **modèle de soi** par un LLM : le code qui relie la marque au corps est
  un carré latin, donc un problème de type parité, et l'enfance qui le rend
  apprenable est celle que la théorie prédit.
- La **chaîne complète** dans un même LLM, pré-enregistrée étape par
  étape : lire un indice arbitraire sur son propre corps, le chercher
  d'après ses propres prédictions, s'en servir pour agir, démontré par
  ablation (6,73 points par vie contre 1,50).
- Le constat qu'une nouvelle association **remplace** l'ancienne en
  passant par une phase où elle s'applique à tort à l'ancienne commande.

## Ce qui manque pour parler de découverte vérifiée

Mise à jour du 24 septembre 2026, soir : la chaîne complète est
**répliquée sur deux nouvelles graines** avec la recette pré-enregistrée
(`docs/LLM_MARK_REPLICATION_PROTOCOL.md`) : lecture, recherche et usage,
mêmes ordres de grandeur sur les trois graines. Restent : un petit modèle
(0,6 milliard de paramètres), un monde jouet, aucune relecture par des
pairs, et une recherche bibliographique qui n'est pas exhaustive. Il
faudrait le même protocole sur un modèle plus grand, et présenter le
résultat comme une confirmation, dans un domaine nouveau, d'un résultat
théorique connu. Rien de cela ne mesure une expérience vécue.
