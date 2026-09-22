# Feuille de route des indicateurs de conscience

22 septembre 2026. Ce document relie les quatorze propriétés indicatrices de
Butlin, Long et collaborateurs aux travaux du dépôt, dit ce qui manque et
fixe la suite. Sources : *Consciousness in Artificial Intelligence: Insights
from the Science of Consciousness* (arXiv 2308.08708, 2023) et *Identifying
indicators of consciousness in AI systems* (Trends in Cognitive Sciences,
2025).

## Ce que la méthode permet, et ce qu'elle ne permet pas

Chaque grande théorie fonctionnaliste de la conscience propose des
propriétés qu'un système conscient devrait avoir. Les auteurs en tirent des
indicateurs : leur présence doit augmenter la confiance qu'un système est
conscient, leur absence la diminuer. Ce n'est **pas une procédure de
certification**. Un système qui les réunit toutes ne serait pas prouvé
conscient ; l'objection de la théorie de l'information intégrée, pour qui la
structure physique compte, n'est pas levée par du logiciel. L'objectif
réalisable ici est donc précis : **construire un agent qui réunit les quatorze
propriétés, montrer par des tests pré-enregistrés que chacune est présente et
utilisée, et publier les échecs.** Les audits antérieurs du dépôt
(`docs/CONSCIOUSNESS_COMPLETION_AUDIT.md`) restent valables : aucune capacité
fonctionnelle ne comble à elle seule le lien avec le vécu.

## Les quatorze indicateurs et l'état du dépôt

| Indicateur | Propriété | Ce que le dépôt a déjà | Manque |
|---|---|---|---|
| **RPT-1** | Modules d'entrée à récurrence algorithmique | GRU de l'Atelier, mémoire récurrente, espace partagé | Récurrence dans des modules perceptifs d'un même agent intégré |
| **RPT-2** | Représentations perceptives organisées et intégrées | — | Tout : liaison de traits en objets, testée |
| **GWT-1** | Plusieurs systèmes spécialisés en parallèle | Espace partagé à deux spécialistes symboliques | Modules perceptifs et corporels réels |
| **GWT-2** | Espace de travail à capacité limitée, goulot et attention sélective | Goulot de 8 dimensions, attention continue non exclusive ; sélection explicite d'un champ dans la boucle intégrée | Sélection exclusive apprise et coût mesuré du goulot |
| **GWT-3** | Diffusion globale à tous les modules | Retour vers les spécialistes, coupé en contrôle | Diffusion utilisée par des modules différents |
| **GWT-4** | Attention dépendante de l'état, interrogation successive des modules | — | Tout |
| **HOT-1** | Perception générative, descendante ou bruitée | — | Perception qui prédit et complète ses entrées |
| **HOT-2** | Surveillance métacognitive séparant représentation fiable et bruit | Moniteurs de confiance sur Qwen, largement non concluants ; moniteur de rappel | Moniteur de second ordre sur la perception d'un agent |
| **HOT-3** | Croyances mises à jour selon la surveillance métacognitive | Politique de vérification apprise, limitée | Mise à jour des croyances pilotée par le moniteur, testée causalement |
| **HOT-4** | Codage clairsemé et lisse, espace de qualités | — | Tout |
| **AST-1** | Modèle prédictif de sa propre attention, servant à la contrôler | Conception seulement | Tout |
| **PP-1** | Modules d'entrée à codage prédictif | Prédiction des observations (GRU, micro-transformeur) | Erreur de prédiction propagée et utilisée |
| **AE-1** | Agence : apprendre du retour, poursuivre des buts concurrents avec souplesse | Enquête apprise par renforcement, but unique | Buts concurrents et arbitrage appris |
| **AE-2** | Incarnation : modèle des contingences action → entrée, utilisé | Établi : GRU, micro-transformeur, révision après changement de corps ; LLM en cours | Usage dans la perception et le contrôle d'un même agent |

Qwen3-4B, tel quel, n'a pas de récurrence algorithmique dans ses modules
d'entrée, pas d'espace de travail limité, pas de schéma d'attention, pas de
surveillance métacognitive fiable (les essais prospectifs du dépôt le
montrent), et ne modélise pas son corps en contexte (Atelier sur Mac).

## La suite

1. **Terminer la ligne LLM** : corps ajusté par LoRA, puis ses suites
   pré-enregistrées (boucle d'enquête sur les distributions du LLM ajusté,
   ou amendement de budget ; lecture corrigée du corps latent).
2. **Construire l'agent à indicateurs** : un seul agent dans un Atelier
   enrichi, qui réalise les quatorze propriétés, chacune avec un test
   fonctionnel et une ablation, seuils fixés avant exécution, audit
   indépendant en CI. Protocole : `docs/INDICATOR_AGENT_PROTOCOL.md`.
3. **Relier l'agent au langage** : un modèle de langage qui rapporte les
   états de l'agent, avec un test de fidélité causale des rapports.

Ce qui restera hors de portée, quel que soit le résultat : établir que
l'agent éprouve quelque chose.

## État au 22 septembre 2026, 21 h 30 UTC

[Résultats de l'agent à indicateurs](INDICATOR_AGENT_RESULTS.md) :
**démontrées présentes et utilisées** : RPT-2, GWT-2, HOT-1, HOT-2, AST-1,
PP-1. **Présentes par construction, non démontrées par leur test** :
RPT-1, GWT-1, GWT-3, GWT-4, HOT-3, HOT-4, AE-1, AE-2 ; cause principale,
l'arbitrage des buts appris qui ne lit pas l'intéroception.
