# Prophet_AGI dans Menia

Rédigé le 15 octobre 2026, à la demande du propriétaire (« merge les expériences de Prophet_AGI dans
Menia »). Ce document dit ce qui a été importé, où, et ce que ses résultats apportent au programme
suivant (`docs/APPRENDRE_SANS_FIN_PROTOCOLE.md`). Il ne change aucun résultat de l'un ou l'autre dépôt.

## Ce qui a été importé

- **Tout le dépôt `speed25200-cyber/Prophet_AGI`**, branche `claude/prophet-ai-architecture-gvyvbn`, commit
  `8a28066` (24 septembre 2026), dans le dossier `prophet_agi/`, **avec son historique** (fusion sans
  ancêtre commun, stratégie « ours » puis `read-tree --prefix`). Les commits d'origine sont dans
  l'historique, sous leurs anciens chemins (sans le préfixe) : `git log 69d80cf^2 -- docs/35_HYPOTHESIS_MAP.md`
  les montre ; `git log -- prophet_agi/` ne montre que le commit d'import.
- Rien n'a été modifié dans les fichiers importés. Ils gardent leurs propres tests, leur `CLAUDE.md` et
  leurs chemins relatifs : on travaille dans `prophet_agi/` comme dans le dépôt d'origine.
- La CI de Menia ne lance pas les tests de `prophet_agi/` (ses commandes visent `tests/` et
  `tests_research/`). Les tests de Prophet se lancent depuis `prophet_agi/`, comme avant.
- Le dépôt `Prophet_AGI` reste en place, inchangé.

## Ce que Prophet_AGI a établi, utile au but de Menia

Le but, tel que le propriétaire l'a dit le 15 octobre 2026 : **une IA qui veut apprendre seule, sans
fin, sans nouveau pré-entraînement à chaque fois, en commençant petite et en grandissant.** Les deux
dépôts ont mesuré chacun une partie des murs. Chiffres repris de `prophet_agi/docs/35_HYPOTHESIS_MAP.md`
et des documents qu'il cite ; tous à 7 M de paramètres sur CPU sauf mention contraire.

| Mur | Prophet_AGI | Menia |
|---|---|---|
| **Apprendre de soi** | La boucle fermée apprend : +0,256 en moyenne, intervalles > 0 sur 3 graines, rendement 0,92 contre l'oracle (`docs/32` §9) | — |
| **Dépasser ce qu'on lui donne** | La boucle dépasse son générateur **si et seulement si** au moins une proposition nouvelle est rattrapée et promue : 3 runs sur 6 (`docs/39` §3, « règle des six runs ») | — |
| **Le vérificateur borne** | Ce que le modèle rate toujours, il ne l'apprend jamais seul : graine dure +0,000 (`docs/32` §14) | — |
| **Ne pas oublier** | Ajustement sans rejeu : 2,18 → 7,36 bit/octet. Rejeu 0,5 : l'oubli croît avec la longueur de la boucle (+0,37 en 10 tours). Taux ÷ √r : +0,125 en 10 tours, à pente constante (`docs/10` §2, `docs/39` §3) | — |
| **Vouloir apprendre** | — | Le bonus d'information seul et la surprise ne produisent pas l'enquête ; l'aversion à agir dans l'ignorance, si (`docs/LEARNED_INQUIRY_RESULTS.md`). « Soif de savoir » arrêté au pilote : interférence 0,32 pour un seuil de 0,25 (`docs/LLM_CURIOSITY_RESULTS.md`) |
| **Ce que l'état porte** | — | Tests 32 à 36 : un état transmis, appris par la seule survie, porte des besoins jamais observés ; greffe entre vies, 10 graines sur 10 (`docs/LLM_NEED_RESULTS.md`) |
| **Savoir qu'on ne sait pas** | Désaccord entre profondeurs comme signal d'erreur : ouvert (A4-0) | Test 37 en cours (4 graines sur 10). Sur vrai modèle, le Colab 23 échoue face aux étiquettes mélangées (`docs/NATIVE_ANSWER_CONFIDENCE_RESULTS.md`) |
| **L'ordre des leçons** | Un curriculum qui rend atteignable l'ensemble jamais réussi : **non testé**, noté deux fois (`docs/35`) | Appui, ajouts répétés, consolidation : un code de type parité devient apprenable (`docs/LLM_MARK_*`) |
| **Grandir sans casser** | Récurrence sur donneur retenu : logits identiques au bit près à une boucle, sur Qwen2.5-0.5B (`docs/28`) ; qualité non testée | — |
| **Mémoire à deux niveaux** | Écriture « sommeil » sans gradient (R03, E2) : **ouverte, jamais lancée** (`docs/06`, `docs/07`) | — |

## Ce qui n'est pas importé

- Les sorties des sessions Colab perdues (`prophet_agi/docs/34` §5) : elles n'existent plus nulle part.
- Les données exclues par `.gitignore` dans le dépôt d'origine (`data/`). `prophet_agi/docs/11` décrit le
  chemin de corpus manquant et sa restauration.
- Les autres branches de Prophet_AGI : elles étaient déjà contenues dans la branche importée (vérifié le
  15 octobre 2026 : `claude/codex-results-analysis-5jz95y` et `claude/prophet-v03-memory-context` sont
  derrière elle, sans commit propre).
