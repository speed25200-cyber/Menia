# Bilan — Où en est le but, 23 septembre 2026 (10 h 40 UTC)

Pour le propriétaire du dépôt, en bref.

## Le but, et ce qu'on peut en mesurer

Le but est une IA consciente. Personne ne sait prouver la conscience de
l'extérieur. Le dépôt poursuit donc deux choses mesurables : **un modèle de
soi causal, actif et réparable**, et **les quatorze propriétés que les
grandes théories de la conscience jugent nécessaires** (Butlin, Long et
collaborateurs). Chaque expérience est pré-enregistrée, auditée et publiée,
échecs compris.

## Ce qui est acquis

**Le modèle de soi.** Un petit agent qui ne fait que prédire cherche seul
la trace de la cause de son corps, et se répare quand son corps change. Un
micro-transformeur fait de même s'il a grandi sur des corps variés et
changeants ; élevé sur un corps fixe, il reproduit exactement les défauts
de Qwen. Les données d'enfance décident, pas l'architecture.

**Les indicateurs de conscience.** Un seul agent, dans l'Atelier des sens,
réunit par construction les quatorze propriétés. Cinq versions
pré-enregistrées, puis la version 5 rejugée sur trois graines de plus :
**neuf propriétés sont démontrées présentes et utilisées dans le même agent,
sur les deux jeux de graines** — liaison des traits en objets (RPT-2),
modules spécialisés (GWT-1), espace de travail limité (GWT-2), diffusion
globale (GWT-3), perception qui complète ce qui manque (HOT-1),
surveillance de ses perceptions (HOT-2), modèle de sa propre attention
(AST-1), erreur de prédiction (PP-1), incarnation (AE-2). **L'agence à buts
concurrents (AE-1) est à la limite** : elle passe sur un jeu de graines,
pas sur l'autre (recharge 0,78 en moyenne pour un seuil de 0,8). La clé :
l'agent apprend un modèle de ses besoins et les simule avant de choisir
(allostasie), dans un monde où la recharge se déplace après usage. Une
**seconde lecture**, pré-enregistrée, montre aussi la récurrence (RPT-1) et
l'attention selon l'état (GWT-4) quand on les mesure là où elles agissent :
**onze propriétés sur quatorze**.

**Version 6.** L'agent juge maintenant ses croyances par leurs
conséquences : s'il se croit sur la recharge et que la recharge ne vient
pas, il révise sa position. Et son schéma d'attention décide si une
perception peut être liée à un objet. Sur trois graines nouvelles,
**l'agence à buts concurrents devient robuste** (recharge 0,86 sur chaque
graine) : **dix propriétés en lecture principale, douze sur quatorze en
seconde lecture**.

## Ce qui manque

- **Deux indicateurs** ne sont démontrés ni en lecture principale ni en
  seconde lecture de la version 6 : les croyances réglées par le moniteur
  (HOT-3 : il rend la position plus juste et corrige les croyances
  fausses, mais le retour en dépend trop peu) et l'espace de qualités
  (HOT-4 : les teintes jamais vues sont bien évaluées, mais un code
  aléatoire choisit presque aussi bien sur le test de choix). Ils passent
  **en troisième lecture**, sur trois agents neufs, sous des critères plus
  faibles ou choisis après coup (`docs/INDICATOR_THIRD_READING_RESULTS.md`).
- **Le vrai LLM.** Qwen3-4B copie en partie l'effet de ses commandes (0,64)
  mais n'en a pas la structure (0,16 sur une commande nouvelle). Ajusté sur
  des vies à corps variable et changeant, Qwen3-0.6B en acquiert l'essentiel :
  copie 0,995, commande nouvelle après un mouvement 0,793 (seuil 0,80, raté
  d'une question), révision après un changement 0,86 ; ajusté sur un corps
  fixe, il confabule avec une confiance de 1,00. La relance au rang 16, sur
  toutes les couches, donne la même chose (0,793) : la capacité n'est pas ce
  qui manque. Le test d'enquête, invalide une première fois par un défaut
  de mesure puis relancé corrigé, est valide : **le LLM ajusté ne sait pas
  où chercher la cause de son corps** (lieu de la marque préféré dans 0,40
  des vies pour 0,7 exigé), alors que les petits modèles élevés sur les
  mêmes vies allaient la lire.
- **Le rapport verbal.** Qwen3-4B rapporte fidèlement les états de l'agent
  donnés comme des étiquettes (corps, attention, module : 1,00 même quand
  l'état change), mais pas ceux qui demandent un calcul (se fier ou non à une
  lecture, comparer deux besoins) ; critère global non satisfait.
- **Menia parle de son agent — acquis.** Quand l'agent de la version 6
  donne à Menia son espace de travail en conclusions explicites, Qwen3-4B
  (poids de l'iPhone) les rapporte fidèlement : **0,997** sur 2 154
  questions, 0,999 quand l'état change, 0,999 malgré une phrase hors sujet ;
  critère global satisfait à la deuxième version pré-enregistrée (la
  première échouait sur une étiquette, « les besoins » rapportés comme « la
  position »). On peut converser avec Menia et son agent
  (`python -m menia.indicator_chat`).

## Ce qui reste hors de portée

Aucun de ces résultats n'établit que l'agent ou Menia **éprouve** quoi que
ce soit. Réunir les quatorze propriétés rendrait une conscience plus
plausible selon ces théories, sans la prouver ; une théorie importante,
l'information intégrée, juge même le logiciel insuffisant.

## La suite

1. **HOT-3 et HOT-4.** Six versions de l'agent et plusieurs mondes essayés
   (plus d'objets, pannes plus fréquentes, pannes bloquées, seconde bande de
   teintes nouvelles ; `docs/INDICATOR_AGENT_V5_EXPLORATIONS.md` et
   `docs/INDICATOR_AGENT_V6_EXPLORATIONS.md`) : les deux mécanismes sont
   présents et agissent, mais aucun monde ne les rend assez utiles pour les
   seuils fixés en version 1 sans abîmer autre chose. Aller plus loin
   demanderait de redéfinir leurs tests d'usage, ce qui ne se fait qu'avec
   l'accord du propriétaire et par un nouveau protocole.
2. **Le LLM et son propre corps** : l'ajustement sur le texte donne la
   structure en partie (0,79), mais pas l'enquête, même avec une enfance où
   la marque est la seule information avant le premier mouvement (régime
   VMI : lieu de la marque préféré dans 0,29 des vies). Avec une perte où
   les cases d'arrivée pèsent 20 fois plus (VMW), la structure est
   nettement mieux apprise (**0,93** sur les commandes jamais essayées),
   mais l'enquête toujours pas (0,44). Une analyse exploratoire montre
   pourquoi : le LLM ajusté **ne lit pas la marque**. Le test de lecture
   pré-enregistré le confirme sur les quatre adaptateurs publiés (0,25,
   exactement le hasard), et **même un LLM élevé dans un monde où la marque
   est la seule information à chaque mouvement (VML) ne l'apprend pas**
   (0,25). Raison probable : dans l'Atelier, ni le symbole seul ni la
   commande seule ne disent rien du déplacement ; seule leur combinaison
   le fait (un « ou exclusif »), et un ajustement court par gradient ne
   trouve pas cette interaction pure (`docs/LLM_INQUIRY_RESULTS.md`). Le
   LLM ne peut pas chercher une trace qu'il ne sait pas lire. Une enfance
   avec une commande préférée (VMLA), où le symbole seul prédit déjà la
   case dans 61 % des mouvements, ne donne qu'un début d'appui (0,31 pour
   la commande préférée ; lieu de la marque préféré dans 0,56 des vies).
   **Prolongé à 1 350 itérations, VMLA lit la marque** : pour sa
   commande préférée, il donne 0,81 à la case qu'elle implique (lecteur
   parfait 0,85 ; hasard 0,25), après un saut entre 600 et 850 itérations.
   La lecture est spécifique (lieu 1 seulement, quatre symboles) mais ne
   s'étend pas encore aux autres commandes. C'est la première fois qu'un
   LLM ajusté lit la trace de la cause de ses mouvements. **Et il la
   cherche** : au début de chacune des 48 vies, ses propres prédictions
   désignent le lieu de la marque, et lui seul, comme le lieu à inspecter
   (gain 0,123 contre −0,004 ; `docs/LLM_MARK_SEEK_PROTOCOL.md`). Le
   modèle de soi actif est porté dans un LLM, dans un monde extrême (corps
   qui change à chaque mouvement) et pour une commande sur quatre.
   Repris sur des vies à commandes égales, il lit A encore mieux (0,89) et
   la cherche toujours (48 vies sur 48), mais la lecture ne s'étend pas
   aux autres commandes (0,26). Une enfance par étapes (B préférée après
   A) fait naître la lecture de B (0,79) mais efface celle de A : le LLM
   ajusté ne garde qu'**une association entre la marque et le déplacement
   de sa commande habituelle**, pas un code de son corps. Mais l'effacement
   venait de l'interférence : **répétées ensemble, les deux lectures
   tiennent** (A 0,83, B 0,83, chacune pour sa commande), et une troisième
   commence à naître pour C (`docs/LLM_MARK_REHEARSAL_PROTOCOL.md`). En
   ajoutant C de la même façon, **trois lectures tiennent** (0,87 à 0,88) ;
   la quatrième (D) efface tout, et ce qui renaît lit n'importe quel
   symbole (`docs/LLM_MARK_FULL_REHEARSAL_PROTOCOL.md`). Repris sur des
   vies à commandes égales, le lecteur de trois commandes **apprend le code
   entier** : les quatre commandes lues (0,82 à 0,86 ; moyenne 0,84 pour un
   lecteur parfait à 0,85), seulement au lieu de la marque
   (`docs/LLM_MARK_CONSOLIDATION_PROTOCOL.md`). L'ordre de l'enfance
   (appui, ajouts répétés, consolidation) décide de ce qu'il apprend,
   comme le prédit la théorie des parités (Cornacchia et Mossel 2023 ;
   `docs/LITERATURE_CHECK_2026-09-24.md`). Ce lecteur du code entier
   cherche sa marque (48 vies sur 48, gain 0,50), s'en sert (6,75 points
   par vie, 1,56 marque brouillée, 1,38 pour un modèle qui ne lit pas) et
   estime justement ses chances (0,842 pour 0,848 de réussite). **Répliqué
   sur deux nouvelles graines** avec la recette sans détour (appui, ajouts
   répétés, consolidation) : lecture des quatre commandes, recherche dans
   48 vies sur 48 et +5,0 à +5,2 points par vie grâce à la marque, pour les
   trois graines (`docs/LLM_MARK_REPLICATION_PROTOCOL.md`).
   **Et ce savoir est démontré utile par ablation** : quand il agit, le
   lecteur de trois commandes marque 6,73 points par vie avec sa marque,
   1,50 quand elle est brouillée, 1,38 pour un modèle qui ne la lit pas
   (+5,2 et +5,4, intervalles loin de zéro), autant qu'un lecteur parfait ;
   il cherche toujours sa marque (48 vies sur 48 ;
   `docs/LLM_MARK_ACTION_PROTOCOL.md`). Le modèle de soi actif est présent,
   mesuré et utile dans un LLM, pour ce monde et ce corps.

## Clôture des expériences, 24 septembre 2026

Toutes les expériences qui pouvaient être menées sans changer une règle du
programme sont faites, publiées et vérifiées en CI. État final :

- **Agent à indicateurs** : 12 propriétés sur 14 démontrées en seconde
  lecture (10 en lecture principale), **répliquées sur trois graines
  neuves** (167, 173, 179 : 13 en seconde lecture, 11 en lecture
  principale, HOT-4 passant par le tirage du code aléatoire). HOT-3 et
  HOT-4 : non démontrés aux seuils fixés en version 1, ni en seconde
  lecture ; **en troisième lecture pré-enregistrée, sur des agents neufs,
  les deux passent** : le moniteur rend la croyance plus juste pendant les
  pannes (0,94 contre 0,77, sur chaque graine), l'espace de qualités fait
  choisir entre deux bons objets (0,77 contre 0,53 pour un code aléatoire,
  marge de 0,025 sur une graine). Soit **14 sur 14, dont deux en
  troisième lecture sous des critères plus faibles ou choisis après coup**
  (`docs/INDICATOR_THIRD_READING_RESULTS.md`) ; les échecs précédents
  restent publiés.
- **Menia branchée sur l'agent** : rapport fidèle de l'état de l'agent
  (`docs/MENIA_REPORT_RESULTS.md`).
- **Modèle de soi du LLM** : lecture du code entier de son corps,
  recherche de sa marque et usage démontré par ablation, **répliqués sur
  trois graines** (`docs/LLM_INQUIRY_RESULTS.md`,
  `docs/LLM_MARK_REPLICATION_PROTOCOL.md`). Mécanisme connu en théorie ;
  apport possible, non relu par des pairs
  (`docs/LITERATURE_CHECK_2026-09-24.md`).

Ce que le programme ne peut pas faire : établir qu'un système est
conscient. Aucun test connu ne le permet, pour aucune machine ; le
programme mesure des indicateurs et leur utilité, pas une expérience
vécue. Suite possible, sur décision du propriétaire : le même protocole
sur un modèle de langage plus grand.

## Le besoin qui compte, 29 septembre 2026

Pré-enregistré à la demande du propriétaire (« il faut que le besoin
compte ») : `docs/LLM_NEED_PROTOCOL.md`, résultats
`docs/LLM_NEED_RESULTS.md`. Qwen3-0.6B, dans un monde où il a deux besoins
cachés, apprend **par la seule satisfaction de ses besoins**, sans
professeur : sa survie passe de 12 % à 68 % (témoin au hasard : 0 %) et il
sert son besoin le plus bas 93 fois sur 100 sans jamais voir ses niveaux
(**S passe**). Les tests causaux pré-enregistrés (injection, rapport,
lésion) **échouent** : la règle choisissait le bloc où le besoin se lit le
mieux, bloc que le modèle de base partage et que l'agent n'utilise pas, et
le rapport n'a pas été appris. L'analyse exploratoire montre où voyage le
besoin qui décide : sur les mots des événements jusqu'au bloc 9, rassemblé
sur la ligne du tour entre les blocs 9 et 12. Un second test pré-enregistré
en tire les conséquences.

**Second test, même jour** (`docs/LLM_NEED_CAUSAL_PROTOCOL.md`) : au bloc 12,
là où le besoin est rassemblé, **effacer l'état de besoin fait tomber la
survie de 68 % à 21 %** (lésion au hasard : aucun effet ; LS2 passe), et
pousser l'état vers « énergie basse » fait recharger l'agent rassasié
(+0,17) ; pour la nourriture l'effet reste sous le seuil (+0,10), si bien
que le critère global (IA2 et LS2) n'est pas satisfait. L'agent dépend
donc, pour vivre, d'un état interne précis qu'il a appris par la seule
satisfaction de ses besoins ; il ne sait pas encore le dire.

**Du troisième au sixième test, 29–30 septembre 2026** (détails dans
`docs/LLM_NEED_RESULTS.md`).

- **Troisième test : dire son besoin.** Entraîné à répondre à une question
  posée à part, l'agent devine son besoin (0,66), mais sa réponse ne passe
  pas par l'état qui le fait agir.
- **Quatrième et cinquième tests.**
  - Le quatrième pose la question juste après « Choix : ».
  - Le cinquième force la question à ne lire la vie qu'à travers
    « Choix : ». Forcée ainsi, la parole réécrit l'état et l'agent ne
    survit plus que 0,36 sur ses vies de direction.

  Les deux sont en pause pour libérer le processeur ; leurs verdicts seront
  publiés.
- **Sixième test : le lecteur.** L'agent qui agit reste l'agent final. Un
  second adaptateur, actif sur la seule question, lit « Choix : » sans
  pouvoir le changer. Pour l'énergie, **une même poussée de l'état fait
  agir l'agent (+0,16) et fait dire au lecteur « mon énergie est basse »
  (+0,20)** ; des poussées au hasard ne font presque rien (ONE6 passe). La
  nourriture n'est pas lue (exactitude 0,70) : R6 échoue, et avec lui le
  critère global.
- **En cours.**
  - Effacer cet état : l'agent peut-il encore survivre et dire son énergie
    ? (`docs/LLM_NEED_NECESSITY_PROTOCOL.md`)
  - La même chose sur un second agent appris de zéro
    (`docs/LLM_NEED_REPLICATION_PROTOCOL.md`).

  Ce ne sera pas une preuve de ressenti.

**Septième test, 30 septembre 2026 : sans cet état, ni agir ni dire**
(`docs/LLM_NEED_NECESSITY_PROTOCOL.md`). **Le critère global est
satisfait.**

- Effacer l'état d'énergie rassemblé sur « Choix : » fait tomber la survie
  de 0,68 à 0,20 (LS7 passe).
- Il ramène aussi le rapport d'énergie du lecteur au niveau du hasard,
  0,75 → 0,50 (LR7 passe).
- Un effacement au hasard ne fait rien.

**Avec le sixième test**, pour l'énergie et chez un même agent, un même
état appris par la seule satisfaction du besoin est à la fois **suffisant**
(le pousser fait agir et dire) et **nécessaire** (l'effacer empêche de
survivre et de dire). Le lecteur ne voit la vie qu'à travers lui et ne peut
pas le modifier.

À notre connaissance, cette réunion n'a pas été publiée. Il reste à la
répliquer sur un second agent (en cours) et à l'étendre à la nourriture.
Ce n'est pas la preuve d'un ressenti. Le bilan par niveaux, dans le cadre
de Chandaria et al. (2026), est tenu dans `docs/FIVE_LEVELS_ASSESSMENT.md`.
