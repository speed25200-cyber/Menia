# Bilan — Où en est le but, 23 septembre 2026 (7 h 15 UTC)

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

- **Deux indicateurs** ne sont démontrés dans aucune lecture, même en
  version 6 : les croyances réglées par le moniteur (HOT-3 : il rend la
  position plus juste et corrige les croyances fausses, mais le retour en
  dépend trop peu) et l'espace de qualités (HOT-4 : les teintes jamais vues
  sont bien évaluées, mais un code aléatoire choisit presque aussi bien sur
  le test de choix).
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

1. HOT-3 et HOT-4 : chercher un monde ou un test où le moniteur et l'espace
   de qualités sont indispensables, sans viser les seuils.
2. Le LLM et son propre corps : l'ajustement donne la structure en partie
   (0,79) mais pas l'enquête ; une autre recette (données où la marque est
   la seule source d'information, ou un modèle plus grand) serait à
   pré-enregistrer.
