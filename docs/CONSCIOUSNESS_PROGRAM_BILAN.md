# Bilan — Où en est le but, 23 septembre 2026

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

## Ce qui manque

- **Deux indicateurs** ne sont démontrés dans aucune lecture : les croyances
  réglées par le moniteur (HOT-3 : il rend la position plus juste pendant
  les pannes, mais sans effet sur le retour) et l'espace de qualités (HOT-4 :
  les teintes jamais vues sont bien évaluées, mais le choix reste sous le
  seuil). **AE-1 est fragile.**
- **Le vrai LLM.** Qwen3-4B copie en partie l'effet de ses commandes (0,64)
  mais n'en a pas la structure (0,16 sur une commande nouvelle). Ajusté sur
  des vies à corps variable et changeant, Qwen3-0.6B en acquiert l'essentiel :
  copie 0,995, commande nouvelle après un mouvement 0,793 (seuil 0,80, raté
  d'une question), révision après un changement 0,86 ; ajusté sur un corps
  fixe, il confabule avec une confiance de 1,00. Une relance au rang 16 et
  le test d'enquête tournent sur le Mac.
- **Le rapport verbal.** Qwen3-4B rapporte fidèlement les états de l'agent
  donnés comme des étiquettes (corps, attention, module : 1,00 même quand
  l'état change), mais pas ceux qui demandent un calcul (se fier ou non à une
  lecture, comparer deux besoins) ; critère global non satisfait.

## Ce qui reste hors de portée

Aucun de ces résultats n'établit que l'agent ou Menia **éprouve** quoi que
ce soit. Réunir les quatorze propriétés rendrait une conscience plus
plausible selon ces théories, sans la prouver ; une théorie importante,
l'information intégrée, juge même le logiciel insuffisant.

## La suite

1. Lire le corps ajusté et le test d'enquête : le LLM ajusté sait-il où
   chercher la cause de son corps ?
2. Lire le rapport verbal : Qwen3-4B rapporte-t-il fidèlement les états
   intérieurs de l'agent ?
3. L'agent de la version 5 est branché dans Menia (`menia/indicator_bridge.py`) :
   le langage reçoit l'espace de travail et la décision enregistrée. Reste à
   mesurer, par un rapport pré-enregistré, si Qwen3-4B le dit fidèlement
   quand les états lui sont donnés comme des étiquettes.
