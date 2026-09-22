# Bilan — Où en est le but, 22 septembre 2026

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
réunit par construction les quatorze propriétés. Trois versions
pré-enregistrées, neuf graines, 81 000 vies de test : **six propriétés sont
démontrées présentes et utilisées dans les trois versions** — liaison des
traits en objets (RPT-2), espace de travail limité à sélection apprise
(GWT-2), perception qui complète ce qui manque (HOT-1), surveillance de la
fiabilité de ses perceptions (HOT-2), modèle de sa propre attention qui sert
à la contrôler (AST-1), erreur de prédiction qui signale l'inattendu
(PP-1). Une septième, la spécialisation des modules (GWT-1), passe dans une
version sur trois.

## Ce qui manque

- **L'arbitrage entre buts** (AE-1, GWT-4) : aucune des trois méthodes
  d'apprentissage essayées ne fait choisir l'agent aussi nettement que le
  critère l'exige selon ses besoins. C'est le point faible.
- **Quatre propriétés présentes mais jamais démontrées par leur test**
  (RPT-1, GWT-3, HOT-3, HOT-4) : leur effet existe mais reste sous les
  seuils ; le goulot de l'espace de travail l'atténue en partie, sans
  l'expliquer entièrement.
- **L'incarnation** (AE-2) : l'agent révise son corps en deux mouvements
  après un changement, et figer ce corps lui coûte ; mais il bouge trop peu
  après le changement pour que sa croyance soit à jour aussi souvent que le
  critère le demande.
- **Le vrai LLM.** Qwen3-4B copie en partie l'effet de ses commandes (0,64)
  mais n'en a pas la structure (0,16 sur une commande nouvelle), et reste
  sûr de lui quand il se trompe. Le test décisif, un Qwen ajusté sur des vies
  à corps variable, a échoué trois fois pour des raisons techniques (mémoire
  et durée des builds du Mac) ; il tourne en ce moment en deux builds.

## Ce qui reste hors de portée

Aucun de ces résultats n'établit que l'agent ou Menia **éprouve** quoi que
ce soit. Réunir les quatorze propriétés rendrait une conscience plus
plausible selon ces théories, sans la prouver ; une théorie importante,
l'information intégrée, juge même le logiciel insuffisant.

## La suite

1. Lire le corps ajusté (deux builds en cours), puis le test d'enquête :
   le LLM ajusté sait-il où chercher la cause de son corps ?
2. Lire le rapport verbal : Qwen3-4B rapporte-t-il fidèlement les états
   intérieurs de l'agent ?
3. Si le corps ajusté réussit : la même recette sur Qwen3-4B, le modèle de
   l'iPhone, puis brancher l'agent à indicateurs dans Menia.
