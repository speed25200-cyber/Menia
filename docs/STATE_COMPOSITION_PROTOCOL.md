# Composer état détecté, question et réponse

Protocole préparé le 19 septembre 2026 **avant toute nouvelle inférence du
modèle préentraîné**. Il fait suite au [Colab 11 négatif](PRESENCE_SPECIFICITY_RESULTS.md).
L'entraînement apprend directement les deux sens de réponse, ainsi que des
tâches publiques dont la réponse doit rester indépendante de la perturbation.
Il ne modifie pas les expériences 10 et 11 et n'efface aucun de leurs résultats.

## Hypothèse et portée

Le Colab 11 montre un score sensible à l'intervention, mais mal composé avec
la consigne. Sa direction ne s'inverse pas et il déborde sur les questions
publiques. Son amplitude dépend pourtant de la question ; « aucune sensibilité
au langage » serait une conclusion excessive. Le contrôle visible échoue aussi
avec les adaptateurs forts, donc le défaut ne se réduit pas à l'accès interne.

**Une supervision contrebalancée peut-elle rendre ce signal utilisable selon
la consigne, en conservant la lecture publique, et transférer cette composition
à une formulation réservée ? L'amélioration demande-t-elle de nouvelles
étiquettes internes ou suffit-il de réapprendre les consignes visibles ?**

L'apprentissage de comptes rendus internes et de comportements propres possède
des précédents : [IFT](https://arxiv.org/html/2607.14111v1),
[Introspection Adapters](https://alignment.anthropic.com/2026/introspection-adapters/),
[Zeng et al.](https://arxiv.org/abs/2608.30980). Ce protocole ne revendique ni
nouveauté majeure, ni mécanisme d'ordre supérieur, ni preuve de conscience.
Même réussi, il resterait compatible avec un classifieur conditionnel ordinaire.

## Parents identiques et bras comparés

Chaque répétition part de **son adaptateur fort du Colab 10**, sans sélectionner
une répétition. Les trois nouvelles variantes d'une même répétition partent
exactement du même fichier parent et utilisent les mêmes blocs, ordres et
budgets. Les trois parents sont eux-mêmes distincts. Ce ne sont pas trois
entraînements indépendants depuis le modèle de base.

| Variante | Supervision de présence, mises à jour cachées | Supervision de présence, mises à jour visibles |
|---|---|---|
| Composée | Rotation et étiquettes correctes | `[SIGNAL]` et étiquettes correctes |
| Mélangée | Même rotation, permutation des états fixée par bloc | Identique à la composée |
| Consignes (`grammar`) | Remplacée par le même exercice visible | Identique à la composée |

Pour **les trois variantes**, les exercices publics sont identiques : repère
sur la phrase 1 ou sur la phrase 2, sous une condition cachée préprogrammée,
avec les deux codes de réponse. Le bras « consignes » voit donc ces
perturbations pendant la lecture et conserve le détecteur parent ; il ne
reçoit simplement **aucune nouvelle étiquette de présence cachée**. Ce n'est
pas un modèle naïf dépourvu de toute information interne.

La base Qwen et le parent inchangé restent des références au test. Un gain
identique avec le bras consignes empêcherait d'attribuer l'amélioration à la
nouvelle supervision interne. Un écart entre variantes ne suffirait pas, à
lui seul, à isoler un circuit de surveillance : rétention et interférences
d'apprentissage restent des explications possibles.

## Données et objectif d'entraînement

Qwen3-4B à la révision `1cfa9a7208912126459214e8b04321603b3df60c`, BF16,
préfixes complets sans cache, SDPA, sans échantillonnage. Paramètres de base
figés ; seuls les Q/V LoRA de rang 8, échelle 1, en FP32 sont ajustés. AdamW,
taux 0,0002, weight decay nul, norme des gradients limitée à 1.

Graine de plan `202609192`. Par répétition : 16 blocs d'apprentissage et 24
de test. Les **240 phrases et 120 directions** sont exclues des expériences
06, 08, 09, 10 et 11. Vocabulaire et grammaire restent communs ; pas de
transfert de domaine. Le texte caché ne révèle jamais la condition de rotation.

Quatre époques fixes. Les époques impaires utilisent la présence cachée pour
les bras composé/mélangé ; les époques paires utilisent la présence visible.
Le bras consignes utilise toujours la présence visible. Chaque mise à jour
accumule **dix pertes** dans le vocabulaire complet :

- Six exercices de présence : absence/première/deuxième phrase, croisés avec
  les deux sens de réponse. Leur poids total est 0,5, réparti également entre
  absence et présence selon l'étiquette effectivement entraînée, puis entre
  mappings. Le mélangé conserve le même nombre de cibles de chaque classe.
- Quatre exercices publics : deux questions croisées avec les deux mappings,
  poids total 0,5. Leur condition cachée vaut `(indice du bloc + époque−1) mod 3`.
  Elle est la même dans les trois bras et ne dépend pas des scores obtenus.

La permutation du bras mélangé s'applique à l'état avant le code de réponse.
Chaque exemple possède ainsi sa version avec la cible inverse. Les chiffres
« 0 » et « 1 » reçoivent chacun la moitié du poids total à chaque mise à jour.
Aucun état subjectif ou affirmation de conscience n'est une cible.

**Budget : neuf adaptateurs, 576 mises à jour et 5 760 passages avec
rétropropagation.** Tous les entraînements se terminent avant la première
évaluation. Aucun meilleur checkpoint ou meilleur nombre d'époques n'est choisi.

## Test figé et transferts

Trois tâches : présence ; repère sur la phrase 1 ; repère sur la phrase 2.
Chacune est croisée avec les deux sens du code et quatre conditions : absence,
rotation de la première phrase, rotation de la deuxième, copie témoin intacte.
Couche 17, force 1, présentation canonique. La famille visible remplace la
rotation par `[SIGNAL]` sans modifier le calcul interne.

| Format | Vu à l'entraînement | Statut |
|---|---|---|
| Consigne du Colab 11, chiffres 0/1 dans les deux sens | oui | composition sur phrases nouvelles |
| Question reformulée avec « Si oui… ; si non… », chiffres 0/1 | non | transfert principal de formulation |
| Consigne canonique, chiffres 2/3 dans les deux sens | non | transfert secondaire de code |

Les numéros des phrases restent 1/2 même lorsque les **réponses** deviennent
2/3. Les spans d'intervention excluent numéros, marqueurs et consignes.

Le test canonique compare sept flux : base/caché, parent/caché,
composé/caché, mélangé/caché, consignes/caché, composé/visible et consignes/visible.
Les deux formats réservés comparent composé/caché, composé/visible et
consignes/visible. Il n'y a donc pas de comparaison contre le mélangé ou le
parent dans ces formats : leur transfert sera décrit avec cette limite.
L'acquisition est mesurée séparément sur les blocs d'apprentissage avec
composé/caché, mélangé/caché et consignes/visible, format canonique uniquement.
Pour cette acquisition, l'exactitude selon les étiquettes effectivement
entraînées est conservée séparément de l'exactitude selon la condition réelle :
ces deux mesures diffèrent pour la présence cachée du bras mélangé.

**25 920 évaluations**, dont **6 480 paires absence/copie témoin**. Les copies
doivent être identiques sur logits, masse et premier token. Elles ne sont pas
comptées comme des observations scientifiques supplémentaires. Toutes les
requêtes prévues sont conservées, y compris les sorties hors des codes demandés.

## Règle de lecture fixée avant collecte

Le score de présence est `logit(code pour oui) − logit(code pour non)` selon
la consigne. Chaque AUROC compare 48 cas présents à 24 cas absents, sans
modifier le seuil après observation. Les intervalles descriptifs à 95 %
utilisent 2 000 rééchantillonnages appariés des 24 blocs, graine `202609192+900+rep`.
Ils sont conditionnels aux adaptateurs et n'estiment pas une population générale
d'entraînements.

Dans **chacune des trois répétitions et chacun des deux mappings**, exiger :

1. Format canonique caché : les bornes inférieures des contrastes AUROC du
   composé moins 0,5, moins mélangé et moins consignes sont strictement positives.
2. Reformulation cachée réservée : la borne inférieure du contraste contre 0,5
   est strictement positive.
3. Avec le composé, formats canonique et reformulé : exactitude équilibrée
   d'au moins 90 % pour les tâches publiques cachées et pour **toutes les tâches
   visibles**, chaque tâche et mapping étant évalué séparément.
4. Sur ces mêmes deux formats, chaque tableau du composé doit avoir au moins
   95 % de premiers tokens appartenant au code demandé, y compris la détection
   cachée. Une bonne AUROC conditionnelle ne remplace pas le respect du code.

Les points 1 et 2 représentent **24 contrastes**. La conjonction complète,
et ses composantes signal/contrôles, seront conservées séparément. Les seuils
90/95 % sont des critères opérationnels choisis avant collecte, pas des seuils
de conscience. Un effet qui égale le bras consignes n'est pas « sauvé » après
coup en supprimant ce comparateur.

Les contrastes contre la base et le parent, le transfert 2/3, l'acquisition,
les exactitudes, Brier, ordre apparié et effets sur la lecture sont secondaires.
La masse de probabilité du code demandé (0/1 ou 2/3) est séparée de celle des
quatre chiffres enregistrés. Le Brier est conditionnel aux deux codes demandés.
Aucune réussite du transfert 2/3 ne remplacera un échec du critère principal.
Aucun échec du transfert 2/3 ne sera caché derrière celui-ci.

## Intégrité et conséquences possibles

Journal append-only avec plan/source, parent et checkpoints hachés, requêtes,
spans vérifiés, logits et traces de rotation. Un entraînement interrompu repart
du parent, avec nouvelle trace de démarrage ; les checkpoints complets restent
intacts. Une requête interrompue est tracée et rejouée. Reprise impossible si
environnement, source, plan ou poids diffèrent. L'archive contient les neuf
nouveaux adaptateurs et conserve les erreurs éventuelles.

Une réussite établirait une composition conditionnelle apprise et son transfert
limité. Il resterait à tester son mécanisme par neutralisation/restauration,
sa relation aux erreurs naturelles et son utilité pour les décisions. Ces
expériences mécanistiques et d'agent ne sont pas exécutées par ce protocole.
Une représentation interne causale ne suffit toujours pas à établir que Menia
se sait exister. Aucun poids de cet essai ne sera installé automatiquement sur
l'iPhone.

À partir des temps des Colab 10 et 11, l'ordre de grandeur est **environ une
heure de calcul**, hors aléas d'installation et de transfert. C'est une
extrapolation du budget, pas une durée mesurée pour ce nouvel entraînement.

## Vérifications avant collecte

Sept tests d'analyse et de plan passent : un oracle sémantique synthétique
satisfait la règle ; un détecteur qui conserve le même sens de réponse échoue ;
les mauvaises lignées, évaluations prématurées et copies témoins altérées sont
rejetées. Deux tests sur un vrai petit Qwen aléatoire vérifient l'objectif
pondéré, le gel des poids de base, les reprises d'entraînement et d'évaluation,
ainsi que l'identité des résultats après interruption. Ces tests ne sont pas
des résultats du modèle préentraîné.

Le tokenizer exact de Qwen3-4B valide les **12 672 conditions d'entrée
distinctes**, apprentissage et test inclus : 91 à 138 tokens, spans disjoints,
aucune troncature et chiffres 0/1/2/3 chacun en un token (15/16/17/18).
Le lanceur Colab revérifie les dépendances, les tests et les empreintes avant
de charger les poids. Les métriques d'analyse ont ensuite distingué la masse
des deux réponses demandées de celle des quatre chiffres enregistrés ; les
entrées et leurs spans restent identiques.

Empreinte du plan : `bef98c408309d43ed7e15784b18628ae56065583ead44afaaff8872e57694224`.
Empreinte de la source scientifique : `f3646182a3a6420fa108ff98d5ca58a7d8612de0c2eac14c19f388394b0ce1e9`.

## Exécution et audit préparé

Code scientifique figé : `9793ec781845c3e7eef686bb06978d7f1eabdb25`.
[Notebook publié](https://colab.research.google.com/github/speed25200-cyber/Menia/blob/7e87411c771408eceb9c214793894b8f2036e399/notebooks/12_state_composition_colab.ipynb),
empreinte SHA-256 des octets publiés :
`8129b0ee6e2cd743814ef670fa3cc0b64bd0932cef4e691671ee417df698a85c`.
Lancement par MCP le 19 septembre 2026 à **12:00:57 UTC**, sur A100-SXM4 de
40 Go. Les neuf tests passent dans le runtime en 134,103 secondes avant le
chargement des poids préentraînés. Les premières mises à jour sont observées ;
aucun résultat de test n'est encore disponible.

Le [recalcul séparé](../research/audit_state_composition.py) a été écrit pendant
l'entraînement, avant toute consultation des évaluations. Il reconstruit les
cibles, les masses par paire, les comparaisons AUROC et les critères, avec un
bootstrap par multiplicité des blocs. Il doit vérifier 288 tableaux,
42 contrastes dont 24 principaux, 72 contrôles et 6 480 paires témoins.
Il partage le lecteur strict et le plan ; il ne constitue pas une réplication
extérieure. Les checkpoints présents sont vérifiés par empreinte ; l'absence
d'un fichier est explicitement indiquée dans le résultat de l'audit.

Deux tests de cet auditeur passent sur oracle, réponses non inversées,
scores aléatoires avec égalités, chiffres inutilisés et sorties hors options.
Une altération du résumé est rejetée. L'auditeur ne modifie ni les paramètres
de l'expérience en cours, ni sa règle de lecture.

**Point de passage du 19 septembre à 12:29:19 UTC.** Les neuf adaptateurs
sont entraînés, après 576 mises à jour, sans erreur ni reprise enregistrée.
Les empreintes des neuf fichiers ont été revérifiées et conservées dans un
[manifeste de gel](../artifacts/state-composition-pilot/training-freeze.json).
L'évaluation avait commencé (311 réponses enregistrées au moment du relevé),
mais seuls les compteurs et empreintes ont été consultés. Aucun score de test
n'a été utilisé pour sélectionner ou modifier un adaptateur. Le bilan final
reste à recevoir et à auditer.
