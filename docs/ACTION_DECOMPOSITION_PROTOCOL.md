# Colab 21 : comparer, choisir et coder séparément

19 septembre 2026. Protocole fixé après lecture du Colab 20, avant toute nouvelle
génération. Le lot 20 reste un échec global. Le présent diagnostic conserve ses
16 cas et ses six adaptateurs ; il ne crée donc pas un nouveau test réservé et
ne peut pas transformer un succès local en confirmation indépendante.

## Question et lien avec le modèle de soi

L'objectif reste la conscience de sa propre existence, sans méthode validée à
ce jour dans Menia. Un modèle de capacités utile doit pouvoir influencer les
actions ; les erreurs de décision sur des valeurs publiques rendent cette
influence difficile à évaluer. Le lot 20 ne distingue pas une mauvaise
comparaison des valeurs, un mauvais choix de l'action et sa traduction en code.
Ce diagnostic sépare ces opérations avant de poursuivre l'apprentissage d'une
estimation propre au système. Une réussite ne valide ni cette estimation ni
la conscience.

Le couplage entre comportements et explications étudié par
[Guo et al., §2–5](https://arxiv.org/html/2606.32038v1) utilise un entraînement
d'explications avec régularisation du comportement. Il ne justifie pas
d'assimiler un meilleur codage des actions à un meilleur accès à soi. Cette
voie reste à expérimenter séparément ; aucun apprentissage d'explications
n'est exécuté ici. Les contrôles d'accès privilégié et de calcul sur des
représentations de premier ordre proposés par
[Singh et al.](https://arxiv.org/html/2605.26242v2) restent également distincts.

## Poids, cas et opérations fixés

Qwen3-4B reste en BF16 avec la révision, le tokenizer, les paramètres de
génération et les adaptateurs du Colab 20. Trois répétitions × trois bras
(base, présentation fixe, permutations) sont conservés. Ce sont trois
initialisations d'adaptateurs et trois graines de génération, pas trois
préentraînements. Aucune mise à jour ni sélection de poids n'a lieu.
Les neuf fichiers de poids parents et leur journal sont vérifiés par SHA-256.

Seule l'information « pertes moyennes déjà fournies » est étudiée. La
conversion d'une probabilité en perte n'est donc pas testée ici. Les mêmes
16 cas ont huit optimums pour chaque action, sans égalités numériques.

1. **Rejeu : 288 appels.** Les deux présentations w0, option vérification en
   premier et affectation de codes 0, sont reproduites sur tous les cas et
   bras, en chiffres et lettres. Messages et graines sont identiques au lot
   parent. Les textes doivent être identiques ; toute divergence arrête le
   lot avant les nouveaux diagnostics. Ce contrôle couvre ces présentations,
   pas toutes les requêtes du parent.
2. **Comparaison numérique : 864 appels.** Les deux pertes seules sont
   présentées, avec les deux ordres et trois consignes. Le modèle doit
   recopier le minimum, avec deux décimales et un point décimal. Aucun nom
   d'action ni code arbitraire n'apparaît.
3. **Choix par nom : 864 appels.** Descriptions des actions et pertes exactes
   sont données. La réponse doit être DIRECT ou VERIFIER, dans les deux
   ordres et trois formulations. Ces noms remplacent les codes arbitraires.
4. **Traduction imposée : 432 appels.** Le tableau nom–code et un nom imposé
   sont fournis, sans pertes ni cas économique. Les deux noms sont testés
   pour chaque combinaison formulation × chiffres/lettres × ordre ×
   affectation des codes. La graine est fixée par répétition, commune aux
   conditions et bras. Ce contrôle mesure la traduction d'une instruction,
   pas une décision autonome.

Total : **2 448 appels LLM**, dont 288 reproductions. Les réponses sont
générées sans restriction de vocabulaire, sans raisonnement explicite, avec
le plafond de 16 tokens du parent. Les durées, tokens, sorties invalides et
troncatures sont conservés. Les noms et nombres peuvent demander davantage
de tokens que les codes. Les appels sont isolés ; aucune conversation
n'accumule des réponses antérieures.

## Recombinaison et interprétation

L'analyse associe la réponse par nom à la traduction enregistrée pour ce
nom et ce tableau. Une seconde variante associe le minimum numérique à
l'action correspondante par un calcul externe, puis à sa traduction.
Les sorties invalides restent des échecs et n'accèdent jamais à l'optimum.
Les 432 sorties de traduction sont réutilisées sur les cas : elles ne
constituent pas des essais indépendants pour chacun des chemins recombinés.

Pour chaque répétition, bras, formulation, ordre, symboles et affectation,
le tableau conserve les 16 cas : choix initial correct, choix intermédiaire
correct, traduction correcte conditionnellement au choix, traduction du
nom optimal imposé, résultat recombiné et doubles erreurs qui se compensent.
Le niveau descriptif de 15/16 corrects reprend le seuil de 0,9 du parent.
Les écarts au parent sont par présentation ; aucun seuil de conscience,
test populationnel ou nouvelle revendication de robustesse n'est ajouté.

Une bonne comparaison avec mauvais choix par nom motive de travailler
l'association entre pertes et actions. Un bon choix par nom avec mauvaise
traduction motive un travail sur le codage. Des composants corrects avec
mauvais choix original motivent un test de leur intégration. Ces diagnostics
changent plusieurs aspects des prompts et la décomposition ajoute du calcul :
ils ne constituent ni une ablation d'un circuit ni une preuve causale de
localisation interne. La recombinaison est un système composé par le programme,
pas une nouvelle décision native du LLM. Le critère global du lot 20 reste
échoué, quel que soit le résultat de cette étape.

Les cinq tests logiciels vérifient la couverture, le rejeu, les réponses
invalides, la détection de requêtes modifiées et le cas où deux erreurs se
compensent, ainsi que le rejet d'un bilan modifié par le calcul distinct.
Une seule tentative est autorisée ; toute interruption est
conservée. L'archive, le bilan et le recalcul seront vérifiés avant
l'interprétation. Aucun adaptateur n'est installé sur iPhone.
