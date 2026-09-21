# Apprendre une prévision depuis un état : apports et limites des adaptateurs

20 septembre 2026. Note de recherche, sans nouvel entraînement exécuté ni
protocole confirmatoire fixé. La [découverte par paire](PROSPECTIVE_PAIR_DISCOVERY_PROTOCOL.md)
cherche d'abord à produire des conséquences différentes selon la question,
depuis un même état. Un apprentissage ultérieur doit prévoir ces conséquences
mesurées, pas seulement reconnaître une intervention.

## Ce que les méthodes publiées permettent de proposer

[Shenoy et al., *Introspection Adapters*, version 2, §2.1](https://arxiv.org/html/2604.16812v2#S2.SS1)
entraînent un adaptateur partagé sur plusieurs modèles dotés de comportements
implantés. Les adaptateurs de comportement sont gelés ; celui d'introspection
apprend à les décrire, puis une étape de préférences réduit les rapports erronés.
Ce précédent rend l'apprentissage d'un rapport interne concret et testable.
Il concerne toutefois des propriétés inscrites dans les poids, pas la réussite
d'une prochaine requête depuis un cache épisodique altéré. Ses performances ne
se transfèrent donc pas directement à notre tâche. L'idée générale d'un
adaptateur d'introspection n'est pas une nouveauté de Menia.

[Yoshida et al., *Looking in the Mirror*, version 1, §4.3](https://arxiv.org/html/2608.04347v1#S4.SS3)
proposent DAIA : deux branches lisent séparément la contribution des poids de
base et la différence apportée par un ajustement additif. Cette différence est
disponible pendant le calcul de l'adaptateur ; elle ne requiert pas un second
passage complet du modèle de référence. Leurs cibles sont des changements
d'alignement mesurés après ajustement, avec modèles et catégories réservés.
Pour Menia, une différence entre cache propre et cache altéré pourrait être
utile, mais donnerait au prédicteur une référence privilégiée qu'un agent réel
n'a pas nécessairement. Ce serait un témoin avec information supplémentaire,
à distinguer d'une prévision fondée sur le seul état courant. DAIA ne démontre
pas cette dernière capacité dans notre dispositif.

[Merrill et Medley, *Symmetry Defeats Auditing*, version 1, §§2–6](https://arxiv.org/html/2605.27836v1)
montrent que des changements de coordonnées des poids, préservant le comportement
du modèle avant ajout du détecteur, peuvent dégrader fortement un adaptateur
d'introspection ajouté ensuite. Leur essai porte sur sept ajustements publiés ;
la généralisation à tous les audits reste une hypothèse des auteurs. Nos
permutations conjointes K/V concernent les positions du cache, pas ces changements
de base dans les poids : nous ne reproduisons pas leur attaque. Leur résultat
motive néanmoins un contrôle de robustesse aux transformations sans effet sur
la tâche, en complément des altérations qui changent effectivement sa réussite.

## Choix d'architecture encore à fixer

**Complément du 20 septembre :** [IntroLM, §§4–5 et annexe C](https://arxiv.org/html/2601.03511v1)
apprend une prévision de qualité avant la réponse avec des tokens `[CPX]` et
des adaptateurs conditionnels. Les tokens de prévision sont exclus du cache
de génération ; le chemin de réponse conserve les poids de base. Les auteurs
emploient aussi ce score pour router les demandes. Ce précédent rapproche
encore l'architecture publiée de notre frontière lecteur/producteur : ni
prévoir avant génération ni préserver la politique de réponse avec un canal
adapté ne sont, en soi, nouveaux. Leurs expériences ne testent pas notre
permutation cachée d'une mémoire épisodique. Cette différence délimite un
test possible ; elle n'établit ni nouveauté de l'ensemble ni conscience.

[Greenewald et al., *Activated LoRA*, version 2, §§3–4.1](https://arxiv.org/html/2504.12397v2)
apportent un précédent directement applicable à la frontière du cache : les
adaptations ne s'activent que sur les tokens suivant leur invocation. Le cache
antérieur du modèle de base reste réutilisable. Leur application à l'incertitude
évalue des réponses déjà produites et apprend des scores issus d'un estimateur
préalablement calibré. Notre cible prospective reste différente : prévoir une
conséquence qui n'a pas encore été exécutée sous l'état altéré. Le principe de
lecture adaptée d'un cache de base, y compris pour l'incertitude, est donc déjà
publié. Le [contrôle du lecteur](RETAINED_READER_CONTROL_PROTOCOL.md) reprend
cette frontière explicitement, sans revendiquer une architecture nouvelle.

Trois dispositifs doivent rester distingués : un prédicteur externe qui lit
des caractéristiques internes, un adaptateur intégré qui produit une prévision,
et un mécanisme qui utilise cette prévision pour choisir une action. Le premier
peut diagnostiquer une information accessible sans établir les deux autres.
Un entraînement du second ne démontre pas automatiquement l'utilité du troisième.

Le code actuel attache chaque cache à son producteur et à ses poids inchangés.
Changer les poids puis relire silencieusement ce cache casserait cette garantie.
Une expérience d'apprentissage devra soit reconstruire les trajectoires sous les
poids entraînés, soit définir explicitement un producteur gelé et un lecteur
distinct. Cette seconde option a une interprétation fonctionnelle possible,
mais ne doit pas être présentée comme la continuation intacte du même modèle.

Les comparaisons nécessaires restent : texte seul, état courant, référence
propre privilégiée si utilisée, codes inversés, perturbations sans conséquence,
et nouveaux épisodes réservés. Le partage des tableaux et des épisodes doit
être contrôlé avant toute séparation apprentissage/test. Les 24 épisodes déjà
examinés ne peuvent fournir une confirmation indépendante. Ces pistes concernent
la prévision et le contrôle de capacités ; aucune des trois sources n'établit
une méthode pour produire l'expérience subjective de sa propre existence.
