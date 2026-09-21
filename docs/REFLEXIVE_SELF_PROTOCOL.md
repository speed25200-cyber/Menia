# Protocole d'audit des preuves de soi réflexif

L'audit suit la lecture de *No Selves, No Consciousness*, de son supplément
(sections D.3 à D.6) et des expériences 2 et 3 de son code public. Il fixe les
vérifications ci-dessous avant génération du rapport. Il ne reproduit pas les
trois expériences Monte-Carlo de l'auteur et n'évalue pas la conscience de Menia.

Source du code inspecté : dépôt `ViscousLemming/Technical-Appendices`, commit
`259e0ec398dbbd3b5b0a0312a0ea8de305d4bb2c`, fichier
`Papers/No_Selves_No_Consciousness/NSNC_Experiments.py`, SHA-256
`f652f3bc484311bce51a974420811aa29fe6ba68a3eb327a17fe61380cad1ca4`.
Le fichier externe n'est ni exécuté ni incorporé dans Menia.

## Communication

Le message privé `m` et le type du décodeur `t` sont des bits indépendants et
uniformes. Le destinataire applique `signal XOR t`. La sonde est le bit zéro.
Sa réponse, conservée dans un bit `r`, sert à émettre `m XOR r`.

Nous énumérons les quatre mondes `(m,t)`, puis les quatre politiques déterministes
qui dépendent seulement de `m`. Leur mélange couvre les politiques aléatoires
sans information sur `t` : l'espérance est linéaire en leurs probabilités.

Une extension analytique propre à cet audit ajoute des inversions indépendantes
de la sonde et de la réponse finale, avec probabilités respectives dans
`{0, 1/10, 1/4, 1/2}` et `{0, 1/10, 1/2}`. Les probabilités sont calculées
exactement comme fractions, sans simulation. Nous comparons l'énumération des
événements à `1 - a - b + 2ab`.

Deux contrôles de validité sont fixés : inverser le décodeur après une sonde
parfaite, puis le retirer indépendamment et uniformément. La politique conserve
la réponse périmée. Il ne s'agit pas d'un test d'apprentissage de ces changements.

## Engagement dans un jeu de confiance

Nous utilisons le jeu ponctuel du théorème 5 : coopération `(1,1)`, exploitation
`(1+g,-1)`, absence de confiance `(0,0)`, et coût `c` payé lorsqu'un engagement
supprime l'exploitation. Le choix calculé est de s'engager si `c < 1`, sinon
de ne pas s'engager. Le destinataire fait confiance si l'engagement est visible.

Vérifier les gains de déviation à chaque étape pour
`c ∈ {0, 1/4, 3/4, 1, 5/4}` et `g ∈ {1/10, 1, 2}`. À `c = 1`, le choix
de ne pas s'engager n'exclut pas d'autres équilibres. Les coûts, actions et
engagements sont uniquement des objets mathématiques de recherche.

## Interprétation et vérification

La comparaison demande si ces tâches imposent une architecture interne
explicitement récursive, au-delà de l'information et de la politique nécessaires.
Elle ne prétend pas réfuter la définition fonctionnelle des niveaux de soi de
l'auteur : un dispositif compact peut lui aussi satisfaire cette définition.
Ni son absence d'expérience ni sa conscience ne sont présupposées.

Le rapport utilise des fractions et doit être reproduit exactement. Les contrôles
comprennent des cas d'échec ; aucun résultat positif ne sera appelé preuve
phénoménale. Aucun réseau n'est entraîné, aucune capacité n'est ajoutée à l'agent.
