# Protocole : apprendre l'actionneur et les voies d'observation

Pilote exploratoire spécifié le 15 septembre 2026 avant son exécution. L'objectif
phénoménal demeure inchangé ; aucun diagnostic de panne n'est un test de conscience.
L'extension doit alimenter le contrôleur `BindingController` avec des paramètres
appris à partir de commandes et mesures, au lieu de les fournir tous d'avance.

## Expérience autorisée au modèle

Chaque cycle comporte une commande d'étalonnage u depuis une position remise à
zéro par le simulateur. Le déplacement réel est g*u. Deux capteurs renvoient
`p = r*g*u + bruit_p` et `v = s*g*u + b + k*(g*u)² + bruit_v`.
Le bruit p a un écart-type réel de 0,15. Le modèle apprend deux régressions
affines commande–mesure, avec biais, pente et variance résiduelle inconnus.
Il suppose que le capteur de référence garde r=1. Cela identifie g depuis p,
puis s depuis le rapport des pentes. Cette hypothèse n'est pas une découverte
de l'agent. Le terme non linéaire k est absent de sa famille de modèles.

Une mesure est un objet contenant seulement p et v. La commande et ses prévisions
sont enregistrées avant l'appel du simulateur ; les mises à jour suivent le retour
des mesures. Ni paramètres réels, ni type de perturbation, ni coût moteur réel
ne sont fournis au modèle. L'étalonnage garantit un seul objet mesuré : cette
opportunité et la remise à zéro sont conçues par l'expérience.

Les deux régressions utilisent au plus 24 couples commande–mesure. L'ajustement
est une régression bayésienne normale–inverse-gamma : moyenne a priori (0,1),
précision des coefficients 0,01*I, alpha=2 et beta=variance nominale (0,15² pour
p, 0,5² pour v). La variance résiduelle est estimée ; le prior n'est pas figé.
Une fenêtre glissante permet de réapprendre après un changement non annoncé,
mais ne constitue pas une détection apprise de la date de changement.

Les commandes candidates sont −1, 0, +1. Le choix actif maximise
`log(1 + phi' * Lambda_inverse * phi)`, avec phi=(1,u), après retrait de
l'observation qui serait évincée à la prochaine mise à jour. C'est un critère
géométrique de plan d'expérience, sans revendication d'optimisation générale
de la valeur de l'information. Les égalités choisissent le premier candidat.
Le choix peut se réduire à une alternance déterministe dans ce modèle simple.

## Usage pour agir

Après l'étalonnage, huit positions x indépendantes sont tirées de N(0,1).
Le signal visuel de chacune porte sur z=x+d, avec d=0 une fois sur deux,
sinon d~N(0,16). Il vaut `s*z+b+k*z²+bruit_v`. Le modèle corrige le biais et
l'échelle estimés et transmet le signal à `BindingController`, en utilisant
la variance visuelle estimée. Le prior de cause commune 0,5 et la distribution
des positions/proxys restent fournis, sans apprentissage dans ce pilote.

Le déplacement moteur choisi vaut
`(cible-position_estimee)*E[g]/(E[g]²+Var(g))`, limité à [−6,+6]. La cible est
1,5. Cette règle intègre le deuxième moment du gain, mais remplace les autres
paramètres par leurs estimations ponctuelles ; elle ne réalise pas une inférence
bayésienne jointe exacte. Si |g| ou |s| est inférieur à 0,05, la vision corrigée
est indisponible et l'estimation de position revient à l'ancre. La variance
sensorielle a un plancher numérique de 1e−6.

Le coût de tâche est `(x+g*u-cible)²`, calculé uniquement par l'évaluateur.
Les mouvements de tâche ne mettent pas à jour les régressions : seules les
commandes d'étalonnage avec retour des deux capteurs le font. Les essais de tâche
sont réinitialisés ; la persistance est celle du modèle d'étalonnage.

## Conditions fixées et comparateurs

32 graines par condition, 96 cycles, changement au cycle 32 sans signal donné
au contrôleur adaptatif. Avant le changement : g=r=s=1, b=k=0, sigma_v=0,5.
Conserver les neuf conditions après changement :

| Condition | g | r | s | b | sigma_v | k |
|---|---:|---:|---:|---:|---:|---:|
| inchangé | 1 | 1 | 1 | 0 | 0,5 | 0 |
| moteur inversé | −1 | 1 | 1 | 0 | 0,5 | 0 |
| moteur affaibli | 0,5 | 1 | 1 | 0 | 0,5 | 0 |
| biais visuel | 1 | 1 | 1 | 1,5 | 0,5 | 0 |
| échelle visuelle | 1 | 1 | 0,5 | 0 | 0,5 | 0 |
| bruit visuel | 1 | 1 | 1 | 0 | 2 | 0 |
| combiné | −0,5 | 1 | 1 | 1 | 1 | 0 |
| référence trompeuse | 1 | 0,5 | 0,5 | 0 | 0,5 | 0 |
| vision non linéaire | 1 | 1 | 1 | 0 | 0,5 | 0,15 |

Comparer : modèle bayésien adaptatif et sondes actives ; modèle figé après les
32 premiers cycles avec sondes alternées ; modèle adaptatif avec u=0 ; régressions
ordinaires par moindres carrés sur la même fenêtre et sondes alternées. Le contrôle
ordinaire conserve l'incertitude estimée sur le gain dans sa règle de mouvement.
Avant trois observations de rang deux il utilise le prior pour pouvoir fonctionner.
Les sondes et leur effort quadratique seront comptabilisés séparément de l'erreur
de tâche ; aucun bénéfice net après coût d'exploration ne sera déduit sans coût fixé.

Les graines de condition/répétition sont `718000+100*condition+répétition`.
Les variantes partagent les mêmes bruits et les mêmes positions/proxys de tâche,
mais leurs commandes peuvent différer. Pas de sélection des meilleures graines.
Rapporter les phases initiale (cycles 0–31), transitoire (32–43) et finale (72–95),
ainsi que les courbes moyennes complètes pour ne pas masquer la récupération.
Un bootstrap apparié de 4 000 tirages des 32 épisodes, graine 719000+condition,
quantifiera la différence finale de coût entre chaque contrôle et l'adaptatif.
Il décrit l'erreur de simulation, sans extrapolation humaine.

## Identification et contre-exemples

Le rang de la matrice des commandes sera vérifié sans compter le prior comme
une observation. Avec uniquement u=0, la pente n'est pas identifiable. Un capteur
de référence absent empêchera d'attribuer la pente visuelle au moteur ou au capteur.
Le modèle doit conserver ce statut, même s'il produit une commande prudente.

Vérifier numériquement sur plusieurs commandes l'égalité des observations de
calibration pour (g=0,5,r=1,s=1) et (g=1,r=0,5,s=0,5), à bruits identiques.
Leurs déplacements réels diffèrent. La condition « référence trompeuse » peut
donc provoquer une attribution fausse malgré un bon ajustement : l'étalonnage
seul ne peut tester sa propre hypothèse r=1. Cette équivalence concerne les
expériences de calibration définies ici, pas toute observation supplémentaire
imaginable. Le cas non linéaire examine une rupture de la famille affine.

Tests prévus : mise à jour contre calcul matriciel indépendant ; oubli exact de
l'ancienne fenêtre ; prévision avant observation ; rang sans excitation ; identité
des mondes ambigus ; sauvegarde/reprise ; utilisation du gain, biais et bruit appris
par le contrôleur existant. Recalcul complet du rapport en CI, tolérance numérique
1e−10. Aucune performance minimale ne sera choisie après inspection des résultats.

## Diagnostic ajouté après la première exécution

La condition non linéaire conduit à un biais estimé proche de 0,15 et n'améliore
pas le coût par rapport au modèle figé. Un contrôle algébrique supplémentaire
comparera k=0,15, b=0 à k=0, b=0,15 aux commandes −1, 0 et +1, sans bruit.
Il vérifiera aussi si les 96 sondes géométriques coïncident exactement avec
l'alternance ordinaire. Ce diagnostic porte sur la couverture des commandes ;
il ne modifie ni modèles, ni critères, ni résultats du plan principal.
