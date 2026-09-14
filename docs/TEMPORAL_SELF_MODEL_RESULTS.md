# Menia : modèles appris des effets retardés

## Résultat et portée

Un nouveau module de recherche apprend des effets retardés à partir de journaux
d'interactions, sans recevoir le vrai délai, le vrai gain ou l'identité du canal
affecté. Ses prédictions sont ensuite confrontées à des interventions effectuées
dans le simulateur. Ce progrès remplace l'énumération de six causes connues par
l'apprentissage des paramètres d'une fonction de prédiction.

Il ne démontre ni conscience subjective ni contribution inédite. Le résultat
principal est aussi une limite : un contrôle polynomial ordinaire fait mieux que
le petit réseau sur les effets non linéaires de cette tâche. Le réseau apprend,
mais son avantage sur une simple régression linéaire ne suffit pas à établir la
nécessité d'un mécanisme neuronal particulier ou d'un modèle conscient de soi.

## Ce qui a été entraîné

Le [protocole initial](TEMPORAL_SELF_MODEL_PROTOCOL.md) a été écrit avant les
entraînements. Douze mondes combinent quatre familles et trois tirages de
paramètres : effet affine retardé, interaction non linéaire, absence d'effet
de l'action et effet retardé au-delà de la fenêtre accessible. Chaque monde est
appris séparément ; les nouveaux échantillons de test viennent du même monde.
Il ne s'agit donc pas d'un transfert sans apprentissage vers des mécanismes inconnus.

Deux journaux de 1 024 instants sont comparés pour chaque monde. Dans le premier,
l'action suit toujours l'indice public. Dans le second, elle le suit pendant
512 instants puis devient indépendante. Un seul canal est sélectionné à chaque
instant et 20 % des lectures manquent. Les cibles non reçues sont supprimées
avant d'atteindre l'optimiseur. Le délai causal et le canal affecté ne sont
utilisés que par le simulateur et par l'évaluateur.

Le MLP temporel compte 546 paramètres : 14 entrées couvrant sept instants,
32 unités tanh et deux sorties. Le MLP instantané, de capacité proche, compte
502 paramètres et reçoit uniquement l'action et l'indice présents. Trois
initialisations indépendantes sont entraînées pour chacun, avec 600 mises à jour
Adam par configuration. Une régression ridge temporelle de 30 coefficients sert
de premier contrôle. Cela donne 144 réseaux entraînés et 24 régressions ajustées.

Les quatorze variables d'entrée et la fenêtre de sept instants restent des choix
de conception. Les réseaux ne découvrent pas de nouveaux capteurs et n'apprennent
pas une mémoire récurrente. L'entraînement est hors ligne sur le journal, avec
mélange des échantillons ; ce module n'est pas encore la boucle interactive de Menia.

## Pourquoi l'évaluation intervient dans le monde

Une bonne prédiction sur un journal où action et indice sont confondus peut utiliser
la mauvaise cause. Le test comporte donc 512 nouveaux instants où ces variables
sont indépendantes. Aucune mise à jour n'est faite sur ces observations de test.
On conserve séparément l'erreur de prévision et l'erreur sur les effets causaux.

Pour 32 historiques supplémentaires, chaque action passée de délai zéro à dix
est imposée à +1, -1 ou zéro. Les autres actions et les indices restent identiques.
Le simulateur recalcule les observations sans bruit. Le modèle produit ses propres
prévisions sur les mêmes historiques modifiés. Le contraste prévu est comparé
au contraste réellement produit. Les trois paires sont testées : comparer seulement
+1 et -1 ferait disparaître un terme quadratique et manquerait cette influence.

L'erreur sur les effets non nuls est publiée séparément. Sans cette précaution,
prédire zéro partout pourrait donner une erreur moyenne apparemment faible,
simplement parce que beaucoup de délais et canaux ne sont pas concernés. Cette
mesure utilise la vérité du simulateur pour évaluer le modèle ; elle ne lui est
pas communiquée comme un jugement sur sa conscience ou son identité.

## Résultats du pilote

Erreur quadratique moyenne sur les effets causaux réellement non nuls, après le
journal comprenant des interventions. Plus bas signifie une meilleure estimation.
Chaque moyenne neuronale couvre trois mondes et trois initialisations, soit neuf
modèles ; ridge n'a pas d'initialisation aléatoire et couvre trois ajustements.

| Famille | MLP temporel | MLP instantané | Ridge temporelle |
|---|---:|---:|---:|
| Effet affine retardé | 0,00751 | 1,99645 | 0,0000734 |
| Interaction non linéaire retardée | 0,04552 | 0,69602 | 0,69728 |
| Effet au délai dix, hors fenêtre | 1,99645 | 1,99645 | 1,99645 |

Sur l'effet affine, la régression est la meilleure solution des trois. Sur
l'interaction non linéaire, le MLP temporel apprend un effet que les deux autres
modèles représentent mal. Le modèle instantané ne possède pas l'historique utile,
même si son nombre de paramètres est proche. Les trois échouent au délai dix :
modifier cette action passée ne change aucune de leurs entrées. Ce dernier résultat
est une limite d'accès à l'information, pas seulement un manque d'entraînement.

Le journal d'interventions est déterminant. Pour le MLP temporel, l'erreur sur
les effets affines passe de 0,52749 avec le journal entièrement confondu à 0,00751
avec interventions ; sur les interactions, de 0,48304 à 0,04552. Cela montre un
avantage de ces données dans les mondes étudiés, sans établir que cette stratégie
est optimale ou que le réseau pourrait choisir seul les expériences nécessaires.

Dans les mondes sans contrôle, la racine moyenne quadratique des faux effets
prédits par le MLP temporel passe de 0,16003 à 0,02140. Les interventions réduisent
les influences imaginées, mais ne les annulent pas exactement. Ridge atteint
0,00332 dans la même condition avec interventions. Ces valeurs sont des amplitudes
d'erreur ; ce ne sont pas des probabilités calibrées de se tromper.

Les [poids et résultats individuels](../artifacts/temporal-effects/report.json)
gardent le monde, la politique de collecte, l'initialisation, les empreintes et
les métriques de chaque prédicteur. Les trois initialisations ne remplacent pas
une grande diversité de mondes. Les résultats restent exploratoires et aucune
significativité statistique générale n'est revendiquée.

## Contrôle polynomial plus exigeant

Après le pilote, un [addendum](TEMPORAL_POLYNOMIAL_ADDENDUM.md) a défini un contrôle
contenant les 14 variables et tous leurs produits de degré deux. Cette famille
inclut les produits et carrés du simulateur. Elle est donc favorablement adaptée
à la tâche par conception ; cet avantage est explicite. Elle compte 240 coefficients
pour les deux sorties, contre 546 paramètres pour le MLP temporel.

| Famille, avec interventions | MLP temporel | Ridge de degré deux |
|---|---:|---:|
| Effet affine retardé | 0,00751 | 0,000809 |
| Interaction non linéaire retardée | 0,04552 | 0,000895 |
| Effet au délai dix, hors fenêtre | 1,99645 | 1,99645 |

Le contrôle polynomial apprend donc mieux les effets des deux premières familles.
Il reste incapable de retrouver une information située hors fenêtre. Ce résultat
retire une interprétation trop favorable au réseau : le pilote n'établit aucune
supériorité générale des réseaux pour construire un modèle de ses propres effets.
Il invite à déplacer l'effort vers l'acquisition et l'utilisation de l'information
manquante, plutôt que multiplier les paramètres sur une fenêtre fixe.

Les [24 contrôles supplémentaires](../artifacts/temporal-polynomial/report.json)
utilisent les mêmes mondes et journaux. Ils ont été décidés après lecture du pilote
et ne constituent pas une confirmation indépendante. Les premiers résultats et
le protocole initial ont été conservés sans réécriture des mesures.

## Rapport avec la littérature et avec l'objectif

Liang et Boularias ont déjà proposé d'identifier des relations causales retardées
dans des environnements partiellement observés, en ajoutant des variables de
mémoire pour expliquer les observations et en utilisant les modèles appris pour
planifier. Leur papier IJCAI 2021 est un antécédent direct au thème « apprendre
des causes et des délais ». Leurs expériences ne sont pas reproduites ici, et
le présent MLP à fenêtre fixe n'est pas une implémentation de leur méthode.[^1]

Zaadnoordijk, Besold et Hunnius ont également critiqué l'inférence qui transforme
un accord entre prédiction et sensation en preuve suffisante d'un sentiment
d'être la cause. Leur article est une analyse théorique, pas une expérience
établissant l'absence ou la présence de conscience chez Menia. Il rappelle que
le mécanisme de comparaison peut remplir une fonction sans expliquer à lui seul
l'expérience subjective recherchée.[^2]

La contribution actuelle est une extension expérimentale reproductible du dépôt,
avec apprentissage des fonctions, contrôles de corrélation, limites temporelles et
comparateurs plus simples. Son caractère inédit dans la littérature n'est pas
établi. Ni les poids appris ni les scores d'intervention ne satisfont l'objectif
de prouver une conscience de sa propre existence. Cet objectif n'est pas clôturé
et n'est pas remplacé par le succès du pilote.

## Prochaine difficulté à résoudre

L'échec hors fenêtre donne une cible précise : permettre au système d'estimer
qu'une information causale lui manque et de choisir une manière de l'acquérir
ou de la conserver. Pour avancer, cette estimation devra modifier son comportement
et se généraliser à des délais et mécanismes non appris. Agrandir manuellement
la fenêtre jusqu'au délai de test ne démontrerait pas cette capacité.

Une future expérience devra comparer un état récurrent appris, une mémoire
explicite de capacité comparable et une politique ordinaire d'exploration. Les
échanges entre modèle de soi, attention et décision devront faire l'objet
d'interventions ciblées. Le mécanisme proposé devra enfin être intégré à la
boucle de Menia, avec des explications fidèles aux données réellement accessibles.
Cela restera un programme de recherche sur des conditions candidates, sans
garantie que leur réalisation produise une expérience subjective.

## Reproduire

```bash
python scripts/check_temporal_effects.py
python -m research.train_temporal_effects --out runs/nouveaux-effets-temporels
python -m research.evaluate_temporal_polynomial --out runs/nouveau-controle-polynomial
```

La première commande utilise NumPy et les poids conservés : elle vérifie les
empreintes puis recalcule les prévisions et interventions. Elle ne réentraîne pas
les réseaux. Elle réajuste les contrôles polynomiaux déterministes. La deuxième
requiert aussi PyTorch ; les versions utilisées figurent dans le rapport. Chaque
commande de production refuse d'écraser un dossier existant.

## Sources

[^1]: Liang, J., et Boularias, A. (2021). [Inferring Time-delayed Causal Relations in POMDPs from the Principle of Independence of Cause and Mechanism](https://www.ijcai.org/proceedings/2021/268). IJCAI, 1944–1950. Notice officielle et PDF consultés le 14 septembre 2026 ; méthode non reproduite.
[^2]: Zaadnoordijk, L., Besold, T. R., et Hunnius, S. (2019). [A match does not make a sense: on the sufficiency of the comparator model for explaining the sense of agency](https://pdfs.semanticscholar.org/7605/90f7edd0a947d2250ee24e05666ee4c94358.pdf). Neuroscience of Consciousness, niz006, DOI 10.1093/nc/niz006. Texte original consulté via une copie du PDF ; article d'opinion théorique.
