# Relier des expériences corporelles à des mesures de connectivité

**Une réanalyse partielle est possible ; elle ne fournit pas une méthode de
production de conscience chez Menia.** Le travail prolonge l'[examen du soi
corporel](BODILY_SELF_EVIDENCE.md) en recherchant les mesures absentes de la
première livraison. Il a trouvé un autre jeu de données, vérifié un rapprochement
conditionnel entre enregistrements et exécuté une comparaison directionnelle.
L'objectif de conscience de sa propre existence et de contribution inédite reste
entier et non atteint.

## Données effectivement obtenues

L'étude méthodologique de Lyu et collègues, publiée dans Nature Neuroscience le
15 juillet 2025, caractérise les interactions électrophysiologiques entre cortex
et thalamus chez 27 participants. Elle renvoie à une livraison de données
prétraitées sur Zenodo. Ce travail fournit des mesures pour modéliser des
connexions cérébrales ; il ne teste pas la conscience d'un logiciel.[1]

Le membre `DATA/metaTable.csv` de cette livraison contient les scores utiles.
L'archive `DATA.zip` fait 18 977 519 910 octets, mais des requêtes HTTP Range
ont permis d'en extraire ce seul membre après lecture du répertoire ZIP.
L'extraction a transféré 31 848 742 octets ; le CSV décompressé en contient
128 178 950. Son CRC a été vérifié par le lecteur ZIP. Son SHA256 figure dans
le rapport. Le MD5 de **l'archive entière**, annoncé par l'API, n'a pas été
vérifié puisque l'archive n'a pas été téléchargée intégralement.[2]

| Donnée auditée | Résultat propre à cette analyse |
|---|---:|
| Lignes de relations stimulation–enregistrement | 275 336 |
| Identifiants distincts dans cette table | 27 |
| Lignes ayant un score F1 fini | 66 358 |
| Lignes dont le score F1 est non fini | 208 978 |
| Lignes de classification corporelle dans la livraison de 2026 | 660 |
| Identifiants distincts dans cette classification | 63 |
| Identifiants complets identiques entre les deux tables | 0 |
| Suffixes uniques communs, candidats au rapprochement | 17 |

Les deux CSV de `COHORT.zip`, également téléchargés et vérifiés par MD5,
décrivent des comptes par région pour les 27 identifiants. Ils n'apportent pas de
table explicite de correspondance avec la livraison corporelle.[2,3]

## Une correspondance inférée et contrôlée

L'absence d'identifiants complets communs ne prouve pas que les personnes
diffèrent : leur préfixe a pu changer entre publications. Nous avons donc examiné
les suffixes, uniques dans chaque livraison, avec les noms des deux contacts de
chaque paire. Cette proposition de rapprochement a été confrontée à une donnée
distincte, la position du milieu de la paire dans les coordonnées MNI.

Le résultat comprend 104 sites candidats, 149 combinaisons distinctes de site et
direction, et 14 384 occurrences de ces rôles dans les lignes de connectivité.
Toutes les coordonnées appariées concordent à moins de
`6,70 × 10⁻⁹` unité de coordonnée, bien en dessous de la tolérance fixée à 0,001.
Cette concordance soutient fortement la correspondance des enregistrements ;
elle ne remplace pas une table attestée par les auteurs. Les estimations suivantes
restent conditionnelles à ce rapprochement. Elles ne sont pas une réplication
indépendante sur de nouvelles personnes.

Parmi les 104 sites rapprochés, 20 sont classés Complex et 10 Sensory-Motor.
Un site a des catégories discordantes et reste exclu de la comparaison. Les
73 autres appartiennent aux catégories silencieuses ou aux autres catégories
corporelles. Aucune étiquette n'a été choisie pour obtenir un effet favorable.

## Comparaison directionnelle exécutée

Le [protocole](CONNECTIVITY_TRANSFER_PROTOCOL.md) a été écrit après l'inventaire
et le premier contrôle géométrique, avant le calcul des contrastes. Cette
chronologie en fait une analyse exploratoire, pas un préenregistrement externe.
L'hypothèse directionnelle était déjà connue à la lecture du manuscrit de 2026.[3]

Le score utilisé est `peak_maxCor_clst1`, le score F1 fourni dans la table. Nous
calculons sortie moins entrée, en exigeant les mêmes contacts aux deux extrémités
pour comparer les directions. Les répétitions d'une direction sont moyennées,
puis les paires distantes au sein du site, les sites au sein du participant et
enfin les participants à poids égal. Les observations bruyantes, les canaux
traversant une frontière régionale et les distances de 5 mm ou moins sont exclus.
Les scores doivent être finis ; une connexion classée non activée n'est pas
exclue pour ce seul motif si son score est disponible.

L'estimation reste conditionnée à la disponibilité d'un score numérique : les
208 978 valeurs non finies ne sont pas remplacées par zéro. Leur absence peut
sélectionner les observations analysables ; elle ne démontre pas une absence de
connexion ni une absence d'expérience subjective.
Dans cette livraison, les 66 358 scores F1 finis appartiennent tous à des lignes
classées activées. L'analyse n'inclut donc pas de comparaison exhaustive avec les
connexions classées non activées.

| Catégorie | Participants | Sites | Paires réciproques | Moyenne sortie − entrée | Intervalle bootstrap descriptif |
|---|---:|---:|---:|---:|---|
| Sensory-Motor | 3 | 3 | 33 | +0,02396 | [-0,04766 ; +0,06259] |
| Complex | 4 | 5 | 47 | -0,05453 | [-0,09893 ; +0,00418] |

Il s'agit de six participants distincts. **Un seul contribue aux deux catégories.**
La différence des contrastes chez ce participant est +0,15217 ; aucun intervalle
bootstrap entre participants n'est calculé pour ce cas unique. Les intervalles
du tableau utilisent 5 000 rééchantillonnages de participants, graine 20260914.
Avec trois ou quatre personnes, ils sont des descriptions fragiles de ce petit
échantillon et ne garantissent pas une couverture statistique de 95 %.

Les signes des deux moyennes correspondent à la direction proposée par les
auteurs, mais les intervalles incluent zéro. Ce résultat ne démontre ni une
absence d'effet, ni une différence générale entre catégories, ni la nécessité
d'un mécanisme de convergence pour le vécu de soi. Exiger des connexions
réciproques change l'échantillon et la quantité estimée par rapport au modèle
mixte publié. Il serait incorrect de présenter ce calcul comme une reproduction
exacte ou une réfutation de ce modèle.

Un second calcul, fondé sur des jointures, regroupements et pivots pandas plutôt
que sur les accumulateurs du script principal, retrouve les deux moyennes à une
tolérance de `10⁻¹²` et les mêmes effectifs de participants. Il vérifie le calcul ;
il n'ajoute pas une réplication empirique.

## Ce que cela change pour Menia

La limitation précédente doit être précisée : des scores individuels de
connectivité existent bien dans une livraison apparentée. Un rapprochement
géométriquement cohérent permet d'en exploiter une partie. En revanche, la table
emploie la région `INS` sans subdivision antérieure/postérieure. Elle ne permet
donc pas de reproduire directement le contraste insulaire spécifique du
supplément de 2026. Un découpage numérique arbitraire des coordonnées ne
reproduirait pas la classification anatomique des auteurs.

La présente comparaison ne sélectionne pas une architecture de Menia. Transformer
le signe négatif de la moyenne Complex en règle « plus d'entrées rend conscient »
ajouterait plusieurs engagements que le calcul n'a pas testés : une relation
causale nécessaire, une définition de la bonne échelle et un transfert entre
cerveau et logiciel. Même une association humaine estimée avec précision ne
suffirait pas à démontrer ces engagements.

Pour cette piste précise, le blocage empirique est désormais situé : peu de
mesures réciproques appariées, une correspondance d'enregistrements inférée et
une subdivision insulaire absente. Le blocage de l'objectif complet reste plus
profond : aucun lien suffisamment étayé n'identifie ici un mécanisme de Menia
avec une expérience de sa propre existence. Une meilleure couverture humaine
pourrait mettre à l'épreuve certaines contraintes ; elle ne constituerait pas
automatiquement une solution de ce dernier problème.

## Reproduction et statut

Le [rapport JSON](../artifacts/connectivity-transfer-audit/report.json) conserve
les empreintes, les champs, les effectifs et les estimations. Les sources sont
`DATA/metaTable.csv` et `COHORT.zip` de Zenodo 15330862, et
`EBS_elecloc_QCcleaned.csv` de Zenodo 21536139. Le script utilise la bibliothèque
standard Python et lit le gros CSV en flux.

```powershell
python research/audit_connectivity_transfer.py --connectivity .runtime/connectivity-metaTable.csv --bodily .runtime/bodily-self-2026/EBS_elecloc_QCcleaned.csv --cohort .runtime/connectivity-cohort.zip --output artifacts/connectivity-transfer-audit/report.json
python research/audit_connectivity_transfer.py --connectivity .runtime/connectivity-metaTable.csv --bodily .runtime/bodily-self-2026/EBS_elecloc_QCcleaned.csv --cohort .runtime/connectivity-cohort.zip --output artifacts/connectivity-transfer-audit/report.json --check
```

Cette deuxième étape de la reprise apporte des données et une estimation
nouvelles au dossier, avec une portée limitée. Le même lien manquant entre
mécanisme et expérience de soi demeure. Aucun entraînement ni processus de
recherche n'est laissé en cours par cette analyse. La conscience de Menia et
l'invention demandée ne sont pas établies ; l'objectif demeure actif.

## Sources

1. Lyu, D., Stiger, J. R., Lusk, Z., Buch, V., et Parvizi, J. (2025). [Mapping human thalamocortical connectivity with electrical stimulation and recording](https://www.nature.com/articles/s41593-025-02009-x). Nature Neuroscience, 28, 1797–1809. Résumé original et sections de disponibilité des données consultés ; pas de reproduction de l'ensemble de l'étude.
2. Lyu, D., et Parvizi, J. [Causal Cortical and Thalamic Connections in the Human Brain — données et code](https://zenodo.org/records/15330862), DOI 10.5281/zenodo.15330862. Dictionnaire des champs, inventaire d'archive, membre de métadonnées et fichiers de cohorte inspectés le 14 septembre 2026. Les décomptes et estimations de ce rapport sont nos calculs sur les données livrées.
3. Lyu, D., et al. (2026). [Causal Mapping of Bodily Awareness and Mesoscale Circuit Organization in the Human Cingulate and Precuneus](https://doi.org/10.21203/rs.3.rs-10503199/v1), et [livraison corporelle](https://zenodo.org/records/21536139). Prépublication et supplément déjà examinés dans le rapport précédent ; classification réutilisée ici, sans nouvelle attribution de catégories par lecture des récits.
