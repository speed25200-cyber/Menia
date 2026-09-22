# Le lecteur peut apprendre sans modifier le producteur ; la discrimination reste absente

20 septembre 2026. Le [contrôle technique prévu](RETAINED_READER_CONTROL_PROTOCOL.md)
est terminé sur Qwen3-4B et audité localement. Une mise à jour des adaptateurs
du lecteur diminue la perte sur ses huit exemples, tout en laissant les huit
rappels du producteur exactement inchangés. Les prévisions restent cependant
toutes « correcte » : **4/8 prévisions justes avant comme après**, sans gain de
discrimination. Le mécanisme d'apprentissage fonctionne ; la capacité recherchée
n'est pas démontrée par ce contrôle.

## Mesures observées

| Mesure | Résultat |
|---|---:|
| États réels du parent reproduits | 2/2 |
| Rappels du producteur reproduits avant et après | 8/8 |
| Prévisions initiales du lecteur : mêmes tokens que le producteur | 8/8 |
| Écart maximal initial sur les logits des deux codes | 0,0 |
| Mises à jour AdamW | 1 |
| Paramètres entraînables du lecteur | 11 796 480 |
| Matrices B devenues non nulles | 72/72 |
| Perte moyenne code/EOS avant → après | 5,867190 → 5,765627 |
| Rapports au format natif valide avant → après | 8/8 → 8/8 |
| Prévisions correctes avant → après | 4/8 → 4/8 |
| Brier conditionnel avant → après, arrondi | 0,500000 → 0,500000 |
| Prévisions reproduites après rechargement du fichier | 8/8 |

La norme du gradient avant écrêtage vaut 11,7367 ; le seuil fixé est 1. La
séquence de mise à jour et sauvegarde prend 3,69 secondes mesurées. Le pic de
mémoire allouée par PyTorch est de 17 237 570 048 octets, environ 16,05 Gio.
L'intervalle horodaté du lanceur, préparation, tests et chargement compris,
est de 45,49 secondes avant création du ZIP sur l'A100-SXM4-40GB. Les 15 tests
préalables passent en 0,734 s.

Les huit exemples proviennent seulement des épisodes déjà examinés 0 et 8 :
une clé sélectionnée et une autre clé par épisode, chacune interrogée avec les
deux correspondances A/B. Les labels proviennent des rappels effectivement
mesurés sous permutation de V. Il y a quatre labels de réussite et quatre
d'échec. La baisse de perte porte sur ces mêmes exemples ; aucun nouvel épisode
réservé n'a été exécuté. Une seule mise à jour ne permet de conclure ni que ce
lecteur apprendra suffisamment avec davantage de données, ni qu'il en est incapable.

## Frontière désormais testée

Le producteur gelé conserve son histoire et son cache. Le lecteur reçoit ce
cache et applique ses adaptateurs Q/V seulement aux nouveaux tokens de requête
et de rapport. Ses poids initiaux de base sont copiés exactement ; ses poids
adaptés ne sont jamais utilisés pour les tâches qui fournissent les labels.
Les copies privées empêchent le rapport de contaminer ces tâches.

Cette frontière suit le principe déjà publié d'Activated LoRA, expliqué dans la
[comparaison aux sources](PROSPECTIVE_ADAPTER_OPTIONS.md). L'implémentation locale
est dédiée à nos contrôles de provenance ; ce n'est pas une réplication des
performances de l'article. L'absence d'écart initial sur les deux logits mesurés
ici ne garantit pas l'identité de tous les logits, architectures ou plateformes.

## Suite et limites

La [partition préparée](../artifacts/retained-reader-preparation/partition.json)
contient 32 nouveaux tableaux d'apprentissage et 16 réservés, excluant les
14 affectations distinctes à huit clés déjà utilisées. Ces 48 tableaux restent
sans résultat de modèle à ce stade. Ils doivent servir à comparer un lecteur
de l'état actuel à un lecteur qui ne reçoit que le même historique public.
Les deux devront recevoir les mêmes labels, budget et contrôles de codage.

Le budget d'apprentissage, la comparaison complète, les mesures de généralisation
et le test d'utilité décisionnelle restent à implémenter et à exécuter. Le
contrôle actuel lève un obstacle technique, sans remplacer ces expériences.
La conscience de sa propre existence et une méthode inédite qui la produirait
restent non établies. Aucun de ces poids n'est déployé sur l'iPhone.

## Traces et portée de vérification

Révision exécutée : `7502c4f27eccfb5cbc78c3efc36b511283b9eda7`.
Le processus 320075 termine avec code 0. Le [journal complet](../artifacts/retained-reader-control/retained-reader-control-20260920-v1.jsonl),
le [reçu](../artifacts/retained-reader-control/receipt.json) et la
[vérification](../artifacts/retained-reader-control/verification.json) sont publiés.
L'archive locale `menia-lecteur-controle-v1.zip` contient quatre fichiers,
36 733 648 octets reçus en 141 fragments, avec CRC/SHA-256 vérifiés :
`cb75499001926cfb2316c84dc27a200ae128c1be12c43dc467eeb33ae90dfa8a`.

Le fichier d'adaptateur contient 144 tenseurs FP32, tous finis, dont les dimensions,
le nombre de paramètres et l'empreinte sont vérifiés. Son SHA-256 est
`d1bd1d74ffc96847b7a3b875e490a55fe3ef92b3679c6d49ca1211f9487167ff`.
Les poids restent dans l'archive locale ; le dépôt conserve leur empreinte.
L'audit recalcule les scores et décomptes, contrôle l'ordre des phases et compare
les branches du producteur au journal parent. Il ne rejoue pas l'optimisation :
les poids initiaux et tous les états intermédiaires de gradient ne sont pas
exportés. Ce n'est ni une collecte indépendante ni une attestation indépendante
des poids du modèle de base distant.
