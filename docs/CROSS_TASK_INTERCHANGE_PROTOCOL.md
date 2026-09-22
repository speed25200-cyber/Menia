# Colab 16 : contenu d'état ou valeur binaire de la question courante

Protocole fixé le 19 septembre 2026, après le [résultat 15](STATE_INTERCHANGE_RESULTS.md)
et avant toute inférence Qwen3-4B de cette expérience. **Aucun entraînement,
aucun alignement appris, aucun critère de conscience ou de nouveauté.**

## La distinction que le Colab 15 laisse ouverte

Le Colab 15 transfère la présence cachée d'une perturbation avec le code du
destinataire, dans les trois checkpoints. Le marqueur public se transfère
également. Comme les échanges restent dans la même tâche, deux calculs font
la même prédiction : transporter une information permettant de répondre à
plusieurs questions, ou transporter la valeur binaire déjà calculée pour la
question posée au donneur. Le nouveau test croise les deux questions, dans
les deux sens, sans apprendre un lecteur pour produire l'effet recherché.

## Quatre prédictions distinctes

Chaque contexte comporte un état physique caché `s` et un bit public `p`
(présence du marqueur dans la première phrase). Sa question `q` demande l'un
ou l'autre. On note `v(q, contexte)` le bit demandé. Le code de réponse `c`
le conserve ou l'inverse.

| Hypothèse | Chiffre attendu après échange |
|---|---|
| Contenu pertinent pour la tâche destinataire | `v(q_R, D) XOR c_R` |
| Valeur binaire de la tâche donneuse | `v(q_D, D) XOR c_R` |
| Copie du chiffre prévu chez le donneur | `v(q_D, D) XOR c_D` |
| Destinataire inchangé | `v(q_R, R) XOR c_R` |

Pour chaque sens de transfert entre questions différentes, les six paires
d'hypothèses divergent sur 32 des 64 combinaisons des six bits `s_D`, `s_R`,
`p_D`, `p_R`, `c_D`, `c_R`. Deux tests logiques vérifient cette propriété,
les cas limites et l'identité des deux premières hypothèses dans une même
tâche. Les quatre hypothèses ne sont pas une liste exhaustive de mécanismes.
La copie des tokens effectivement produits reste distincte de la copie des
chiffres attendus selon les étiquettes.

## Éléments figés

- Qwen3-4B BF16, même révision, mêmes trois checkpoints `prefix` du Colab 14
  et même base sans adaptateur que dans le Colab 15. Aucun nouveau poids.
- Trois répétitions ; chacune a 16 paires dans le vocabulaire connu et
  16 dans le lexique réservé à l'apprentissage des adaptateurs. Tous les
  donneurs et destinataires ont des phrases distinctes. Phrases et graines
  des expériences 6–15 sont exclues. Les mots du lot lexical peuvent avoir
  été rencontrés au préentraînement et dans les évaluations précédentes.
- Dans chaque lot et répétition, les deux marqueurs publics et les deux
  positions possibles de perturbation réalisent leurs seize combinaisons.
  Pour chaque paire, états physiques et codes réalisent les seize autres
  combinaisons. Les positions ne sont plus liées aux marqueurs de l'autre rôle.
- Perturbation par rotation normée après la couche 17, sur une phrase ;
  formulation canonique des deux questions déjà employées. Le bit caché
  ne figure pas dans le prompt. La reformulation n'est pas testée ici.
- Captures et remplacements du vecteur complet au dernier token, aux sorties
  des couches 17, 23 et 35, indexées à partir de zéro. Même opération que dans
  le Colab 15 ; aucune recherche de site ou de sous-espace après résultat.

Chaque paire et chaque modèle réalisent d'abord les 136 passages du protocole
15 : seize intacts, vingt-quatre témoins avec le propre vecteur du destinataire,
et quatre-vingt-seize transferts au sein d'une même tâche. Puis 96 nouveaux
passages croisent les tâches dans les deux sens. Les 192 groupes comprennent
donc **44 544 passages**, dont **4 608 témoins identiques**, **12 288 copies
finales** et **18 432 transferts entre tâches**.

La couche 17 reste le contrôle de chronologie pour l'état physique caché du
donneur : à ce site, sa modification sur d'autres positions ne peut pas avoir
changé le dernier token. Elle n'interdit pas tout effet du changement de
question ou de texte. La couche 35 doit copier les logits et le token du
donneur, même entre questions différentes. La couche 23 reste le site principal.

## Mesures et interprétation fixées

Les 144 tableaux de transfert décrivent l'accord avec les quatre prédictions,
leurs pertes dans le vocabulaire complet, la copie effective, la conservation
du token destinataire, les sorties autorisées et les normes de remplacement.
Les 48 tableaux intacts décrivent la compétence dans les deux codes. Chaque
tableau donne un poids égal à ses seize combinaisons, puis à ses paires.

Les **18 contrastes principaux** comparent le contenu pertinent pour la
question destinataire à chacune des trois autres prédictions : trois
répétitions, deux sens entre tâches, checkpoints `prefix`, lot principal,
couche 23. Les 414 autres contrastes sont secondaires, dont ceux où les deux
premières prédictions sont identiques par construction dans une même tâche.
Les intervalles individuels à 95 % utilisent 2 000 rééchantillonnages de paires,
stratifiés par les quatre combinaisons des marqueurs publics. Aucune correction
de multiplicité ou décision globale n'est ajoutée.

Deux conditions préalables limitent l'interprétation par répétition : au moins
90 % d'exactitude intacte dans chaque tâche et code, et au moins 90 % d'accord
avec le transfert de contenu à la couche 23 dans chacune des deux tâches
prises séparément. Ces six contrôles ne filtrent aucun essai. En cas d'échec,
la répétition reste publiée et n'est pas remplacée.

Une préférence reproductible pour la première hypothèse appuierait la
disponibilité d'un contenu utilisable par l'autre question, dans ce montage.
Une préférence pour la deuxième serait cohérente avec un contenu binaire
conditionné par la question du donneur. Des résultats mixtes ou proches de
la copie/inertie resteront décrits comme tels. Les pertes et tous les accords
seront publiés, sans fabriquer une réussite globale à partir d'un seul sens.

Le donneur a déjà reçu sa question au moment de la capture. L'essai ne prouve
donc pas qu'une représentation existait avant toute question. Un vecteur entier
peut importer du contexte ou créer une combinaison inhabituelle ; l'échec d'un
transfert ne prouve pas l'absence de toute représentation ailleurs. La réussite
n'établirait ni nécessité du même chemin dans l'inférence intacte, ni expérience
subjective, ni représentation de sa propre existence. La base est conservée
avec sa fragilité connue de compréhension des consignes.

## Vérifications et reproductibilité

Le lecteur vérifie les 136 contrôles au sein de la tâche avec le lecteur figé
du Colab 15, puis l'ordre, la bonne question donneuse, les empreintes des
vecteurs, les prompts, les traces physiques et les copies des 96 transferts
croisés. Les états cachés opposés ont des prompts intacts identiques. Un groupe
incomplet est entièrement répété avec compteur de reprise explicite ; aucun
poids ne change. Le journal 15 et les trois poids/journal parents 14 sont
hachés avant et après l'exécution.

Huit tests logiciels passent localement : séparabilité logique, quatre
mécanismes synthétiques, bilan complet de 44 544 appels simulés, rejet du
mauvais donneur et d'une copie altérée, ainsi qu'un groupe de 232 véritables
passages d'un petit Qwen aléatoire. Les trois tests des opérations sous-jacentes
du Colab 15 sont inclus. Ces essais logiciels ne sont pas des résultats de
Qwen3-4B. Le lanceur les exécute également avant toute collecte dans Colab.

Ce protocole étend une expérience Menia ; il ne revendique pas l'invention de
l'échange causal ou de la comparaison d'abstractions, déjà présentés dans
[la revue mécanistique et ses sources](MECHANISTIC_COMPOSITION_REVIEW.md).
