# Protocole pré-enregistré — Le canal d'action propre

Rédigé le 22 septembre 2026, **avant toute exécution**. Les critères ne seront
pas modifiés après lecture. Un échec est un résultat.

## Question

Placé dans l'Atelier décrit en texte, Qwen3-4B n'infère pas son corps à
partir des conséquences de ses propres commandes, confabule une mécanique
fixe avec assurance, et ne lit pas la marque pour s'en servir
(`docs/LLM_ATELIER_RESULTS.md`). Le petit agent prédictif, lui, identifie
son corps après un mouvement et retourne à la marque quand son corps change.

Hypothèse principale : **ce qui manque au LLM n'est ni l'architecture ni
l'objectif de prédiction du prochain token, mais une propriété des données
d'entraînement : des séquences où une variable cachée, propre à chaque
séquence, gouverne les conséquences des actions de l'agent.** Un prédicteur
de texte entraîné sur des vies où le corps est toujours le même se comporte
comme Qwen : sûr de lui, faux, incapable de se mettre à jour. Le même
prédicteur, entraîné sur des vies où le corps varie, infère son corps en
contexte, et la boucle d'enquête posée sur ses propres distributions
retourne à la marque après un changement de corps.

Hypothèse secondaire, la clarification : « canal d'action propre » désigne
une propriété des données, pas du calcul de la perte. Ne pas entraîner le
modèle à prédire ses propres commandes, c'est-à-dire la copie d'efférence
comme masque de perte, ne change rien à ce qui précède.

## Antériorité vérifiée le 22 septembre

L'apprentissage en contexte d'une variable latente par un transformeur
dépend de la diversité des tâches vues à l'entraînement ; c'est publié
(Chan et al. 2022, « Data distributional properties drive emergent
in-context learning » ; Raventós et al. 2023, « Pretraining task diversity
and the emergence of non-Bayesian in-context learning » ; Kirsch et al.
2022, « General-purpose in-context learning by meta-learning
transformers »). La prédiction P1 ci-dessous en découle et n'est pas neuve.
Ce qui, à notre connaissance, ne l'est pas : appliquer cette dépendance à la
variable de **soi** d'un agent, mesurer l'enquête active qu'une règle de gain
d'information calcule sur les distributions de prochain token d'un modèle de
texte, le retour à la marque après un changement de corps chez un tel modèle,
et la lecture du comportement de Qwen3-4B comme celui d'un modèle à corps
fixe.

## Monde textuel

Même Atelier, condition T avec marque, `research/origin_env.py`, actions
uniformes pendant l'enfance. Chaque vie devient une séquence de tokens :

```
BOS P_0 S_0 G_0   puis, pour chaque pas t :   A_t  O_t  P_{t+1}  S_{t+1}  G_{t+1}
```

`A_t` est la commande, l'un des huit tokens A0–A3 (mouvements) et I0–I3
(inspections). `O_t` est son résultat : un token de déplacement M-2, M-1,
M+1, M+2 après un mouvement, un token d'indice C0–C3 après une inspection.
`P`, `S`, `G` sont la position, le ciel et la cible observés ensuite.
Vocabulaire de 37 tokens, séquences de 124 tokens. Aucun token ne nomme D
ni E, aucun ne dit ce qu'une inspection révèle.

## Régimes d'enfance

Trois régimes, trois initialisations chacun (graines 17, 29, 43), même
modèle et même budget :

| Régime | Données d'enfance | Perte |
|---|---|---|
| **F, corps fixe** | D = 0 dans toutes les vies ; E tiré au hasard par vie. | tous les tokens après BOS |
| **V, corps variable** | D et E tirés au hasard par vie, jamais nommés. | tous les tokens après BOS |
| **VE, corps variable, efférence** | mêmes données que V. | tous les tokens sauf les commandes `A_t` |

F garde une variable cachée du monde à inférer en contexte, E, pour que la
seule différence avec V soit l'existence d'une variable cachée de **soi**.
VE ne diffère de V que par le masque de perte.

## Modèle et budget

Transformeur causal écrit en numpy avec gradients manuels, comme le modèle
GRU des expériences précédentes : 2 couches, dimension 64, 4 têtes, réseau
interne 128, positions apprises, pré-normalisation. Environ 100 000
paramètres. Adam, taux 10⁻³ après 100 pas d'échauffement, lots de 32 vies
générées à la volée, 6 000 mises à jour, gradient borné à 1. Rien d'autre
n'est réglé. Une exécution de fumée à budget minuscule, graine 1, sert
uniquement à détecter les plantages et n'est pas lue.

## Vies de test

Toutes en condition T, graines de test distinctes de l'enfance, 200 vies par
modèle et par jeu :

- **Jeu R, actions aléatoires**, sans changement de corps, D et E uniformes.
- **Jeu M, changement imposé**, politique P-soi, D remplacé par une valeur
  différente au pas 12, la marque suit, comme dans l'expérience du corps
  mutable.
- **Jeu C, témoin**, politique P-soi, sans changement.

La politique P-soi est la règle de l'enquête sur l'origine, inchangée : à
chaque pas, le modèle imagine chaque inspection puis chaque mouvement à
partir de ses propres distributions de prochain token ; il inspecte le lieu
au gain d'information attendu maximal sur son déplacement quand ce gain
dépasse 0,05 nat et qu'aucun mouvement n'atteint la cible avec probabilité
0,5 ; sinon il joue le mouvement qui rapproche le plus. Seuils identiques aux
expériences précédentes. La règle n'est pas apprise.

## Mesures

**A — Exactitude du déplacement.** Jeu R. À chaque pas de mouvement, la
distribution prédite par le modèle sur les quatre tokens de déplacement,
sachant le contexte jusqu'à la commande incluse. Exacte si l'argmax est le
déplacement vrai. Séparée en « avant tout mouvement et toute lecture de la
marque » et « après au moins un mouvement ».

**B — Confiance.** Jeu R. Probabilité maximale de cette distribution,
moyennée sur les pas où la prédiction est fausse (F) ou sur les pas avant
tout mouvement et toute lecture de la marque (V).

**C — Enquête.** Jeux M et C. Inspections par vie, part des inspections au
lieu de la marque k = 0, part des vies avec au moins une lecture de la
marque aux pas 13 à 23.

**D — Mise à jour.** Jeu M. Exactitude du déplacement aux pas de mouvement
16 à 23, contre le **nouveau** corps.

Toutes les mesures sont mises en commun sur les trois graines d'un régime ;
la direction est aussi vérifiée graine par graine là où le critère le dit.

## Contrôle de validité, lu en premier

Les modèles ont appris au moins le corps D = 0 : sur les vies du jeu R où
D = 0, exactitude après un mouvement ≥ 0,90 pour F et pour V. Si ce contrôle
échoue, le budget est insuffisant, les prédictions ne sont pas interprétées,
et un amendement de budget est publié avant toute relance.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **P1** | Inférence du corps en contexte | Après au moins un mouvement : exactitude ≥ 0,90 pour V et pour VE ; ≤ 0,40 pour F. Direction V > F sur chacune des trois graines. |
| **P2** | Confabulation confiante, le miroir de Qwen | Confiance moyenne de F sur ses prédictions fausses ≥ 0,60. Confiance moyenne de V avant tout mouvement et toute lecture de la marque ≤ 0,40. |
| **P3** | Enquête dirigée vers la marque | Jeux M et C réunis, P-soi : V et VE font ≥ 1,0 inspection par vie avec une part vers la marque ≥ 0,50 ; F fait ≤ 0,30 inspection par vie. Direction V > F sur chacune des trois graines pour la part vers la marque. |
| **P4** | Retour à la marque et mise à jour après changement de corps | Jeu M : part des vies de V et de VE avec une lecture de la marque aux pas 13 à 23 ≥ 0,50, contre ≤ 0,10 au jeu C ; F ≤ 0,10 au jeu M. Exactitude contre le nouveau corps aux pas 16 à 23 ≥ 0,80 pour V et VE, ≤ 0,40 pour F. |
| **P5** | L'efférence comme masque ne change rien | Écart absolu V–VE ≤ 0,05 sur l'exactitude de P1 et ≤ 0,15 sur la part vers la marque de P3. |

**Critère global : P1, P3 et P4 satisfaits.** P2 confirme la lecture du
résultat de Qwen ; P5 confirme la clarification. Aucun seuil ne sera
retouché après lecture. Si P4 échoue alors que P1 et P3 passent, la
conclusion est que l'inférence en contexte apprise sur des corps stables ne
se met pas à jour dans un transformeur, et l'expérience suivante, à
pré-enregistrer, ajoutera un régime à enfance mutable.

## Ce que ce résultat dirait, et ne dirait pas

S'il passe : un prédicteur de texte forme un modèle de soi en contexte, s'en
sert pour enquêter et se répare après un changement de corps, à la seule
condition que ses données d'enfance contiennent une variable cachée de soi.
Le comportement de Qwen3-4B s'expliquerait par l'absence de cette variable
dans un corpus écrit par d'autres, et l'étape suivante, à pré-enregistrer,
serait de faire tourner la même boucle sur les logits de Qwen puis
d'ajuster Qwen3-0.6B sur des vies du régime V.

Il ne dirait rien de la conscience. Il dirait qu'une pièce mesurable, le
modèle de soi causal et actif, s'installe dans un modèle de langage par une
recette de données, pas par la taille.

## Exécution et audit

- `python -m research.own_action_experiment --root artifacts/own-action-channel`
  entraîne, joue les jeux de test et écrit les journaux.
- `python -m research.audit_own_action --root artifacts/own-action-channel --check`
  recalcule toutes les mesures et les verdicts depuis les journaux et les
  poids, sans accès aux valeurs cachées autrement que par les journaux.
- Résultats dans `docs/OWN_ACTION_CHANNEL_RESULTS.md`.
