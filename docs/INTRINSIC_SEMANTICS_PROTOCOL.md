# Protocole de l'audit d'information sémantique

Date : 15 septembre 2026. Protocole consigné avec les calculs, non préenregistré.
Voir les [résultats et leur portée](INTRINSIC_SEMANTICS_EVIDENCE.md).

## Objet et provenance

Rejeu du modèle fini associé à
[Kolchinsky et Wolpert, 2018](https://doi.org/10.1098/rsfs.2018.0041), puis extension
de la famille des interventions. Ce protocole ne teste pas la conscience.

Source : `artemyk/semantic_information`, commit
`bc56a371130a4073062c766654ca4796110d5c8e`, daté du 3 mars 2019.

| Fichier lu | SHA-256 |
|---|---|
| `model.ipynb` | `55b33ca1e23468f925a04f4b6f6dc90bbfb590bdc6f032bb4ceb8744a580613c` |
| `utils.py` | `2d369987fb87e822989c79ff5a51c02eca59b2bdccb51651c8646f6eece656e6` |

Le code source téléchargé n'est pas exécuté. La réexpression locale conserve
le modèle et les détails numériques pertinents ; son attribution et la
[licence MIT](../licenses/semantic_information-MIT.txt) sont incluses.

## Dynamique conservée

- 900 états conjoints : position (5), cible (6), niveau (5), nourriture (6).
- Niveaux `0..4`, départ au centre, niveau 4, cible identique à la nourriture,
  nourriture uniforme sur les cinq positions. Le sixième symbole signifie
  absence ; il n'a pas de masse initiale.
- Rayon de consommation 1 ; taux de disparition 0,1 ; coût de niveau 10 ;
  terme nourriture 100 ; terme de niveau nul `−100 ln(2)` dans l'énergie libre.
- Conserver l'ordre des états et les **affectations** successives des taux
  avant/arrière du notebook, sans les remplacer par des accumulations.
- Retirer les taux diagonaux, diviser par le maximum des sommes de lignes,
  compléter la diagonale pour obtenir une matrice stochastique.
- Modes vers la cible et opposé à la cible ; horizons 0, 1, 2, 3, 5, 8 et 10.

Cet audit reproduit la procédure du code. Il ne constitue pas un audit général
de sa réalisation thermodynamique ou de toutes ses affirmations physiques.

## Interventions et lectures distinctes

Pour une partition `g` des cinq positions, la distribution initiale est :

`P(cible=i, nourriture=j) = 1/(5 × taille du groupe g(i))` si `g(i)=g(j)`,
et zéro sinon. Les deux marginales restent uniformes.

Énumérer les 203 partitions des six symboles sous forme canonique ; leurs
restrictions au support initial donnent exactement les 52 partitions des cinq
positions. Les partitions contiguës donnent 32 et 16, respectivement. Propager
les 52 distributions distinctes avec une dynamique identique par mode.

Mesures : `I(cible;nourriture)`, `V = −H(position,cible,niveau) − 100 P(niveau=0)`,
probabilité de niveau positif, et valeur `V_initial − V_brouillé`.

Conserver trois lectures séparées :

1. Minimum de l'information des partitions dont la viabilité égale la référence.
2. Maximum de viabilité pour chaque niveau d'information : la courbe supérieure.
3. Premier niveau de cette courbe ayant la viabilité de référence : lecture du
   notebook, qui peut différer de la première après extension de la recherche.

Employer l'arrondi à cinq décimales du notebook pour les égalités et niveaux.
Conserver les 208 points non arrondis des horizons 1 et 5 (52 × 2 modes × 2
horizons), et les résumés des 728 évaluations. Ne pas assimiler ces tolérances
numériques à une preuve d'égalité analytique.

Le diagnostic sans entropie interne change uniquement la lecture des mêmes
distributions. Il ne change pas les taux et ne représente pas un autre modèle
physique validé. La dynamique, la frontière du système et les paramètres restent
fixés ; seule une sensibilité aux horizons et familles d'interventions est étudiée.

## Rejeu et contrôles

```bash
python -m research.audit_intrinsic_semantics
python -m research.audit_intrinsic_semantics --check
```

NumPy suffit. Aucun téléchargement n'est nécessaire. Le rapport est comparé
avec une tolérance absolue de `1e-10`, sans tolérance relative. Les structures,
libellés et comptes sont comparés exactement. À chaque pas, la normalisation et
la non-négativité sont vérifiées. L'information est calculée à la fois depuis
les marginales et l'entropie des groupes.

Contrôles locaux supplémentaires exécutés : comparaison des 32 lignes imprimées
du notebook ; comparaison propagation creuse/matrice dense pour deux dynamiques,
52 partitions et cinq pas. Le rapport versionné conserve les valeurs mesurées ;
la CI rejoue le calcul local, sans importer le notebook.

Aucun agent autonome supplémentaire, adaptation du LLM, contrôle matériel,
évaluation de souffrance ou indice de conscience n'est créé.
