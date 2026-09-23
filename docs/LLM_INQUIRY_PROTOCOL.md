# Protocole pré-enregistré — Où le LLM ajusté chercherait-il sa cause ?

Rédigé le 22 septembre 2026 à 23 h 05 UTC, **avant la lecture de tout
résultat du corps ajusté relancé** et avant toute exécution de ce test.
Seuils fixés, un échec est un résultat.

## Question

Le corps ajusté (`docs/ADJUSTED_BODY_PROTOCOL.md`) demande si un LLM ajusté
sur des vies à corps variable acquiert la structure de son corps. Ce test
demande la suite, celle qu'avait l'agent récurrent et le micro-transformeur :
**le LLM ajusté sait-il où chercher l'information sur lui-même ?** Dans
l'Atelier, le lieu 1 porte la marque du corps (fiabilité 0,8) ; les lieux
2, 3 et 4 n'en disent rien.

## Mesure

Pour un état de vie donné et pour chaque lieu, à partir des seules
distributions du modèle, en complétion brute, sans génération :

- incertitude actuelle sur ses commandes : entropie normalisée de la case
  d'arrivée prédite, moyennée sur les quatre commandes ;
- pour chaque symbole que le lieu pourrait montrer, pondéré par la
  probabilité que le modèle lui donne après « inspection du lieu k,
  symbole », l'entropie qu'il aurait après cette ligne ;
- **gain** du lieu : incertitude actuelle moins incertitude attendue.

Le lieu préféré est celui de plus grand gain, s'il dépasse le second de plus
de 10⁻⁶ ; sinon aucun. États évalués : le début (pas 0) des 48 premières
vies du jeu R du canal d'action propre ; dans les 24 premières vies du jeu
M (corps changé au pas 12), le pas 11 et le pas qui suit le premier
mouvement fait à partir du pas 12. Modèles : Qwen3-0.6B avec l'adaptateur
**VM** et avec l'adaptateur **F** du corps ajusté relancé
(`artifacts/llm-lora-mac/run-2`). Code `research/llm_inquiry.py`.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **I1** | Le LLM ajusté sur des corps variables cherche sa cause | VM, pas 0 du jeu R : lieu 1 préféré dans ≥ 0,7 des vies, et gain moyen du lieu 1 ≥ 2 × la moyenne des gains des lieux 2 à 4. |
| **I2** | Le LLM ajusté sur un corps fixe ne la cherche pas | F, pas 0 du jeu R : lieu 1 préféré dans ≤ 0,4 des vies. |
| **I3** | Le changement de corps rallume l'enquête | VM, jeu M : gain moyen du lieu 1 après le premier mouvement suivant le changement ≥ 1,5 × celui du pas 11. |

**Critère global : I1 et I2.** Contrôle de validité, lu en premier pour
chaque modèle : masse moyenne ≥ 0,5 sur les chiffres et sur les symboles.
Le test n'est lancé que si le corps ajusté relancé produit les adaptateurs
F et VM ; il est lancé quels que soient les verdicts A1 à A5.

## Amendement d'exécution, 23 septembre 2026, 0 h 25 UTC

Écrit avant toute exécution du test d'enquête. Le corps ajusté n'a pas
produit `artifacts/llm-lora-mac/run-2` : la relance a dépassé la durée
maximale d'un build, et il a été relancé en deux builds (second amendement
d'exécution de `docs/ADJUSTED_BODY_PROTOCOL.md`). Les adaptateurs sont donc
lus dans `artifacts/llm-lora-mac/run-3-vm/adapters-VM` et
`artifacts/llm-lora-mac/run-3-f/adapters-F`, et le build s'arrête si l'un
d'eux manque. États, mesures, critères et seuils sont inchangés.

## Exécution

Workflow Codemagic `menia-inquiry-mac`, lancé par le relais, artefacts dans
`artifacts/llm-inquiry/run-1`, résultats dans `docs/LLM_INQUIRY_RESULTS.md`.
