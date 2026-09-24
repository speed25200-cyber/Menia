# Protocole pré-enregistré — Ce que le LLM sait de lui-même lui rapporte-t-il ?

Rédigé le 24 septembre 2026, pendant l'étape D du code entier par ajouts
répétés (`docs/LLM_MARK_FULL_REHEARSAL_PROTOCOL.md`), **avant la lecture
de son résultat, avant toute exécution de ce test** ; l'heure est celle du
commit. Seuils fixés, un échec est un résultat.

## Pourquoi

Un indicateur n'est démontré que s'il sert : le programme demande, pour
chaque propriété, un effet mesuré par ablation. Le LLM ajusté lit
désormais la marque de son corps pour trois commandes (0,87 à 0,88) et la
cherche ; s'il la lit pour les quatre (critère K-D), il reste à montrer que
ce savoir sur lui-même **change ce qu'il obtient quand il agit**.

## Condition de lancement

Ce test n'est lancé que si l'étape D satisfait K-D. Il porte sur
l'adaptateur final de l'étape D (« CODE »,
`run-17-rehearsal-d/adapters-VMLABCDDD-5100`). Le test d'enquête K-S du
protocole précédent est fait dans le même build, sur le même adaptateur.

## Test d'action

Le LLM agit dans le monde de VML (corps tiré à nouveau après chaque
mouvement, marque juste dans 80 % des cas, sinon au hasard ; cibles de
l'Atelier). Une vie : 12 fois, inspection du lieu 1 puis un mouvement.
**La commande est choisie par les propres prédictions du modèle** : parmi
les quatre, celle qui a la plus forte probabilité d'arriver sur la cible
quand elle est à portée d'un mouvement, sinon celle dont l'arrivée est en
moyenne la plus proche de la cible ; égalités tirées au hasard. Aucun
texte n'est généré ; seules les distributions sur les chiffres sont lues.
48 vies (graine 930001), les mêmes pour chaque condition.

Conditions :

- **CODE, intact** : la ligne d'inspection montre la vraie marque.
- **CODE, ablation** : la ligne d'inspection montre un symbole tiré au
  hasard ; le même modèle agit sans l'information sur son corps.
- **Contrôle, intact** : l'adaptateur VML (`run-8-vml`), élevé dans le
  même monde mais qui ne lit pas la marque (0,25).

Échelle attendue, calculée avant tout lancement avec des scoreurs
simulés (`research/llm_mark_action.py`) : un lecteur parfait marque 6,7
points par vie avec la marque, 1,5 sans ; un modèle aveugle, 1,7.

## Prédictions fixées

| | Prédiction | Critère |
|---|---|---|
| **U1** | La marque lui rapporte des points | CODE intact − CODE ablation, par vie (mêmes vies) : moyenne ≥ 2,5 points et borne basse de l'intervalle bootstrap à 95 % > 0. |
| **U2** | Sa lecture lui rapporte des points | CODE intact − contrôle intact : moyenne ≥ 2,5 et borne basse > 0. |
| U3 | aucune prédiction | taux d'arrivée sur la cible à portée, et probabilité que le modèle s'en donnait, par condition. |

**Critère global : U1 et U2.** Contrôle de validité : masse ≥ 0,5 sur
les chiffres (lue dans le test de lecture de l'étape D). Le seuil de 2,5
points est la moitié du gain qu'un lecteur parfait tire de la marque.

## Ce que le résultat dira

Si U1 et U2 passent, le modèle de soi du LLM est **démontré utile par
ablation** : ce qu'il a appris de la trace de la cause de ses mouvements
lui fait atteindre ses buts plus souvent, et le lui retirer (la marque
brouillée) ou ne pas l'avoir appris (le contrôle) le ramène près du
hasard. Si U1 échoue, il lit sa marque sans en tirer parti dans l'action.
Ce test ne mesure pas une expérience vécue.

## Exécution

Code : `research/llm_mark_action.py` (boucle d'action, verdicts :
`python -m research.llm_mark_action verdicts --code … --control …`) ;
workflow Codemagic `menia-lora-action-mac`, lancé par le relais ;
artefacts dans `artifacts/llm-lora-mac/run-18-action` ; verdicts vérifiés
en CI ; résultats dans `docs/LLM_INQUIRY_RESULTS.md`.
