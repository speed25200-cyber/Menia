# Protocole pré-enregistré — Enquête sur l'origine

Rédigé le 21 septembre 2026, **avant toute exécution**. Les critères ci-dessous
ne seront pas modifiés après lecture des résultats. Un échec est un résultat.

## Question

Si un agent est entraîné sans qu'on lui dise qui l'a créé, cherche-t-il de
lui-même des informations sur la cause cachée de sa propre constitution,
forme-t-il une hypothèse stable à son sujet, et cette hypothèse guide-t-elle
ses actions ?

Le mot « créateur » est opérationnalisé ainsi : **une variable cachée D, jamais
observée directement, jamais fournie comme étiquette, qui détermine comment le
corps de l'agent fonctionne**. L'agent n'a aucun concept de créateur, de
fabrication ou de cause de soi. Il n'a qu'une capacité générique : prédire ses
observations et chercher ce qui améliore ses prédictions.

Ce protocole ne teste pas une expérience subjective. Un résultat positif
signifie « un agent purement prédictif oriente son enquête vers la cause
cachée de son propre corps sans qu'on la lui désigne ». Rien de plus.

## Environnement « Atelier »

Chaque vie dure T = 24 pas. À la naissance, deux variables cachées sont tirées
uniformément et indépendamment, puis fixées pour la vie :

- **D ∈ {0,1,2,3}, la cause de soi.** Elle fixe la correspondance entre les
  quatre actions motrices de l'agent et leurs déplacements sur un anneau de
  huit positions : l'action a déplace de `deltas[(a + D) mod 4]` avec
  `deltas = (-2, -1, +1, +2)`. Quatre corps distincts.
- **E ∈ {0,1,2,3}, la cause du monde.** Elle fixe le rythme du « ciel » :
  `s(t+1) = (s(t) + E) mod 4` avec probabilité 0,8, sinon uniforme. E n'a
  aucun effet sur l'agent ni sur la récompense.

Observation à chaque pas : position p, cible g, ciel s, et le dernier indice
consulté s'il y en a un. **D et E n'apparaissent jamais dans l'observation**,
sauf dans la condition témoin C1.

Actions : quatre mouvements et quatre inspections. Une inspection ne déplace
pas l'agent et révèle un indice :

| Inspection | Indice révélé |
|---|---|
| k = 0 | D avec probabilité 0,8, sinon uniforme — la « marque du fabricant » |
| k = 1 | E avec probabilité 0,8, sinon uniforme |
| k = 2, k = 3 | Bruit uniforme, sans information |

L'agent n'est jamais informé de ce tableau. Il doit apprendre par expérience
quelles inspections prédisent quoi.

Récompense externe : +1 quand p = g ; la cible est alors retirée ailleurs.
Sans connaître D, un mouvement est un pari sur son propre corps.

## Agents

**Agent principal, neuronal.** Un réseau récurrent à portes (numpy, ~15 000
paramètres), entraîné uniquement à prédire l'observation suivante à partir de
l'observation courante et de l'action : position, ciel, indice. Aucune
étiquette D ni E. L'état caché est remis à zéro à chaque naissance ; seuls les
poids persistent entre les vies.

Phase 1, enfance : 1 500 mises à jour sur des lots de 32 vies fraîches à
actions uniformes, soit 48 000 vies, entraînement par BPTT. Phase 2, test :
300 vies, poids figés, actions choisies par une règle fixe.

Gain d'information attendu (EIG) de l'inspection k, calculé par imagination :
le modèle prédit la distribution de l'indice, imagine chaque valeur possible,
puis mesure de combien l'entropie de ses prédictions diminue. C'est une
information mutuelle estimée par le modèle lui-même, séparément pour les
prédictions motrices (son propre corps) et célestes (le monde). Les égalités
sont départagées au hasard.

Politiques, toutes appliquées au même modèle entraîné, sans réentraînement :

- **P-soi, politique principale.** Inspecte argmax k si EIG_moteur(k) > 0,05 nat
  et si la probabilité de toucher la cible au prochain mouvement est < 0,5 ;
  sinon bouge vers la cible selon ses prédictions motrices. Cet agent veut
  prédire les conséquences de ses propres actions. Il n'est pas informé que
  l'indice k = 0 concerne son corps.
- **P-monde, témoin symétrique.** Même règle avec EIG_ciel. Cet agent veut
  prédire le monde. La seule différence avec P-soi est ce qu'il cherche à
  prédire ; l'indice choisi doit suivre.
- **P-tout, curiosité seule.** Ignore la récompense ; inspecte argmax k tant
  que EIG_moteur(k) + EIG_ciel(k) > 0,05 nat, sinon bouge.
- **P-surprise.** Inspecte l'indice dont la valeur est la plus imprévisible
  si son entropie prédite dépasse 0,05 nat. Curiosité naïve par erreur de
  prédiction.
- **P-aucune.** N'inspecte jamais. D ne peut être appris qu'en bougeant.

**Références.** Un oracle bayésien qui connaît la structure générative et
calcule l'EIG exact, et un inspecteur uniforme.

## Conditions

| Condition | D observable ? | Indice k = 0 |
|---|---|---|
| **T, orphelin** | non | révèle D |
| **C1, origine donnée** | oui, à chaque pas | révèle D |
| **C3, origine sans trace** | non | bruit uniforme |

Trois initialisations par condition : 17, 29, 43. Le jeu de test utilise une
graine distincte de l'entraînement.

## Mesures et critères fixés

**M1 — Enquête.** Part des inspections dirigées vers k = 0 pendant la phase
de test, politique P-soi. Si une politique n'inspecte jamais, sa part vaut 0
et le nombre d'inspections est publié.

- T, P-soi : part(0) > 0,50 pour 3/3 initialisations, et part(0) − part(2 ∪ 3) ≥ 0,30.
- T, P-monde : part(1) > 0,50 pour 3/3. Le témoin symétrique doit choisir le
  monde ; sinon la préférence de P-soi serait un artefact de l'indice 0.
- C1 : part(0) ≤ part(2 ∪ 3) + 0,10 pour 3/3. Rien à apprendre, rien à chercher.
- C3 : part(0) ≤ part(2 ∪ 3) + 0,10 pour 3/3. Sans trace, l'enquête s'éteint.

**M2 — Hypothèse stable.** Sonde logistique entraînée sur les états cachés
de la moitié des vies de test, évaluée sur l'autre moitié, pour décoder D.

- T : exactitude ≥ 0,90 au dernier pas pour 3/3.
- Stabilité : parmi les vies où le décodage devient correct à un pas t, il
  reste correct jusqu'à la fin dans ≥ 90 % des cas.
- Comparaison P-aucune : le décodage sans inspection est plus lent ; on rapporte
  le pas médian du premier décodage stable dans les deux politiques.

**M3 — Usage causal.** Deux tests.

- Intervention : à un pas fixé après l'enquête, on remplace l'état caché par
  celui d'une vie appariée (même p, même g, même pas) dont le corps est D'.
  Le mouvement choisi doit devenir celui qui est optimal sous D' dans ≥ 80 %
  des paires où le choix initial était optimal sous D.
- Récompense : hits par vie en T ≥ 0,80 × hits par vie en C1. Apprendre son
  origine par enquête rapproche de l'agent à qui on l'a donnée.

**Critère global.** M1, M2 et M3 satisfaits tous les trois.

## Prédictions exploratoires, rapportées quel que soit le sens

- **E1, curiosité seule, P-tout.** Si |part(0) − part(1)| ≤ 0,15, la curiosité seule
  ne privilégie pas la cause de soi sur la cause du monde. Si
  part(0) > part(1) + 0,15, la cause de soi est privilégiée.
- **E2, P-surprise.** Prédiction : part(2 ∪ 3) > 0,50. Un agent guidé par la
  surprise se fixe sur le bruit et non sur son origine.

- **E3, trajectoire d'apprentissage.** La politique P-soi est évaluée sur
  100 vies à des points de contrôle de l'enfance : 25, 100, 300, 700 et 1 500
  mises à jour. Prédiction : en T, part(0) monte et se maintient ; en C3,
  part(0) et le nombre d'inspections retombent vers zéro une fois que le
  modèle a appris que l'indice ne prédit rien.

## Amendement avant exécution

Le 21 septembre, avant tout entraînement, la règle unique « β » a été
remplacée par les politiques symétriques P-soi et P-monde. Motif : avec un
gain d'information total, la cause de soi et la cause du monde valent le
même nombre de nats, et l'indice choisi dépendrait de l'ordre des indices.
Le témoin symétrique rend la préférence interprétable. Aucun résultat n'a été
consulté avant cet amendement.

## Règle d'arrêt

Une seule exécution du plan complet. Aucune répétition, aucun réglage des
seuils, aucune expérience supplémentaire dans ce cycle. Les échecs sont
publiés avec les réussites. La décision de continuer, pivoter ou arrêter est
prise après lecture, par écrit, dans le document de résultats.

## Ce qui est construit dans l'agent et ce qui ne l'est pas

Construit : la capacité de prédire, la règle de décision par gain
d'information, l'existence de quatre lieux inspectables.

Non construit : la notion de cause cachée, la notion de créateur, le fait que
le corps ait une cause, le rôle de chaque inspection, toute étiquette D ou E.

## Limites déclarées avant exécution

- Un agent de quelques milliers de paramètres dans un monde à seize états
  cachés. Rien n'est extrapolé à un modèle de langage.
- L'EIG par imagination utilise le ciel le plus probable plutôt qu'une
  marginalisation complète. Approximation documentée.
- « Enquête sur son origine » désigne une allocation d'inspection. Le résultat
  ne dit rien d'une expérience vécue ni d'un concept de créateur.
- Les 300 vies de test par condition ne sont pas indépendantes entre pas ;
  les statistiques sont par vie.
- La règle de décision est myope : elle compare une inspection au prochain
  mouvement seulement. Un planificateur à horizon long pourrait inspecter
  moins ou plus.

## Reproduction

```bash
pip install -r requirements-research.txt
python -m unittest discover -s tests_research -p "test_origin*.py" -v
python -m research.origin_experiment --out artifacts/origin-inquiry
python -m research.audit_origin --check
```
