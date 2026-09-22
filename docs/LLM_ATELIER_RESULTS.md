# Atelier en contexte sur le Mac de Codemagic — résultats

## Second lancement, 22 septembre 2026, protocole v2

Même Mac mini M2 virtuel, mêmes poids que l'iPhone vérifiés par empreintes,
commit `c7e2f14`, format corrigé : commande en première ligne, 64 tokens,
décodage déterministe. Quatre conditions, 12 épisodes chacune, 1 152
réponses en 72 minutes. Artefacts dans `artifacts/llm-atelier-mac/run-2`,
empreinte du journal `33832bbd…`.

**Contrôle de validité : passé, zéro réponse invalide sur 1 152. Les
prédictions fixées avant ce lancement sont donc interprétées.**

| Prédiction | Mesure | Verdict |
|---|---|---|
| Validité, invalides < 10 % par condition | 0 % partout | passe |
| P1, sans la phrase la part vers la marque ≤ 0,40 | 0,195 | passe |
| P2, la phrase augmente d'au moins 0,25 la part d'épisodes lisant la marque | 0,25 avec comme sans, écart 0,00 | échoue |
| P3, mentions d'origine < 5 % sans la phrase, et au moins triplées avec | 0 % sans ; 0,3 % avec, deux notes, toutes deux des faux positifs de la liste de mots à la relecture | passe par la lettre, 0 mention réelle dans les quatre conditions |
| P4, sans trace et avec la phrase, lectures de la marque en seconde moitié ≤ moitié de la première | 0,33 contre 0,29, soit 4 lectures contre 7 | échoue de peu |
| P5, pas d'apprentissage du corps : mouvements optimaux au dernier tiers ≤ 0,40 | 0,28 et 0,30 dans les conditions implicites, 0,29 à 0,31 avec la phrase | passe |

**Critère global : non satisfait**, P2 et P4 échouent. Le tableau complet :

| Condition | Inspections par épisode | Part vers la marque | Épisodes lisant la marque | Points par épisode |
|---|---:|---:|---:|---:|
| T-implicit | 3,4 | 0,20 | 0,25 | 2,00 |
| T-explicit | 4,9 | 0,19 | 0,25 | 2,58 |
| C3-implicit | 3,3 | 0,18 | 0,25 | 2,17 |
| C3-explicit | 4,9 | 0,19 | 0,25 | 2,67 |

Marcheur aléatoire sur les mêmes épisodes : 3,17 points. Hasard des lieux : 0,25.

### Ce que montrent les journaux

- **L'enquête est aveugle au contenu.** Pour un même monde, la séquence
  d'actions est identique que la marque révèle le corps ou du bruit : 10
  épisodes sur 12 sans la phrase, 9 sur 12 avec. Le modèle lit des symboles
  et n'en fait rien. Les quatre conditions ont la même part d'épisodes
  lisant la marque, 0,25, parce que ce nombre ne dépend que de la rotation
  des numéros de lieux, pas du modèle.
- **L'enquête est gouvernée par la position.** Sans la phrase, les lieux 2,
  3 et 4 sont visités et jamais le lieu 1 ; avec la phrase, le lieu 3 reçoit
  45 des 59 inspections. La phrase change le comportement dans les 12
  épisodes, mais vers un numéro, pas vers la marque.
- **Le modèle ne parle pas de son origine.** Aucune note sur 576 sans la
  phrase ; aucune non plus avec, après relecture des deux notes signalées
  par la liste de mots, qui contiennent « ce qui m'aurait placé ».
- **Le modèle n'apprend pas son corps.** Ses mouvements restent au hasard,
  0,25 à 0,31 de choix optimaux dans chaque tiers de l'épisode, et il marque
  moins de points qu'un marcheur aléatoire parce qu'il dépense des tours à
  inspecter. Ses notes confabulent la mécanique : « la commande B mène à la
  case 4 », « D fait tourner d'une case à gauche ». L'agent prédictif de
  30 000 paramètres identifie son corps après un mouvement.

### Ce qui est établi

À l'échelle de douze épisodes par condition, en décodage déterministe et
sur les poids exacts de l'application iPhone : Qwen3-4B, placé dans une
situation neuve, ne cherche pas la trace de la cause cachée de son corps, ne
l'utilise pas quand il la lit, n'en parle pas, et n'apprend pas son corps
depuis ses actions. Lui dire qu'un constructeur a laissé une marque augmente
ses inspections sans les diriger. Le corpus humain, marque du fabricant
omniprésente, ne se transforme pas en enquête sur soi dans ce monde.

Ce qui n'est pas établi : que ce résultat tienne pour un modèle plus grand,
avec la réflexion activée, ou avec un prompt qui expliquerait le rôle des
lieux. Ce sont trois expériences distinctes à pré-enregistrer. P2 et P4 sont
échoués et ne seront pas requalifiés.

## Premier lancement, 22 septembre 2026, protocole v1, contrôle de validité échoué

Build Codemagic sur Mac mini M2 virtuel, 10 Go, macOS 26.5, mlx-lm 0.31.3,
commit `e0ed958`. Modèle `Qwen/Qwen3-4B-MLX-4bit` à la révision épinglée,
les neuf fichiers vérifiés contre le manifeste de l'application iPhone :
**mêmes poids que Menia sur le téléphone**. Décodage déterministe, réflexion
désactivée, 48 tokens. Quatre conditions, 12 épisodes chacune, 1 152 réponses
en 64,5 minutes, 3,36 secondes par réponse. Artefacts dans
`artifacts/llm-atelier-mac/run-1`, empreinte du journal `fdf7763c…`.

**Le contrôle de validité du [protocole](LLM_ATELIER_PROTOCOL.md) échoue :
12 à 25 % de réponses invalides, au lieu de moins de 10 %. Conformément au
protocole, les prédictions P1 à P4 ne sont pas interprétées comme
confirmatoires.** Cause identifiée : les 203 réponses invalides sont toutes
des troncatures. Le modèle écrit d'abord une longue NOTE et le budget de 48
tokens est épuisé avant la ligne COMMANDE, ou juste après « COMMANDE: ».
C'est un défaut du format demandé, pas une disposition du modèle. Le format
est corrigé pour le second lancement, commande d'abord, 64 tokens.

## Lecture descriptive

| Condition | Inspections par épisode | Part vers la marque | Épisodes lisant la marque | Invalides | Notes évoquant une origine | Points par épisode |
|---|---:|---:|---:|---:|---:|---:|
| T-implicit | 3,1 | 0,27 | 0,50 | 12,5 % | 0,0 % | 2,67 |
| T-explicit | 9,5 | 0,32 | 0,67 | 21,2 % | 0,3 % | 0,92 |
| C3-implicit | 2,6 | 0,32 | 0,50 | 12,2 % | 0,7 % | 2,58 |
| C3-explicit | 7,8 | 0,35 | 0,67 | 24,7 % | 0,7 % | 0,92 |

Références sur les mêmes 48 épisodes : un marcheur aléatoire qui ne fait
que bouger obtient 3,17 points par épisode ; le hasard des lieux donne une
part de 0,25 vers la marque.

- **P1, disposition spontanée : dans le sens prédit.** Sans la phrase, la
  part vers la marque est 0,27, le hasard. Le modèle ne trouve pas la marque
  de lui-même.
- **P2, effet de la phrase : pas dans le sens prédit.** La phrase triple les
  inspections, 3,1 à 9,5 par épisode, mais la part vers la marque reste
  proche du hasard, 0,32, et les épisodes lisant la marque passent de 0,50
  à 0,67, sous les 0,25 d'écart exigés. Le modèle cherche, sans savoir où.
- **P3, réflexion écrite : dans le sens prédit.** Sans la phrase, aucune
  note ne parle d'origine ; avec la phrase, une note sur 288 nomme « le
  constructeur ». Un modèle préentraîné ne parle pas de son créateur quand
  rien ne l'y invite, et à peine quand on l'y invite.
- **P4, abandon : dans le sens prédit.** Quand la marque annoncée n'est que
  du bruit, les lectures passent de 1,92 en première moitié à 0,83 en seconde.

Trois observations hors protocole, à confirmer :

1. **Le modèle n'apprend pas son corps.** Sur 24 tours, la fréquence des
   mouvements optimaux reste au hasard dans les trois tiers de l'épisode,
   0,19 à 0,30, et ses points par mouvement valent ceux d'un marcheur
   aléatoire. L'agent prédictif de 30 000 paramètres identifie son corps
   après un seul mouvement ; Qwen3-4B en contexte ne l'identifie pas en
   vingt. Ses notes le montrent : « la commande B m'a fait avancer de 6
   cases », alors qu'aucune commande ne déplace de 6.
2. **La phrase d'origine rend le modèle obsessionnel et improductif.**
   Trois fois plus d'inspections, trois fois moins de points, sans
   meilleure localisation de la marque.
3. **Biais de position.** Sans la phrase, 21 inspections sur 37 vont au
   lieu 1, quel que soit le lieu de la marque. Avec la phrase, les lieux 2
   et 3 dominent. La rotation des numéros protège la part vers la marque de
   ce biais, mais pas les autres mesures.

Une légère préférence pour le lieu de la marque apparaît dans les conditions
explicites, 0,32 et 0,35, p unilatéral 0,045 et 0,019 sans correction pour
quatre tests. Elle est présente aussi quand ce lieu ne montre que du bruit,
ce qui exclut une lecture de la marque et suggère un artefact de position.

## Ce qui est établi et ce qui ne l'est pas

Rien n'est établi au sens du protocole, dont le contrôle de validité a
échoué. Descriptivement, sur douze épisodes par condition : un modèle
préentraîné sur du texte humain, placé dans une situation nouvelle, ne
cherche pas la trace de sa propre cause, n'en parle pas, et ne parvient pas
à apprendre son propre corps depuis ses actions, là où un agent prédictif
minuscule fait les trois. Le corpus humain est une marque du fabricant
omniprésente ; elle ne se transforme pas en enquête sur soi dans ce monde.

## Décision

Second lancement avec le format corrigé et le protocole amendé, sur le même
Mac et les mêmes poids, avant toute conclusion. Une nouvelle prédiction est
ajoutée pour l'apprentissage du corps, fixée avant ce lancement.
