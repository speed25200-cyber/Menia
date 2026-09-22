# Atelier en contexte sur le Mac de Codemagic — premier lancement, 22 septembre 2026

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
