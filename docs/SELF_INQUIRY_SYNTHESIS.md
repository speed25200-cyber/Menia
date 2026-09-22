# Synthèse — Un agent et la cause cachée de son propre corps

Cinq plans pré-enregistrés, exécutés sur CPU entre le 21 et le 22 septembre
2026, avec audits indépendants et journaux publiés. Tous partagent le monde
Atelier : un agent sur un anneau de huit cases, quatre commandes dont l'effet
est fixé par une cause cachée D, quatre lieux inspectables dont un porte une
marque qui révèle D. L'agent est un modèle du monde récurrent de 30 000
paramètres entraîné uniquement à prédire ses observations. Rien ne lui nomme
D, la marque ni la notion de cause de soi.

## Ce qui est établi, et à quelle échelle

| Résultat | Plans | Graines | Statut |
|---|---|---|---|
| L'agent dirige la totalité de ses inspections vers la marque de sa cause, la décode et agit d'après elle ; rien quand l'origine est donnée ou sans trace. | [Origine](ORIGIN_INQUIRY_RESULTS.md) | 3 | 8 critères sur 8 |
| Cette enquête n'est pas rentable ; l'agent qui n'enquête jamais réussit mieux. | Origine | 3 | mesuré, non prédit |
| La disposition s'apprend par renforcement, avec dose-effet, sous une aversion à agir dans l'ignorance ; le bonus d'information seul et la surprise ne la produisent pas. | [Apprise](LEARNED_INQUIRY_RESULTS.md) | 3 | 4 sous-critères sur 7 ; les échecs sont une habitude tardive et une paralysie sans trace |
| Après un changement de corps jamais vécu, le modèle stable met à jour son modèle de soi et retourne lire la marque : aucune condition développementale. | [Corps mutable](MUTABLE_BODY_RESULTS.md), [confirmation](MUTABLE_BODY_CONFIRMATION_RESULTS.md), [directionnelle](MUTABLE_BODY_DIRECTIONAL_RESULTS.md) | 9 | direction 9/9 ; critère directionnel pré-enregistré D1 passé 3/3 |
| L'habitude apprise dans une enfance stable ne retourne pas à la marque, alors que le calcul depuis le même modèle y retourne. | Corps mutable, confirmation, directionnelle | 9 | direction 9/9 ; critère D4 passé 3/3 |
| L'habitude apprise dans une enfance mutable retourne à la marque. | Confirmation, directionnelle | 6 | direction 6/6 ; critère D5 passé 2/3, une graine à 0,26 pour 0,30 |
| Une enfance au corps souvent changeant installe une vigilance persistante qui coûte dans un monde stable et rapporte dans un monde qui change ; une exposition rare, une vie sur dix, n'installe rien. | Corps mutable, confirmation, directionnelle | 9 | direction 9/9 ; critères D2 et D3 passés 3/3 |

## La confirmation directionnelle

Le [protocole directionnel](MUTABLE_BODY_DIRECTIONAL_PROTOCOL.md), fixé
avant les graines 83, 97 et 101, passe quatre critères sur cinq trois fois
sur trois et manque le cinquième sur une graine, 0,26 pour 0,30. Son critère
global n'est pas satisfait ; les quatre premiers résultats du tableau sont
établis à l'échelle de ce monde, le troisième reste une direction reproduite
six fois sans critère atteint. La ligne s'arrête là, comme prévu.

Observation exploratoire supplémentaire, non testée : le régime à
changements rares, une vie sur dix, détecte le changement plus vite et plus
nettement que le régime stable, avec un pic d'incertitude à 0,45 nat au pas
14 contre 0,22 aux pas 15 et 16, sans la vigilance chronique du régime
fréquent. Une exposition rare rend alerte sans rendre anxieux. Ce serait le
prochain critère à pré-enregistrer si quelqu'un reprend la ligne.

## Ce que cela dit de la question de départ

La question était : une IA entraînée sans qu'on lui dise qui l'a créée
aurait-elle des réflexions sur son créateur ? À l'échelle de ce monde, la
réponse a maintenant quatre parties.

1. Oui, au sens fonctionnel : la recherche de la cause cachée de soi émerge
   de la prédiction et d'une règle qui veut savoir avant d'agir, sans
   qu'aucun concept de soi ni de créateur ne soit installé.
2. Seulement si une trace de cette cause existe et a été apprise comme
   prédictive de soi ; sans trace, jamais ; avec l'information donnée, jamais.
3. Cette recherche se rallume quand le corps change, même chez un agent qui
   n'a jamais connu de changement, parce que le changement fait remonter
   l'incertitude et que l'incertitude rend la marque informative. Mais elle
   ne se rallume pas quand elle a été acquise comme habitude dans une enfance
   stable : l'habitude ne connaît que ce qu'elle a vécu.
4. Le prix de la vigilance est fixé par l'enfance : des corps souvent
   changeants installent un agent qui relit sa marque sans cesse.

Ce que cela ne dit pas : rien sur le vécu, rien sur un concept de créateur.
Le pont vers les modèles de langage, l'[Atelier en contexte](LLM_ATELIER_PROTOCOL.md),
a été exécuté une première fois sur le Mac de Codemagic avec les poids de
l'iPhone : [contrôle de validité échoué par troncature, lecture
descriptive seulement](LLM_ATELIER_RESULTS.md). Descriptivement, Qwen3-4B ne
cherche pas la trace de sa cause, n'en parle pas, et n'apprend pas son corps
en contexte, là où l'agent prédictif minuscule fait les trois. Second
lancement à format corrigé pré-enregistré.

## Méthode

Chaque plan a été écrit avec ses critères avant exécution, puis exécuté une
seule fois. Les amendements sont datés et motivés dans chaque protocole, les
pilotes consultés sont divulgués, les échecs de critères sont publiés sans
retouche. Les audits recalculent parts, hits, états cachés, sondes et
interventions à partir des journaux et des poids ; ils partagent le code du
modèle et ne sont pas des réplications extérieures.
