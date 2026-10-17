# Vérification bibliographique — l'alternance du conscient et du supprimé (17 octobre 2026)

Faite pendant le test 37, **avant la fin de ses apprentissages (graines 44 à
49) et sans lecture de ses verdicts**, à la suite d'un préprint nouveau. Elle
sert à savoir ce qui est connu avant d'écrire le plan d'un pilote. Recherche
web (moteur de recherche, puis lecture des articles accessibles). Elle n'est
pas exhaustive. Elle ne change rien au test 37.

## Le préprint

Hesse, Lanfranchi, McGrory, Shi, Engel et Tsao, « Rapid alternation of
conscious and suppressed stimulus representations in the primate brain »,
*bioRxiv*, 16 octobre 2026, doi 10.64898/2026.10.15.759299 (**texte intégral
lu**, PDF de 50 pages ; non relu par des pairs).

- **Dispositif.** Deux macaques, sondes Neuropixels dans les zones des
  visages ML et AM (cortex inférotemporal) et dans deux régions
  préfrontales. Suppression binoculaire par flash : une image seule pendant
  1 ou 2 s, puis une autre dans l'autre œil ; la nouvelle domine. 1 500
  visages tirés d'un espace de traits.
- **Pas de rapport chez le singe.** Les singes fixent un point. « Conscient »
  repose sur 14 humains : l'image flashée est rapportée dans 82 % des essais
  (87 % pour une image seule), de 300 à 1 000 ms après le flash.
- **Méthode.** Par pas de 10 ms, un rapport de vraisemblance de Poisson
  demande si les décharges ressemblent plus à la réponse au visage A seul
  ou au visage B seul (gabarits pris sur les essais sans conflit). Un
  modèle de Markov caché donne les durées.
- **Résultats.**
  - Deux états, occupés à parts égales (environ 50 % chacun). Durée médiane
    64 ms (conscient) et 59 ms (supprimé), plateau vers 60 ms quelle que
    soit la taille des pas. Ce n'est pas une oscillation démontrée : aucune
    analyse de phase ni de potentiel de champ.
  - Dans l'état 1, on reconstruit le visage conscient (plus de 80 % de
    variance expliquée pour certains traits ; plafond sans conflit : plus de
    90 %) et pas le supprimé. Dans l'état 2, l'inverse.
  - Les mêmes neurones : l'axe d'accord du visage supprimé dans l'état 2 est
    indiscernable de celui mesuré sans conflit (p = 0,35) ; corrélation des
    gains entre états r = 0,90.
  - Moyennée, la modulation est faible (pente 0,23). Par état : 0,93 et
    −0,76. Gains égalisés, il ne reste rien (r = −0,02).
  - **Ce qui distingue les états est petit** : gain d'accord 1,21 fois plus
    fort dans l'état conscient ; un décodeur linéaire reconnaît l'état dans
    66 % des cas (hasard 50 %) ; taux moyen un peu plus élevé dans l'état 2.
  - ML et AM basculent **faiblement ensemble**. Le préfrontal montre la même
    structure, de moins bonne qualité.
- **Circularité, traitée par les auteurs.** Ils montrent qu'un modèle sans
  alternance (chaque neurone à la moyenne de ses deux réponses) reproduit les
  reconstructions quand le même échantillon sert à définir les états et à
  les lire. D'où : états inférés sur une moitié des neurones et lus sur
  l'autre ; validation en laissant un neurone de côté ; gabarits mélangés ;
  une batterie de quatre modèles génératifs (deux états coordonnés,
  conscient seul, moyenne, alternance propre à chaque neurone). Seul le
  modèle à deux états coordonnés passe tous les tests ; la moyenne et
  l'alternance non coordonnée s'effondrent quand on lit sur l'autre moitié.
- **Une limite non discutée** (notre lecture). Le visage supprimé est
  toujours celui qui était là depuis 1 à 2 s, le conscient toujours le
  nouveau. Le gain plus fort et le taux plus bas de l'état 1 pourraient
  venir en partie de la nouveauté ou de l'adaptation.
- **Leur interprétation.** Ils écartent le « problème difficile ». Ils
  proposent un **routage dynamique** : les époques conscientes iraient vers
  le rapport, l'introspection et l'action ; les époques supprimées
  exploreraient l'autre interprétation sans être diffusées. Le fil de D. Tsao
  sur X (« plus près de la recette ») va au-delà de l'article.

## La question pour Menia

Quand une entrée admet deux causes incompatibles, un petit réseau appris
porte-t-il les deux hypothèses **séparément, à des moments différents**
d'un même état (alternance), ou un **mélange** fixe des deux (moyenne) ? Et
une sortie de confiance (le pari du test 37) lit-elle cette alternance ?

## Ce qui existe

- **Alternance entre options pendant une décision.** Rich et Wallis,
  « Decoding subjective decisions from orbitofrontal cortex », *Nature
  Neuroscience* 19, 2016 (https://link.springer.com/article/10.1038/nn.4329 ;
  page non lisible ici, résumé d'après la recherche). Un décodeur appris sur
  des essais à une seule image, appliqué aux essais de choix : la
  représentation **alterne** entre les deux options, d'autant plus que leurs
  valeurs sont **proches**, et cela prédit le temps de réponse et le choix.
  C'est la même méthode de gabarits que Hesse et al.
- **Multiplexage temporel de deux stimuli par les mêmes neurones.** Caruso
  et al., « Single neurons may encode simultaneous stimuli by switching
  between activity patterns », *Nature Communications* 9, 2715, 2018
  (https://doaj.org/article/a6d99ceb52a94256ad330919d3864eab ; résumé
  d'après la recherche). Colliculus inférieur, deux sons : des neurones
  passent d'une réponse à l'autre au cours d'un essai ou d'un essai à
  l'autre ; fluctuations corrélées entre paires ; des fluctuations
  semblables dans le cortex inférotemporal pour plusieurs images.
- **La rivalité comme échantillonnage de la croyance.** Gershman, Vul et
  Tenenbaum, « Multistability and perceptual inference », *Neural
  Computation* 24, 2012 (https://dspace.mit.edu/handle/1721.1/70125 ; résumé
  d'après la recherche). Les alternances de la rivalité s'expliquent si le
  système visuel approche la croyance sur les causes par des **échantillons**
  successifs (Monte-Carlo par chaînes de Markov). Version de conférence :
  NeurIPS 2009. Voir aussi « Neuronal Adaptation for Sampling-Based
  Probabilistic Inference in Perceptual Bistability », NeurIPS 2011
  (adaptation des neurones et échantillonnage).
- **Un réseau artificiel qui sépare deux entrées dans le temps.** Duecker,
  Idiart, van Gerven et Jensen, « Oscillations in an artificial neural
  network convert competing inputs into a temporal code », *PLoS
  Computational Biology* 20(9), e1012429, 2024
  (https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012429 ;
  début du texte lu). Un réseau à deux couches cachées, déjà entraîné, reçoit
  une réfraction et une inhibition pulsée à 10 Hz **ajoutées après coup** :
  deux lettres présentées ensemble sortent l'une après l'autre au lieu d'un
  mélange. Le mécanisme est imposé, pas appris.
- **Interprétations concurrentes dans les modèles de langage.**
  - « Verbalizable Representations Form a Global Workspace in Language
    Models », arXiv 2607.15495, juillet 2026 (déjà dans
    `docs/LITERATURE_CHECK_READER_2026-09-30.md` ; résumé d'après la
    recherche) : pour un mélange construit de deux concepts, la lecture est
    graduelle dans les premières couches puis **bascule** vers l'un ou
    l'autre à partir d'une couche (environ la 38e), rapprochée de
    l'« ignition ». Une bascule le long des couches, pas une alternance dans
    le temps.
  - Takatsuki, Doi, Watahiki, Seth et Yanaka, « (How) Do MLLMs Report
    Bistable Images Like Humans? », arXiv 2609.13254, septembre 2026
    (résumé lu) : LLaVA rapporte le canard-lapin surtout de façon
    **exclusive**, modulable par l'image et la langue ; les auteurs relient
    cela à des représentations concurrentes dans les tokens d'image.
  - « Divergent large language model predictions from convergent
    representations in ambiguous word pairs », arXiv 2608.01816 (résumé
    d'après la recherche) : les deux lectures d'un mot ambigu divergent au
    milieu du réseau et se rapprochent en fin de réseau.
- **Rythme de l'attention.** Fiebelkorn et Kastner, « A rhythmic theory of
  attention », *Trends in Cognitive Sciences*, 2019 (cité par Hesse et al.,
  non relu ici) : la sensibilité fluctue avec un rythme, comme un
  projecteur qui échantillonne.

## Ce que nous n'avons pas trouvé

Un agent appris par renforcement (et non un réseau auquel on ajoute
l'oscillation) chez qui l'on chercherait, avec les contrôles de Hesse et al.
(gabarits sans ambiguïté, lecture sur une moitié des unités, modèles
« moyenne » et « non coordonné »), si une alternance entre deux causes
cachées **apparaît** quand on lui donne un temps interne, et si sa
confiance la lit. La recherche est rapide ; cela peut exister.

## Ce que cela veut dire pour Menia

- **Déjà montré** : qu'un cortex peut porter deux contenus dans les mêmes
  unités en alternant (Caruso 2018, Rich et Wallis 2016, Hesse et al. 2026) ;
  que l'alternance peut s'interpréter comme un échantillonnage de la
  croyance (Gershman 2012) ; qu'un réseau artificiel peut séparer deux
  entrées dans le temps si on lui **impose** une oscillation (Duecker 2024) ;
  qu'un modèle de langage bascule vers une lecture le long des couches
  (2607.15495) et rapporte une image ambiguë de façon exclusive (2609.13254).
- **Pour nos textes.** Le contenu non conscient peut être riche jusque dans
  le préfrontal : la présence d'un contenu dans un module « haut » n'est pas
  un marqueur d'accès. Le marqueur, dans ce préprint, est une étiquette
  faible (gain ×1,21) et, selon les auteurs, un routage vers l'aval. Dans
  l'agent à indicateurs, la sélection est déjà un routage (un seul module
  écrit par pas, la politique ne lit que l'espace de travail), mais il n'y a
  ni deux hypothèses sur une même entrée dans un même module, ni temps
  interne. Le témoin `round_robin` est une alternance imposée : ce n'est pas
  un « non-mécanisme » au sens de ce préprint.
- **Méthode à reprendre** : inférer l'état sur une moitié des unités et le
  lire sur l'autre ; comparer aux modèles « moyenne » et « non coordonné ».
  Moyenner peut faire croire à une information faible quand la structure
  est forte ; nos lecteurs d'états internes (Colab 10 à 24) lisaient l'état
  final ou des moyennes.
- **Ce qui serait propre à un pilote de Menia**, s'il réussissait : une
  alternance **apprise et non imposée**, mesurée avec ces contrôles, dans un
  agent dont la confiance (le pari) la lit. Ce serait une contribution
  **étroite**, de méthode et de contrôle, pas une découverte de principe. Un
  échec (un mélange fixe, ou une alternance que le pari ne lit pas) serait
  publié de même.
- Aucun de ces travaux, ni le nôtre, ne dit rien d'un ressenti. Le préprint
  étudie un corrélat chez le singe sans rapport ; il ne dit pas qu'un
  mécanisme semblable suffirait à produire une expérience.

## Sources

- https://www.biorxiv.org/content/10.64898/2026.10.15.759299v1
- https://x.com/doristsao/status/2111316522098733078
- https://link.springer.com/article/10.1038/nn.4329
- https://doaj.org/article/a6d99ceb52a94256ad330919d3864eab
- https://dspace.mit.edu/handle/1721.1/70125
- https://proceedings.nips.cc/paper_files/paper/2011/hash/b51a15f382ac914391a58850ab343b00-Abstract.html
- https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012429
- https://arxiv.org/abs/2607.15495
- https://arxiv.org/abs/2609.13254
- https://arxiv.org/abs/2608.01816
