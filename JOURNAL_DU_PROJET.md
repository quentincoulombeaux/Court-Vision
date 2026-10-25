# CourtVision — Journal complet du projet

Ce document retrace tout ce qui a été fait sur CourtVision, du premier modèle de détection de balle
jusqu'à l'interface web : les choix de technologies, de modèles et de datasets, les essais qui ont
échoué, les réglages et les mesures. Il suit l'ordre chronologique (19 au 22 octobre 2026), puis
récapitule l'état final et les limites connues.

---

## Sommaire

1. [Point de départ (19 octobre)](#1-point-de-départ-19-octobre)
2. [Exploration de RacketVision et de TrackNetV3 (20 octobre)](#2-exploration-de-racketvision-et-de-tracknetv3-20-octobre)
3. [Nouveau dataset de balle pour YOLO (20 octobre)](#3-nouveau-dataset-de-balle-pour-yolo-20-octobre)
4. [Entraînement de YOLO11s sur Google Colab (20–21 octobre)](#4-entraînement-de-yolo11s-sur-google-colab-2021-octobre)
5. [Suivi de la balle et traînée avec YOLO](#5-suivi-de-la-balle-et-traînée-avec-yolo)
6. [Détection des rebonds : version 1 (méthode géométrique)](#6-détection-des-rebonds--version-1-méthode-géométrique)
7. [Détection des rebonds : mesure et améliorations](#7-détection-des-rebonds--mesure-et-améliorations)
8. [Détection du terrain (YOLO11s-pose + homographie)](#8-détection-du-terrain-yolo11s-pose--homographie)
9. [Pipeline complet et jugement in / out](#9-pipeline-complet-et-jugement-in--out)
10. [Interface web](#10-interface-web)
11. [Corrections sur une vidéo sur dur (Alcaraz–Shelton)](#11-corrections-sur-une-vidéo-sur-dur-alcarazshelton)
12. [Interface v2 : mini-carte, anglais, son](#12-interface-v2--mini-carte-anglais-son)
13. [Vitesse des frappes](#13-vitesse-des-frappes)
14. [État final du système](#14-état-final-du-système)
15. [Tous les réglages](#15-tous-les-réglages)
16. [Limites connues et pistes](#16-limites-connues-et-pistes)
17. [Inventaire des fichiers](#17-inventaire-des-fichiers)

---

## 1. Point de départ (19 octobre)

**Ce qui existait :**

- Un petit dataset Roboflow (*Tennis-Analysis-system-with-YOLO*) de **384 images annotées** au
  format COCO (`train\`, `valid\`, `test\`), avec une seule classe, « tennis ball ».
- `scripts\prepare_yolo.py` : conversion COCO → YOLO dans `yolo\`, avec une séparation pensée pour
  éviter les fuites. Les scènes « clay » (terre battue) et « fed » (gazon) sont réservées au test.
  Les autres scènes sont découpées en blocs de frames consécutives, la fin de chaque bloc va en
  validation, et un tampon de 15 frames sépare les deux.
- `scripts\train.py` : entraînement de **YOLO11n** sur CPU, en 1280 px, 50 époques, avec des
  augmentations adaptées aux petits objets :
  - `scale=0.25` (au lieu de 0.5, qui réduirait une balle de 12 px à 6 px) ;
  - pas de rotation, pas de miroir vertical, `close_mosaic=10`.
- `scripts\predict_image.py` : test sur images.

**Résultat : le modèle `runs\ball_v1`**, entraîné environ 6 h 45 sur le CPU (à peu près
10 minutes par époque, arrêt à l'époque 39 sur 50).

| Split | Précision | Rappel | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Validation (mêmes scènes que l'entraînement) | 0,93 | 0,76 | 0,80 | 0,41 |
| Test (scènes jamais vues) | 0,85 | **0,26** | **0,32** | 0,10 |

Ce modèle ne généralisait pas : il ne trouvait qu'une balle sur quatre sur des scènes nouvelles.
Le dataset était trop petit et déséquilibré.

---

## 2. Exploration de RacketVision et de TrackNetV3 (20 octobre)

**Source :** le dépôt GitHub `OrcustD/RacketVision`, publié avec un article présenté à AAAI 2026.
C'est un benchmark couvrant trois sports (badminton, tennis de table, tennis) et trois tâches :
suivi de la balle, pose de la raquette et prédiction de trajectoire. Seule la détection de la
balle de tennis nous intéressait.

**Ce qui a été récupéré :**

- **Le code**, cloné dans `RacketVision\`.
- **Les poids `balltrack_best.pth`** (47 Mo), via `download_checkpoints.py`. C'est un modèle
  TrackNetV3 unique pour les trois sports.
- **Le dataset, partie tennis seulement** (1,8 Go) : 431 vidéos de rallyes en 1920x1080, un CSV de
  positions de balle par vidéo, et les splits train/val/test.
  - Hugging Face a **limité notre débit** (erreur 429) à cause des milliers de petits fichiers JSON
    de raquette. Nous les avons exclus du téléchargement, qui s'est fait en 2 workers avec reprise
    automatique.

**Pourquoi TrackNetV3 marche bien :** au lieu d'une seule image, il prend en entrée **4 images
consécutives et l'image de fond médiane** de la vidéo (15 canaux en 512x288), et sort une carte de
chaleur de la balle. Il voit donc le mouvement.

**Écrit pour ce test : `scripts\predict_video_tracknet.py`.** C'est un script autonome qui lit
n'importe quelle vidéo, calcule le fond médian sur 200 images échantillonnées, puis fait
l'inférence par lots.

- **Bug trouvé dans le code officiel.** Le `inference.py` du dépôt utilise les images *t-4 à t-1*
  pour prédire l'image *t*, alors qu'à l'entraînement le dataloader utilise *t-3 à t*. Notre script
  reprend l'alignement de l'entraînement.
- **Dépendance :** `mmengine` est nécessaire, car le checkpoint contient des objets mmengine dans
  son pickle.

**Résultats :**

- Sur une vidéo du dataset, comparée à la vérité terrain : **erreur médiane de 1,4 px** en 1080p,
  et 24 balles trouvées sur 24.
- Sur `video_test.mp4` (Roland-Garros) : balle trouvée sur **671 frames sur 700 (96 %)**, au plus
  5 frames manquées d'affilée, aucun saut aberrant.
- **Défaut rédhibitoire :** c'est lent sur le CPU, environ **1,1 à 1,4 s par frame**.

L'utilisateur a choisi de garder **YOLOv11** comme modèle de balle, en s'appuyant sur les données
de RacketVision.

---

## 3. Nouveau dataset de balle pour YOLO (20 octobre)

**Script : `scripts\prepare_racketvision_yolo.py`**, qui produit `yolo_rv\` (3,9 Go).

- **Seules les frames annotées sont extraites** : 21 544 sur environ 150 000. Les annotations ne
  portent que sur une frame sur 4 ou 5.
- **Les images sont redimensionnées en 1280x720**, en JPG qualité 92.
- **La séparation officielle de RacketVision est conservée** (un clip = un rallye) :
  - train : 17 494 images, dont 1 157 sans balle ;
  - val : 1 900 images, dont 118 sans balle ;
  - test : 2 150 images, dont 203 sans balle.
- **Les boîtes sont créées à partir des centres.** RacketVision ne donne que le centre de la balle :
  on crée donc une **boîte carrée de 16 px** (en 1920x1080). Ce choix a été vérifié visuellement
  sur des crops : la balle fait de 6 à 12 px, et le flou de mouvement déborde un peu.
- **Les images sans balle** (Visibility = 0) reçoivent un fichier label vide : ce sont des
  exemples négatifs.
- Le traitement est **parallélisé** sur 4 processus.

Ce dataset est environ **55 fois plus grand** que l'ancien, et couvre plusieurs surfaces.

---

## 4. Entraînement de YOLO11s sur Google Colab (20–21 octobre)

**Pourquoi Colab :** sur le CPU, il fallait environ 2 s par image, soit **environ 10 h par
époque**. Ce n'était pas faisable. La contrainte de l'utilisateur était un entraînement de
moins de 12 heures.

**Choix du modèle : YOLO11s.** Le nano détectait moins bien les très petits objets, et les
versions medium et large étaient 2 à 4 fois plus lentes à l'inférence pour un gain faible.

**Notebook : `colab\train_yolo11s_colab.ipynb`**

1. Vérification du GPU, puis montage de Google Drive.
2. Reconstruction du dataset **directement dans Colab** à partir de Hugging Face, ce qui évite à
   l'utilisateur d'envoyer 3,9 Go. Le téléchargement réessaie automatiquement en cas de limite de
   débit.
3. Entraînement avec ces réglages :
   - `YOLO('yolo11s.pt')`, `imgsz=1280`, `epochs=100` ;
   - **`time=10`**, un plafond en heures qui ajuste lui-même le nombre d'époques ;
   - `batch=16`, `cache=False` ;
   - les mêmes augmentations pour petits objets qu'avant ;
   - `patience=15`.
4. **Sauvegarde sur Drive à chaque époque, avec reprise automatique** si la session coupe.
5. Évaluation finale sur val et test.

**Incidents et corrections :**

- **« credential propagation was unsuccessful »** au montage de Drive. Cause : la fenêtre
  d'autorisation n'avait pas été entièrement validée. Correction : une solution de secours sans
  Drive a été ajoutée au notebook, avec les consignes « Tout sélectionner » et une fenêtre de
  navigation privée.
- **Mémoire GPU saturée avec `batch=16`.** Ultralytics est redescendu tout seul à 8.
- **Crash à l'époque 19**, après environ 5 h : c'est la limite d'usage GPU de Colab gratuit. Les
  poids `best.pt` ont été récupérés sur Drive.

**Résultat : `runs\ball_rv\weights\best.pt`**, la meilleure époque autour de la 16.

| | mAP50 | mAP50-95 | Rappel |
|---|---|---|---|
| Ancien `ball_v1` (val) | 0,80 | 0,41 | 0,76 |
| **YOLO11s RacketVision (val)** | **0,94** | **0,61** | **0,87** |

Le scénario officiel de test n'a pas été lancé, puisque la session a coupé avant.

---

## 5. Suivi de la balle et traînée avec YOLO

**Script : `scripts\predict_video_yolo.py`.** YOLO ne voit qu'une image à la fois ; un filtre de
suivi choisit donc la balle parmi les candidats de chaque frame. La traînée relie les 10 à 12
dernières positions : elle n'est pas apprise, elle est dessinée.

**Version 1 :** le candidat le plus confiant dans un rayon de 8 % de la largeur de l'image autour
de la dernière position. Après 5 frames perdues, on accepte un saut.

- Problème d'environnement : le Python système avait une version d'Ultralytics trop ancienne pour
  YOLO11, et il manquait `pandas` et `tqdm` dans le `.venv`. Le script a été réécrit sans eux.
- Résultat avec `ball_v1` : balle vue sur 38 frames sur 150 seulement. Avec `ball_rv`, le suivi est
  bon, de l'aveu même de l'utilisateur.

**Version 2 (22 octobre), voir la section 11 :** le suivi est découpé en deux étapes,
`detect_candidates` (YOLO, lent, mis en cache) puis `link_track` (chaînage avec prédiction de la
vitesse, rapide).

---

## 6. Détection des rebonds : version 1 (méthode géométrique)

**Idée de l'utilisateur :** repérer des segments de trajectoire rectilignes, puis signaler un
rebond quand la direction change d'un angle compris entre deux seuils.

**Implémentation dans `scripts\detect_bounces.py`, qui travaille sur le CSV de suivi :**

1. **Nettoyage.** La vidéo de test a des **frames dupliquées** : la même position revient 2 ou
   3 frames de suite. On fusionne ces doublons, puis on retire les points isolés aberrants.
2. **Deux droites.** À chaque échantillon, une droite des moindres carrés est ajustée sur les
   5 positions d'avant et une autre sur les 5 d'après. Les deux segments doivent être rectilignes
   (écart de 4 px au maximum) et assez longs.
3. **Angle** entre les deux droites, compris entre 25° et 150°. Le point exact est l'intersection
   des deux droites.
4. **Premier essai : 35 rebonds**, alors que la vidéo en contient environ 14. L'angle seul ne
   suffit pas : une frappe du joueur proche casse aussi la trajectoire à environ 130°, autant
   qu'un rebond côté éloigné.
5. **Ajout d'une structure.** On repère les **frappes**, c'est-à-dire les demi-tours verticaux
   francs de Y avec une grande amplitude (« prominence »). On impose ensuite **au plus un rebond
   entre deux frappes** : celui qui casse le plus la trajectoire.
6. Le seuil bas a été relevé de 20° à 25°, ce qui a éliminé une frappe (frame 474) et un cas
   douteux (frame 592).

**Résultat sur Roland-Garros : 14 rebonds**, que l'utilisateur a jugés justes à environ 80–90 %.

---

## 7. Détection des rebonds : mesure et améliorations

### 7.1 Les données de mesure

Le dataset tennis original de **TrackNet (Huang et al., 2019)** annote chaque frame avec un
**statut** : 0 en vol, 1 frappe, 2 rebond. Il a été trouvé sur Hugging Face
(`oweng/tracknet_preprocessd_data`, `tennis.zip`, 2,5 Go). **Seuls les 95 fichiers `Label.csv` ont
été lus à distance** dans le zip, sans télécharger les images.

On obtient **523 rebonds et 516 frappes** sur 95 clips en 1280x720 à 30 fps. Ils sont stockés dans
`bounce_eval\tracknet_labels\`.

**Script de mesure : `scripts\eval_bounces.py`.** Un rebond détecté est juste s'il tombe à ±2
frames d'un rebond annoté. Le script calcule précision, rappel et F1, l'erreur de position, et le
rappel par type de rebond :

- **« axial »** : la balle arrive dans l'axe de la caméra ;
- **« filet »** : rebond dans la bande centrale de l'image ;
- **« autre »**.

L'option `--noise` simule un vrai détecteur : 1,5 px de bruit et 8 % de détections manquées.

### 7.2 Toutes les étapes, avec leurs mesures

| Étape | F1 | Axial | Filet | Commentaire |
|---|---|---|---|---|
| Version 1 (angle + frappes) | 0,655 | 32 % | 73 % | point de départ |
| **« Coup vers le haut »** (vitesses au lieu d'angles) | 0,79 | 47 % | 79 % | précision 0,72 → 0,90 |
| Fourchette d'angle haute retirée, test de paraboles | ~0,79 | 56–60 % | | paraboles : meilleur rappel, plus de faux positifs |
| Seuil de prominence des frappes à 0,16 | 0,805 | 57 % | 74 % | |
| Règle d'alternance des frappes | 0,805 | | | sans effet |
| **Rebond haut = la balle ralentit** (ratio < 0,8), qui sert de frontière | 0,817 | **70 %** | 74 % | |
| Tolérance de rectitude de 4 → **8 px**, mise à l'échelle 720p | 0,843 | 77 % | 75 % | bruité : 0,695 → 0,831 |
| Rectitude jugée sur la **géométrie** (pas le temps) | 0,844 | **81 %** | 76 % | sinon les frames dupliquées faussent tout |
| Bug « intervalle effacé » corrigé + **frappes par coup vers le bas** | **0,870** | **91 %** | **88 %** | rebond derrière le filet et rebond out retrouvés |
| **Frappe rapide vers le haut** = frappe du joueur proche (vitesse > 0,8 H/s, accélération > 1,2) | **0,884** | 91 % | **90 %** | |
| Seuil du coup vers le haut à **0,4** (section 11) | **0,888** | 91 % | 90 % | bruité : 0,868 |

### 7.3 Les principes retenus

1. **Signature physique du rebond.** Le sol renvoie toujours la balle **vers le haut de l'image**,
   alors que la gravité la tire toujours vers le bas. On compare donc les **vitesses** avant et
   après (et non plus seulement les directions), avec le temps reconstruit malgré les frames
   dupliquées. `kick = −Δvy / vitesse` doit valoir au moins 0,4. Ce critère voit aussi les rebonds
   dans l'axe de la caméra, où la direction change à peine mais où la vitesse verticale chute.
2. **Les frappes servent de frontières** : au plus un rebond entre deux frappes.
   - Une frappe se reconnaît à un demi-tour vertical franc en Y (prominence ≥ 16 % de la hauteur
     de l'image)…
   - …ou à un **coup brutal vers le bas** (kick ≤ −0,8), dont seule une raquette est capable…
   - …ou à un **départ rapide vers le haut en accélérant**, la frappe du joueur proche.
3. **Un rebond haut dans l'axe ressemble à une frappe** (demi-tour de Y côté joueur proche). On le
   distingue par la vitesse : **le rebond ralentit la balle, la frappe l'accélère**. Mesuré sur les
   annotations : vitesse après / avant médiane de 2,56 pour les frappes, contre 0,65 pour ces
   rebonds.

### 7.4 Les cas signalés par l'utilisateur et corrigés

- **Rebond derrière le filet et rebond out non détectés.** Cause : un bug où un « rebond haut » en
  fin de vidéo effaçait l'intervalle qui le précédait, plus une frappe du joueur du fond sans grand
  demi-tour. Corrigé par la règle du « coup vers le bas ».
- **Frappe annoncée comme rebond sur Wimbledon (frame 426).** Corrigé par la règle de la frappe
  rapide vers le haut.
- **Frame 160 de Wimbledon.** Ce n'est pas un problème de rebond : YOLO suit la tête de raquette
  pendant le geste. Cela relève du suivi.

---

## 8. Détection du terrain (YOLO11s-pose + homographie)

### 8.1 Le dataset

Le dossier `data\` (7,3 Go) a été fourni par l'utilisateur. Sa structure ressemble à celle du
projet open source *TennisCourtDetector*.

- **8 841 images** en 1280x720, tirées de **498 vidéos YouTube** ;
- **14 points** par image, toujours dans le même ordre :
  - 0 à 3 : les coins (lignes de double) ;
  - 4 à 7 : les bouts des lignes de simple ;
  - 8 à 11 : les croisements des lignes de service avec les lignes de simple ;
  - 12 et 13 : les « T ».
- **Deux pièges trouvés :**
  - le split d'origine **mélangeait les mêmes vidéos** en train et en val (434 vidéos sur 442) ;
  - **401 points étaient hors de l'image.**

### 8.2 La conversion

**Script : `scripts\prepare_court_pose.py`**, qui produit `court_pose\` et `court_pose.zip`
(1,52 Go, en JPG).

- **Nouvelle séparation par vidéo**, 80 / 10 / 10 : 6 802 / 1 077 / 962 images (399 / 50 / 49
  vidéos).
- Les points hors de l'image reçoivent la **visibilité 0**.
- **La boîte** est l'enveloppe des points visibles, plus 3 % de marge.
- **`flip_idx`**, pour que le miroir gauche/droite échange correctement les points :
  `[1,0,3,2,6,7,4,5,9,8,11,10,12,13]`.

### 8.3 Le choix du modèle : YOLO11s-pose

Les alternatives étudiées :

- **Réseau à cartes de chaleur** de type TrackNet / TennisCourtDetector : un peu plus précis au
  pixel, mais du code sur mesure, plus lourd, et en dehors de l'outillage Ultralytics.
- **YOLO11-pose** : même outil que pour la balle, rapide, et suffisamment précis une fois l'erreur
  rattrapée par l'**homographie**. C'est celui qui a été retenu.

### 8.4 L'entraînement

**Notebook : `colab\train_court_pose_colab.ipynb`**, qui enregistre dans un nouveau dossier Drive,
`CourtVision_court`.

- `yolo11s-pose.pt`, `imgsz=960` (le terrain est un grand objet), `time=8` h, `batch=16` ;
- **`mosaic=0`** (la mosaïque découpe le terrain en morceaux), `degrees=0`, `scale=0.3`,
  `translate=0.1`, `fliplr=0.5`.

**Incidents :**

- quota GPU épuisé, d'où le passage par un deuxième compte Google ;
- session déconnectée, reprise grâce à `last.pt` sur Drive.

**Résultat : `runs\court_pose\weights\best.pt`.** Ses métriques finales ne m'ont pas été
transmises. Mesures faites ensuite :

- sur une image d'Alcaraz–Shelton, les 14 points sont trouvés avec une confiance de 1,0 et
  **0,3 px** d'erreur de reprojection après homographie ;
- sur un gros plan sans terrain, le modèle ne trouve **aucun terrain**, ce qui est le comportement
  attendu.

### 8.5 L'homographie

**Script : `scripts\predict_court.py`.**

- **Le court réglementaire est codé en mètres** : 23,77 m de long, 10,97 m de large en double,
  8,23 m en simple, couloirs de 1,37 m, lignes de service à 6,40 m du filet, poteaux du filet à
  0,914 m de la ligne de double.
- **`cv2.findHomography` avec RANSAC**, sur les points dont la confiance est d'au moins 0,5 (il en
  faut 4 au minimum).
- **Validation sur les points annotés : erreur médiane de 0,36 px**, 95 % sous 0,73 px.
- **Détection toutes les 15 frames**, avec lissage (moyenne glissante de l'homographie). Les
  frames intermédiaires gardent la dernière homographie.
- **Correction ajoutée en section 11 :** quand le terrain apparaît ou disparaît entre deux
  détections (changement de plan), la frame exacte est retrouvée par **dichotomie**, pour environ
  4 détections de plus.

---

## 9. Pipeline complet et jugement in / out

**Script : `scripts\pipeline.py`.** L'ordre a été choisi par l'utilisateur :
**terrain → balle → rebonds → jugement**. Commencer par le terrain permet de savoir quand le court
est visible (les ralentis et gros plans sont alors marqués « non jugeables »).

- **Cache par étape** dans `runs\analysis\<nom>\` : `court.json`, `candidates.json`, `ball.csv`,
  puis `analysis.json`. Relancer ne refait que ce qui manque.
- **Le jugement est fait par l'interface, pas par le pipeline.** On stocke la position de chaque
  rebond **en mètres sur le court**, grâce à l'inverse de l'homographie de sa frame. L'interface
  décide ensuite selon le mode choisi.
- **Les règles :**
  - **Simple** par défaut, couloirs fautes. **Double** au choix.
  - **Orange « limite »** quand le rebond est à moins de ±5 cm de la ligne. Le seuil se règle de
    0 à 20 cm.
  - Le service n'est pas encore jugé sur le carré de service (choix de l'utilisateur, pour plus
    tard).
- **Le point de contact au sol** (section 11) : la balle est ramenée d'un rayon (3,3 cm) vers le
  sol avant d'être projetée.
- **Les rebonds impossibles sont écartés** : à plus de **3 m** hors des lignes, c'est en réalité
  le lancer de service ou une erreur de suivi.

---

## 10. Interface web

**Choix technique : une application web locale.** `app\server.py` est un serveur **FastAPI +
Uvicorn**. `app\static\` contient la page en **HTML / CSS / JavaScript pur**, sans compilation.

- **Raison principale :** les calques sont **dessinés en direct par le navigateur** sur la vidéo
  d'origine (`<canvas>` par-dessus `<video>`), à partir de `analysis.json`. Changer un réglage
  (simple/double, zone limite, calques) est donc instantané, sans rien recalculer.

**Le serveur :**

- bibliothèque `videos\` ;
- **une tâche à la fois en arrière-plan** (analyse ou export), le CPU ne permettant pas mieux ;
- route de lecture vidéo avec requêtes partielles (206) pour pouvoir se déplacer dans le lecteur ;
- **import** avec conversion en H.264 par ffmpeg si nécessaire ;
- **export MP4** via `scripts\render_overlay.py`.

**Le style « Centre Court de nuit » :**

- fond `#0D1117` ;
- accent jaune balle `#D9F24B`, réservé à la balle et aux actions ;
- lignes blanches façon craie ;
- verdicts : vert `#34D399` pour bonne, rouge `#F43F5E` pour faute, orange `#F59E0B` pour limite ;
- polices Inter et JetBrains Mono.

**L'animation d'impact :** deux ondes qui s'étalent **à plat sur le sol, dans la perspective du
court** (cercles en mètres projetés par l'homographie), un flash, et une étiquette IN / OUT /
LIMITE qui monte puis s'efface. Ensuite, une **marque reste au sol**.

**Les autres éléments :**

- la frise des rebonds (colorés) et des frappes, avec clic sur un rebond pour y sauter au ralenti ;
- les statistiques et la liste des rebonds ;
- les raccourcis clavier.

**Les tests :** toutes les routes ont été testées, ainsi qu'un parcours complet (import → analyse
→ export). L'interface elle-même n'a pas pu être vue par l'assistant ; c'est l'utilisateur qui l'a
validée.

---

## 11. Corrections sur une vidéo sur dur (Alcaraz–Shelton)

L'utilisateur a remonté plusieurs problèmes sur `alcaraz_shelton.mp4` (US Open, 1080p).

| Problème | Cause | Correction |
|---|---|---|
| Bonne annoncée faute près de la ligne (frame 350) | Le point pris était l'image d'avant le contact, encore en l'air. À la ligne de fond éloignée, 1 px = 8 cm. | **Point de contact recalculé** : meilleure coupure descente / remontée, **paraboles** (avec des droites, la gravité les fait se croiser trop haut, biais mesuré), puis contact = instant où les deux trajectoires verticales se rejoignent, puis **descente d'un rayon de balle**. Résultat : bonne, 33 cm à l'intérieur (au lieu de faute à 76 cm). |
| Faute annoncée au lancer de service (frame 26) | Saut du suivi pendant le lancer, projection au sol loin derrière la ligne | **Contrôle des sauts** dans les segments, et rebonds à plus de 3 m hors du terrain écartés |
| Rebond détecté à la frappe (frame 281) | Point aberrant que le filtre laissait passer (seuil fixe prévu pour 720p) | **Filtre d'aberrations proportionnel à la résolution** |
| Faux rebond pendant la volée (frame 560) | Balle en vol au-dessus du filet, kick 0,38 | **Seuil du coup vers le haut à 0,4** (0,5 faisait perdre un vrai rebond à 0,49) |
| Terrain affiché sur le gros plan de fin | Détection toutes les 15 frames seulement | **Dichotomie** pour retrouver la frame exacte du changement de plan |
| YOLO saute sur d'autres objets | Rayon de recherche fixe de 154 px en 1080p | **Nouveau suivi** avec prédiction de la vitesse (sauts de plus de 80 px : 21 → 3) |

**Le nouveau suivi** (`link_track`) fonctionne ainsi :

- position prédite = dernière position + vitesse estimée sur les deux dernières positions ;
- fenêtre de recherche = 3 % de la largeur + 2 fois la vitesse, élargie à chaque frame perdue ;
- choix du candidat qui équilibre confiance et proximité de la prédiction ;
- réacquisition après 5 frames perdues, sur un candidat d'au moins 0,3 de confiance ;
- **les détections brutes de YOLO sont mises en cache** (`candidates.json`), ce qui permet de
  refaire le suivi en quelques secondes.

**Ce qui reste non résolu :** le coup gagnant final (vers la frame 780), perdu par YOLO au moment
du rebond.

---

## 12. Interface v2 : mini-carte, anglais, son

- **Mini-carte « Bounce map ».** C'est un SVG à l'échelle en mètres, côté éloigné en haut comme sur
  la vidéo. Les rebonds apparaissent au fil de la lecture, le dernier est entouré, avec une onde à
  son arrivée. Au survol, une bulle donne les détails ; un clic saute au rebond. En simple, les
  couloirs sont teintés en rouge.
  - Bug corrigé : de grosses zones blanches apparaissaient. Un style global (contour de 2 unités,
    prévu pour les icônes) valait **2 m** dans le repère de la carte.
- **L'anglais par défaut**, avec un sélecteur EN / FR mémorisé. Il s'applique aux étiquettes vidéo
  (CLOSE ou LIMITE, SERVE ou SERVICE) et à l'export.
- **La disposition :** règles et export dans la colonne de gauche ; calques en grille, statistiques
  compactes et mini-carte visible en entier dans la colonne de droite.
- **Le mode démo est supprimé**, tout comme les numéros 1 à 4 des calques et les mentions de côté
  sur la carte.
- **Le son** est activé, avec un bouton muet et un volume. Il est aussi **conservé à l'export**
  (piste AAC de la vidéo d'origine) et à l'import.
- **Cache du navigateur :** les nouveautés n'apparaissaient pas. Correction : le serveur envoie un
  en-tête `Cache-Control: no-cache`, les fichiers sont versionnés (`?v=4`) et les requêtes du
  script sont faites en `fetch(..., {cache: "no-store"})`.

---

## 13. Vitesse des frappes

### 13.1 Premier essai abandonné : calibrer toute la caméra

L'idée était de retrouver la focale et la pose 3D de la caméra à partir d'une seule homographie, en
supposant le centre optique au centre de l'image.

- US Open : caméra plausible, environ 10 m de haut et 20 à 30 m derrière le court.
- Roland-Garros : caméra à **275 m**. Le zoom est si fort que la perspective est presque plate, et
  le calcul devient instable.
- Wimbledon : caméra **sur le court**. La vidéo est recadrée, donc l'hypothèse du centre optique
  est fausse.

Cette méthode n'était pas fiable et a été abandonnée.

### 13.2 La méthode retenue : une approximation locale

1. **Pixels par mètre de hauteur** à un endroit du court : on prend l'échelle en largeur fois
   `cos(e)`, où `sin(e)` = échelle en profondeur / échelle en largeur. C'est l'angle sous lequel la
   caméra voit le sol à cet endroit. Il suffit de l'homographie.
2. **Le point de frappe** est la balle à l'image de la frappe, supposée à **1 m** de haut, puis
   ramenée au sol.
   - **Service** : on cherche par dichotomie la hauteur qui place le serveur **40 cm à l'intérieur
     de sa ligne de fond**. On obtient environ 2,9 à 3,3 m. Une hauteur fixe de 2,7 m plaçait le
     serveur 3 à 5 m derrière sa ligne, ce qui est impossible.
   - **Reconnaissance du service** : c'est la première frappe du point (aucun rebond dans les 2,5 s
     qui précèdent) et la balle est presque immobile avant (le lancer, < 0,006 hauteur d'image par
     frame). Mesuré : 0,0022 à 0,0041 pour les services, au moins 0,0047 pour les coups d'échange.
3. **Vitesse moyenne** = distance 3D frappe → rebond / durée du vol.
4. **Vitesse au départ** (comparable aux radars) : `v0 = (e^{kD} − 1) / (k·t)`, avec
   **k = 0,019 m⁻¹** (frottement de l'air d'une balle de tennis, `ρ·Cd·A / 2m`).
5. **Garde-fous :**
   - la frappe doit être **de l'autre côté du filet** que le rebond, sinon on essaie la frappe
     précédente ;
   - la vitesse doit rester entre 30 et 260 km/h.

### 13.3 Les corrections suite aux remarques de l'utilisateur

1. **« Les vitesses sont un peu faibles. »** La frappe était repérée au **sommet de la courbe** de
   la balle dans l'image, 1 à 4 images avant le vrai contact. La durée était donc trop longue, et la
   vitesse sous-estimée de 5 à 25 %.
   - Correction : le contact est maintenant l'instant de **l'accélération brusque**
     (`contact_time`), c'est-à-dire la dernière position lente avant le départ rapide, en tenant
     compte des images où la balle est perdue.
2. **« Alcaraz ne sert pas à 217 km/h. »** La vidéo de Wimbledon n'a que ~15 positions distinctes
   par seconde.
   - Correction : le contact est placé au milieu entre la dernière position lente et la première
     rapide, en ignorant les doublons, et l'instant du rebond garde sa valeur fractionnaire.
   - Résultat : **197 km/h**.

| | Version 1 | Frappe à l'accélération | + temps fractionnaires |
|---|---|---|---|
| Service de Shelton (US Open) | 129 | 159 | **183** |
| Service d'Alcaraz (Wimbledon) | 143 | 217 | **197** |
| Échanges | 100–160 | 100–165 | **100–162** |

**Dans l'interface :**

- le calque « Speed », une étiquette jaune au départ de la trajectoire, avec la mention SERVE pour
  les services ;
- les statistiques : vitesse max, vitesse moyenne des échanges, service le plus rapide ;
- la vitesse dans la liste des rebonds et dans l'export.

**Précision estimée : environ ±10 %**, et environ ±7 % de timing en plus sur les vidéos à 15
images distinctes par seconde. L'effet de la balle (lift, slice) n'est pas modélisé.

### 13.4 Six améliorations (23 octobre)

**Modèles**

1. **Combler les trous du suivi** (`predict_video_yolo.fill_gaps`).
   - D'abord, une détection YOLO écartée est reprise si elle tombe là où la balle devrait être.
   - Sinon, une seule parabole relie les deux morceaux de trajectoire.
   - Si une parabole ne suffit pas, un rebond a eu lieu dans le trou : chaque morceau est prolongé
     par une droite jusqu'à leur point de rencontre, ce qui reconstitue le rebond. Des paraboles
     extrapolées s'emballaient, d'où le choix des droites.
   - Mesure avec simulation de balle perdue 2 à 6 images : F1 0,827 → **0,851** (18 rebonds
     retrouvés). Trajectoires propres : inchangé (0,888).
   - Sur Alcaraz–Shelton : 48 positions reconstituées.
2. **Détection des joueurs** (`scripts/detect_players.py`).
   - YOLO11n COCO « person », une image sur deux. Les pieds sont projetés sur le court.
   - Zone plausible : le court élargi, en excluant les abords des poteaux.
   - Un joueur de chaque côté du filet, suivi par continuité avec une **vitesse humaine maximale de
     9 m/s**. Une première version, trop tolérante (6 m en 4 images), avait accroché une personne
     immobile près de la chaise d'arbitre.
   - Utilisations : calque « Joueurs », positions en direct sur la carte, **règle du carré de
     service** (`serve_boxes` : carré en diagonale du serveur), et hauteur de frappe.
3. **Vitesse plus précise** (`fit_shot_3d`).
   - Reconstruction 3D du vol : gravité, frottement de l'air, et projection avec l'approximation
     locale de la caméra. L'ajustement (Levenberg-Marquardt, 6 inconnues) porte sur **toutes** les
     positions observées, avec la contrainte « touche le sol au rebond ».
   - Hauteur de frappe de départ = **taille du joueur** : 1,85 m × (pieds − balle) / hauteur de la
     boîte. La mesure par l'échelle du court donnait des valeurs irréalistes (3 m) près du joueur
     proche.
   - Contrainte serrée (`H_SIGMA = 0,1 m`) : avec 0,6 m, l'ajustement remontait la frappe à 3 m
     pour compenser le modèle de caméra simplifié.
   - Repli sur la méthode à deux points si l'ajustement dépasse 4 px d'écart.
   - Résultats : Alcaraz (service, Wimbledon) 197 km/h, Shelton (service) 147 km/h, échanges 100 à
     168 km/h.

**Interface**

4. **Relecture de l'impact** (`features.js`, touche R). Rendu 3D sur canvas, avec une caméra
   virtuelle placée derrière le frappeur :
   - la balle suit la trajectoire 3D reconstruite, au ralenti ;
   - puis la caméra descend à la verticale de la trace, dessinée en ellipse dans le sens du coup ;
   - les lignes sont dessinées à leur vraie largeur (5 cm, 10 cm pour les lignes de fond) ;
   - la distance trace → ligne est affichée en cm.
5. **Carte de chaleur** : densité des rebonds sur une grille de 10 cm (noyau gaussien de 0,9 m,
   rampe jaune → orange → rouge), filtrable (tous / bonnes / fautes).
6. **Rapport du match** : page imprimable / PDF avec résumé, carte des rebonds, carte de chaleur,
   graphique des vitesses et tableau des rebonds. Plus un export CSV.

### 13.5 Corrections sur deux nouvelles vidéos (Rune–Kecmanovic, 23 octobre)

- **La cause principale des erreurs n'était pas le calcul.** Ces vidéos avaient été analysées par un
  serveur lancé avant les dernières mises à jour, qui faisait tourner l'ancien code : pas de
  joueurs, pas de carré de service. Deux corrections :
  - le serveur recharge le code d'analyse avant chaque tâche ;
  - chaque analyse porte un `PIPELINE_VERSION`, et l'interface affiche le badge « À mettre à jour »
    quand il est dépassé.
- **Frappes prises pour des rebonds.** Une cassure en demi-tour (≥ 140°) dans la zone de frappe d'un
  joueur détecté devient une frappe. Mesuré sur les annotations : un angle supérieur à 150° concerne
  45 % des frappes, contre 7 % des rebonds. La règle s'applique aussi aux « rebonds hauts », mais
  seulement s'il y a demi-tour : un rebond juste devant le joueur proche apparaît au-dessus de ses
  pieds dans l'image.
- **Rebonds manqués** sur une vidéo à images dupliquées (environ 15 positions distinctes par
  seconde, parfois une seule position entre le rebond et la frappe). Une deuxième passe avec des
  segments de 3 positions cherche un rebond dans les intervalles entre frappes qui n'en ont pas
  (au moins 0,7 s). Effet neutre sur les annotations (F1 0,887).

- **Instant et position du contact au sol.** Sur `rune_kecmanovic_1`, le rebond 512 était annoncé
  dans le couloir alors que la balle était bonne. Le contact était placé à la frame 513,3, soit après
  que la balle avait franchi la ligne en remontant ; le vrai point bas était à la frame 510,3. La
  coupure « deux paraboles » glissait d'un cran sur les images dupliquées.
  - Correction : quand la trajectoire a un vrai point bas (descente puis remontée d'au moins 1 px de
    chaque côté), le contact est le sommet de la parabole passant par ce point et ses voisins, avec
    la position latérale interpolée au même instant (`lowest_point`).
  - Un simple palier (rebond dans l'axe côté proche) garde l'ancienne méthode. Sans cette exigence,
    le rebond 138 d'Alcaraz–Shelton, sur la ligne, passait à tort en faute.
  - Résultat : 512 bon, à 17 cm à l'intérieur ; 138 reste « limite ». Sur les annotations, l'erreur
    médiane passe de 3,9 à **3,0 px**, et 77 % des contacts tombent sous le centre annoté de la
    balle (physiquement attendu), contre 60 % avant. `PIPELINE_VERSION = 9`.

- **Service non reconnu et verdict trop tranché** (`rune_kecmanovic_1`, premier rebond).
  - Le service est maintenant reconnu aussi par la **signature du lancer** (`is_toss`) : la balle
    redescend du lancer et accélère déjà un peu avant l'impact. Deux garde-fous : au moins 0,7 s de
    vidéo avant la frappe, et un rebond à moins de 2,5 m du carré (sinon l'échange continuait, ce
    n'était pas un service).
  - **Zone « limite » adaptée à la précision locale** (`position_error`). Environ 1,5 px de
    précision sur le centre de la balle, convertis avec l'échelle locale du court. Côté éloigné
    d'une vidéo en 632 px de haut, 1 px vaut près de 10 cm en profondeur ; la zone orange vaut au
    moins cette incertitude, dans l'axe de la ligne la plus proche. L'interface affiche « ±x cm ».
  - Résultat : le service passe de faute (12 cm) à **limite (±15 cm)**. `PIPELINE_VERSION = 11`.

- **Frappe de Rune prise pour un rebond** (`rune_kecmanovic_2`, 6e rebond, frame 234 au lieu de 226).
  - La règle « demi-tour dans la zone d'un joueur » testait le point de contact recalculé, sorti de
    la zone de 3 px. Elle teste maintenant la **position observée** de la balle, avec une zone élargie
    à ±0,8 fois la largeur du joueur.
  - Pour ne pas emporter les rebonds hauts à côté d'un joueur (rebond 489–491 d'Alcaraz–Shelton,
    cassure de 155°), deux conditions sont ajoutées :
    - un seuil de demi-tour à **155°** (les frappes concernées font 167–171°) ;
    - une **accélération** : dans les 0,2 s qui suivent, la balle repart au moins 1,5 fois plus vite.
      Après un rebond, au contraire, elle ralentit.
  - Résultat : seul ce rebond change (234 → 226) sur les 5 vidéos. F1 sur les annotations : 0,889.

- **Federer–Nadal (Open d'Australie 2017).**
  - La vidéo avait encore été analysée avec un ancien code (« version None »), d'où le 2e rebond
    annoncé faute ; il est bon avec le code actuel.
  - Corrections, sans régression sur les annotations (F1 0,889 → **0,892**, 0,883 avec bruit) :
    - au moins **0,15 s** entre une frappe et le rebond suivant, et **0,35 s** si la frappe a eu
      lieu dans la raquette d'un joueur détecté, car la balle doit traverser le court (faux rebond
      de 7 s sur la balle qui accélère encore après la frappe). La seconde passe applique la même
      règle ;
    - un **départ accéléré vers le haut** depuis la zone d'un joueur est une frappe, même sans
      demi-tour (14,67 s, ce qui libère le rebond éloigné de 15,4 s) ;
    - un point bas suivi d'un **ralentissement franc** sur les 2 positions suivantes est un rebond,
      même si la frappe suit 5 images plus tard (rebond de 11,93 s). Un « rebond haut » dans la
      raquette n'est converti en frappe que si la balle accélère dans les 0,1 s.
  - `PIPELINE_VERSION = 13`.

- **Continuité du point** (idée de l'utilisateur ; les vidéos contiennent un seul point).
  - Une balle faute termine l'échange. Si un rebond est jugé faute (en simple) alors que l'échange
    continue, c'est-à-dire qu'un rebond bon suit avec au moins une frappe entre les deux, ce n'était
    pas un rebond. Il est imposé comme frappe et la détection est relancée, ce qui trouve le vrai
    rebond de l'intervalle (`continuity_violations`, boucle dans `analyze`, `extra_shots` dans
    `find_bounces`).
  - Exceptions : une faute au service suivie d'un nouveau service (seconde balle), et les rebonds
    après la fin du point (aucune frappe entre la faute et ces rebonds).
  - Testé sur des cas construits ; aucun effet sur les 8 vidéos, dont les fautes sont toutes en fin
    de point. Sur `alcaraz_djokovic_wim`, l'erreur signalée venait de l'ancien code du serveur : avec
    le code actuel, le premier rebond (frame 15) est bien dans le carré de service.
    `PIPELINE_VERSION = 14`.

### 13.6 Deux règles de cohérence du point (24 octobre)

Les vidéos ne contiennent qu'un seul point, ce qui permet de vérifier la logique de l'échange.

- **Règle 1 : une faute suivie d'un bon rebond n'était pas un rebond.** Un rebond « out »
  directement suivi d'un rebond bon est forcé en frappe, puis la détection est relancée sur
  l'intervalle (`fault_then_good`).
  - Exception : une faute au service suivie d'un second service.
- **Règle 2 : pas deux rebonds d'affilée du même côté** si un rebond de l'autre côté suit dans le
  point (`same_side_pairs`). Le moins fiable des deux est écarté, puis la détection est relancée.
  - Exception : une volée entre les deux, c'est-à-dire au moins deux frappes.
  - Seul un rebond situé dans la zone de frappe d'un joueur est retiré : c'est alors une frappe
    mal classée. Sinon, il manque plutôt un rebond entre les deux, et en supprimer un serait pire
    (cas réels : Alcaraz–Shelton 530/594, Federer 644).
- **Le terrain doit être crédible.** Un rebond n'est placé que si le terrain est détecté à
  ±3 images près.
  - La détection du terrain est refusée si la confiance médiane des 14 points est sous 0,85 ou si
    moins de 10 points sont sûrs.
  - Cela supprime les terrains « hallucinés » sur les gros plans. Sur Roland-Garros 2023, un faux
    rebond « bon » à l'image 430, pendant le gros plan final, faisait effacer par la règle 1 la
    vraie faute finale de l'image 410.
- `PIPELINE_VERSION = 16`. Toutes les vidéos ont été ré-analysées.
- **Complément (v17).** Quand il n'y a aucune frappe entre deux rebonds du même côté, la balle
  n'a pas pu changer de côté : un seul des deux est le vrai rebond. On garde le point le plus bas
  à l'image, même hors de la zone de frappe d'un joueur. Cela corrige Federer–Nadal 2 : le rebond
  9.97 s est gardé, celui de 10.10 s (une cassure sur la montée) est retiré. Les autres vidéos sont
  inchangées. `PIPELINE_VERSION = 17`.
- **Reste à corriger :** sur Roland-Garros 2023, le rebond côté éloigné vers l'image 293 est pris
  pour une frappe (291), car il tombe juste devant le joueur.

### 13.7 Vidéo best_point (25 octobre)

L'analyse affichée avait été faite par l'ancien serveur (« version None »), donc sans les règles du
point. Avec le code à jour, il restait quatre erreurs, toutes corrigées. `PIPELINE_VERSION = 20`.

- **Rebond du service (frame 40) manqué.** La seconde passe retenait à la place la frappe du
  relanceur (frame 47). Elle ignore maintenant les demi-tours (≥ 155°) dans la zone de raquette
  d'un joueur.
- **Rebond de 31,82 s (frame 955) pris pour la frappe.** La balle touche le sol au pied du joueur,
  qui la reprend pendant un trou du suivi (`hidden_hit`). Si la balle file au moins 1,6 fois plus
  vite juste après le trou, le point bas était le rebond et la frappe a eu lieu dans le trou.
  - Condition ajoutée : pas de rebond net dans la demi-seconde précédente. Sinon la règle se
    trompait sur Rune–Kecmanovic 2 (frame 541).
- **Rebond de 32,90 s (frame 986) remplacé par une fausse faute à 1010.** La cassure la plus forte
  de l'intervalle était le second rebond du coup gagnant, hors du terrain.
  - `find_bounces` garde une cassure nette plus précoce (kick ≥ 0,8, au moins 0,3 s avant) dans
    `first`.
  - Si le rebond retenu tombe faute et que `first` tombe dans le terrain, le pipeline prend
    `first`.
  - Cette décision est prise dans le pipeline et non dans le détecteur, car choisir
    systématiquement le premier rebond faisait baisser le F1 à 0,886.
- **Deux rebonds d'affilée du même côté (392 / 461).** Il manquait le rebond éloigné (frame 427),
  tombé juste devant le joueur du fond. Les fausses frappes au sommet du lob (416, 421, 432) le
  masquaient.
  - Nouvelle règle : un joueur ne frappe pas deux fois de suite sans rebond. Deux « frappes »
    autour du même joueur, à moins de 1 s d'intervalle et sans rebond entre elles : seule la
    dernière compte (`same_player_gap`, joueur identifié avec `serve_zone`, qui renvoie maintenant
    son côté).
  - Cela corrige aussi le rebond 293 de Roland-Garros 2023.
- **Paire 659 / 724 laissée telle quelle.** Les deux rebonds sont réels : le rebond côté proche
  (~697) a lieu alors que la balle est cachée derrière le joueur, et le suivi s'accroche à sa tête.
  Ce rebond est invisible.
- **Évaluation sur les annotations TrackNet :** F1 0,892 sans bruit, 0,883 avec bruit, inchangé.

### 13.8 Rebond caché derrière un joueur (25 octobre)

**Le cas.** Dans `best_point`, à 23,20 s, le rebond côté proche a lieu alors que la balle est
cachée par le joueur, et le suivi s'accroche à sa tête. Le rebond n'est pas visible, mais on
peut le déduire (`hidden_bounces`).

**Quand la reconstruction se déclenche.** Toutes ces conditions doivent être réunies :
- deux frappes de part et d'autre du filet, sans rebond détecté entre elles, à au moins 0,6 s
  d'intervalle ;
- le receveur n'est pas au filet (à plus de 4,5 m), sinon ce serait une volée.

**Comment le rebond est retrouvé.**
1. On garde les positions vues entre la frappe (+4 images) et le moment où la balle « entre »
   dans la boîte du receveur. Les images dupliquées sont écartées.
2. Les instants sont régularisés avec `sample_times`, parce que la vidéo vient d'une caméra à
   ~20 im/s convertie en 30. Sans ça, les écarts alternaient et la reconstruction échouait.
3. On reconstruit le vol en 3D (gravité + frottement de l'air), avec une hauteur de frappe
   supposée de 1 m (± 0,4 m).
4. On cherche l'instant où la balle touche le sol.

**Conditions d'acceptation :**
- erreur moyenne ≤ 4 px (pour du 720p) ;
- contact au plus 2 images avant la dernière position vue ;
- contact de l'autre côté du filet, à moins de 1 m des lignes.

**Résultat.**
- Le rebond est marqué `hidden`. Son incertitude est d'au moins ±40 cm en largeur et ±60 cm en
  profondeur.
- L'interface affiche « estimé, balle cachée ».
- `best_point` : rebond reconstitué à 23,17 s, en (2,79 ; 18,05) m, bon. Aucun effet sur les
  12 autres vidéos.
- `PIPELINE_VERSION = 22`. Fichiers de l'interface en `?v=12`.

---

## 14. État final du système

```
Vidéo ──► 1. Terrain : YOLO11s-pose (toutes les 15 frames) ──► homographie RANSAC + dichotomie aux changements de plan
      ──► 2. Balle : YOLO11s (1280 px) ──► candidats (cache) ──► chaînage avec prédiction de la vitesse
      ──► 3. Rebonds : nettoyage, segments, « coup vers le haut », frappes (3 critères), point de contact
      ──► 4. Jugement : position en mètres, contact au sol, filtre 3 m, vitesse des frappes
      ──► analysis.json ──► interface web (calques en direct, mini-carte, statistiques) / export MP4
```

**Les performances :**

- détection de la balle : environ 0,6 à 1 s par frame sur le CPU de l'utilisateur, ce qui en fait
  l'étape longue (85 % du temps) ;
- terrain : quelques dizaines de secondes par vidéo ;
- rebonds, jugement et vitesses : moins d'une seconde ;
- export : environ 1 minute par vidéo.

**Les mesures :**

- rebonds : **F1 0,888** sur les 523 rebonds annotés (0,868 avec bruit simulé) ;
- rebonds dans l'axe : 91 % trouvés ; rebonds près du filet : 90 % trouvés.

---

## 15. Tous les réglages

### Rebonds (`detect_bounces.py`, seuils en pixels donnés pour du 720p et mis à l'échelle)

| Paramètre | Valeur | Rôle |
|---|---|---|
| `n` | 5 | échantillons par segment |
| `kick` | 0,4 | coup vers le haut minimal |
| `max-dev` / `max-rel-dev` | 8 px / 8 % | rectitude |
| `min-len` | 15 px | longueur minimale |
| `shot-prom` | 0,16 × H | amplitude d'un demi-tour = frappe |
| `shot-ratio` | 0,8 | sous ce rapport de vitesses, un pic côté joueur proche est un rebond |
| `shot-kick` | 0,8 | coup vers le bas = frappe |
| `hit-speed` / `hit-ratio` | 0,8 H/s / 1,2 | frappe rapide vers le haut |
| `min-gap` | 4 frames | écart minimal frappe ↔ rebond |
| `max-frame-gap` | 6 | trou maximal dans un segment |
| `dedup` / `outlier` | 2 px / 120 px | doublons, aberrations |
| `refine` | 1 | point de contact par paraboles |

### Suivi (`link_track`)

| Paramètre | Valeur |
|---|---|
| `conf` de YOLO | 0,10 |
| Confiance minimale de suivi / de réacquisition | 0,15 / 0,3 |
| Fenêtre de recherche | 3 % de la largeur + 2 fois la vitesse |
| Frames perdues avant réacquisition | 5 |

### Terrain et pipeline

| Paramètre | Valeur |
|---|---|
| Taille d'image du modèle de terrain | 960 |
| Confiance minimale d'un point | 0,5 |
| Fréquence de détection | toutes les 15 frames |
| Lissage | 0,5 |
| Rayon de balle | 0,033 m |
| Filtre « hors du terrain » | 3 m |
| Hauteur de frappe | 1,0 m |
| Service | 0,4 m à l'intérieur de la ligne de fond |
| `k_drag` | 0,019 m⁻¹ |
| Zone limite | 5 cm |

---

## 16. Limites connues et pistes

- **Le suivi de la balle** est le facteur limitant : balle perdue au rebond (coup gagnant de la
  frame 780 d'Alcaraz–Shelton), raquette suivie à la place de la balle (frame 160 de Wimbledon).
  - Pistes : TrackNet, plus fiable car il voit plusieurs images d'affilée, mais lent sur CPU ;
    un filtre de Kalman ; un réentraînement sur des vidéos plus variées.
- **Le jugement in / out** n'a pas la précision du Hawk-Eye. À 30 fps, la balle parcourt jusqu'à
  environ 1 m entre deux images, d'où une incertitude de 10 à 30 cm près des lignes.
- **Le service** n'est pas jugé sur le carré de service.
- **Les volées** lentes ne sont pas toujours marquées comme frappes.
- **La vitesse** est une estimation à ±10 % (hauteur de frappe supposée, effet non modélisé).
  - Piste : la calibrer avec la vitesse officielle affichée à l'écran pendant les retransmissions.
- **Les métriques finales du modèle de terrain** n'ont pas été relevées.
- **La reconnaissance des joueurs** n'existe pas encore. Elle aiderait à identifier les frappes (une
  frappe a toujours lieu près d'un joueur).

---

## 17. Inventaire des fichiers

| Catégorie | Éléments |
|---|---|
| Système utilisé | `app\`, `scripts\pipeline.py`, `predict_video_yolo.py`, `predict_court.py`, `detect_bounces.py`, `render_overlay.py`, `runs\ball_rv\`, `runs\court_pose\`, `runs\analysis\`, `videos\`, `.venv\` |
| Réentraîner / mesurer | `bounce_eval\` + `eval_bounces.py`, `colab\` (2 notebooks), `prepare_racketvision_yolo.py`, `prepare_court_pose.py`, `train.py`, `data\` (7,3 Go), `court_pose\` (1,5 Go), `yolo_rv\` (3,9 Go), `RacketVision\` (1,9 Go) |
| Obsolète (supprimé lors du ménage du 22 octobre) | `train\`, `valid\`, `test\`, `yolo\`, `README.*.txt`, `yolo11n.pt`, `runs\ball_v1*`, `runs\predict_images`, `runs\tracknet`, `runs\yolo_video`, `runs\bounces`, `runs\court`, `prepare_yolo.py`, `predict_image.py`, `predict_video_tracknet.py`, `court_pose.zip`, les vidéos en double à la racine |

**Les versions utilisées :** Python 3.11, Ultralytics 8.4.182, PyTorch 2.14 (CPU), OpenCV 5.0,
NumPy 2.4, FastAPI 0.159, ffmpeg 8.1. Sur Colab : Ultralytics 8.4.x avec un GPU Tesla T4.
