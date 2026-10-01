# match-report — Rapport post-match automatique depuis une vidéo

Pipeline MVP : vidéo de match (caméra fixe bord de terrain) → détection
joueurs/ballon (YOLOv8) → tracking (ByteTrack) → équipes (clustering couleur)
→ projection terrain (homographie) → rapport HTML (heatmaps, distances,
sprints, field tilt, proxy de possession).

## Setup (VS Code)

```bash
git clone <ton-repo>   # ou décompresse ce dossier
cd match-report
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

GPU fortement recommandé. Sans GPU, utilise `--model yolov8s.pt --stride 5`
(compte quand même ~1-2h pour un match complet sur CPU).

## Usage

```bash
# 1. Mets ta vidéo dans data/
# 2. Calibration (une fois par position de caméra) — clique 4 coins du terrain
python -m src.calibrate data/match.mp4

# 3. Pipeline complet
python main.py data/match.mp4 --label "Mon Club vs Adversaire — P1"

# Rapport : output/report.html
# Pour re-générer le rapport sans refaire la détection :
python main.py data/match.mp4 --skip-detect
```

## Synchronisation GitHub

```bash
cd match-report
git init
git add .
git commit -m "MVP: video -> post-match report pipeline"
# Crée un repo vide sur github.com (sans README), puis :
git remote add origin git@github.com:<ton-user>/match-report.git
git branch -M main
git push -u origin main
```

Workflow quotidien : `git add -A && git commit -m "..." && git push`.
Dans VS Code, l'onglet Source Control (Ctrl+Shift+G) fait la même chose en UI.

## Limitations à connaître (lis ça avant de vendre quoi que ce soit)

1. **Caméra fixe obligatoire.** L'homographie est calculée une fois ; si la
   caméra bouge/zoome, toutes les coordonnées terrain sont fausses.
2. **Les tracks se cassent.** Occlusions, sorties de champ → un joueur =
   plusieurs track_ids. Les distances sont des minimums par segment, pas des
   totaux par joueur. La ré-identification (ReID) est l'étape suivante, et
   elle est difficile.
3. **Le ballon est peu fiable** sur vidéo amateur (petit, flou). Le rapport
   affiche le taux de détection et masque la possession si < 30%.
4. **Gardiens et arbitre** polluent le clustering couleur (cluster "other",
   heuristique fragile).
5. **Pas d'événements** (passes, tirs, duels) : ça, c'est l'étape d'après et
   c'est un ordre de grandeur plus dur. Ne le promets pas.

## Structure

```
main.py                  # orchestration CLI
src/calibrate.py         # homographie (4 clics)
src/detect_track.py      # YOLOv8 + ByteTrack -> tracks.parquet
src/team_assignment.py   # KMeans couleurs maillots
src/stats.py             # coords terrain, distances, sprints, possession
src/report.py            # heatmaps mplsoccer + HTML
templates/report.html.j2 # template du rapport
```


## Tests

```bash
python run_tests.py --controls   # ~7 s, aucune dépendance lourde
python run_tests.py              # ajoute main.py de bout en bout (~30 s)
python run_tests.py --list       # ce que chaque test couvre
```

Les *controls* sont les contrôles synthétiques de chaque détecteur : un
ballon promené par-dessus chaque ligne, une remise en jeu depuis chaque
repère, une possession juste, inversée, ou tirée à pile ou face. Ils ne
demandent ni vidéo ni modèle. Le second niveau lance le vrai `main.py` sur
un match synthétique — c'est le seul test qui attrape une régression du
rapport.

Aucun de ces tests ne mesure la précision : les chiffres de
`EVENT_ACCURACY.md` viennent de vidéos sous NDA absentes du dépôt. Les tests
vérifient que le code fait ce qu'il annonce sur des cas à réponse connue,
pas qu'il marche sur du football.

## Ce qui est mesuré, et ce qui ne l'est pas

Le pipeline d'origine (ci-dessus) produit un rapport. Ce qui suit a été
ajouté ensuite et, surtout, **mesuré** : chaque chiffre vient d'un contrôle
reproductible, et les limites sont écrites au même endroit que les
résultats. Les documents détaillés sont en anglais ; `EVENT_ACCURACY.md`
commence par un résumé de l'état actuel.

| | état | où c'est mesuré |
|---|---|---|
| Position du ballon sur le terrain | 8 à 45 % des images selon le clip ; erreur médiane **0,5 m** | `EVENT_ACCURACY.md` |
| Position de la caméra | 5 clips sur 6 localisés, le 6e par la hauteur du même match | `EVENT_ACCURACY.md` |
| Distance, angle et xG des tirs | les 6 tirs annotés, xG total 0,47 | `EVENT_ACCURACY.md` |
| Détection des cages | 6 sur 6 aux moments de tir | `EVENT_ACCURACY.md` |
| Ballon sorti | précision **2 sur 2**, rappel 2 sur 4 | `EVENT_ACCURACY.md` |
| Tirs et buts détectés | **0 sur 10 et 0 sur 2** | `EVENT_ACCURACY.md` |
| Équipes (maillots) | 0,99 | `TEAM_ASSIGNMENT.md` |
| Équipes (écarter les non-joueurs) | 0,80 | `TEAM_ASSIGNMENT.md` |
| Possession (part) | erreur 0,06 | `EVENT_ACCURACY.md` |

### La géométrie : d'où viennent les mètres

Tout ce qui est en mètres dépend de savoir où est la caméra et où elle
regarde. Trois étapes, chacune vérifiée contre quelque chose qu'elle ne voit
pas :

1. **Les coins des cages**, cliqués une fois par clip, donnent une droite
   sur laquelle se trouve la caméra — pas un point : focale et distance se
   compensent.
2. **Le rond central et la ligne médiane** donnent une seconde droite. La
   caméra est à leur croisement (`src/camera_position.py`). Vérifié sur des
   lignes peintes que rien n'ajuste : le côté des six mètres à 0,6 et 0,0 m,
   la ligne de surface droite à 16,8 m pour 16,5.
3. **Le détecteur de cages**, image par image, donne le reste de la pose.
   Les boîtes annotées vont du haut du poteau gauche au pied du poteau
   droit, pas autour de la cage ; les lire ainsi a fait passer l'erreur de
   18 m à 0,5 m.

### Ce qui ne marche pas encore

**Les tirs.** Aucun tir réel détecté (0 sur 10). La géométrie n'est plus en
cause, et le ballon n'est rare que pour 2 des 12 moments annotés : sur les
autres, il est suivi sur 31 à 47 des 50 images. Le détecteur n'accepte qu'un
ballon dirigé vers le cadre, donc un tir à côté ou au-dessus ne compte
jamais ; l'analyse tir par tir reste à faire. Le seul « tir » trouvé un temps
était le centre qui le précédait.

**Le ballon en l'air.** Placer le ballon suppose qu'il est au sol ; en l'air,
il est projeté bien trop loin. C'est la limite qui revient : elle a coûté un
tir et une sortie réelle. Une estimation de la hauteur par la trajectoire
(`src/ball_height.py`) marche sur des vols simulés d'une seconde mais pas
encore sur ces vols-ci, trop courts.

**Le ballon sorti** est passé de 3 sur 15 à 2 sur 2 en précision, en
exigeant que le ballon soit suivi *en train de franchir* la ligne (les
fausses alertes étaient des panneaux publicitaires pris pour le ballon) et
en interpolant la caméra entre les images où sa pose est calculée.
Quatre sorties annotées ne suffisent pas à valider un détecteur.

### Reproduire

```bash
# Coins des cages (une page à cliquer, envoyée en fichier, jamais publiée)
python make_corner_labeller.py --labels goal_labels.json

# Scorer un clip contre des annotations écrites à la main
python score_hand_labels.py labels.txt --out output_clip \
    --goal-weights best.pt --corners goal_corners.json \
    --midfield --camera-store cameras.json
```

Les poids du détecteur de cages, les annotations et les vidéos viennent de
SoccerNet, sous accord non commercial : ils ne sont pas dans ce dépôt, et
rien qui en dérive ne doit y entrer. Les scripts les cherchent dans
`VEO_DATA_DIR` (par défaut `./data`) et gardent leurs résultats
intermédiaires dans `VEO_CACHE_DIR` (par défaut `./.cache`), deux dossiers
ignorés par git.

Le reste du dépôt suit la même règle : `VEO_FOOTAGE.md` et
`EVENT_ACCURACY.md` contiennent aussi les corrections d'erreurs commises en
cours de route, y compris celles qui annulent une conclusion publiée la
veille.
