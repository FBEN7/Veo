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
