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


## Ce qui est mesuré, et ce qui ne l'est pas

Le pipeline d'origine (ci-dessus) produit un rapport. Ce qui suit a été
ajouté ensuite et, surtout, **mesuré** : chaque chiffre vient d'un contrôle
reproductible, et les limites sont écrites au même endroit que les
résultats. Les documents détaillés sont en anglais.

| | état | où c'est mesuré |
|---|---|---|
| Tirs et xG | fonctionne sur diffusion TV | `EVENT_ACCURACY.md` |
| Occasions et passes décisives | fonctionne | `EVENT_ACCURACY.md` |
| Équipes (maillots) | 0.99 | `TEAM_ASSIGNMENT.md` |
| Équipes (écarter les non-joueurs) | 0.80 | `TEAM_ASSIGNMENT.md` |
| Ballon sorti, buts, coups de pied arrêtés | **ne fonctionne pas** | `EVENT_ACCURACY.md` |

Deux précisions qui comptent plus que le tableau.

**Les tirs.** Un tir simulé, placé dans la vraie trajectoire du ballon, est
retrouvé sur chaque clip à 0.3 m près et son xG à 0.007 près. Sur six
minutes de football vérifié sans tir, rien ne se déclenche. Ce qui n'a
jamais été mesuré, faute d'images annotées en contenant, c'est le rappel :
rien ne prouve que ce détecteur trouve un tir qu'on ne lui a pas donné.

**Le ballon sorti.** Correct sur entrée synthétique, 1 sur 3 sur du vrai
football, avec 4 à 6 fausses alertes par 90 secondes. La cause est connue et
n'est pas la géométrie : les repères viennent du rond central, donc du milieu
du terrain, et le ballon sort sur les côtés. Sur les deux sorties manquées,
le ballon est détecté 74 et 51 fois et positionné zéro fois. Les coups de
pied arrêtés en héritent, puisqu'une sortie manquée est une remise en jeu
manquée.

Le reste du dépôt suit la même règle : `VEO_FOOTAGE.md` contient aussi les
corrections d'erreurs commises en cours de route, y compris celles qui
annulent une conclusion publiée la veille.
