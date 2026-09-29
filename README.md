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
résultats. Les documents détaillés sont en anglais.

| | état | où c'est mesuré |
|---|---|---|
| Tirs et xG | **ne trouve aucun tir réel** (0 sur 10) | `EVENT_ACCURACY.md` |
| Buts | **aucun but réel détecté** (0 sur 2) | `EVENT_ACCURACY.md` |
| Occasions et passes décisives | fonctionne | `EVENT_ACCURACY.md` |
| Équipes (maillots) | 0.99 | `TEAM_ASSIGNMENT.md` |
| Équipes (écarter les non-joueurs) | 0.80 | `TEAM_ASSIGNMENT.md` |
| Ballon sorti, buts, coups de pied arrêtés | **ne fonctionne pas** | `EVENT_ACCURACY.md` |

Deux précisions qui comptent plus que le tableau.

**Les tirs.** Le rappel a fini par être mesuré, sur six fenêtres découpées
autour de tirs annotés, dans deux matchs : **0 tir trouvé sur 10, et 0 but
sur 2**. Aucune fausse alerte non
plus, mais un détecteur qui ne se déclenche jamais obtient ce score-là
aussi.

La cause n'est pas le détecteur de tirs. Sur deux des trois fenêtres la
couverture est bonne (61 % du ballon positionné) et le ballon est quand
même placé à quarante mètres de sa vraie position : le repère vient du rond
central, la caméra est zoomée dans la surface quand il y a un tir, et le
rond central n'est plus dans l'image. Détail dans `EVENT_ACCURACY.md`.

Ce qui reste vrai : un tir simulé, placé dans la vraie trajectoire du
ballon, est retrouvé à 0.3 m près et son xG à 0.007 près. Cela mesure la
géométrie et le modèle xG, pas la détection — le tir simulé était inséré à
travers un repère valide, ce qui garantissait silencieusement une bonne
carte.

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
