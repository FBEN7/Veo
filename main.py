"""Pipeline complet : vidéo -> rapport post-match PDF.

Étapes préalables (une fois par vidéo) :
  1. python -m src.calibrate data/match.mp4   # clique 4 points -> homography.npy
  2. python main.py data/match.mp4 --label "RFC Virton - P1, 12/07"

Options utiles :
  --stride 5          # traite 1 frame sur 5 (plus rapide, moins précis)
  --model yolov8s.pt  # plus léger si pas de GPU
  --skip-detect       # réutilise output/tracks.parquet existant

Téléchargement d'une vidéo de test :
  python scripts/download_video.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import detect_track, stats, report
from src.team_assignment import assign_teams
from src import events as ev_module
from src import quality
from src import player_rating
from src.database import MatchDatabase
from src.auto_calibrate import auto_calibrate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--label", default="Match")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--model", default="yolov8m.pt")
    p.add_argument("--conf-player", type=float, default=0.25,
                   help="seuil confiance YOLO pour joueurs")
    p.add_argument("--conf-ball", type=float, default=0.08,
                   help="seuil confiance YOLO pour ballon")
    p.add_argument("--imgsz", type=int, default=640,
                   help="input image size for YOLO (default: 640)")
    p.add_argument("--skip-detect", action="store_true",
                   help="réutilise output/tracks.parquet existant")
    p.add_argument("--db", default="output/match.db",
                   help="chemin vers la base SQLite (défaut: output/match.db)")
    p.add_argument("--out", default="output/report.pdf",
                   help="chemin du PDF de sortie (défaut: output/report.pdf)")
    p.add_argument("--max-seconds", type=int, default=0,
                   help="traite seulement les N premières secondes (0 = tout)")
    p.add_argument("--auto-calibrate", action="store_true",
                   help="détection automatique des coins du terrain (pas de clic manuel)")
    p.add_argument("--homography", default="",
                   help="chemin vers un fichier .npy d'homographie à utiliser")
    args = p.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Homographie (auto ou manuelle)                                    #
    # ------------------------------------------------------------------ #
    if args.homography:
        h_path = Path(args.homography)
        if not h_path.exists():
            raise FileNotFoundError(f"Homography file not found: {h_path}")
    else:
        h_path = Path("output/homography.npy")
        if not h_path.exists() or args.auto_calibrate:
            print("Calibration automatique du terrain...")
            auto_calibrate(args.video, str(h_path), debug=True)

    print(f"Homographie utilisée: {h_path}")
    H = np.load(h_path)

    # ------------------------------------------------------------------ #
    # 2. Détection + tracking                                              #
    # ------------------------------------------------------------------ #
    if args.skip_detect and Path("output/tracks.parquet").exists():
        tracks = pd.read_parquet("output/tracks.parquet")
    else:
        tracks = detect_track.run(args.video, stride=args.stride,
                                  model_name=args.model,
                                  conf_player=args.conf_player,
                                  conf_ball=args.conf_ball,
                                  max_seconds=args.max_seconds,
                                  imgsz=args.imgsz)

    # ------------------------------------------------------------------ #
    # 3. Assignation des équipes                                           #
    # ------------------------------------------------------------------ #
    print("Assignation des équipes...")
    tracks = assign_teams(args.video, tracks)
    print("Ré-identification des joueurs...")
    tracks = detect_track.reidentify_tracks(args.video, tracks)
    tracks.to_parquet("output/tracks.parquet")
    tracks.to_parquet("output/tracks_teams.parquet")

    # ------------------------------------------------------------------ #
    # 4. Projection terrain + statistiques physiques                       #
    # ------------------------------------------------------------------ #
    print("Projection terrain + stats physiques...")
    tracks = stats.to_pitch_coords(tracks, H)
    phys = stats.physical_stats(tracks)
    tilt = stats.field_tilt(tracks)
    poss = stats.possession_proxy(tracks)
    ball_rate = stats.ball_detection_rate(tracks)

    # ------------------------------------------------------------------ #
    # 5. Détection des événements (passes, tirs, buts, hors-jeu)          #
    # ------------------------------------------------------------------ #
    print("Détection des événements...")
    events = ev_module.detect_events(tracks)
    epp = ev_module.events_per_player(events)
    tet = ev_module.team_event_totals(events)

    # ------------------------------------------------------------------ #
    # 6. Évaluation des joueurs                                            #
    # ------------------------------------------------------------------ #
    print("Calcul des ratings joueurs...")
    rated_phys = player_rating.compute(phys, epp, poss)
    bw = player_rating.best_worst(rated_phys)

    # ------------------------------------------------------------------ #
    # 7. Stockage en base de données                                       #
    # ------------------------------------------------------------------ #
    print("Stockage en base SQLite...")
    db = MatchDatabase(args.db)
    db.init()

    import cv2
    cap = cv2.VideoCapture(args.video)
    duration_s = (
        cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
    )
    cap.release()

    quality_report = quality.compute_quality(
        tracks_pitch=tracks,
        phys=phys,
        events=events,
        ball_rate=ball_rate,
        duration_s=duration_s,
    )

    match_id = db.insert_match(
        label=args.label,
        video_path=args.video,
        duration_s=duration_s,
    )
    db.insert_events(match_id, events)
    db.insert_player_stats(match_id, rated_phys.to_dict("records"))

    # Build team-level rows for DB
    team_stat_rows = []
    for team_key in ["team_A", "team_B"]:
        tp = rated_phys[rated_phys.team == team_key]
        team_stat_rows.append({
            "team": team_key,
            "possession_pct": poss.get(team_key, 0.0) or 0.0,
            "total_distance_km": tp["distance_m"].sum() / 1000 if not tp.empty else 0,
            "total_sprints": int(tp["n_sprints"].sum()) if not tp.empty else 0,
            "n_passes": tet.get(team_key, {}).get("n_passes", 0),
            "n_shots":  tet.get(team_key, {}).get("n_shots", 0),
            "n_goals":  tet.get(team_key, {}).get("n_goals", 0),
            "field_tilt_x": tilt.get(team_key, 0.0),
        })
    db.insert_team_stats(match_id, team_stat_rows)

    summary = db.event_summary(match_id)
    print(f"  Résumé événements DB : {summary}")
    print(f"  Qualité données : {quality_report}")

    # ------------------------------------------------------------------ #
    # 8. Rapport PDF                                                       #
    # ------------------------------------------------------------------ #
    print("Génération du rapport PDF...")
    report.build_pdf(
        tracks=tracks,
        phys=rated_phys,
        events=events,
        best_worst=bw,
        tilt=tilt,
        possession=poss,
        ball_rate=ball_rate,
        quality_report=quality_report,
        match_label=args.label,
        team_event_totals=tet,
        out_path=args.out,
    )

    print(f"\nTerminé !")
    print(f"  PDF     : {Path(args.out).resolve()}")
    print(f"  Base DB : {Path(args.db).resolve()}")


if __name__ == "__main__":
    main()
