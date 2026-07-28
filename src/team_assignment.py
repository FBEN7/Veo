"""Assignation d'équipe par clustering couleur des maillots.

Pour chaque track, échantillonne quelques crops du torse, calcule la couleur
moyenne (HSV), puis KMeans k=3 sur l'ensemble des tracks :
2 équipes + 1 cluster 'autres' (arbitre/gardiens), identifié comme le plus petit.

Limite connue : gardiens mal classés, tracks courts bruités. Suffisant pour un MVP.
"""
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def _torso_color(frame: np.ndarray, px: float, py: float, h: float) -> np.ndarray | None:
    # bbox approx reconstruite depuis bottom-center + hauteur
    w = h * 0.4
    x1, x2 = int(px - w / 2), int(px + w / 2)
    y2 = int(py)
    y1 = int(py - h)
    # torse = moitié haute, centre
    ty1 = y1 + int(h * 0.2)
    ty2 = y1 + int(h * 0.55)
    crop = frame[max(ty1, 0):max(ty2, 1), max(x1, 0):max(x2, 1)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0)


def assign_teams(video_path: str, tracks: pd.DataFrame,
                 samples_per_track: int = 5) -> pd.DataFrame:
    players = tracks[tracks.cls == "player"]
    cap = cv2.VideoCapture(video_path)

    feats, tids = [], []
    for tid, g in players.groupby("track_id"):
        if len(g) < 10:  # ignore les tracks trop courts
            continue
        sample = g.sample(min(samples_per_track, len(g)), random_state=0)
        colors = []
        for _, r in sample.iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame)
            ok, frame = cap.read()
            if not ok:
                continue
            c = _torso_color(frame, r.px, r.py, r.crop_h)
            if c is not None:
                colors.append(c)
        if colors:
            feats.append(np.mean(colors, axis=0))
            tids.append(tid)
    cap.release()

    if len(feats) < 3:
        raise RuntimeError("Pas assez de tracks pour le clustering.")

    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(np.array(feats))
    labels = km.labels_
    # le cluster le plus petit = 'other' (arbitre, gardiens)
    counts = np.bincount(labels, minlength=3)
    other = int(np.argmin(counts))
    team_ids = [c for c in range(3) if c != other]
    mapping = {team_ids[0]: "team_A", team_ids[1]: "team_B", other: "other"}

    team_map = {tid: mapping[l] for tid, l in zip(tids, labels)}
    tracks = tracks.copy()
    tracks["team"] = tracks.track_id.map(team_map)
    tracks.loc[tracks.cls == "ball", "team"] = "ball"
    return tracks
