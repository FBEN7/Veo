"""Team-assignment variants over cached crops (tuning harness, not shipped)."""
import json, pickle, cv2, numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src import team_assignment_v2 as ta

def torso_pixels(crop, fx, fy, h):
    """HSV pixels of the torso box the pipeline uses, grass removed."""
    w = h * 0.4
    x1, x2 = int(fx - w * ta.TORSO_WIDTH / 2), int(fx + w * ta.TORSO_WIDTH / 2)
    y1, y2 = int(fy - h + h * ta.TORSO_TOP), int(fy - h + h * ta.TORSO_BOTTOM)
    H, W = crop.shape[:2]
    x1, x2, y1, y2 = max(0, x1), min(W, x2), max(0, y1), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None, None
    bgr = crop[y1:y2, x1:x2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    ng = ~((hsv[:, 0] >= ta.GRASS_HUE_LO) & (hsv[:, 0] <= ta.GRASS_HUE_HI) & (hsv[:, 1] >= ta.GRASS_MIN_SAT))
    if ng.sum() >= 4:
        hsv, lab = hsv[ng], lab[ng]
    return hsv, lab

def crop_feature(crop, fx, fy, h, mode):
    hsv, lab = torso_pixels(crop, fx, fy, h)
    if hsv is None or len(hsv) < 4:
        return None
    if mode == 'current':
        return ta._torso_feature(crop, fx, fy, h)
    ang = hsv[:, 0].astype(float) * (2 * np.pi / 180)
    s, v = hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    if mode == 'cone_all':
        return np.array([np.median(s * np.cos(ang)), np.median(s * np.sin(ang)), np.median(v)])
    if mode == 'cone_top':
        k = max(2, int(len(s) * ta.SATURATION_KEEP)); i = np.argsort(s)[-k:]
        return np.array([np.median(s[i] * np.cos(ang[i])), np.median(s[i] * np.sin(ang[i])), np.median(v[i])])
    if mode == 'lab':
        return np.median(lab, axis=0) / 255.0
    raise ValueError(mode)

def track_features(crops, mode):
    out = {}
    for tid, items in crops.items():
        f = [crop_feature(c, fx, fy, h, mode) for _, c, fx, fy, h in items]
        f = [x for x in f if x is not None]
        if f:
            out[tid] = np.median(f, axis=0)
    return out

def track_geometry(w):
    """Per player track: rows, median depth inside the outline / height, movement / height."""
    t = pd.read_parquet(f'/home/user/Veo/output_{w}/tracks_teams.parquet')
    p = t[t.cls == 'player'].copy()
    cut = json.loads(open(f'/home/user/Veo/output_{w}/pitch.json').read())
    poly = np.asarray(cut['pitch'], np.float32).reshape(-1, 1, 2)
    p['depth'] = [cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in zip(p.px, p.py)]
    g = p.groupby('track_id').agg(rows=('frame', 'size'), h=('crop_h', 'median'), depth=('depth', 'median'),
                                  x0=('px', 'min'), x1=('px', 'max'), y0=('py', 'min'), y1=('py', 'max'))
    g['depth_h'] = g.depth / g.h
    g['move_h'] = np.hypot(g.x1 - g.x0, g.y1 - g.y0) / g.h
    return g

def assign(feats, geom, scale=True, k=3, select='count', member=None, residual_cut=ta.RESIDUAL_CUT, seed=0):
    """team per track. member(geom_row) -> bool decides who takes part in the fit;
    excluded tracks are 'other'."""
    tids = [t for t in feats if t in geom.index and geom.rows[t] >= ta.MIN_TRACK_FRAMES]
    fit = [t for t in tids if member is None or member(geom.loc[t])]
    X = np.array([feats[t] for t in fit])
    sc = StandardScaler().fit(X) if scale else None
    tr = (lambda a: sc.transform(a)) if scale else (lambda a: np.asarray(a))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(tr(X))
    lab = km.labels_
    weight = np.array([1.0 if select == 'count' else geom.rows[t] for t in fit])
    size = np.bincount(lab, weights=weight, minlength=k)
    big = np.argsort(size)[::-1][:2]
    mapping = {int(big[0]): 'team_A', int(big[1]): 'team_B'}
    team = {t: mapping.get(int(l), 'other') for t, l in zip(fit, lab)}
    cent = km.cluster_centers_[big]
    Z = tr(X)
    dist = np.linalg.norm(Z[:, None] - cent[None], axis=2).min(1)
    inside = np.array([team[t] != 'other' for t in fit])
    s = np.median(dist[inside]) if inside.any() else np.median(dist)
    for t, d in zip(fit, dist / max(s, 1e-9)):
        if d > residual_cut and team[t] != 'other':
            team[t] = 'other'
    for t in tids:
        team.setdefault(t, 'other')
    return team
