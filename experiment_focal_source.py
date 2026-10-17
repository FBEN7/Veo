"""Which focal length is right? Ask the measurements, not the conditioning.

`ground_plane.py` gets its focal length from the camera's pan, and that
estimate has been the prime suspect in this project for a while: it spreads
0.16 to 0.27 across baselines, and the plane built on it makes pass F1,
speed reliability and distance reliability all worse.

Two orthogonal vanishing points give an independent one, straight from
(v_a - c) . (v_b - c) = -f^2. It is steadier -- interquartile spread 0.02 to
0.14 -- and larger by 1.6 to 1.8 times: 2331, 2983, 2648 and 3295 px against
1663, 1614, 1790 and 1799. On conditioning alone it is the better number, and
swapping it in looked like the obvious repair.

It is not. Run over the four labelled windows, every instrument that can be
checked gets slightly worse and the one plausibility anchor moves out of
range:

    clip      pass F1          carry F1         km/90        speed r
    w1        0.886 -> 0.914   0.687 -> 0.687   10.5 -> 11.6  0.45 -> 0.46
    w2        0.698 -> 0.690   0.731 -> 0.654   12.5 -> 15.1  0.57 -> 0.58
    w3        0.767 -> 0.754   0.702 -> 0.678   12.2 -> 14.3  0.47 -> 0.52
    reading   0.679 -> 0.630   0.625 -> 0.667   10.5 -> 13.5  0.51 -> 0.31

Mean pass F1 falls 0.758 to 0.747, carry 0.686 to 0.672, split-half speed
0.50 to 0.47, distance-rate reliability 0.20 to 0.13, and km per 90 rises
from 11.4 to 13.6 where football produces 10 to 12. Five measures, all
pointing the same way, at the estimate that is worse conditioned.

So the pan-derived focal stays, and the hypothesis that it was the plane's
problem is not supported. Either it is right despite being noisy, or the
vanishing-point constraint is biased here -- it assumes a centred principal
point and square pixels, and broadcast footage is cropped and rescaled before
anyone sees it. What can be said from the evidence available is only that the
downstream measurements prefer the pan.

    python experiment_focal_source.py
"""
import json, sys, dataclasses
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/user/Veo")

from analyse_pass_outcome import WINDOWS, UPLOADS
from score_soccernet import (run_pipeline, load_window, match_one_to_one,
                             OURS_TO_GROUP)
from src import ground_plane, stats
from fit_pitch_anchor import horizons_from_lines

TOL = 2.0
rng = np.random.default_rng(0)

def f1(events, lf, off, dur, group):
    labels = load_window(str(UPLOADS / lf), off, dur)
    truths = sorted(l["t"] for l in labels if l["group"] == group)
    preds = sorted(e["timestamp_s"] for e in events
                   if OURS_TO_GROUP.get(e["event_type"]) == group)
    pairs, _, _ = match_one_to_one(preds, truths, TOL)
    p = len(pairs)/len(preds) if preds else 0.0
    r = len(pairs)/len(truths) if truths else 0.0
    return 2*p*r/(p+r) if (p+r) else 0.0

def dist_rel(tracks, min_s=6.0):
    pl = tracks[(tracks.cls=="player") & (tracks.team.isin(["team_A","team_B"]))]
    a, b = [], []
    for tid, g in pl.groupby("track_id"):
        g = g.sort_values("frame")
        if g.time_s.max()-g.time_s.min() < min_s or len(g) < 40: continue
        mid = len(g)//2; rates=[]
        for half in (g.iloc[:mid], g.iloc[mid:]):
            h = stats._smooth(half)
            dx,dy,dt = h.x.diff(), h.y.diff(), h.time_s.diff()
            sp = np.sqrt(dx**2+dy**2)/dt*3.6
            ok = sp < stats.MAX_SPEED_KMH
            secs = float(h.time_s.max()-h.time_s.min())
            if secs<=0 or not ok.any(): rates=[]; break
            rates.append(float(np.sqrt(dx[ok]**2+dy[ok]**2).sum())/secs)
        if len(rates)==2: a.append(rates[0]); b.append(rates[1])
    return float(np.corrcoef(a,b)[0,1]) if len(a)>=5 else float("nan")

# Focal from vanishing points, per clip.
vp_focal = {}
for name, d, lf, off in WINDOWS:
    _, hor = horizons_from_lines(Path(d), 30, rng)
    fs = [h[2] for h in hor if h[2]]
    vp_focal[name] = float(np.median(fs)) if fs else None
print("focal from vanishing points:", {k: (round(v) if v else None) for k,v in vp_focal.items()})

orig_load = ground_plane.load_or_build
print(f"\n{'clip':>16s} {'config':>10s} {'passF1':>7s} {'carryF1':>8s} "
      f"{'km/90':>6s} {'speed r':>8s} {'dist r':>7s}")
for name, d, lf, off in WINDOWS:
    ci = json.loads((Path(d)/"clip.json").read_text())
    for label in ("shipped", "vp focal"):
        newf = vp_focal[name]
        if label == "vp focal" and not newf:
            continue
        def patched(video, tracks, w, h, cache, verbose=True, _l=label, _f=newf):
            plane = orig_load(video, tracks, w, h, cache, verbose=False)
            if plane is None or _l == "shipped":
                return plane
            return dataclasses.replace(plane, focal_px=_f)
        ground_plane.load_or_build = patched
        try:
            events, profile, metric = run_pipeline(ci["path"], Path(d),
                                                   return_tracks=True)
        finally:
            ground_plane.load_or_build = orig_load
        dur = profile.n_frames/profile.fps
        ph = stats.physical_stats(metric); ph = ph[ph.minutes_tracked>0]
        km90 = float((ph.distance_m/ph.minutes_tracked*90/1000).median())
        rel = stats.top_speed_reliability(metric)
        print(f"{name:>16s} {label:>10s} {f1(events,lf,off,dur,'pass'):7.3f} "
              f"{f1(events,lf,off,dur,'carry'):8.3f} {km90:6.1f} "
              f"{rel['r']:8.2f} {dist_rel(metric):7.2f}")
