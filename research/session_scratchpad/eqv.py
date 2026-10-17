import sys, io, contextlib, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
spec = importlib.util.spec_from_file_location("old_tav2", "/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/old/old_tav2.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
import src.team_assignment_v2 as new

class FakeCap:
    def __init__(self, path): self.i = 0
    def read(self):
        self.i += 1
        return (self.i <= 400), np.zeros((4, 4, 3), np.uint8)
    def release(self): pass

def make_feat_fn(seed):
    def f(frame, px, py, h):
        r = np.random.default_rng(int(px * 1000 + py) + seed)
        if h < 0: return None
        g = int(px) % 4
        base = [np.array([1, 0, .9, .5]), np.array([0, 1, .3, .4]),
                np.array([-1, 0, .8, .9]), np.array([0, -1, .1, .1])][g]
        return base + r.normal(0, .2 if g < 2 else .4, 4)
    return f

for mod in (old, new):
    mod.cv2.VideoCapture = FakeCap

def tracks_for(seed, n_tracks, nframes=60, drop=()):
    r = np.random.default_rng(seed)
    rows = []
    for t in range(n_tracks):
        L = int(r.integers(5, nframes))
        s = int(r.integers(0, 300))
        for k in range(L):
            rows.append(dict(frame=s + k, track_id=t, cls="player",
                             px=float(t % 4) + 0.001 * k, py=float(k),
                             crop_h=-1.0 if t in drop else 40.0))
    for k in range(50):
        rows.append(dict(frame=k, track_id=9999, cls="ball", px=1., py=1., crop_h=5.))
    return pd.DataFrame(rows)

def run(mod, tr, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            out = mod.assign_teams_v2("x", tr, **kw)
        except Exception as e:
            out = repr(e)
    return out, buf.getvalue()

cases = 0
for seed in range(40):
    for n in (0, 2, 3, 5, 12, 40):
        for kc in (2, 3, 4):
            tr = tracks_for(seed, n, drop=set(range(0, n, 3)) if seed % 5 == 0 else ())
            for mod in (old, new):
                mod._torso_feature = make_feat_fn(seed)
            a, pa = run(old, tr, n_clusters=kc)
            b, pb = run(new, tr, n_clusters=kc)
            if isinstance(a, str) or isinstance(b, str):
                assert a == b, (seed, n, kc, a, b)
            else:
                pd.testing.assert_frame_equal(a, b)
                assert a.team.dtype == b.team.dtype
            assert pb.startswith(pa), (seed, n, kc, pa, pb)
            if pb != pa:
                print("extra print:", repr(pb[len(pa):]))
            cases += 1
# only-ball and empty
tr = tracks_for(0, 0)
a, _ = run(old, tr); b, _ = run(new, tr)
pd.testing.assert_frame_equal(a, b)
print("cases", cases, "all equal")
