import sys
sys.path.insert(0, "/home/user/Veo")
import numpy as np, pandas as pd
import build_ball_strips as bs
import label_ball_strips as lb
h, _ = bs.synthetic_half()
poly = np.array([[700, 120], [1700, 120], [2340, 660], [60, 660]], np.float32)
box = bs.pitch_box((700, 2400), poly)
rows = pd.DataFrame(bs.samples(h, bs.KINDS, box, poly, seed=0))
for r in rows[rows.kind == "flight"].itertuples():
    out = lb.flight_codes(r, h)
    if out is None:
        print("flight", r.frame, r.category, "dropped"); continue
    c, nar = out
    print("flight", r.frame, r.category, "s %.2f th %.2f L %.1f" % (r.s, r.t_hat, r.L), "narrow", nar,
          {k: int((c == k).sum()) for k in (1, 0, -1, -3)})
A, B = np.array([r.a_x, r.a_y]), np.array([r.b_x, r.b_y])
r = next(rows[(rows.kind == "flight") & (rows.category == "completed")].itertuples())
A, B = np.array([r.a_x, r.a_y]), np.array([r.b_x, r.b_y]); d = B - A
on_line = A + r.t_hat * d; far = on_line + (0, -8.0)
uv, sx, sy = bs.scales([on_line, far], h.cal)
print("on/far uv", uv - (r.x0, r.y0), sx, sy)
print("far box", lb.body_slices(uv[1], sx[1], sy[1], r.x0, r.y0, bs.CELLS))
for k in ("lofted", "action", "dribble"):
    for r in rows[rows.kind == k].itertuples():
        fn = {"lofted": lb.lofted_codes, "action": lb.action_codes, "dribble": lb.dribble_codes}[k]
        c = fn(r, h)
        print(k, r.frame, r.label, None if c is None else {v: int((c == v).sum()) for v in (1, 0, -1, -3)})
