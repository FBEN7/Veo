import sys, numpy as np, pandas as pd, cv2
from scipy.optimize import linear_sum_assignment
D, GT, TR = sys.argv[1:4]
H = np.load(f"{D}/117093_homography.npy")
z = np.load(f"{D}/117093_camera_intrinsics.npz", allow_pickle=False)
K, Dd, Knew = z["K"], z["D"], z["Knew"]
Hinv = np.linalg.inv(H)
def to_pitch(uv):
    und = cv2.fisheye.undistortPoints(np.ascontiguousarray(uv, dtype=np.float64).reshape(-1,1,2), K, Dd, P=Knew).reshape(-1,2)
    q = np.c_[und, np.ones(len(und))] @ Hinv.T
    return q[:, :2] / q[:, 2:] - [52.5, 34.0]
def to_image(xy):
    q = np.c_[np.asarray(xy, float) + [52.5, 34.0], np.ones(len(xy))] @ H.T
    und = q[:, :2] / q[:, 2:]
    n = cv2.undistortPoints(und.reshape(-1,1,2), Knew, None)
    return cv2.fisheye.distortPoints(n, K, Dd).reshape(-1,2)

g = pd.read_parquet(GT); g = g[g.bx.notna() & g.x.notna()]
uv = np.c_[g.bx + g.bw/2, g.by + g.bh].astype(float); gt = g[["x","y"]].to_numpy(float)
e = to_pitch(uv) - gt
inside = (np.abs(e[:,0]) <= 0.525 + 1e-6) & (np.abs(e[:,1]) <= 0.34 + 1e-6)
print("share of mapped feet inside the GT's 1.05x0.68 m grid cell: %.1f%%" % (100*inside.mean()))
rng = np.random.default_rng(0)
q = np.c_[rng.uniform(-.525,.525,len(gt)), rng.uniform(-.34,.34,len(gt))]
dq = np.linalg.norm(q, axis=1)
print("quantisation floor alone: median %.2f p90 %.2f" % (np.median(dq), np.percentile(dq,90)))
# pixel-space residual: GT pitch projected to image vs box feet
pr = to_image(gt) - uv
print("image-space: GT pitch -> pixel minus box foot: median |d| %.1f px, p90 %.1f; median du %+.1f dv %+.1f" % (
    np.median(np.linalg.norm(pr,axis=1)), np.percentile(np.linalg.norm(pr,axis=1),90), np.median(pr[:,0]), np.median(pr[:,1])))
# by region
d = np.linalg.norm(e, axis=1)
for name, m in (("far third (y<-11)", gt[:,1] < -11), ("middle", (gt[:,1]>=-11)&(gt[:,1]<11)), ("near third (y>=11)", gt[:,1]>=11),
                ("|x|>40", np.abs(gt[:,0])>40)):
    print(f"  {name}: n={m.sum()} median {np.median(d[m]):.2f} p90 {np.percentile(d[m],90):.2f}")

# pipeline detections vs GT (every 25th frame)
t = pd.read_parquet(TR); t = t[(t.cls=="player") & (t.frame % 25 == 0)]
gb = {f: x for f, x in g.assign(fu=uv[:,0], fv=uv[:,1], gx=gt[:,0], gy=gt[:,1]).groupby("frame")}
rows = []
for f, dd in t.groupby("frame"):
    G = gb.get(f)
    if G is None: continue
    a = dd[["px_raw","py_raw"]].to_numpy(float); b = G[["fu","fv"]].to_numpy(float)
    dist = np.linalg.norm(a[:,None]-b[None], axis=2)
    r, c = linear_sum_assignment(np.where(dist <= 43, dist, 1e6))
    for i, j in zip(r, c):
        if dist[i,j] <= 43:
            rows.append((*a[i], *b[j], G.gx.iloc[j], G.gy.iloc[j]))
m = np.array(rows)
off = m[:,0:2] - m[:,2:4]
print(f"pipeline detections matched: {len(m)}; det foot minus box foot: median du {np.median(off[:,0]):+.1f} dv {np.median(off[:,1]):+.1f} px, median |d| {np.median(np.linalg.norm(off,axis=1)):.1f}")
ed = to_pitch(m[:,0:2]) - m[:,4:6]; dd_ = np.linalg.norm(ed, axis=1)
print(f"  det feet -> pitch vs GT: median {np.median(dd_):.2f} p90 {np.percentile(dd_,90):.2f}; median dx {np.median(ed[:,0]):+.2f} dy {np.median(ed[:,1]):+.2f}")
ed2 = to_pitch(m[:,0:2] - np.median(off,axis=0)) - m[:,4:6]; d2 = np.linalg.norm(ed2, axis=1)
print(f"  after constant image shift {(-np.median(off,axis=0)).round(1)} px: median {np.median(d2):.2f} p90 {np.percentile(d2,90):.2f}")
ed3 = ed - np.median(ed, axis=0); d3 = np.linalg.norm(ed3, axis=1)
print(f"  after constant pitch shift {(-np.median(ed,axis=0)).round(2)} m: median {np.median(d3):.2f} p90 {np.percentile(d3,90):.2f}")
gx, gy = m[:,4], m[:,5]
for name, mk in (("det far (y<-11)", gy < -11), ("det middle", (gy>=-11)&(gy<11)), ("det near (y>=11)", gy>=11), ("det |x|>35", np.abs(gx)>35)):
    print(f"  {name}: n={mk.sum()} median {np.median(dd_[mk]):.2f} p90 {np.percentile(dd_[mk],90):.2f}")
import json
kp = json.load(open(f"{D}/117093_keypoints.json"))
P = np.array([[float(v) for v in k.strip("()").split(",")] for k in kp]); U = np.array(list(kp.values()), float)
pi = to_image(P - [52.5, 34.0]); ep = np.linalg.norm(pi - U, axis=1)
print("keypoints: pitch->image reprojection px: median %.1f p90 %.1f, n within 3px %d/%d" % (np.median(ep), np.percentile(ep,90), (ep<=3).sum(), len(ep)))
em = np.linalg.norm(to_pitch(U) - (P - [52.5,34.0]), axis=1)
for i in np.argsort(-ep)[:8]:
    print("   kp", P[i], "px err %.1f  m err %.2f" % (ep[i], em[i]))
cap = cv2.VideoCapture("/home/user/Veo/data/soccertrack_v2/st2_117093_1st_f015000.mp4")
print("video", cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("GT box foot range u %.0f..%.0f v %.0f..%.0f" % (uv[:,0].min(), uv[:,0].max(), uv[:,1].min(), uv[:,1].max()))
