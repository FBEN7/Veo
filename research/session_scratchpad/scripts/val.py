import sys, numpy as np, pandas as pd, cv2
D, GT = sys.argv[1], sys.argv[2]
H = np.load(f"{D}/117093_homography.npy")
z = np.load(f"{D}/117093_camera_intrinsics.npz", allow_pickle=False)
K, Dd, Knew = z["K"], z["D"], z["Knew"]
rvec = np.array([-1.2871781691075368, 0.005286106112673476, -0.012193678411224752])
tvec = np.array([-51.58332609565938, -16.137515514829506, 82.7531937548315])
R, _ = cv2.Rodrigues(rvec); C = -R.T @ tvec
Hinv = np.linalg.inv(H)

def route_h(uv):
    und = cv2.fisheye.undistortPoints(uv.reshape(-1,1,2).astype(np.float64), K, Dd, P=Knew).reshape(-1,2)
    q = np.c_[und, np.ones(len(und))] @ Hinv.T
    return q[:, :2] / q[:, 2:] - [52.5, 34.0]

def route_ray(uv):
    n = cv2.fisheye.undistortPoints(uv.reshape(-1,1,2).astype(np.float64), K, Dd).reshape(-1,2)
    d = np.c_[n, np.ones(len(n))] @ R      # = (R.T @ d_cam) row-wise
    s = -C[2] / d[:, 2]
    return (C[None, :2] + s[:, None] * d[:, :2]) - [52.5, 34.0]

g = pd.read_parquet(GT)
g = g[g.bx.notna() & g.x.notna()]
uv = np.c_[g.bx + g.bw / 2, g.by + g.bh].astype(float)
gt = g[["x", "y"]].to_numpy(float)

# round-trip check of undistortPoints
und = cv2.fisheye.undistortPoints(uv.reshape(-1,1,2), K, Dd, P=Knew)
nrm = cv2.undistortPoints(und, Knew, None)   # back to normalised
back = cv2.fisheye.distortPoints(nrm, K, Dd).reshape(-1,2)
print("undistortPoints round trip px: max %.4f" % np.abs(back - uv).max())

def rep(name, est):
    e = est - gt; d = np.linalg.norm(e, axis=1)
    print(f"{name}: median {np.median(d):.2f} m, p90 {np.percentile(d,90):.2f}, p99 {np.percentile(d,99):.2f}, mean dx {e[:,0].mean():+.2f} dy {e[:,1].mean():+.2f}, median dx {np.median(e[:,0]):+.2f} dy {np.median(e[:,1]):+.2f}")
    return e
eh = rep("homography (fisheye undistort -> inv(H))", route_h(uv))
er = rep("ray/plane (rvec,tvec)            ", route_ray(uv))
# constant shift in pitch
for name, e in (("H", eh), ("ray", er)):
    d = np.linalg.norm(e - np.median(e, axis=0), axis=1)
    print(f"  {name} after constant pitch shift {np.median(e,axis=0).round(2)}: median {np.median(d):.2f}, p90 {np.percentile(d,90):.2f}")
# image-space foot offset: best constant dv (pixels) on foot point
for dv in (-20,-15,-10,-5,0,5,10):
    for du in (0,):
        e = route_h(uv + [du, dv]) - gt; d = np.linalg.norm(e, axis=1)
        print(f"  H with foot shift dv={dv:+d}px: median {np.median(d):.2f} p90 {np.percentile(d,90):.2f}")
np.save(sys.argv[3], np.c_[uv, gt, route_h(uv), route_ray(uv)])
