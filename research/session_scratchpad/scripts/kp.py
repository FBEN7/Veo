import sys, json, zipfile, io, numpy as np, cv2
D = sys.argv[1]
H = np.load(f"{D}/117093_homography.npy")
z = np.load(f"{D}/117093_camera_intrinsics.npz", allow_pickle=False)
K, Dd, Knew = z["K"], z["D"], z["Knew"]
print("rms", z["rms"])
# rvecs/tvecs are pickled object arrays of 3 floats (checked with pickletools: only numpy _reconstruct)
rvec = np.array([-1.2871781691075368, 0.005286106112673476, -0.012193678411224752])
tvec = np.array([-51.58332609565938, -16.137515514829506, 82.7531937548315])
kp = json.load(open(f"{D}/117093_keypoints.json"))
P = np.array([[float(v) for v in k.strip("()").split(",")] for k in kp])
U = np.array(list(kp.values()), float)
und = cv2.fisheye.undistortPoints(U.reshape(-1,1,2), K, Dd, P=Knew).reshape(-1,2)
def ap(M, X):
    Y = np.c_[X, np.ones(len(X))] @ M.T
    return Y[:, :2] / Y[:, 2:]
# direction 1: H maps pitch -> undistorted px
e1 = np.linalg.norm(ap(H, P) - und, axis=1)
print("H pitch->und px err px: median %.2f max %.2f" % (np.median(e1), e1.max()))
# direction 2: H maps und px -> pitch
e2 = np.linalg.norm(ap(H, und) - P, axis=1)
print("H und px->pitch err m: median %.2f max %.2f" % (np.median(e2), e2.max()))
e3 = np.linalg.norm(ap(np.linalg.inv(H), und) - P, axis=1)
print("inv(H) und px->pitch err m: median %.2f p90 %.2f max %.2f" % (np.median(e3), np.percentile(e3,90), e3.max()))
# refit check
Hf, _ = cv2.findHomography(P, und, cv2.RANSAC, 3.0)
print("refit H/H33", Hf/Hf[2,2]); 
# fisheye extrinsics
proj, _ = cv2.fisheye.projectPoints(np.c_[P, np.zeros(len(P))].reshape(-1,1,3), rvec, tvec, K, Dd)
e4 = np.linalg.norm(proj.reshape(-1,2) - U, axis=1)
print("fisheye project (X,Y,0) w/ rvec,tvec -> px err: median %.2f p90 %.2f max %.2f" % (np.median(e4), np.percentile(e4,90), e4.max()))
R,_ = cv2.Rodrigues(rvec); C = -R.T @ tvec
print("camera centre in pitch frame (corner origin, z):", C)
# worst keypoints for H route
for i in np.argsort(-e3)[:5]:
    print(" kp", P[i], "px", U[i], "err m %.2f" % e3[i])
