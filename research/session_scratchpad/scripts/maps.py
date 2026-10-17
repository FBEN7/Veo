import sys, numpy as np, cv2
D = sys.argv[1]
z = np.load(f"{D}/117093_camera_intrinsics.npz", allow_pickle=False)
K, Dd, Knew = z["K"], z["D"], z["Knew"]
m1 = np.load(f"{D}/117093_mapx.npy"); m2 = np.load(f"{D}/117093_mapy.npy")
# decode CV_16SC2 + CV_16UC1 fixed-point maps (INTER_BITS=5)
sx = m1[...,0] + (m2 & 31) / 32.0
sy = m1[...,1] + (m2 >> 5) / 32.0
fx, fy = cv2.convertMaps(m1, m2, cv2.CV_32FC1)
print("decode == convertMaps:", np.abs(fx - sx).max(), np.abs(fy - sy).max())
rx, ry = cv2.fisheye.initUndistortRectifyMap(K, Dd, np.eye(3), Knew, (4096, 1080), cv2.CV_32FC1)
d = np.hypot(rx - sx, ry - sy)
ok = np.isfinite(d)
print("vs fisheye.initUndistortRectifyMap(K,D,I,Knew,(4096,1080)): max %.3f median %.4f px" % (d[ok].max(), np.median(d[ok])))
# source-pixel coverage
inside = (sx >= 0) & (sx < 4096) & (sy >= 0) & (sy < 1080)
print("undistorted px whose source lies inside the frame: %.1f%%" % (100*inside.mean()))
ys, xs = np.nonzero(inside); print("undistorted bbox of valid region x %d..%d y %d..%d" % (xs.min(), xs.max(), ys.min(), ys.max()))
