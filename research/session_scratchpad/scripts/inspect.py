import sys, numpy as np
D = sys.argv[1]
H = np.load(f"{D}/117093_homography.npy", allow_pickle=False)
print("H", H.dtype, H.shape); print(H)
z = np.load(f"{D}/117093_camera_intrinsics.npz", allow_pickle=False)
for k in z.files:
    v = z[k]; print(k, v.dtype, v.shape); print(v)
for n in ("mapx", "mapy"):
    m = np.load(f"{D}/117093_{n}.npy", mmap_mode="r", allow_pickle=False)
    print(n, m.dtype, m.shape, float(np.nanmin(m)), float(np.nanmax(m)))
    print(" corners", m[0,0], m[0,-1], m[-1,0], m[-1,-1], " centre", m[m.shape[0]//2, m.shape[1]//2])
