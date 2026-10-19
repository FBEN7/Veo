import sys, numpy as np, pandas as pd, cv2

def load_calib(root, match="117093"):
    z = np.load(f"{root}/{match}_camera_intrinsics.npz", allow_pickle=False)  # K, D, Knew read fine; rvecs/tvecs are pickled
    return {"K": z["K"], "D": z["D"], "Knew": z["Knew"],
            "Hinv": np.linalg.inv(np.load(f"{root}/{match}_homography.npy"))}

def image_to_pitch(uv, cal):
    """uv: (N,2) pixels of the original 4096x1080 panorama frame -> (N,2) metres,
    origin centre spot, x to the right as seen from the camera, y towards the camera
    (the GSR / BAS convention). NaN where the point is at or above the horizon."""
    uv = np.ascontiguousarray(uv, dtype=np.float64).reshape(-1, 1, 2)   # cv2 misreads strided views
    und = cv2.fisheye.undistortPoints(uv, cal["K"], cal["D"], P=cal["Knew"]).reshape(-1, 2)
    q = np.c_[und, np.ones(len(und))] @ cal["Hinv"].T
    xy = q[:, :2] / q[:, 2:] - (52.5, 34.0)      # keypoint frame is corner-origin 0..105 x 0..68, y=68 the near touchline
    xy[q[:, 2] <= 0] = np.nan
    return xy

cal = load_calib(sys.argv[1])
print("Hinv row3", cal["Hinv"][2])
g = pd.read_parquet(sys.argv[2]); g = g[g.bx.notna()]
xy = image_to_pitch(np.c_[g.bx + g.bw / 2, g.by + g.bh], cal)
d = np.hypot(*(xy - g[["x", "y"]].to_numpy()).T)
print(len(d), "nan", np.isnan(d).sum(), "median %.3f p90 %.3f p99 %.3f max %.3f" % tuple(np.nanpercentile(d, [50, 90, 99, 100])))
