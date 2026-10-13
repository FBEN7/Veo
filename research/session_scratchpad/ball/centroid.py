import sys, cv2, numpy as np
S, D = sys.argv[1], sys.argv[2]
for k in map(int, sys.argv[3:]):
    sh = (k - 1) // 30 + 1; j = (k - 1) % 30; r, c = j // 5, j % 5
    im = cv2.imread(f"{S}/{D}/sheet_{sh}.jpg")
    t = im[r * 238 + 22:(r + 1) * 238, c * 216:(c + 1) * 216].astype(int)
    g = t.sum(2)
    for thr in (540, 600):
        m = g > thr
        ys, xs = np.nonzero(m)
        if len(xs):
            print(k, "thr", thr, "px", m.sum(), "centroid", round(xs.mean(), 1), round(ys.mean(), 1), "(box 72..144)")
