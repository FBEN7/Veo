import sys, cv2, numpy as np
S = sys.argv[1]
D = "tiles_SV"
ks = [int(k) for k in sys.argv[2:]]
out = []
for k in ks:
    sh = (k - 1) // 30 + 1
    j = (k - 1) % 30
    im = cv2.imread(f"{S}/{D}/sheet_{sh}.jpg")
    r, c = j // 5, j % 5
    t = im[r * 238:(r + 1) * 238, c * 216:(c + 1) * 216]
    out.append(cv2.resize(t, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
cv2.imwrite(f"{S}/{D}/z.png", np.hstack(out))
