"""Zoom S-V tiles: python zoom_tiles.py OUT.png TILE [TILE ...] -> the tiles
side by side at 3x (box = the 4 red corner marks; each mark's corner vertex
is the box corner). Reads only tiles_SV/sheet_*.jpg."""
import sys, cv2, numpy as np
D = "/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/judge2/sheets"
out, ks = sys.argv[1], [int(k) for k in sys.argv[2:]]
parts = []
for k in ks:
    sh, j = (k - 1) // 30 + 1, (k - 1) % 30
    im = cv2.imread(f"{D}/sheet_{sh}.jpg")
    r, c = j // 5, j % 5
    parts.append(cv2.resize(im[r * 238:(r + 1) * 238, c * 216:(c + 1) * 216], None,
                            fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
cv2.imwrite(out, np.hstack(parts))
