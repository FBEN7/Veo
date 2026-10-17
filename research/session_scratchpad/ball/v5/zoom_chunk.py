"""Zoom tiles of a verification chunk: python zoom_chunk.py CHUNK_DIR OUT.png TILE [TILE ...]
-> the tiles side by side at 3x. Reads only CHUNK_DIR/sheet_*.jpg and CHUNK_DIR/layout.json."""
import json, sys
import cv2, numpy as np
d, out, ks = sys.argv[1], sys.argv[2], [int(k) for k in sys.argv[3:]]
lay = json.load(open(f"{d}/layout.json"))
parts = []
for k in ks:
    sh, r, c = lay[str(k)]
    im = cv2.imread(f"{d}/sheet_{sh}.jpg")
    t = im[r * 238:(r + 1) * 238, c * 216:(c + 1) * 216]
    parts.append(cv2.resize(t, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
cv2.imwrite(out, np.hstack(parts))
