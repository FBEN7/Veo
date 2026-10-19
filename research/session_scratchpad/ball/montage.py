"""Crops around the dataset's ball track: at action frames and between actions (look only)."""
import sys, json, io, numpy as np, pandas as pd, cv2
sys.path.insert(0, '/home/user/Veo')
from src import soccertrack_v2 as st
w, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
info = json.load(open(f'data/soccertrack_v2/{w}.json'))
match, half, start = info['match'], info['half'], info['start_frame']
cal = st.calibration(match)
z = np.load(io.BytesIO(st.open_stream(f"ball/{match}_{st._half(half)}_ball.npz").read()), allow_pickle=False)
gb = pd.DataFrame({'frame': z['frame'] - 1 - start, 'x': z['x'], 'y': z['y'], 'status': z['status']}).set_index('frame')
lab = pd.read_csv(f'data/soccertrack_v2/{w}_events.csv')
H = np.linalg.inv(cal['Hinv'])
def p2i(xy):
    p = np.c_[np.asarray(xy, float) + (52.5, 34.0), np.ones(len(xy))] @ H.T
    und = p[:, :2] / p[:, 2:]
    norm = (und - cal['Knew'][:2, 2]) / np.diag(cal['Knew'])[:2]
    return cv2.fisheye.distortPoints(np.ascontiguousarray(norm.reshape(-1, 1, 2)), cal['K'], cal['D']).reshape(-1, 2)
if mode == 'actions':
    frames = list(lab.frame[:24])
else:   # midway between consecutive actions at least 1.5 s apart
    f = lab.frame.to_numpy(); frames = [int((a + b) / 2) for a, b in zip(f[:-1], f[1:]) if b - a > 37][:24]
pts = {fr: p2i(gb.loc[[fr], ['x', 'y']].to_numpy())[0] for fr in frames if fr in gb.index}
cap = cv2.VideoCapture(f'data/soccertrack_v2/{w}.mp4'); idx = 0; tiles = []
while idx <= max(pts):
    ok, im = cap.read()
    if not ok: break
    if idx in pts:
        x, y = map(int, pts[idx]); c = im[max(0, y - 48):y + 48, max(0, x - 48):x + 48].copy()
        c = cv2.copyMakeBorder(c, 0, 96 - c.shape[0], 0, 96 - c.shape[1], cv2.BORDER_CONSTANT)
        cv2.circle(c, (48, 48), 2, (0, 0, 255), -1)
        tiles.append(cv2.resize(c, (192, 192), interpolation=cv2.INTER_NEAREST))
    idx += 1
rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles) - len(tiles) % 6, 6)]
cv2.imwrite(out, np.vstack(rows))
print(len(tiles), 'crops')
