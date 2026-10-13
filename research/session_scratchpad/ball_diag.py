"""How often the detector sees the ball at labelled action frames, by confidence (diagnostic)."""
import sys, json, io, numpy as np, pandas as pd, cv2
sys.path.insert(0, '/home/user/Veo')
from ultralytics import YOLO
from src import soccertrack_v2 as st
w = sys.argv[1]
info = json.load(open(f'data/soccertrack_v2/{w}.json'))
match, half, start = info['match'], info['half'], info['start_frame']
cal = st.calibration(match)
z = np.load(io.BytesIO(st.open_stream(f"ball/{match}_{st._half(half)}_ball.npz").read()), allow_pickle=False)
gb = pd.DataFrame({'frame': z['frame'] - 1 - start, 'x': z['x'], 'y': z['y'], 'status': z['status']})
lab = pd.read_csv(f'data/soccertrack_v2/{w}_events.csv')
H = np.linalg.inv(cal['Hinv'])
def pitch_to_image(xy):
    p = np.c_[np.asarray(xy) + (52.5, 34.0), np.ones(len(xy))] @ H.T
    und = p[:, :2] / p[:, 2:]
    norm = (und - cal['Knew'][:2, 2]) / np.diag(cal['Knew'])[:2]
    return cv2.fisheye.distortPoints(np.ascontiguousarray(norm.reshape(-1, 1, 2)), cal['K'], cal['D']).reshape(-1, 2)
poly = np.asarray(info['pitch'], np.float32).reshape(-1, 1, 2)
frames = sorted(set(lab.frame))
want = {f: pitch_to_image(gb[gb.frame == f][['x', 'y']].to_numpy())[0] for f in frames if (gb.frame == f).any()}
model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture(f'data/soccertrack_v2/{w}.mp4'); idx = 0; rows = []
while idx <= max(want):
    ok, fr = cap.read()
    if not ok: break
    if idx in want:
        r = model(fr, imgsz=4096, conf=0.02, classes=[32], verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.conf.cpu().numpy()
        ctr = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2] if len(b) else np.zeros((0, 2))
        bot = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]] if len(b) else np.zeros((0, 2))
        inside = np.array([cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -10 for x, y in ctr], bool)
        d = np.hypot(*(bot - want[idx]).T) if len(b) else np.zeros(0)
        rows.append(dict(frame=idx, conf=c, inside=inside, d=d))
    idx += 1
print(w, len(rows), 'action frames')
for t in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
    hit = np.mean([((r['conf'] >= t) & r['inside'] & (r['d'] <= 20)).any() for r in rows])
    fp = np.mean([((r['conf'] >= t) & r['inside'] & (r['d'] > 20)).sum() for r in rows])
    print(f'  conf >= {t}: ball found within 20 px of the GT ball at {hit:.2f} of action frames; other in-pitch candidates per frame {fp:.2f}')
