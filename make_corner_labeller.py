"""Build a one-file page for clicking the four corners of a goal mouth.

The box detector works -- 6 of 6 shot moments, IoU 0.89 to 0.96 -- but a box
is a landmark, not a geometry. Put one through the ground plane and measure
a quantity the laws of the game fix exactly and it reads 4.7 m where a goal
is 7.32 m wide, short on almost every frame. The same measurement taken
vertically, where an oblique view does not foreshorten, reads 2.5 m against
a true 2.44. See `probe_goal_ruler.py`.

Width wrong, height right: the plane's scale is sound and the axis-aligned
rectangle is the problem. It discards the slant that encodes orientation,
which the labeller said at the time -- "with the perspective, the goal is
not a rectangle but rather a paralelipede rectangle".

Four corners fix that. The goal mouth is a rectangle of known size, 7.32 by
2.44 m, so four image points of it give the camera's pose by PnP, and with
it a distance and an angle to goal at the moment of a shot. Two points give
no pose but still train a keypoint model, which is why partial labels are
kept rather than refused.

## What is done to keep it short

Only the 25 frames that already have a hand-drawn goal, not the 150 of the
first round -- a corner cannot be clicked where there is no goal. That is
100 clicks rather than 600.

Each frame is **cropped to the box it already carries and blown up**. A
distant goal 70 px wide arrives on screen at seven times that, so the
corners are large targets rather than a few pixels in a wide shot. This
matters beyond comfort: PnP turns a few pixels of click error into metres
of position error, and the magnification divides that error by the zoom.

Clicks are converted back to full-frame coordinates in the page, so nothing
downstream needs to know a crop happened.

## The order, and why it is fixed

    1  left post, base        3  right post, top
    2  left post, top         4  right post, base

Left and right as seen on screen. A fixed order means the four points
arrive already identified, with no post-hoc guessing about which corner is
which -- and `space` skips a corner that is out of frame or hidden, so a
goal with two visible corners costs two clicks and two taps.

## Why the page is not published

The frames are SoccerNet, under an NDA that forbids passing the data to
anyone who has not signed for it. The page is handed to the one person who
already has it, as a file, never to a URL. The generator is committed; what
it makes is not.

    python make_corner_labeller.py --labels goal_labels.json
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

from propagate_goal_labels import DIRS, REPLAY_AREA, area

# Padding round the known box, as a share of its larger side, with a floor
# so a small distant goal still arrives with context round it. Context is
# what makes a post base identifiable: the corner is where the white meets
# the grass, and that needs some grass in shot.
PAD_RATIO = 0.35
PAD_MIN_PX = 40

# The crop is scaled to fit this, which is what provides the magnification.
# Capped so a close-up goal does not arrive larger than a screen.
VIEW_W, VIEW_H = 960, 640
MAX_ZOOM = 8.0

# Higher than the first round's 72: there are 25 crops here rather than 150
# full frames, so the whole page is a couple of megabytes and the quality is
# free. Clicking a post base accurately needs the edge to be crisp.
JPEG_QUALITY = 92

CORNERS = [
    ("left post", "base"),
    ("left post", "top"),
    ("right post", "top"),
    ("right post", "base"),
]


def crop_for(frame, box, width: int, height: int):
    """Crop round a normalised box and scale it up. Returns (jpeg, rect)."""
    x0, y0, x1, y1 = (box[0] * width, box[1] * height,
                      box[2] * width, box[3] * height)
    pad = max(PAD_MIN_PX, PAD_RATIO * max(x1 - x0, y1 - y0))
    cx0 = int(max(0, np.floor(x0 - pad)))
    cy0 = int(max(0, np.floor(y0 - pad)))
    cx1 = int(min(width, np.ceil(x1 + pad)))
    cy1 = int(min(height, np.ceil(y1 + pad)))
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None, None

    patch = frame[cy0:cy1, cx0:cx1]
    zoom = min(VIEW_W / (cx1 - cx0), VIEW_H / (cy1 - cy0), MAX_ZOOM)
    out = cv2.resize(patch, (int(round((cx1 - cx0) * zoom)),
                             int(round((cy1 - cy0) * zoom))),
                     interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY,
                                         JPEG_QUALITY])
    if not ok:
        return None, None
    # The rect is what the page needs to send clicks back to full-frame
    # coordinates; the zoom is only for reporting.
    return base64.b64encode(buf).decode("ascii"), {
        "x": cx0, "y": cy0, "w": cx1 - cx0, "h": cy1 - cy0,
        "zoom": round(float(zoom), 2),
        # The original box, drawn faintly so it is obvious which goal is
        # meant and that the crop found it.
        "box": [round((x0 - cx0) / (cx1 - cx0), 4),
                round((y0 - cy0) / (cy1 - cy0), 4),
                round((x1 - cx0) / (cx1 - cx0), 4),
                round((y1 - cy0) / (cy1 - cy0), 4)],
    }


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Goal corners</title>
<style>
 :root{--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--hit:#00e5ff;--ok:#00c853;
       --warn:#ffab40}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
 header{display:flex;gap:18px;align-items:center;padding:10px 16px;
        background:#0d0f12;border-bottom:1px solid #23262b;position:sticky;
        top:0;z-index:5}
 #bar{flex:1;height:8px;background:#23262b;border-radius:4px;overflow:hidden}
 #fill{height:100%;width:0;background:var(--ok);transition:width .15s}
 .count{font-variant-numeric:tabular-nums;color:var(--dim)}
 main{display:flex;justify-content:center;gap:20px;padding:14px;
      align-items:flex-start;flex-wrap:wrap}
 #stage{position:relative;line-height:0;cursor:crosshair}
 #shot{max-width:100%;height:auto;border-radius:6px;display:block}
 #guide{position:absolute;border:1px dashed rgba(255,255,255,.35);
        pointer-events:none}
 .dot{position:absolute;width:12px;height:12px;margin:-6px 0 0 -6px;
      border-radius:50%;background:var(--hit);border:2px solid #04252b;
      pointer-events:none}
 .dot span{position:absolute;left:14px;top:-4px;font:12px/1 ui-monospace,
           monospace;color:var(--hit)}
 aside{width:230px;color:var(--dim);font-size:14px}
 aside svg{width:100%;height:auto;margin-bottom:8px}
 .step{padding:5px 9px;border-radius:6px;margin:3px 0;
       border:1px solid transparent}
 .step.now{background:#0d2b33;border-color:#12707f;color:#9beaf5}
 .step.got{color:#7bd8a8}
 .step.gone{color:#6b6f75;text-decoration:line-through}
 #ask{text-align:center;font-size:19px;padding:6px 16px 0}
 #ask b{color:var(--hit)}
 footer{display:flex;gap:10px;justify-content:center;align-items:center;
         padding:10px 16px 20px;flex-wrap:wrap}
 button{font:inherit;padding:10px 18px;border-radius:8px;
        border:1px solid #2c3037;background:#1b1e24;color:var(--fg);
        cursor:pointer}
 button:hover{background:#242830}
 .hint{color:var(--dim);text-align:center;padding:0 16px 6px;font-size:14px}
 kbd{background:#23262b;border:1px solid #33373e;border-bottom-width:2px;
     border-radius:5px;padding:1px 7px;font:13px ui-monospace,monospace}
 .done{text-align:center;padding:40px}
</style></head><body>
<header>
  <strong>Goal corners</strong>
  <div id="bar"><div id="fill"></div></div>
  <span class="count" id="count"></span>
  <button id="save">Download corners</button>
</header>
<p id="ask"></p>
<main>
 <div id="stage"><img id="shot" alt=""><div id="guide"></div></div>
 <aside>
  <svg viewBox="0 0 200 130">
   <rect x="0" y="0" width="200" height="130" fill="#1b1e24" rx="6"/>
   <path d="M40 95 L40 35 L160 35 L160 95" fill="none" stroke="#c8ccd2"
         stroke-width="5"/>
   <line x1="20" y1="95" x2="185" y2="95" stroke="#3d6b45" stroke-width="3"/>
   <circle cx="40" cy="95" r="8" fill="#00e5ff"/>
   <circle cx="40" cy="35" r="8" fill="#00e5ff"/>
   <circle cx="160" cy="35" r="8" fill="#00e5ff"/>
   <circle cx="160" cy="95" r="8" fill="#00e5ff"/>
   <text x="40" y="99" font-size="11" text-anchor="middle" fill="#04252b"
         font-family="monospace">1</text>
   <text x="40" y="39" font-size="11" text-anchor="middle" fill="#04252b"
         font-family="monospace">2</text>
   <text x="160" y="39" font-size="11" text-anchor="middle" fill="#04252b"
         font-family="monospace">3</text>
   <text x="160" y="99" font-size="11" text-anchor="middle" fill="#04252b"
         font-family="monospace">4</text>
  </svg>
  <div class="step" id="s0">1 &nbsp;left post, base</div>
  <div class="step" id="s1">2 &nbsp;left post, top</div>
  <div class="step" id="s2">3 &nbsp;right post, top</div>
  <div class="step" id="s3">4 &nbsp;right post, base</div>
 </aside>
</main>
<p class="hint">Click where the <b>post meets the ground</b> and where the
 <b>post meets the crossbar</b> &mdash; the frame itself, not the net.
 If a corner is out of frame or hidden, press <kbd>space</kbd> to skip it.</p>
<footer>
  <button id="none">Can't label this one &nbsp;<kbd>N</kbd></button>
  <button id="undo">Undo &nbsp;<kbd>U</kbd></button>
  <button id="skipc">Corner not visible &nbsp;<kbd>space</kbd></button>
</footer>
<script>
const FRAMES = __DATA__;
const KEY = "goal-corners-v1";
const NAMES = ["left post base","left post top","right post top",
               "right post base"];
let at = 0, pts = [], done = {};
try { done = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}

const img = document.getElementById("shot");
const stage = document.getElementById("stage");
const guide = document.getElementById("guide");
const ask = document.getElementById("ask");

function firstUnlabelled(){
  for (let i=0;i<FRAMES.length;i++) if (done[FRAMES[i].id] === undefined) return i;
  return FRAMES.length;
}
function clearDots(){ stage.querySelectorAll(".dot").forEach(d=>d.remove()); }

function paint(){
  const n = Object.keys(done).length;
  document.getElementById("count").textContent = n + " / " + FRAMES.length;
  document.getElementById("fill").style.width = (100*n/FRAMES.length) + "%";
  for (let i=0;i<4;i++){
    const el = document.getElementById("s"+i);
    el.className = "step" + (i===pts.length ? " now"
                   : i<pts.length ? (pts[i] ? " got" : " gone") : "");
  }
  ask.innerHTML = pts.length<4
    ? "Click the <b>" + NAMES[pts.length] + "</b>"
    : "saving&hellip;";
}
function show(){
  clearDots(); pts = [];
  if (at >= FRAMES.length){ finish(); return; }
  const f = FRAMES[at];
  img.src = "data:image/jpeg;base64," + f.jpeg;
  const b = f.rect.box;
  guide.style.left=(b[0]*100)+"%"; guide.style.top=(b[1]*100)+"%";
  guide.style.width=((b[2]-b[0])*100)+"%";
  guide.style.height=((b[3]-b[1])*100)+"%";
  paint();
}
function addDot(u,v,label){
  const d = document.createElement("div");
  d.className = "dot";
  d.style.left=(u*100)+"%"; d.style.top=(v*100)+"%";
  d.innerHTML = "<span>"+label+"</span>";
  stage.appendChild(d);
}
function record(){
  // Clicks are stored in full-frame normalised coordinates, so nothing
  // downstream needs to know the page cropped and magnified anything.
  const f = FRAMES[at], r = f.rect;
  done[f.id] = pts.map(p => p === null ? null : [
    +((r.x + p[0]*r.w) / __W__).toFixed(5),
    +((r.y + p[1]*r.h) / __H__).toFixed(5)]);
  localStorage.setItem(KEY, JSON.stringify(done));
  at++; show();
}
function push(p){
  if (p) addDot(p[0], p[1], pts.length+1);
  pts.push(p);
  paint();
  if (pts.length === 4) setTimeout(record, 160);
}
stage.addEventListener("click", e => {
  if (at >= FRAMES.length || pts.length >= 4) return;
  const r = img.getBoundingClientRect();
  const u = (e.clientX - r.left)/r.width, v = (e.clientY - r.top)/r.height;
  if (u<0||u>1||v<0||v>1) return;
  push([u,v]);
});
document.getElementById("skipc").onclick = ()=>{
  if (at < FRAMES.length && pts.length < 4) push(null);
};
document.getElementById("none").onclick = ()=>{
  if (at >= FRAMES.length) return;
  done[FRAMES[at].id] = [null,null,null,null];
  localStorage.setItem(KEY, JSON.stringify(done));
  at++; show();
};
document.getElementById("undo").onclick = ()=>{
  if (pts.length){
    pts.pop(); clearDots();
    pts.forEach((p,i)=> p && addDot(p[0],p[1],i+1));
    paint(); return;
  }
  at = Math.max(0, at-1);
  delete done[FRAMES[at].id];
  localStorage.setItem(KEY, JSON.stringify(done)); show();
};
document.addEventListener("keydown", e=>{
  if (e.key === " "){ e.preventDefault();
    document.getElementById("skipc").click(); return; }
  const k = e.key.toLowerCase();
  if (k==="n") document.getElementById("none").click();
  else if (k==="u") document.getElementById("undo").click();
});
document.getElementById("save").onclick = ()=>{
  const out = FRAMES.map(f => ({id:f.id, clip:f.clip, frame:f.frame,
                                box:f.box,
                                corners: done[f.id] === undefined ? null
                                        : done[f.id]}));
  const blob = new Blob([JSON.stringify(
    {width:__W__, height:__H__, order:NAMES,
     labelled:Object.keys(done).length, frames:out}, null, 1)],
    {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "goal_corners.json";
  a.click();
};
function finish(){
  document.querySelector("main").innerHTML =
    "<div class='done'><h2>All " + FRAMES.length + " done &mdash; thank you.</h2>"
    + "<p>Press <b>Download corners</b> above and send me the file.</p></div>";
  ask.textContent = "";
  document.querySelector("footer").style.display = "none";
  document.querySelector(".hint").style.display = "none";
}
at = firstUnlabelled(); show();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True,
                    help="goal_labels.json from the first round")
    ap.add_argument("--out", default="goal_corner_labeller.html")
    args = ap.parse_args()

    rows = json.loads(Path(args.labels).read_text())["frames"]
    wanted = [r for r in rows if r.get("label") and r["label"].get("goal")
              and area(r["label"]["goal"]) <= REPLAY_AREA]

    frames, caps = [], {}
    for row in sorted(wanted, key=lambda r: (r["clip"], r["frame"])):
        out_dir = DIRS.get(row["clip"])
        if not out_dir or not (Path(out_dir) / "clip.json").exists():
            continue
        if row["clip"] not in caps:
            info = json.loads((Path(out_dir) / "clip.json").read_text())
            caps[row["clip"]] = (cv2.VideoCapture(info["path"]),
                                 info["width"], info["height"])
        cap, width, height = caps[row["clip"]]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame"]))
        ok, frame = cap.read()
        if not ok:
            continue
        jpeg, rect = crop_for(frame, row["label"]["goal"], width, height)
        if jpeg is None:
            continue
        frames.append({"id": row["id"], "clip": row["clip"],
                       "frame": row["frame"], "box": row["label"]["goal"],
                       "jpeg": jpeg, "rect": rect})
    for cap, _, _ in caps.values():
        cap.release()

    if not frames:
        raise SystemExit("no labelled goals found")

    zooms = [f["rect"]["zoom"] for f in frames]
    print(f"  {len(frames)} goals to corner, {len(frames) * 4} clicks at most")
    print(f"  magnification {min(zooms):.1f}x to {max(zooms):.1f}x "
          f"(median {float(np.median(zooms)):.1f}x)")

    width, height = next(iter(caps.values()))[1:] if caps else (1280, 720)
    page = (PAGE.replace("__DATA__", json.dumps(frames))
                .replace("__W__", str(width)).replace("__H__", str(height)))
    target = Path(args.out)
    target.write_text(page, encoding="utf-8")
    size = target.stat().st_size / 1e6
    print(f"\n  -> {target} ({size:.1f} MB)")
    if size > 29:
        print("  TOO BIG to upload")


if __name__ == "__main__":
    main()
