"""Build a one-file labelling page for goals: open it, click, download.

Training a goal detector needs annotated goals, and the free route is shut.
Labels could have come from the anchor -- project the goal mouth through a
sound anchor and read off a box -- but goals are in view on **0%** of
anchored frames across seven clips and 122 anchors. Anchors come from the
centre circle at midfield; the goals are fifty metres away and out of shot.
That is the same wall the marking classifier hit, for the same reason.

So the labels have to be drawn by a person, and the only thing worth
optimising is their time.

## The shape of it

One HTML file, opened in a browser. No install, no server, no Python, no
unzipping: the frames are embedded in the page. Two clicks put a box round a
goal and move on; one key says there is no goal and moves on. Work is saved
in the browser after every frame, so it can be stopped and resumed, and a
button downloads the labels at any point -- a half-finished set is still
worth having.

## Why the frames are not published

They are SoccerNet frames, under a non-commercial NDA that forbids passing
the data to anyone who has not signed for it. This page is therefore handed
to the one person who already has it, as a file, and never published to a
URL. That is also why nothing here is committed: the generator is, the page
it makes is not.

    python make_goal_labeller.py [--per-window 15] [--out labeller.html]
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

# Where the goal is usually visible, and where it usually is not. Both are
# wanted: a detector trained only on frames containing goals learns that
# every frame has one.
WINDOWS = [
    ("stoke_1302", "output_stoke_1302", 15),
    ("stoke_4207", "output_stoke_4207", 15),
    ("stoke_7001", "output_stoke_7001", 15),
    ("reading_0737", "output_reading_0737", 15),
    ("reading_1155", "output_reading_1155", 15),
    ("reading_2519", "output_reading_2519", 15),
    ("soccernet_w1", "output_soccernet", 10),
    ("soccernet_w2", "output_soccernet_w2", 10),
    ("soccernet_w3", "output_soccernet_w3", 10),
    ("soccernet_reading", "output_soccernet_reading", 10),
    ("veo", "output_veo", 20),
]

# Big enough to see a distant goal, small enough that 150 of them fit in one
# page under the upload limit.
SHOW_W, SHOW_H = 960, 540
JPEG_QUALITY = 72


def sample_frames(out_dir: str, count: int):
    """Frames spread evenly through a clip, as encoded JPEG bytes."""
    path = Path(out_dir)
    if not (path / "clip.json").exists():
        return []
    info = json.loads((path / "clip.json").read_text())
    cap = cv2.VideoCapture(info["path"])
    total = int(info["n_frames"])
    shots = []
    for idx in np.linspace(0, total - 1, count).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (SHOW_W, SHOW_H))
        ok, buf = cv2.imencode(".jpg", small,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            shots.append({"frame": int(idx),
                          "jpeg": base64.b64encode(buf).decode("ascii")})
    cap.release()
    return shots


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Goal labelling</title>
<style>
 :root{--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--hit:#00e5ff;--ok:#00c853}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
 header{display:flex;gap:18px;align-items:center;padding:10px 16px;
        background:#0d0f12;border-bottom:1px solid #23262b;position:sticky;top:0}
 #bar{flex:1;height:8px;background:#23262b;border-radius:4px;overflow:hidden}
 #fill{height:100%;width:0;background:var(--ok);transition:width .15s}
 .count{font-variant-numeric:tabular-nums;color:var(--dim)}
 main{display:flex;justify-content:center;padding:14px}
 #stage{position:relative;line-height:0;cursor:crosshair}
 #shot{max-width:100%;height:auto;border-radius:6px}
 #box{position:absolute;border:2px solid var(--hit);
      background:rgba(0,229,255,.14);pointer-events:none;display:none}
 .dot{position:absolute;width:10px;height:10px;margin:-5px 0 0 -5px;
      border-radius:50%;background:var(--hit);pointer-events:none}
 footer{display:flex;gap:10px;justify-content:center;align-items:center;
         padding:0 16px 18px;flex-wrap:wrap}
 button{font:inherit;padding:10px 18px;border-radius:8px;border:1px solid #2c3037;
        background:#1b1e24;color:var(--fg);cursor:pointer}
 button:hover{background:#242830}
 button.primary{background:#14342a;border-color:#1d5c46;color:#9bf3cf}
 .hint{color:var(--dim);text-align:center;padding:0 16px 10px}
 kbd{background:#23262b;border:1px solid #33373e;border-bottom-width:2px;
     border-radius:5px;padding:1px 7px;font:13px ui-monospace,monospace}
 .done{text-align:center;padding:40px}
</style></head><body>
<header>
  <strong>Goal labelling</strong>
  <div id="bar"><div id="fill"></div></div>
  <span class="count" id="count"></span>
  <button id="save">Download labels</button>
</header>
<main><div id="stage">
  <img id="shot" alt="">
  <div id="box"></div>
</div></main>
<p class="hint" id="hint"></p>
<footer>
  <button id="none">No goal visible &nbsp;<kbd>N</kbd></button>
  <button id="undo">Undo &nbsp;<kbd>U</kbd></button>
  <button id="skip">Skip &nbsp;<kbd>S</kbd></button>
</footer>
<script>
const FRAMES = __DATA__;
const KEY = "goal-labels-v1";
let at = 0, first = null, labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}

const img = document.getElementById("shot"), box = document.getElementById("box");
const stage = document.getElementById("stage"), hint = document.getElementById("hint");

function firstUnlabelled(){
  for (let i=0;i<FRAMES.length;i++) if (labels[FRAMES[i].id] === undefined) return i;
  return FRAMES.length;
}
function show(){
  clearDots();
  if (at >= FRAMES.length){ finish(); return; }
  const f = FRAMES[at];
  img.src = "data:image/jpeg;base64," + f.jpeg;
  box.style.display = "none"; first = null;
  const n = Object.keys(labels).length;
  document.getElementById("count").textContent = n + " / " + FRAMES.length;
  document.getElementById("fill").style.width = (100*n/FRAMES.length) + "%";
  hint.innerHTML = "Click <b>one corner</b> of the goal, then the <b>opposite "
    + "corner</b>. Include the posts and crossbar, not the net behind. "
    + "If no goal is visible press <kbd>N</kbd>.";
}
function clearDots(){ stage.querySelectorAll(".dot").forEach(d=>d.remove()); }
function record(value){
  labels[FRAMES[at].id] = value;
  localStorage.setItem(KEY, JSON.stringify(labels));
  at++; show();
}
stage.addEventListener("click", e => {
  if (at >= FRAMES.length) return;
  const r = img.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
  if (x<0||x>1||y<0||y>1) return;
  const dot = document.createElement("div");
  dot.className = "dot";
  dot.style.left = (x*100) + "%"; dot.style.top = (y*100) + "%";
  stage.appendChild(dot);
  if (first === null){ first = [x,y]; return; }
  const x0=Math.min(first[0],x), x1=Math.max(first[0],x);
  const y0=Math.min(first[1],y), y1=Math.max(first[1],y);
  box.style.display="block";
  box.style.left=(x0*100)+"%"; box.style.top=(y0*100)+"%";
  box.style.width=((x1-x0)*100)+"%"; box.style.height=((y1-y0)*100)+"%";
  setTimeout(()=>record({goal:[+x0.toFixed(4),+y0.toFixed(4),
                               +x1.toFixed(4),+y1.toFixed(4)]}), 140);
});
document.getElementById("none").onclick = ()=> at<FRAMES.length && record({goal:null});
document.getElementById("skip").onclick = ()=>{ at++; show(); };
document.getElementById("undo").onclick = ()=>{
  if (first !== null){ first=null; clearDots(); box.style.display="none"; return; }
  at = Math.max(0, at-1); delete labels[FRAMES[at].id];
  localStorage.setItem(KEY, JSON.stringify(labels)); show();
};
document.addEventListener("keydown", e=>{
  const k = e.key.toLowerCase();
  if (k==="n") document.getElementById("none").click();
  else if (k==="u") document.getElementById("undo").click();
  else if (k==="s") document.getElementById("skip").click();
});
document.getElementById("save").onclick = ()=>{
  const out = FRAMES.map(f => ({id:f.id, clip:f.clip, frame:f.frame,
                                label: labels[f.id] === undefined ? null
                                      : labels[f.id]}));
  const blob = new Blob([JSON.stringify({width:__W__, height:__H__,
                                         labelled:Object.keys(labels).length,
                                         frames:out}, null, 1)],
                        {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "goal_labels.json"; a.click();
};
function finish(){
  document.querySelector("main").innerHTML =
    "<div class='done'><h2>All " + FRAMES.length + " done — thank you.</h2>"
    + "<p>Press <b>Download labels</b> above and send me the file.</p></div>";
  hint.textContent = ""; document.querySelector("footer").style.display = "none";
}
at = firstUnlabelled(); show();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="goal_labeller.html")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every window's frame count")
    args = ap.parse_args()

    frames = []
    for name, out_dir, count in WINDOWS:
        want = max(1, int(round(count * args.scale)))
        got = sample_frames(out_dir, want)
        for shot in got:
            shot["clip"] = name
            shot["id"] = f"{name}:{shot['frame']}"
        frames += got
        print(f"  {name:>18s} {len(got):3d} frames")

    if not frames:
        raise SystemExit("no clips found -- run the pipeline first")

    page = (PAGE.replace("__DATA__", json.dumps(frames))
                .replace("__W__", str(SHOW_W)).replace("__H__", str(SHOW_H)))
    target = Path(args.out)
    target.write_text(page, encoding="utf-8")
    size = target.stat().st_size / 1e6
    print(f"\n  {len(frames)} frames -> {target} ({size:.1f} MB)")
    if size > 29:
        print("  TOO BIG to upload; lower --scale")


if __name__ == "__main__":
    main()
