"""Build a one-file page for clicking the centre circle on midfield frames.

Built when the constrained midfield fit was passing almost no frames -- 0 of
80 circles on stoke_7001 -- to find out whether the circle detector or the
geometry was at fault. The geometry was: the camera position the fit held
fixed was 25 m out, and locating it properly (`src/camera_position.py`)
brought 21-29 of 30 circle frames per clip into agreement. No labels were
needed for that.

It is kept because it is still the way to measure the circle detector
itself: its hits against hand-clicked arcs, and its false positives, which
nothing else in the pipeline counts.

## What is clicked

    1  centre spot
    2  where the circle crosses the halfway line, left on screen
    3  where the circle crosses the halfway line, right on screen
    then a few points anywhere on the circle's line, Enter when done

The first three are ordered and named, so they arrive identified and are
exact correspondences on the ground. `space` skips one that is hidden or
out of frame. The arc points are unordered: the fit scores each by its
distance to the projected circle, as it does the detector's pixels, so
where on the line they sit does not matter -- only that they are on it.
Spread round the visible arc they pin the ellipse's shape, which is what
carries the tilt.

`N` marks a frame with no centre circle, which is itself a label: the
detector's false positives.

## Which frames

Frames where the detector found a circle on the clips that have goal
corners (without corners there is no camera position to fit with), spread
evenly through each clip. Detector hits only, so its misses are not
measured here.

## Why the page is not published

Same as the corner page: SoccerNet frames under an NDA. Handed to the one
person who already has the data, as a file. The generator is committed;
what it makes is not.

    python make_circle_labeller.py --per-clip 25
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

import detect_shots as ds
from probe_centre_circle import find_circle
from propagate_goal_labels import DIRS

# Clips with goal corners, which is what gives the camera position.
CLIPS = ["stoke_1302", "stoke_4207", "stoke_7001",
         "reading_0737", "reading_1155", "reading_2519"]

# The pipeline samples every fourth frame; label from the same set.
STEP = 4

# Full frames, shown about 1000 px wide. The arc is large; the centre spot
# is not, which is what the loupe is for.
JPEG_QUALITY = 82

ORDERED = ["centre spot", "circle meets halfway, left",
           "circle meets halfway, right"]
MAX_ARC = 8

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Centre circle</title>
<style>
 :root{--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--hit:#00e5ff;--arc:#ffab40;
       --ok:#00c853}
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
 #stage{position:relative;line-height:0;cursor:crosshair;max-width:1000px}
 #shot{width:100%;height:auto;border-radius:6px;display:block}
 .dot{position:absolute;width:10px;height:10px;margin:-5px 0 0 -5px;
      border-radius:50%;background:var(--hit);border:2px solid #04252b;
      pointer-events:none}
 .dot.arc{background:var(--arc);border-color:#3a2400}
 .dot span{position:absolute;left:12px;top:-4px;font:12px/1 ui-monospace,
           monospace;color:var(--hit)}
 #loupe{position:absolute;width:150px;height:150px;border-radius:50%;
        border:2px solid var(--hit);pointer-events:none;display:none;
        background-repeat:no-repeat;box-shadow:0 2px 10px #000}
 #loupe:after{content:"";position:absolute;left:74px;top:74px;width:2px;
        height:2px;background:#ff1744;box-shadow:0 0 0 1px #fff}
 aside{width:240px;color:var(--dim);font-size:14px}
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
 .hint{color:var(--dim);text-align:center;padding:0 16px 6px;font-size:14px;
       max-width:1000px;margin:0 auto}
 kbd{background:#23262b;border:1px solid #33373e;border-bottom-width:2px;
     border-radius:5px;padding:1px 7px;font:13px ui-monospace,monospace}
 .done{text-align:center;padding:40px}
 @media (max-width:700px){aside{width:100%}}
</style></head><body>
<header>
  <strong>Centre circle</strong>
  <div id="bar"><div id="fill"></div></div>
  <span class="count" id="count"></span>
  <button id="save">Download labels</button>
</header>
<p id="ask"></p>
<main>
 <div id="stage"><img id="shot" alt=""><div id="loupe"></div></div>
 <aside>
  <svg viewBox="0 0 200 120">
   <rect x="0" y="0" width="200" height="120" fill="#1b4d2a" rx="6"/>
   <line x1="100" y1="5" x2="100" y2="115" stroke="#e8eaed" stroke-width="2"/>
   <ellipse cx="100" cy="60" rx="70" ry="35" fill="none" stroke="#e8eaed"
            stroke-width="2"/>
   <circle cx="100" cy="60" r="7" fill="#00e5ff"/>
   <circle cx="100" cy="25" r="7" fill="#00e5ff"/>
   <circle cx="100" cy="95" r="7" fill="#00e5ff"/>
   <circle cx="38" cy="44" r="5" fill="#ffab40"/>
   <circle cx="160" cy="44" r="5" fill="#ffab40"/>
   <circle cx="55" cy="87" r="5" fill="#ffab40"/>
   <circle cx="150" cy="84" r="5" fill="#ffab40"/>
   <text x="100" y="64" font-size="10" text-anchor="middle" fill="#04252b"
         font-family="monospace">1</text>
   <text x="100" y="29" font-size="10" text-anchor="middle" fill="#04252b"
         font-family="monospace">2</text>
   <text x="100" y="99" font-size="10" text-anchor="middle" fill="#04252b"
         font-family="monospace">3</text>
  </svg>
  <div style="font-size:12px;margin-bottom:6px">The halfway line can run
   any way across the screen. 2 and 3 are its two crossings with the circle,
   whichever is further left on screen first.</div>
  <div class="step" id="s0">1 &nbsp;centre spot</div>
  <div class="step" id="s1">2 &nbsp;circle meets halfway, left</div>
  <div class="step" id="s2">3 &nbsp;circle meets halfway, right</div>
  <div class="step" id="s3">then 3&ndash;8 points on the circle,
   <kbd>Enter</kbd></div>
 </aside>
</main>
<p class="hint">Click the <b>middle</b> of the painted line. The magnifier
 follows the mouse. <kbd>space</kbd> skips a numbered point that is hidden
 or off screen. For the orange points, click anywhere on the circle's line,
 spread round what is visible, then <kbd>Enter</kbd>. <kbd>N</kbd> if there
 is no centre circle in the picture (penalty arc, replay, close-up&hellip;).</p>
<footer>
  <button id="none">No centre circle &nbsp;<kbd>N</kbd></button>
  <button id="undo">Undo &nbsp;<kbd>U</kbd></button>
  <button id="skipc">Point not visible &nbsp;<kbd>space</kbd></button>
  <button id="next">Done with this frame &nbsp;<kbd>Enter</kbd></button>
</footer>
<script>
const FRAMES = __DATA__;
const KEY = "centre-circle-v1";
const NAMES = __ORDERED__;
const MAX_ARC = __MAX_ARC__, MIN_ARC = 3;
let at = 0, pts = [], arc = [], done = {};
try { done = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
function store(){ try { localStorage.setItem(KEY, JSON.stringify(done)); }
                  catch (e) {} }

const img = document.getElementById("shot");
const stage = document.getElementById("stage");
const ask = document.getElementById("ask");
const loupe = document.getElementById("loupe");
const ZOOM = 4;

function firstUnlabelled(){
  for (let i=0;i<FRAMES.length;i++) if (done[FRAMES[i].id] === undefined) return i;
  return FRAMES.length;
}
function redraw(){
  stage.querySelectorAll(".dot").forEach(d=>d.remove());
  pts.forEach((p,i)=> p && addDot(p[0],p[1],i+1,false));
  arc.forEach(p=> addDot(p[0],p[1],"",true));
}
function paint(){
  const n = Object.keys(done).length;
  document.getElementById("count").textContent = n + " / " + FRAMES.length;
  document.getElementById("fill").style.width = (100*n/FRAMES.length) + "%";
  for (let i=0;i<4;i++){
    const el = document.getElementById("s"+i);
    if (i<3) el.className = "step" + (i===pts.length ? " now"
                  : i<pts.length ? (pts[i] ? " got" : " gone") : "");
    else el.className = "step" + (pts.length===3 ? " now" : "");
  }
  if (pts.length < 3) ask.innerHTML = "Click the <b>" + NAMES[pts.length] + "</b>";
  else ask.innerHTML = "Click <b>points on the circle</b> (" + arc.length
       + " so far" + (arc.length < MIN_ARC ? ", at least " + MIN_ARC
       : "") + "), then <kbd>Enter</kbd>";
}
function show(){
  pts = []; arc = [];
  if (at >= FRAMES.length){ finish(); return; }
  img.src = "data:image/jpeg;base64," + FRAMES[at].jpeg;
  loupe.style.backgroundImage = "url(" + img.src + ")";
  redraw(); paint();
}
function addDot(u,v,label,isArc){
  const d = document.createElement("div");
  d.className = "dot" + (isArc ? " arc" : "");
  d.style.left=(u*100)+"%"; d.style.top=(v*100)+"%";
  if (label !== "") d.innerHTML = "<span>"+label+"</span>";
  stage.appendChild(d);
}
function record(){
  const f = FRAMES[at];
  const r = p => p === null ? null : [+p[0].toFixed(5), +p[1].toFixed(5)];
  done[f.id] = {spot:r(pts[0]), left:r(pts[1]), right:r(pts[2]),
                arc:arc.map(r)};
  store(); at++; show();
}
stage.addEventListener("click", e => {
  if (at >= FRAMES.length) return;
  const b = img.getBoundingClientRect();
  const u = (e.clientX - b.left)/b.width, v = (e.clientY - b.top)/b.height;
  if (u<0||u>1||v<0||v>1) return;
  if (pts.length < 3) pts.push([u,v]);
  else if (arc.length < MAX_ARC) arc.push([u,v]);
  redraw(); paint();
});
stage.addEventListener("mousemove", e => {
  const b = img.getBoundingClientRect();
  const x = e.clientX - b.left, y = e.clientY - b.top;
  if (x<0||y<0||x>b.width||y>b.height){ loupe.style.display="none"; return; }
  loupe.style.display = "block";
  loupe.style.left = (x + 20) + "px"; loupe.style.top = (y - 170) + "px";
  loupe.style.backgroundSize = (b.width*ZOOM) + "px " + (b.height*ZOOM) + "px";
  loupe.style.backgroundPosition = (75 - x*ZOOM) + "px " + (75 - y*ZOOM) + "px";
});
stage.addEventListener("mouseleave", ()=> loupe.style.display = "none");
document.getElementById("skipc").onclick = ()=>{
  if (at < FRAMES.length && pts.length < 3){ pts.push(null); redraw(); paint(); }
};
document.getElementById("next").onclick = ()=>{
  if (at >= FRAMES.length) return;
  while (pts.length < 3) pts.push(null);
  if (arc.length < MIN_ARC && !confirm("Fewer than " + MIN_ARC
      + " circle points -- save anyway?")) { paint(); return; }
  record();
};
document.getElementById("none").onclick = ()=>{
  if (at >= FRAMES.length) return;
  done[FRAMES[at].id] = {none:true}; store(); at++; show();
};
document.getElementById("undo").onclick = ()=>{
  if (arc.length){ arc.pop(); redraw(); paint(); return; }
  if (pts.length){ pts.pop(); redraw(); paint(); return; }
  at = Math.max(0, at-1);
  delete done[FRAMES[at].id]; store(); show();
};
document.addEventListener("keydown", e=>{
  if (e.key === " "){ e.preventDefault();
    document.getElementById("skipc").click(); return; }
  if (e.key === "Enter"){ e.preventDefault();
    document.getElementById("next").click(); return; }
  const k = e.key.toLowerCase();
  if (k==="n") document.getElementById("none").click();
  else if (k==="u") document.getElementById("undo").click();
});
document.getElementById("save").onclick = ()=>{
  // Points are normalised to the full frame; width and height travel with
  // every frame, since the clips are not all one size.
  const out = FRAMES.map(f => Object.assign(
    {id:f.id, clip:f.clip, frame:f.frame, width:f.fw, height:f.fh},
    done[f.id] === undefined ? {unlabelled:true} : done[f.id]));
  const blob = new Blob([JSON.stringify(
    {ordered:NAMES, labelled:Object.keys(done).length, frames:out}, null, 1)],
    {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "circle_labels.json";
  a.click();
};
function finish(){
  document.querySelector("main").innerHTML =
    "<div class='done'><h2>All " + FRAMES.length + " done &mdash; thank you.</h2>"
    + "<p>Press <b>Download labels</b> above and send me the file.</p></div>";
  ask.textContent = "";
  document.querySelector("footer").style.display = "none";
  document.querySelector(".hint").style.display = "none";
}
at = firstUnlabelled(); show();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-clip", type=int, default=25,
                    help="circle frames to take from each clip, at most")
    ap.add_argument("--out", default="circle_labeller.html")
    args = ap.parse_args()

    frames = []
    for clip in CLIPS:
        out_dir = Path(DIRS[clip])
        if not (out_dir / "clip.json").exists():
            print(f"  {clip}: no clip.json, skipped")
            continue
        info = json.loads((out_dir / "clip.json").read_text())
        ball = ds.ball_track(out_dir)
        wanted = sorted({int(f) - int(f) % STEP for f in ball.frame.tolist()})
        cap = cv2.VideoCapture(info["path"])
        hits = []
        for idx in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            if find_circle(frame, np.random.default_rng(idx)) is not None:
                hits.append((idx, frame))
        cap.release()
        if not hits:
            print(f"  {clip}: no circles found")
            continue
        take = np.unique(np.linspace(0, len(hits) - 1,
                                     min(args.per_clip, len(hits)))
                         .round().astype(int))
        for i in take:
            idx, frame = hits[i]
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue
            frames.append({"id": f"{clip}:{idx}", "clip": clip, "frame": idx,
                           "fw": int(info["width"]), "fh": int(info["height"]),
                           "jpeg": base64.b64encode(buf).decode("ascii")})
        print(f"  {clip}: circle on {len(hits)} of {len(wanted)} sampled "
              f"frames, {len(take)} taken")

    if not frames:
        raise SystemExit("no circle frames found")
    print(f"  {len(frames)} frames, about {len(frames) * 7} clicks")

    page = (PAGE.replace("__DATA__", json.dumps(frames))
                .replace("__ORDERED__", json.dumps(ORDERED))
                .replace("__MAX_ARC__", str(MAX_ARC)))
    target = Path(args.out)
    target.write_text(page, encoding="utf-8")
    size = target.stat().st_size / 1e6
    print(f"\n  -> {target} ({size:.1f} MB)")
    if size > 29:
        print("  TOO BIG to upload")


if __name__ == "__main__":
    main()
