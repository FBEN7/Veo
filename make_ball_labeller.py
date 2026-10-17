"""Build a one-file page for clicking the ball, around shots, goals and outs.

The ball classifier (`src/ball_classifier.py`) learned that a ball is on
grass: trained on public labels whose balls almost all are, it scores
reading_1155's ball 1.0 at the strike and 0.0 in the top corner of the net.
What it lacks are balls against the net, the stands and the hoardings --
exactly where shots, goals and balls going out happen.

This page collects those. Frames are taken every `STEP` frames from `BEFORE`
seconds before to `AFTER` seconds after every labelled shot, goal and
out-of-play crossing, on the clips that have labels. On each, click the
centre of the ball, or press N where it cannot be seen. One click is a
positive; every detector candidate on that frame away from the click
becomes a look-alike, so a frame yields several examples.

## Why the page is not published

The frames are SoccerNet footage under an agreement that forbids passing
them on. The page is handed to the one person who already has the data, as
a file, never to a URL. The generator is committed; what it makes is not.

    python make_ball_labeller.py
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import cv2

from src.paths import DATA_DIR

CLIPS = ["stoke_1302", "stoke_4207", "stoke_7001",
         "reading_0737", "reading_1155", "reading_2519"]
EVENTS = ("shot", "goal", "out")

# Around each event: a second before, through the strike, to two seconds
# after, which covers the ball reaching the line or the net.
BEFORE, AFTER = 1.0, 2.0
STEP = 5

# Full frames, shown about 1000 px wide; quality chosen to keep about 200
# frames under the 29 MB upload limit.
JPEG_QUALITY = 72

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Where is the ball?</title>
<style>
 :root{--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--hit:#00e5ff;--ok:#00c853}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
 header{display:flex;gap:18px;align-items:center;padding:10px 16px;
        background:#0d0f12;border-bottom:1px solid #23262b;position:sticky;
        top:0;z-index:5}
 #bar{flex:1;height:8px;background:#23262b;border-radius:4px;overflow:hidden}
 #fill{height:100%;width:0;background:var(--ok);transition:width .15s}
 .count{font-variant-numeric:tabular-nums;color:var(--dim)}
 main{display:flex;justify-content:center;padding:14px}
 #stage{position:relative;line-height:0;cursor:crosshair;max-width:1000px}
 #shot{width:100%;height:auto;border-radius:6px;display:block}
 #loupe{position:absolute;width:170px;height:170px;border-radius:50%;
        border:2px solid var(--hit);pointer-events:none;display:none;
        background-repeat:no-repeat;box-shadow:0 2px 10px #000}
 #loupe:after{content:"";position:absolute;left:84px;top:84px;width:2px;
        height:2px;background:#ff1744;box-shadow:0 0 0 1px #fff}
 #ask{text-align:center;font-size:19px;padding:6px 16px 0}
 #ask b{color:var(--hit)}
 #where{color:var(--dim);font-size:13px}
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
</style></head><body>
<header>
  <strong>Where is the ball?</strong>
  <div id="bar"><div id="fill"></div></div>
  <span class="count" id="count"></span>
  <button id="save">Download labels</button>
</header>
<p id="ask">Click the <b>centre of the ball</b> <span id="where"></span></p>
<main><div id="stage"><img id="shot" alt=""><div id="loupe"></div></div></main>
<p class="hint">The magnifier follows the mouse. The ball counts wherever it
 is: on the grass, in the air, against the net, the stands or the
 hoardings. <kbd>N</kbd> if it cannot be seen (hidden, out of frame, or you
 are not sure). <kbd>U</kbd> goes back one frame.</p>
<footer>
  <button id="none">Ball not visible &nbsp;<kbd>N</kbd></button>
  <button id="undo">Back &nbsp;<kbd>U</kbd></button>
</footer>
<script>
const FRAMES = __DATA__;
const KEY = "ball-click-v1";
let at = 0, done = {};
try { done = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
function store(){ try { localStorage.setItem(KEY, JSON.stringify(done)); }
                  catch (e) {} }
const img = document.getElementById("shot");
const stage = document.getElementById("stage");
const loupe = document.getElementById("loupe");
const ZOOM = 5;
function firstUnlabelled(){
  for (let i=0;i<FRAMES.length;i++) if (done[FRAMES[i].id] === undefined) return i;
  return FRAMES.length;
}
function paint(){
  const n = Object.keys(done).length;
  document.getElementById("count").textContent = n + " / " + FRAMES.length;
  document.getElementById("fill").style.width = (100*n/FRAMES.length) + "%";
}
function show(){
  if (at >= FRAMES.length){ finish(); return; }
  const f = FRAMES[at];
  img.src = "data:image/jpeg;base64," + f.jpeg;
  loupe.style.backgroundImage = "url(" + img.src + ")";
  document.getElementById("where").textContent =
    "(" + f.clip + ", " + f.near + ")";
  paint();
}
function record(value){
  done[FRAMES[at].id] = value; store(); at++; show();
}
stage.addEventListener("click", e => {
  if (at >= FRAMES.length) return;
  const b = img.getBoundingClientRect();
  const u = (e.clientX - b.left)/b.width, v = (e.clientY - b.top)/b.height;
  if (u<0||u>1||v<0||v>1) return;
  record({ball:[+u.toFixed(5), +v.toFixed(5)]});
});
stage.addEventListener("mousemove", e => {
  const b = img.getBoundingClientRect();
  const x = e.clientX - b.left, y = e.clientY - b.top;
  if (x<0||y<0||x>b.width||y>b.height){ loupe.style.display="none"; return; }
  loupe.style.display = "block";
  loupe.style.left = (x + 20) + "px"; loupe.style.top = (y - 190) + "px";
  loupe.style.backgroundSize = (b.width*ZOOM) + "px " + (b.height*ZOOM) + "px";
  loupe.style.backgroundPosition = (85 - x*ZOOM) + "px " + (85 - y*ZOOM) + "px";
});
stage.addEventListener("mouseleave", ()=> loupe.style.display = "none");
document.getElementById("none").onclick = ()=>{
  if (at < FRAMES.length) record({ball:null});
};
document.getElementById("undo").onclick = ()=>{
  at = Math.max(0, at-1); delete done[FRAMES[at].id]; store(); show();
};
document.addEventListener("keydown", e=>{
  const k = e.key.toLowerCase();
  if (k==="n") document.getElementById("none").click();
  else if (k==="u") document.getElementById("undo").click();
});
document.getElementById("save").onclick = ()=>{
  // Normalised to the full frame; width and height travel with each frame.
  const out = FRAMES.map(f => Object.assign(
    {id:f.id, clip:f.clip, frame:f.frame, width:f.fw, height:f.fh},
    done[f.id] === undefined ? {unlabelled:true} : done[f.id]));
  const blob = new Blob([JSON.stringify(
    {labelled:Object.keys(done).length, frames:out}, null, 1)],
    {type:"application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "ball_labels.json";
  a.click();
};
function finish(){
  document.querySelector("main").innerHTML =
    "<div class='done'><h2>All " + FRAMES.length + " done &mdash; thank you.</h2>"
    + "<p>Press <b>Download labels</b> above and send me the file.</p></div>";
  document.getElementById("ask").textContent = "";
  document.querySelector("footer").style.display = "none";
  document.querySelector(".hint").style.display = "none";
}
at = firstUnlabelled(); show();
</script></body></html>
"""


def labelled_events(path: Path):
    """(seconds, kind) for every shot, goal and out in a label file."""
    out = []
    for line in path.read_text().splitlines():
        m = re.match(r"(\d+):(\d+)\s+(\w+)", line.strip())
        if m and m[3].lower() in EVENTS:
            out.append((int(m[1]) * 60 + int(m[2]), m[3].lower()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--out", default="ball_labeller.html")
    args = ap.parse_args()

    frames = []
    for clip in CLIPS:
        labels = sorted(DATA_DIR.glob(f"*-{clip}_2shots.txt"))
        info_path = Path(f"output_{clip}") / "clip.json"
        if not labels or not info_path.exists():
            print(f"  {clip}: no labels or no clip.json, skipped")
            continue
        info = json.loads(info_path.read_text())
        fps = float(info["fps"])
        wanted = {}
        for when, kind in labelled_events(labels[0]):
            first = max(0, int((when - BEFORE) * fps))
            last = min(int(info["n_frames"]) - 1, int((when + AFTER) * fps))
            for f in range(first, last + 1, args.step):
                wanted.setdefault(f, f"{kind} at {when // 60}:{when % 60:02d}")
        cap = cv2.VideoCapture(info["path"])
        for f in sorted(wanted):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, image = cap.read()
            if not ok:
                continue
            ok, buf = cv2.imencode(".jpg", image,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                frames.append({"id": f"{clip}:{f}", "clip": clip, "frame": f,
                               "near": wanted[f], "fw": int(info["width"]),
                               "fh": int(info["height"]),
                               "jpeg": base64.b64encode(buf).decode("ascii")})
        cap.release()
        print(f"  {clip}: {len(wanted)} frames around "
              f"{len(labelled_events(labels[0]))} labelled events")

    page = PAGE.replace("__DATA__", json.dumps(frames))
    target = Path(args.out)
    target.write_text(page, encoding="utf-8")
    size = target.stat().st_size / 1e6
    print(f"\n  {len(frames)} frames -> {target} ({size:.1f} MB)")
    if size > 29:
        print("  TOO BIG to upload: raise --step")


if __name__ == "__main__":
    main()
