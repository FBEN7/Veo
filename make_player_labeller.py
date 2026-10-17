"""A page for clicking who did each event: the ground truth for player identity.

Every event the pipeline emits names a player -- `player_track_id`, the
track nearest the ball -- and nothing has checked it. Tracks are not people
either: after re-joining fragments a 60-90 s window still has 96-138 player
tracks for about 25 people. Identity is built in three steps (an anonymous
person per set of tracks, a shirt number wherever one can be read, then the
squad list), and each needs ground truth this page collects:

- **who did it**: the person clicks the player who played the ball in the
  event (pass, carry, recovery, tackle, shot), so attribution can be
  scored against the click;
- **their number**, typed when it can be read on this frame or the two
  around it, so that two events by the same number say "same person"
  whatever the tracks say -- the test for the anonymous identities;
- **whether the event happened at all**, a by-product: `E` marks a
  detection with no such event.

The pipeline's own answer is not shown, so the clicks are not led by it.
The ball the pipeline tracked is ringed on the frame, to point at the
moment; it is a guess and may be wrong.

Events are sampled per clip and per type; every hand-labelled shot is
included. The frames are under the data agreement, so the page goes to the
person who has the data, as a file, and is never published.

    python make_player_labeller.py --out player_labeller.html
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from make_ball_labeller import CLIPS

PER_CLIP = 24
# Context frames either side of the event, in seconds.
CONTEXT_S = 0.4
MAIN_W, SIDE_W = 1280, 400
MAIN_QUALITY, SIDE_QUALITY = 60, 55


def jpeg(image, width, quality) -> str:
    h, w = image.shape[:2]
    if w != width:
        image = cv2.resize(image, (width, int(h * width / w)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def sample(clip: str, rng) -> list[dict]:
    """Events of one clip, spread over types, plus every labelled shot."""
    from score_hand_labels import read_labels
    from src.paths import DATA_DIR

    out_dir = Path(f"output_{clip}")
    events = json.loads((out_dir / "events.json").read_text())
    by_type = {}
    for e in events:
        by_type.setdefault(e["event_type"], []).append(e)
    chosen, share = [], PER_CLIP
    # Rare types in full, the rest in proportion to what is left.
    for kind, group in sorted(by_type.items(), key=lambda kv: len(kv[1])):
        take = min(len(group), max(1, share // max(len(by_type), 1)))
        if kind in ("tackle", "recovery"):
            take = min(len(group), max(take, 6))
        picks = rng.choice(len(group), size=take, replace=False)
        chosen += [group[i] for i in sorted(picks)]
    items = [{"clip": clip, "kind": e["event_type"],
              "time_s": float(e["timestamp_s"])} for e in chosen]
    labels = sorted(Path(DATA_DIR).glob(f"*-{clip}_2shots.txt"))
    if labels:
        truth, _, _ = read_labels(labels[0])
        items += [{"clip": clip, "kind": e["event_type"],
                   "time_s": float(e["time_s"])} for e in truth
                  if e["event_type"] in ("shot", "goal")]
    return items


def frames_for(items) -> list[dict]:
    from detect_shots import ball_track

    out = []
    for clip in sorted({i["clip"] for i in items}):
        out_dir = Path(f"output_{clip}")
        info = json.loads((out_dir / "clip.json").read_text())
        fps = float(info["fps"])
        ball = ball_track(out_dir, ball_detector="fill")
        at = {int(r.frame): (r.px, r.py) for r in ball.itertuples()}
        cap = cv2.VideoCapture(info["path"])
        for item in sorted((i for i in items if i["clip"] == clip),
                           key=lambda i: i["time_s"]):
            frame = int(round(item["time_s"] * fps))
            shots = {}
            for name, f in (("before", frame - int(CONTEXT_S * fps)),
                            ("main", frame),
                            ("after", frame + int(CONTEXT_S * fps))):
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(f, 0))
                ok, image = cap.read()
                if not ok:
                    continue
                if f in at:
                    cv2.circle(image, (int(at[f][0]), int(at[f][1])), 18,
                               (0, 230, 255), 2)
                shots[name] = jpeg(image, MAIN_W if name == "main" else
                                   SIDE_W, MAIN_QUALITY if name == "main"
                                   else SIDE_QUALITY)
            if "main" not in shots:
                continue
            out.append({"id": f"{clip}:{frame}:{item['kind']}", "clip": clip,
                        "frame": frame, "kind": item["kind"],
                        "time": f"{item['time_s']:.1f} s", **shots})
        cap.release()
    return out


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Who played the ball?</title>
<style>
 :root{--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--hit:#00e5ff;--ok:#00c853;
       --mark:#ff1744}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
 header{display:flex;gap:18px;align-items:center;padding:10px 16px;
        background:#0d0f12;border-bottom:1px solid #23262b;position:sticky;
        top:0;z-index:5}
 #bar{flex:1;height:8px;background:#23262b;border-radius:4px;overflow:hidden}
 #fill{height:100%;width:0;background:var(--ok);transition:width .15s}
 .count{font-variant-numeric:tabular-nums;color:var(--dim)}
 #ask{text-align:center;font-size:19px;padding:8px 16px 0;margin:0}
 #ask b{color:var(--hit)}
 #where{color:var(--dim);font-size:13px}
 main{display:flex;flex-direction:column;align-items:center;gap:8px;
      padding:10px 16px}
 #stage{position:relative;line-height:0;cursor:crosshair;max-width:1100px;
        width:100%}
 #shot{width:100%;height:auto;border-radius:6px;display:block}
 #dot{position:absolute;width:14px;height:14px;margin:-7px 0 0 -7px;
      border:2px solid var(--mark);border-radius:50%;display:none;
      pointer-events:none}
 #loupe{position:absolute;width:190px;height:190px;border-radius:50%;
        border:2px solid var(--hit);pointer-events:none;display:none;
        background-repeat:no-repeat;box-shadow:0 2px 10px #000}
 .side{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
 .side figure{margin:0;text-align:center;color:var(--dim);font-size:12px}
 .side img{width:min(400px,44vw);border-radius:4px;display:block}
 #numrow{display:none;gap:10px;align-items:center;justify-content:center;
         padding:6px}
 #num{font:inherit;width:90px;padding:8px;border-radius:8px;
      border:1px solid #2c3037;background:#0d0f12;color:var(--fg)}
 footer{display:flex;gap:10px;justify-content:center;align-items:center;
        padding:6px 16px 20px;flex-wrap:wrap}
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
  <strong>Who played the ball?</strong>
  <div id="bar"><div id="fill"></div></div>
  <span class="count" id="count"></span>
  <button id="save">Download labels</button>
</header>
<p id="ask">Click the player who played the ball in this <b id="kind"></b>
 <span id="where"></span></p>
<main>
  <div id="stage"><img id="shot" alt=""><div id="dot"></div>
    <div id="loupe"></div></div>
  <div id="numrow">Shirt number, if you can read it:
    <input id="num" inputmode="numeric" autocomplete="off">
    <button id="ok">Next &nbsp;<kbd>Enter</kbd></button></div>
  <div class="side">
    <figure><img id="before" alt=""><figcaption>0.4 s before</figcaption></figure>
    <figure><img id="after" alt=""><figcaption>0.4 s after</figcaption></figure>
  </div>
</main>
<p class="hint">The yellow ring is the system's guess of the ball and may be
 wrong. Click the feet of the player who passed, carried, won or shot the
 ball at this moment, then type the shirt number if it can be read here or
 on the small frames (leave it empty otherwise) and press <kbd>Enter</kbd>.
 <kbd>E</kbd> if no such event happens here; <kbd>N</kbd> if you cannot
 tell who it is; <kbd>U</kbd> goes back.</p>
<footer>
  <button id="noevent">No such event &nbsp;<kbd>E</kbd></button>
  <button id="none">Can't tell &nbsp;<kbd>N</kbd></button>
  <button id="undo">Back &nbsp;<kbd>U</kbd></button>
</footer>
<script>
const ITEMS = __DATA__;
const KEY = "player-click-v1";
let at = 0, done = {}, pending = null;
try { done = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
function store(){ try { localStorage.setItem(KEY, JSON.stringify(done)); }
                  catch (e) {} }
const $ = id => document.getElementById(id);
const img = $("shot"), stage = $("stage"), loupe = $("loupe"), dot = $("dot");
const ZOOM = 4;
function firstUnlabelled(){
  for (let i=0;i<ITEMS.length;i++) if (done[ITEMS[i].id] === undefined) return i;
  return ITEMS.length;
}
function paint(){
  const n = Object.keys(done).length;
  $("count").textContent = n + " / " + ITEMS.length;
  $("fill").style.width = (100*n/ITEMS.length) + "%";
}
function show(){
  pending = null; dot.style.display = "none"; $("numrow").style.display = "none";
  $("num").value = "";
  if (at >= ITEMS.length){ finish(); return; }
  const f = ITEMS[at];
  img.src = "data:image/jpeg;base64," + f.main;
  loupe.style.backgroundImage = "url(" + img.src + ")";
  $("before").src = f.before ? "data:image/jpeg;base64," + f.before : "";
  $("after").src = f.after ? "data:image/jpeg;base64," + f.after : "";
  $("kind").textContent = f.kind;
  $("where").textContent = "(" + f.clip + ", " + f.time + ")";
  paint();
}
function record(value){ done[ITEMS[at].id] = value; store(); at++; show(); }
stage.addEventListener("click", e => {
  if (at >= ITEMS.length) return;
  const b = img.getBoundingClientRect();
  pending = {x: +((e.clientX-b.left)/b.width).toFixed(5),
             y: +((e.clientY-b.top)/b.height).toFixed(5)};
  dot.style.left = (e.clientX-b.left) + "px";
  dot.style.top = (e.clientY-b.top) + "px";
  dot.style.display = "block";
  $("numrow").style.display = "flex";
  $("num").focus();
});
stage.addEventListener("mousemove", e => {
  const b = img.getBoundingClientRect();
  const x = e.clientX-b.left, y = e.clientY-b.top;
  loupe.style.display = "block";
  loupe.style.left = (x-95) + "px"; loupe.style.top = (y-95) + "px";
  loupe.style.backgroundSize = (b.width*ZOOM) + "px " + (b.height*ZOOM) + "px";
  loupe.style.backgroundPosition = (95-x*ZOOM) + "px " + (95-y*ZOOM) + "px";
});
stage.addEventListener("mouseleave", () => loupe.style.display = "none");
function confirm(){
  if (!pending) return;
  const n = $("num").value.trim();
  record({...pending, number: n === "" ? null : n});
}
$("ok").onclick = confirm;
$("noevent").onclick = () => record("no event");
$("none").onclick = () => record(null);
$("undo").onclick = () => { if (at > 0){ at--; delete done[ITEMS[at].id];
                                            store(); show(); } };
document.addEventListener("keydown", e => {
  if (e.key === "Enter"){ confirm(); return; }
  if (document.activeElement === $("num")) return;
  const k = e.key.toLowerCase();
  if (k === "e") $("noevent").click();
  else if (k === "n") $("none").click();
  else if (k === "u") $("undo").click();
});
$("save").onclick = () => {
  const out = {page: KEY, frames: ITEMS.map(f => ({id: f.id, clip: f.clip,
    frame: f.frame, kind: f.kind, label: done[f.id] === undefined
    ? undefined : done[f.id]}))};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)],
                                        {type: "application/json"}));
  a.download = "player_labels.json"; a.click();
};
function finish(){
  paint();
  $("stage").innerHTML = '<div class="done"><h2>All done</h2>' +
    '<p>Press <b>Download labels</b> and send player_labels.json.</p></div>';
  $("numrow").style.display = "none";
}
at = firstUnlabelled(); show();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="player_labeller.html")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    items = []
    for clip in CLIPS:
        if (Path(f"output_{clip}") / "events.json").exists():
            items += sample(clip, rng)
    frames = frames_for(items)
    Path(args.out).write_text(PAGE.replace("__DATA__", json.dumps(frames)))
    kinds = pd.Series([f["kind"] for f in frames]).value_counts()
    print(f"  {len(frames)} events: " + ", ".join(
        f"{n} {k}" for k, n in kinds.items()))
    print(f"  wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
