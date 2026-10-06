"""A page for labelling every action in a stretch of play, frame by frame.

The first labelling round asked about events one detector had found, so
it could say which of them were real but not what was missed, and it
could not measure any other detector fairly. This page asks for all of
it: in a continuous stretch of live play, every pass, carry, shot,
tackle, recovery, clearance and goal, in the format of
`src/event_schema.py` -- type, the frame it starts on, the frame it ends
on, the player who did it and, for a pass, the player who received it,
and whether it succeeded. Lengths come later from where the clicked
players are; nothing has to be typed.

How a stretch is chosen: per clip, the `SECONDS` of live play (no replay)
where the ball is tracked on the most frames -- not where any detector
fired, so the labels do not lean towards one. Frames are shown every
`STEP` frames; the frame numbers saved are the video's own.

The frames are under the data agreement, so the page goes to the person
who has the data, as a file, and is never published.

    python make_action_labeller.py --out action_labeller.html
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

from make_ball_labeller import CLIPS

SECONDS = 10.0
STEP = 2
WIDTH = 896
QUALITY = 50


def stretch(out_dir: Path, fps: float, taken=()):
    """First and last frame of the best-tracked live stretch, not
    overlapping the (first, last) ranges in `taken`."""
    import detect_shots

    ball = detect_shots.ball_track(out_dir, ball_detector="fill")
    seen = np.zeros(int(ball.frame.max()) + 1)
    seen[ball.frame.to_numpy().astype(int)] = 1
    n = int(SECONDS * fps)
    if len(seen) <= n:
        return 0, len(seen) - 1
    counts = np.convolve(seen, np.ones(n), mode="valid")
    for first, last in taken:
        counts[max(first - n + 1, 0):last + 1] = -1
    start = int(np.argmax(counts))
    return start, start + n - 1


def frames(out_dir: Path, taken=()):
    from src.video_frames import frames as read_frames

    info = json.loads((out_dir / "clip.json").read_text())
    first, last = stretch(out_dir, float(info["fps"]), taken)
    out = []
    for index, image in read_frames(info["path"], range(first, last + 1, STEP)):
        h, w = image.shape[:2]
        image = cv2.resize(image, (WIDTH, int(h * WIDTH / w)),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", image,
                               [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        out.append({"f": index, "j": base64.b64encode(buf.tobytes()).decode()})
    return {"clip": out_dir.name.replace("output_", ""),
            "fps": float(info["fps"]), "frames": out}


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Label every action</title>
<style>
 :root{--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--hit:#00e5ff;--ok:#00c853;
       --from:#ff9100;--to:#00e676}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
 header{display:flex;gap:14px;align-items:center;padding:8px 14px;
        background:#0d0f12;border-bottom:1px solid #23262b;flex-wrap:wrap}
 select,button{font:inherit;padding:6px 12px;border-radius:7px;
        border:1px solid #2c3037;background:#1b1e24;color:var(--fg);
        cursor:pointer}
 button:hover{background:#242830}
 .wrap{display:flex;gap:14px;padding:10px 14px;flex-wrap:wrap}
 #left{flex:1 1 640px;min-width:0}
 #stage{position:relative;line-height:0;cursor:crosshair}
 #shot{width:100%;height:auto;border-radius:6px;display:block}
 .mark{position:absolute;width:16px;height:16px;margin:-8px 0 0 -8px;
       border:3px solid;border-radius:50%;pointer-events:none}
 #bar{display:flex;gap:10px;align-items:center;padding:8px 0}
 #slider{flex:1}
 #status{padding:8px 10px;border-radius:7px;background:#1b1e24;
         min-height:40px}
 #status b{color:var(--hit)}
 #right{flex:0 1 330px;min-width:260px}
 #list{max-height:60vh;overflow:auto;border:1px solid #23262b;
       border-radius:7px}
 .act{display:flex;justify-content:space-between;gap:6px;padding:6px 8px;
      border-bottom:1px solid #23262b;cursor:pointer}
 .act:hover{background:#1b1e24}
 .act .x{color:#ff5252}
 kbd{background:#23262b;border:1px solid #33373e;border-bottom-width:2px;
     border-radius:5px;padding:0 6px;font:12px ui-monospace,monospace}
 .help{color:var(--dim);font-size:13px}
 .help td{padding:1px 6px;vertical-align:top}
</style></head><body>
<header>
  <strong>Label every action</strong>
  <select id="clip"></select>
  <span id="frameinfo" class="help"></span>
  <span style="flex:1"></span>
  <button id="save">Download labels</button>
</header>
<div class="wrap">
 <div id="left">
  <div id="stage"><img id="shot" alt=""></div>
  <div id="bar">
    <button id="play">Play <kbd>Space</kbd></button>
    <input id="slider" type="range" min="0" value="0">
  </div>
  <div id="status"></div>
 </div>
 <div id="right">
  <div class="help"><table>
   <tr><td><kbd>&larr;</kbd><kbd>&rarr;</kbd></td><td>step one frame (<kbd>Shift</kbd>: five)</td></tr>
   <tr><td><kbd>P</kbd> <kbd>C</kbd> <kbd>S</kbd></td><td>start a pass, carry, shot on this frame</td></tr>
   <tr><td><kbd>T</kbd> <kbd>R</kbd> <kbd>X</kbd> <kbd>G</kbd></td><td>tackle, recovery, clearance, goal</td></tr>
   <tr><td>click</td><td>the player who does it (feet); for a pass, at its end, the receiver</td></tr>
   <tr><td><kbd>Enter</kbd></td><td>end the action on this frame</td></tr>
   <tr><td><kbd>F</kbd></td><td>mark it failed (pass not received by a team-mate, shot off target...)</td></tr>
   <tr><td><kbd>K</kbd></td><td>the player who did it is the goalkeeper</td></tr>
   <tr><td><kbd>Esc</kbd></td><td>cancel the action being made</td></tr>
  </table>
  <p>A pass starts on the frame the ball leaves the foot and ends on the
  frame it is received. A carry starts at the reception and ends at the
  player's next touch away. Tackles and recoveries are one frame: the
  moment the ball is won. Label everything you see, even when unsure of
  the exact frame.</p></div>
  <div id="list"></div>
 </div>
</div>
<script>
const CLIPS = __DATA__;
const KEY = "__KEY__";
const NAMES = {p:"pass", c:"carry", s:"shot", t:"tackle", r:"recovery",
               x:"clearance", g:"goal"};
const ONE_FRAME = new Set(["tackle","recovery","goal"]);
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) {}
function store(){ try { localStorage.setItem(KEY, JSON.stringify(saved)); }
                  catch(e) {} }
const $ = id => document.getElementById(id);
let ci = 0, at = 0, timer = null, draft = null;
function clip(){ return CLIPS[ci]; }
function acts(){ return saved[clip().clip] = saved[clip().clip] || []; }
CLIPS.forEach((c,i) => { const o = document.createElement("option");
  o.value = i; o.textContent = c.clip; $("clip").appendChild(o); });
$("clip").onchange = e => { ci = +e.target.value; at = 0; draft = null; show(); };
function show(){
  const c = clip(), fr = c.frames[at];
  $("shot").src = "data:image/jpeg;base64," + fr.j;
  $("slider").max = c.frames.length - 1; $("slider").value = at;
  $("frameinfo").textContent = "frame " + fr.f + "  (" +
    (fr.f / c.fps).toFixed(2) + " s)  " + (at+1) + "/" + c.frames.length;
  document.querySelectorAll(".mark").forEach(m => m.remove());
  const marks = [];
  if (draft){ if (draft.from && draft.from.f === fr.f) marks.push([draft.from,"--from"]);
              if (draft.to && draft.to.f === fr.f) marks.push([draft.to,"--to"]); }
  for (const a of acts()){
    if (a.from && a.from.f === fr.f) marks.push([a.from,"--from"]);
    if (a.to && a.to.f === fr.f) marks.push([a.to,"--to"]); }
  for (const [p,col] of marks){
    const m = document.createElement("div"); m.className = "mark";
    m.style.left = (p.x*100) + "%"; m.style.top = (p.y*100) + "%";
    m.style.borderColor = "var(" + col + ")"; $("stage").appendChild(m); }
  status(); list();
}
function status(){
  if (!draft){ $("status").innerHTML = "Step to where an action starts and " +
    "press its key."; return; }
  let s = "<b>" + draft.type + "</b> from frame " + draft.start + ". ";
  if (!draft.from) s += "Click the player who plays the ball.";
  else if (ONE_FRAME.has(draft.type)) s += "Press <kbd>Enter</kbd> to save.";
  else if (draft.type === "pass") s += "Step to the reception, click the " +
    "receiver" + (draft.to ? " (done)" : "") + ", then <kbd>Enter</kbd>.";
  else s += "Step to where it ends and press <kbd>Enter</kbd>.";
  if (draft.result === "fail") s += " Marked <b>failed</b>.";
  if (draft.keeper) s += " By the <b>goalkeeper</b>.";
  $("status").innerHTML = s;
}
function list(){
  const el = $("list"); el.innerHTML = "";
  acts().slice().sort((a,b) => a.start - b.start).forEach(a => {
    const d = document.createElement("div"); d.className = "act";
    d.innerHTML = "<span>" + a.type + " " + a.start + "&rarr;" + a.end +
      (a.result === "fail" ? " (failed)" : "") + (a.keeper ? " (keeper)" : "") +
      "</span><span class='x'>delete</span>";
    d.onclick = e => {
      if (e.target.className === "x"){ acts().splice(acts().indexOf(a),1);
        store(); show(); return; }
      at = Math.max(0, clip().frames.findIndex(fr => fr.f >= a.start)); show(); };
    el.appendChild(d); });
}
function go(n){ at = Math.min(Math.max(at + n, 0), clip().frames.length-1); show(); }
$("slider").oninput = e => { at = +e.target.value; show(); };
$("play").onclick = () => {
  if (timer){ clearInterval(timer); timer = null; return; }
  timer = setInterval(() => { if (at >= clip().frames.length-1){
    clearInterval(timer); timer = null; return; } go(1); },
    1000 * __STEP__ / clip().fps); };
$("stage").addEventListener("click", e => {
  if (!draft) return;
  const b = $("shot").getBoundingClientRect();
  const p = {x: +((e.clientX-b.left)/b.width).toFixed(4),
             y: +((e.clientY-b.top)/b.height).toFixed(4),
             f: clip().frames[at].f};
  if (!draft.from) draft.from = p; else if (draft.type === "pass") draft.to = p;
  show();
});
function finish(){
  if (!draft || !draft.from) return;
  const end = ONE_FRAME.has(draft.type) ? draft.start : clip().frames[at].f;
  if (end < draft.start) return;
  acts().push({type: draft.type, start: draft.start, end: end,
               from: draft.from, to: draft.to || null,
               result: draft.result || "success", keeper: !!draft.keeper});
  draft = null; store(); show();
}
document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (e.key === " "){ e.preventDefault(); $("play").click(); return; }
  if (e.key === "ArrowRight"){ e.preventDefault(); go(e.shiftKey ? 5 : 1); return; }
  if (e.key === "ArrowLeft"){ e.preventDefault(); go(e.shiftKey ? -5 : -1); return; }
  if (e.key === "Enter"){ finish(); return; }
  if (e.key === "Escape"){ draft = null; show(); return; }
  if (k === "k" && draft){ draft.keeper = !draft.keeper; status(); return; }
  if (k === "f" && draft){ draft.result = draft.result === "fail" ? "success" : "fail"; status(); return; }
  if (NAMES[k]){ draft = {type: NAMES[k], start: clip().frames[at].f}; show(); }
});
$("save").onclick = () => {
  const out = {page: KEY, clips: CLIPS.map(c => ({clip: c.clip, fps: c.fps,
    first: c.frames[0].f, last: c.frames[c.frames.length-1].f,
    actions: saved[c.clip] || []}))};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)],
                                        {type: "application/json"}));
  a.download = KEY + ".json"; a.click();
};
show();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="action_labeller.html")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="label files from earlier rounds: their stretches "
                         "are not shown again")
    ap.add_argument("--key", default="action-label-v1",
                    help="where the page keeps its work in the browser; a "
                         "new round needs a new key")
    args = ap.parse_args()
    taken = {}
    for f in args.exclude:
        for c in json.loads(Path(f).read_text())["clips"]:
            taken.setdefault(c["clip"], []).append((c["first"], c["last"]))
    data = [frames(Path(f"output_{c}"), taken.get(c, ())) for c in CLIPS
            if (Path(f"output_{c}") / "clip.json").exists()]
    html = (PAGE.replace("__DATA__", json.dumps(data))
            .replace("__STEP__", str(STEP)).replace("__KEY__", args.key))
    Path(args.out).write_text(html)
    for d in data:
        print(f"  {d['clip']}: frames {d['frames'][0]['f']}-"
              f"{d['frames'][-1]['f']} ({len(d['frames'])} shown)")
    print(f"  wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
