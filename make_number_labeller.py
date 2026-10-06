"""A page for naming players: click a player, give the team and number.

Player identities are joined within a clip and linked across a match by
appearance (`src/player_identity.py`, `src/match_identity.py`); shirt
numbers cannot be read at this resolution, by the reader or, often, by
eye. What a person who knows the teams can say -- "this is Huddersfield's
#9" -- checks both and puts names on them: two clicks with the same team
and number are one player, in one clip or across a match, and the squad
list then names the number.

The page shows `PER_CLIP` frames per clip, chosen where players are
largest (main camera, no replay, spread over the clip), at full width.
On each the person clicks as many players as they can name, picks the
team (1 or 2) and types the number. The system's identities are not
shown, so the answers do not lean on them. Built locally and sent as a
file; never published or committed.

    python make_number_labeller.py --out number_labeller.html
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

PER_CLIP = 6
QUALITY = 72
TEAMS = {"stoke": ("Stoke City", "Huddersfield Town"),
         "reading": ("Reading", "Fulham")}


def pick_frames(out_dir: Path, n: int = PER_CLIP):
    """Frames where players are largest, spread over the clip, skipping
    replays and close-ups (players more than twice their usual size)."""
    from src import replays

    info = json.loads((out_dir / "clip.json").read_text())
    m = pd.read_parquet(out_dir / "tracks_merged.parquet")
    p = m[m.cls == "player"]
    size = p.groupby("frame").crop_h.median()
    count = p.groupby("frame").size()
    spans = replays.for_clip(out_dir, info)
    ok = size[(size < 2 * size.median()) & (count >= 6)]
    ok = ok[~replays.in_replay(ok.index.to_numpy(), spans)]
    chunks = np.array_split(ok.index.to_numpy(), n)
    return [int(ok.loc[c].idxmax()) for c in chunks if len(c)]


def frames(out_dir: Path):
    from src.video_frames import frames as read_frames

    info = json.loads((out_dir / "clip.json").read_text())
    out = []
    for index, image in read_frames(info["path"], pick_frames(out_dir)):
        ok, buf = cv2.imencode(".jpg", image,
                               [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        out.append({"f": index, "j": base64.b64encode(buf.tobytes()).decode()})
    clip = out_dir.name.replace("output_", "")
    return {"clip": clip, "teams": TEAMS[clip.split("_")[0]], "frames": out}


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Name the players</title>
<style>
 :root{--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--hit:#00e5ff;--ok:#00c853;
       --t1:#ff9100;--t2:#40c4ff}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
 header{display:flex;gap:14px;align-items:center;padding:8px 14px;
        background:#0d0f12;border-bottom:1px solid #23262b;flex-wrap:wrap}
 #bar{flex:1;height:8px;background:#23262b;border-radius:4px;overflow:hidden;
      min-width:120px}
 #fill{height:100%;width:0;background:var(--ok)}
 button{font:inherit;padding:6px 12px;border-radius:7px;
        border:1px solid #2c3037;background:#1b1e24;color:var(--fg);
        cursor:pointer}
 button:hover{background:#242830}
 main{padding:10px 14px;display:flex;flex-direction:column;align-items:center}
 #stage{position:relative;line-height:0;cursor:crosshair;max-width:1280px;
        width:100%}
 #shot{width:100%;height:auto;border-radius:6px;display:block}
 .mark{position:absolute;transform:translate(-50%,-100%);font:bold 13px
       system-ui;padding:1px 5px;border-radius:4px;color:#000;
       pointer-events:none;white-space:nowrap}
 #loupe{position:absolute;width:220px;height:220px;border-radius:50%;
        border:2px solid var(--hit);pointer-events:none;display:none;
        background-repeat:no-repeat;box-shadow:0 2px 10px #000}
 #ask{display:none;gap:8px;align-items:center;padding:8px;flex-wrap:wrap;
      justify-content:center}
 #num{font:inherit;width:80px;padding:6px;border-radius:7px;
      border:1px solid #2c3037;background:#0d0f12;color:var(--fg)}
 .team.on{outline:2px solid var(--hit)}
 .hint{color:var(--dim);max-width:1000px;text-align:center;padding:4px 14px}
 kbd{background:#23262b;border:1px solid #33373e;border-bottom-width:2px;
     border-radius:5px;padding:0 6px;font:12px ui-monospace,monospace}
</style></head><body>
<header>
  <strong>Name the players</strong>
  <span id="where" class="hint"></span>
  <div id="bar"><div id="fill"></div></div>
  <button id="prev">&larr; Previous</button>
  <button id="next">Next frame &rarr; <kbd>N</kbd></button>
  <button id="save">Download labels</button>
</header>
<main>
  <div id="stage"><img id="shot" alt=""><div id="loupe"></div></div>
  <div id="ask">Team:
    <button class="team" id="t1"></button><button class="team" id="t2"></button>
    Number: <input id="num" inputmode="numeric" autocomplete="off">
    <button id="ok">Save <kbd>Enter</kbd></button>
    <button id="cancel">Cancel <kbd>Esc</kbd></button></div>
  <p class="hint">Click the feet of every player you can name, choose the
  team (<kbd>1</kbd> or <kbd>2</kbd>), type the shirt number and press
  <kbd>Enter</kbd>. Name as many as you can on each frame, by number or
  because you know who it is; skip anyone you are not sure of. Goalkeepers
  count too. Click a saved label to remove it. <kbd>N</kbd> goes to the
  next frame. The magnifier follows the mouse.</p>
</main>
<script>
const CLIPS = __DATA__;
const KEY = "player-names-v1";
const FR = [];
CLIPS.forEach(c => c.frames.forEach(f => FR.push({clip: c.clip, teams: c.teams,
                                                   f: f.f, j: f.j})));
let at = 0, done = {}, draft = null;
try { done = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) {}
function store(){ try { localStorage.setItem(KEY, JSON.stringify(done)); }
                  catch(e) {} }
const $ = id => document.getElementById(id);
const img = $("shot"), loupe = $("loupe");
const id = fr => fr.clip + ":" + fr.f;
function show(){
  const fr = FR[at];
  img.src = "data:image/jpeg;base64," + fr.j;
  loupe.style.backgroundImage = "url(" + img.src + ")";
  $("where").textContent = fr.clip + ", frame " + fr.f + "  (" + (at+1) +
    " of " + FR.length + ")";
  $("t1").textContent = "1 " + fr.teams[0]; $("t2").textContent = "2 " + fr.teams[1];
  const n = FR.filter(x => (done[id(x)] || []).length).length;
  $("fill").style.width = (100*n/FR.length) + "%";
  draw();
}
function draw(){
  document.querySelectorAll(".mark").forEach(m => m.remove());
  const fr = FR[at];
  (done[id(fr)] || []).forEach((p, k) => {
    const m = document.createElement("div"); m.className = "mark";
    m.style.left = (p.x*100) + "%"; m.style.top = (p.y*100) + "%";
    m.style.background = p.team === 0 ? "var(--t1)" : "var(--t2)";
    m.style.pointerEvents = "auto"; m.style.cursor = "pointer";
    m.textContent = (p.number || "?");
    m.title = "click to remove";
    m.onclick = e => { e.stopPropagation(); done[id(fr)].splice(k,1); store(); draw(); };
    $("stage").appendChild(m); });
  if (draft){ const m = document.createElement("div"); m.className = "mark";
    m.style.left = (draft.x*100)+"%"; m.style.top = (draft.y*100)+"%";
    m.style.background = "#fff"; m.textContent = "..."; $("stage").appendChild(m); }
}
$("stage").addEventListener("click", e => {
  if (e.target.classList.contains("mark")) return;
  const b = img.getBoundingClientRect();
  draft = {x: +((e.clientX-b.left)/b.width).toFixed(4),
           y: +((e.clientY-b.top)/b.height).toFixed(4), team: null};
  $("ask").style.display = "flex"; $("num").value = ""; $("num").focus();
  ["t1","t2"].forEach(t => $(t).classList.remove("on")); draw();
});
$("stage").addEventListener("mousemove", e => {
  const b = img.getBoundingClientRect();
  const x = e.clientX-b.left, y = e.clientY-b.top, Z = 4;
  loupe.style.display = "block";
  loupe.style.left = (x-110) + "px"; loupe.style.top = (y-230) + "px";
  loupe.style.backgroundSize = (b.width*Z) + "px " + (b.height*Z) + "px";
  loupe.style.backgroundPosition = (110-x*Z) + "px " + (110-y*Z) + "px";
});
$("stage").addEventListener("mouseleave", () => loupe.style.display = "none");
function team(t){ if (!draft) return; draft.team = t;
  $("t1").classList.toggle("on", t===0); $("t2").classList.toggle("on", t===1); }
$("t1").onclick = () => team(0); $("t2").onclick = () => team(1);
function save(){
  if (!draft || draft.team === null) return;
  const fr = FR[at]; done[id(fr)] = done[id(fr)] || [];
  done[id(fr)].push({x: draft.x, y: draft.y, team: draft.team,
                     number: $("num").value.trim() || null});
  draft = null; $("ask").style.display = "none"; store(); show();
}
function cancel(){ draft = null; $("ask").style.display = "none"; draw(); }
$("ok").onclick = save; $("cancel").onclick = cancel;
$("next").onclick = () => { cancel(); at = Math.min(at+1, FR.length-1); show(); };
$("prev").onclick = () => { cancel(); at = Math.max(at-1, 0); show(); };
document.addEventListener("keydown", e => {
  if (e.key === "Enter"){ save(); return; }
  if (e.key === "Escape"){ cancel(); return; }
  if (draft && (e.key === "1" || e.key === "2") &&
      document.activeElement !== $("num")){ team(+e.key-1); return; }
  if (draft && (e.key === "1" || e.key === "2") && $("num").value === ""
      && draft.team === null){ e.preventDefault(); team(+e.key-1); return; }
  if (document.activeElement === $("num")) return;
  if (e.key.toLowerCase() === "n") $("next").click();
});
$("save").onclick = () => {
  const out = {page: KEY, frames: FR.map(fr => ({clip: fr.clip, frame: fr.f,
    teams: fr.teams, players: done[id(fr)] || []}))};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)],
                                        {type: "application/json"}));
  a.download = "player_names.json"; a.click();
};
show();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="number_labeller.html")
    args = ap.parse_args()
    data = [frames(Path(f"output_{c}")) for c in CLIPS
            if (Path(f"output_{c}") / "tracks_merged.parquet").exists()]
    Path(args.out).write_text(PAGE.replace("__DATA__", json.dumps(data)))
    for d in data:
        print(f"  {d['clip']}: frames {[f['f'] for f in d['frames']]}")
    print(f"  wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
