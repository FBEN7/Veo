"""A page for naming players: pause the video, click a player, give the
team and number.

Player identities are joined within a clip and linked across a match by
appearance (`src/player_identity.py`, `src/match_identity.py`); shirt
numbers cannot be read on the main camera, where players are 60-150 px
tall at 720p and a number a few pixels. What a person can say -- "this is
Huddersfield's #9" -- checks both and puts names on them: two clicks with
the same team and number are one player, in one clip or across a match,
and the squad list then names the number.

A first version showed still frames chosen by the pipeline: main-camera
frames had no legible number, and frames chosen as close-ups by player
size were as often the bench or a blurred wide shot. This one plays the
clips themselves, at full quality, so the person can stop wherever a
number shows (close-ups, players near the camera) and click there. The
page holds no video: the person opens their own copies of the clips
(matched by file name, e.g. `stoke_1302_2shots.mp4`), so it is small and
nothing is embedded. The system's identities are not shown, so the
answers do not lean on them. Built locally and sent as a file; never
published or committed.

    python make_number_labeller.py --out number_labeller.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from make_ball_labeller import CLIPS

TEAMS = {"stoke": ("Stoke City", "Huddersfield Town"),
         "reading": ("Reading", "Fulham")}


def clip_info(out_dir: Path):
    info = json.loads((out_dir / "clip.json").read_text())
    cap = cv2.VideoCapture(info["path"])
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    clip = out_dir.name.replace("output_", "")
    return {"clip": clip, "fps": float(info["fps"]), "frames": n,
            "teams": TEAMS[clip.split("_")[0]]}


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Name the players</title>
<style>
 :root{--bg:#14161a;--fg:#e8eaed;--dim:#9aa0a6;--hit:#00e5ff;--ok:#00c853;
       --t1:#ff9100;--t2:#40c4ff;--bad:#ff5252}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
 header{display:flex;gap:10px;align-items:center;padding:8px 14px;
        background:#0d0f12;border-bottom:1px solid #23262b;flex-wrap:wrap}
 button,label.btn{font:inherit;padding:6px 12px;border-radius:7px;
        border:1px solid #2c3037;background:#1b1e24;color:var(--fg);
        cursor:pointer}
 button:hover,label.btn:hover{background:#242830}
 .tab.on{outline:2px solid var(--hit)}
 .tab.missing{color:var(--dim)}
 main{padding:10px 14px;display:flex;flex-direction:column;align-items:center}
 #stage{position:relative;line-height:0;cursor:crosshair;max-width:1280px;
        width:100%}
 #vid{width:100%;height:auto;border-radius:6px;display:block;background:#000}
 .mark{position:absolute;transform:translate(-50%,-100%);font:bold 13px
       system-ui;padding:1px 5px;border-radius:4px;color:#000;
       white-space:nowrap;cursor:pointer}
 #loupe{position:absolute;width:240px;height:240px;border-radius:50%;
        border:2px solid var(--hit);pointer-events:none;display:none;
        box-shadow:0 2px 10px #000;background:#000}
 #controls{display:flex;gap:8px;align-items:center;width:100%;
           max-width:1280px;padding:8px 0;flex-wrap:wrap}
 #seek{flex:1;min-width:200px}
 #ask{display:none;gap:8px;align-items:center;padding:8px;flex-wrap:wrap;
      justify-content:center;background:#1b1e24;border-radius:8px;margin:4px}
 #num{font:inherit;width:80px;padding:6px;border-radius:7px;
      border:1px solid #2c3037;background:#0d0f12;color:var(--fg)}
 .team.on{outline:2px solid var(--hit)}
 .hint{color:var(--dim);max-width:1100px;text-align:center;padding:4px 14px}
 #list{max-width:1280px;width:100%;display:flex;flex-wrap:wrap;gap:6px}
 .item{background:#1b1e24;border:1px solid #2c3037;border-radius:6px;
       padding:2px 8px;cursor:pointer}
 .item b{color:#000;border-radius:3px;padding:0 4px}
 kbd{background:#23262b;border:1px solid #33373e;border-bottom-width:2px;
     border-radius:5px;padding:0 6px;font:12px ui-monospace,monospace}
 #load{border-color:var(--hit)}
</style></head><body>
<header>
  <strong>Name the players</strong>
  <label class="btn" id="load">Open the video files&hellip;
    <input type="file" id="files" accept="video/*" multiple hidden></label>
  <span id="tabs"></span>
  <span style="flex:1"></span>
  <span id="count" class="hint"></span>
  <button id="save">Download labels</button>
</header>
<main>
  <p class="hint" id="intro">Open your copies of the six clips (you can
  select them all at once; they are matched by name, e.g.
  <code>stoke_1302_2shots.mp4</code>). Nothing is uploaded; the page only
  plays them.</p>
  <div id="stage"><video id="vid" muted playsinline preload="auto"></video>
    <canvas id="loupe" width="240" height="240"></canvas></div>
  <div id="controls">
    <button id="play">Play <kbd>Space</kbd></button>
    <button id="back">&minus;1 frame <kbd>&larr;</kbd></button>
    <button id="fwd">+1 frame <kbd>&rarr;</kbd></button>
    <input type="range" id="seek" min="0" value="0" step="1">
    <span id="where" class="hint"></span>
  </div>
  <div id="ask">Team:
    <button class="team" id="t1"></button><button class="team" id="t2"></button>
    Number: <input id="num" inputmode="numeric" autocomplete="off">
    <button id="ok">Save <kbd>Enter</kbd></button>
    <button id="cancel">Cancel <kbd>Esc</kbd></button></div>
  <p class="hint">Play or scrub until you can read a player's number
  (close-ups are best), pause, and click the player's feet. Then press
  <kbd>1</kbd> or <kbd>2</kbd> for the team, type the number and press
  <kbd>Enter</kbd>. Click a player while the video plays to pause it.
  <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> step one frame, with <kbd>Shift</kbd>
  one second. Name every player whose number you can read, once or twice
  per clip each is plenty; goalkeepers too. Skip anyone you are not sure
  of. Click a label to remove it; click an entry below to go back to it.</p>
  <div id="list"></div>
</main>
<script>
const CLIPS = __DATA__;
const KEY = "player-names-v2";
const $ = id => document.getElementById(id);
const vid = $("vid"), loupe = $("loupe"), lctx = loupe.getContext("2d");
let cur = 0, done = {}, draft = null, urls = {}, shown = 0;
try { done = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) {}
function store(){ try { localStorage.setItem(KEY, JSON.stringify(done)); }
                  catch(e) {} }
const C = () => CLIPS[cur];
const mine = () => (done[C().clip] = done[C().clip] || []);

// The frame on screen: the presented frame's media time where the browser
// reports it, else the playback position.
function frameNow(){ return Math.min(C().frames - 1,
  Math.max(0, Math.round(shown * C().fps))); }
function track(now, md){ shown = md.mediaTime; info();
  vid.requestVideoFrameCallback(track); }
if ("requestVideoFrameCallback" in HTMLVideoElement.prototype)
  vid.requestVideoFrameCallback(track);
else vid.addEventListener("timeupdate", () => { shown = vid.currentTime; info(); });
vid.addEventListener("seeked", () => {
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) shown = vid.currentTime;
  info(); });
function go(f){ f = Math.max(0, Math.min(C().frames - 1, f));
  vid.pause(); vid.currentTime = (f + 0.1) / C().fps; shown = vid.currentTime; info(); }

function tabs(){
  $("tabs").innerHTML = "";
  CLIPS.forEach((c, k) => { const b = document.createElement("button");
    b.className = "tab" + (k === cur ? " on" : "") + (urls[c.clip] ? "" : " missing");
    const n = (done[c.clip] || []).length;
    b.textContent = c.clip + (n ? " (" + n + ")" : "");
    b.title = urls[c.clip] ? "" : "video not opened yet";
    b.onclick = () => { cancel(); cur = k; open(); }; $("tabs").appendChild(b); });
  const all = Object.values(done).reduce((a, l) => a + l.length, 0);
  $("count").textContent = all + " players named";
}
function open(){
  const c = C(); tabs();
  $("t1").textContent = "1 " + c.teams[0]; $("t2").textContent = "2 " + c.teams[1];
  $("seek").max = c.frames - 1;
  if (urls[c.clip]){ vid.src = urls[c.clip]; $("intro").style.display = "none"; }
  else { vid.removeAttribute("src"); vid.load(); }
  shown = 0; info(); list();
}
$("files").addEventListener("change", e => {
  const unmatched = [];
  for (const f of e.target.files){
    const c = CLIPS.find(c => f.name.includes(c.clip));
    if (c) urls[c.clip] = URL.createObjectURL(f); else unmatched.push(f);
  }
  // A file whose name names no clip goes to the clip on screen.
  if (unmatched.length === 1 && !urls[C().clip])
    urls[C().clip] = URL.createObjectURL(unmatched[0]);
  else if (unmatched.length) alert("Not matched to a clip by name: " +
    unmatched.map(f => f.name).join(", "));
  const k = CLIPS.findIndex(c => urls[c.clip]);
  if (!urls[C().clip] && k >= 0) cur = k;
  open();
});

function info(){
  const f = frameNow();
  $("seek").value = f;
  $("where").textContent = C().clip + ", frame " + f + " / " + (C().frames - 1);
  $("play").innerHTML = (vid.paused ? "Play" : "Pause") + " <kbd>Space</kbd>";
  draw();
}
function draw(){
  document.querySelectorAll(".mark").forEach(m => m.remove());
  const f = frameNow();
  if (!vid.paused) return;
  mine().forEach((p, k) => { if (Math.abs(p.frame - f) > 0) return;
    const m = document.createElement("div"); m.className = "mark";
    m.style.left = (p.x*100) + "%"; m.style.top = (p.y*100) + "%";
    m.style.background = p.team === 0 ? "var(--t1)" : "var(--t2)";
    m.textContent = p.number || "?"; m.title = "click to remove";
    m.onclick = e => { e.stopPropagation(); mine().splice(k, 1); store(); tabs(); list(); draw(); };
    $("stage").appendChild(m); });
  if (draft){ const m = document.createElement("div"); m.className = "mark";
    m.style.left = (draft.x*100)+"%"; m.style.top = (draft.y*100)+"%";
    m.style.background = "#fff"; m.textContent = "...";
    m.style.pointerEvents = "none"; $("stage").appendChild(m); }
}
function list(){
  const l = $("list"); l.innerHTML = "";
  mine().slice().sort((a, b) => a.frame - b.frame).forEach(p => {
    const d = document.createElement("span"); d.className = "item";
    d.innerHTML = "<b style='background:" + (p.team === 0 ? "var(--t1)" : "var(--t2)") +
      "'>" + (p.number || "?") + "</b> " + C().teams[p.team] + ", frame " + p.frame;
    d.onclick = () => go(p.frame); l.appendChild(d); });
}

$("stage").addEventListener("click", e => {
  if (e.target.classList.contains("mark") || !vid.src) return;
  if (!vid.paused){ vid.pause(); return; }
  const b = vid.getBoundingClientRect();
  draft = {x: +((e.clientX-b.left)/b.width).toFixed(4),
           y: +((e.clientY-b.top)/b.height).toFixed(4),
           frame: frameNow(), team: null};
  $("ask").style.display = "flex"; $("num").value = ""; $("num").focus();
  ["t1","t2"].forEach(t => $(t).classList.remove("on")); draw();
});
$("stage").addEventListener("mousemove", e => {
  if (!vid.videoWidth) return;
  const b = vid.getBoundingClientRect();
  const x = e.clientX-b.left, y = e.clientY-b.top, Z = 3;
  const sc = vid.videoWidth / b.width, w = 240 / Z * sc;
  lctx.drawImage(vid, x*sc - w/2, y*sc - w/2, w, w, 0, 0, 240, 240);
  loupe.style.display = "block";
  loupe.style.left = (x - 120) + "px"; loupe.style.top = (y - 260) + "px";
});
$("stage").addEventListener("mouseleave", () => loupe.style.display = "none");
function team(t){ if (!draft) return; draft.team = t;
  $("t1").classList.toggle("on", t===0); $("t2").classList.toggle("on", t===1); }
$("t1").onclick = () => team(0); $("t2").onclick = () => team(1);
function save(){
  if (!draft) return;
  if (draft.team === null){ $("t1").style.borderColor = $("t2").style.borderColor = "var(--bad)";
    return; }
  $("t1").style.borderColor = $("t2").style.borderColor = "";
  mine().push({frame: draft.frame, x: draft.x, y: draft.y, team: draft.team,
               number: $("num").value.trim() || null});
  draft = null; $("ask").style.display = "none"; store(); tabs(); list(); draw();
}
function cancel(){ draft = null; $("ask").style.display = "none"; draw(); }
$("ok").onclick = save; $("cancel").onclick = cancel;
$("play").onclick = () => { cancel(); vid.paused ? vid.play() : vid.pause(); };
$("back").onclick = () => go(frameNow() - 1);
$("fwd").onclick = () => go(frameNow() + 1);
$("seek").addEventListener("input", e => { cancel(); go(+e.target.value); });
vid.addEventListener("play", info); vid.addEventListener("pause", info);
document.addEventListener("keydown", e => {
  if (e.key === "Enter"){ e.preventDefault(); save(); return; }
  if (e.key === "Escape"){ cancel(); return; }
  if (draft && (e.key === "1" || e.key === "2") && draft.team === null
      && $("num").value === ""){ e.preventDefault(); team(+e.key - 1); return; }
  if (draft && document.activeElement === $("num")) return;
  if (e.key === " "){ e.preventDefault(); $("play").click(); }
  else if (e.key === "ArrowLeft"){ e.preventDefault();
    go(frameNow() - (e.shiftKey ? Math.round(C().fps) : 1)); }
  else if (e.key === "ArrowRight"){ e.preventDefault();
    go(frameNow() + (e.shiftKey ? Math.round(C().fps) : 1)); }
});
$("save").onclick = () => {
  const frames = [];
  CLIPS.forEach(c => {
    const by = {};
    (done[c.clip] || []).forEach(p => { (by[p.frame] = by[p.frame] || []).push(
      {x: p.x, y: p.y, team: p.team, number: p.number}); });
    Object.keys(by).sort((a, b) => a - b).forEach(f => frames.push(
      {clip: c.clip, frame: +f, teams: c.teams, players: by[f]}));
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify({page: KEY, frames}, null, 1)],
                                        {type: "application/json"}));
  a.download = "player_names.json"; a.click();
};
open();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="number_labeller.html")
    args = ap.parse_args()
    data = [clip_info(Path(f"output_{c}")) for c in CLIPS
            if (Path(f"output_{c}") / "clip.json").exists()]
    Path(args.out).write_text(PAGE.replace("__DATA__", json.dumps(data)))
    for d in data:
        print(f"  {d['clip']}: {d['frames']} frames at {d['fps']} fps")
    print(f"  wrote {args.out} ({Path(args.out).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
