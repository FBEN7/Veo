"""The click page (page.html) from manifest.json: the frames to click, the
sprite of each shot served beside the page as out/<shot>.jpg, clicks kept in
the artifact's db (collection "clicks", one document per frame).

    python make_page.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

PAGE = r"""<title>Balle en vol</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a review desk — the frame large on the left, the loupe and the shot's six frames beside it; keys do the work. */
:root {
  --bg: #eef1ee; --panel: #ffffff; --ink: #18211b; --muted: #5b675e; --line: #d3dad4;
  --pitch: #1f6b3d; --ball: #e8590c; --ok: #1f7a46; --warn: #b7791f; --none: #6b7280;
  --display: "Barlow Semi Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111613; --panel: #19201b; --ink: #e6ece7; --muted: #9aa69d; --line: #2c372f;
  --pitch: #4fbf7c; --ball: #ff7a3d; --ok: #4fbf7c; --warn: #e0a84a; --none: #9aa1ad; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111613; --panel: #19201b; --ink: #e6ece7; --muted: #9aa69d; --line: #2c372f;
  --pitch: #4fbf7c; --ball: #ff7a3d; --ok: #4fbf7c; --warn: #e0a84a; --none: #9aa1ad; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 15px/1.5 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 16px 40px; }
.wrap { max-width: 1240px; margin: 0 auto; display: grid; gap: 14px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 20px; }
h1 { font: 700 28px/1.1 var(--display); letter-spacing: .01em; margin: 0; text-wrap: balance; }
.count { font: 500 14px var(--mono); color: var(--muted); font-variant-numeric: tabular-nums; }
.bar { height: 6px; background: var(--line); border-radius: 3px; overflow: hidden; }
.bar > i { display: block; height: 100%; width: 0; background: var(--pitch); transition: width .3s; }
.help { color: var(--muted); max-width: 70ch; margin: 0; }
.help b { color: var(--ink); font-weight: 500; }
kbd { font: 500 12px var(--mono); border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 4px; padding: 0 5px; background: var(--panel); }
.desk { display: grid; grid-template-columns: minmax(0, 1fr) 240px; gap: 14px; align-items: start; }
@media (max-width: 900px) { .desk { grid-template-columns: minmax(0, 1fr); } .side .loupe-box { display: none; } }
.stage { background: #000; border-radius: 6px; overflow: auto; max-width: 100%; position: relative; }
#view { display: block; width: 100%; height: auto; cursor: crosshair; image-rendering: auto; }
.stage.zoom #view { width: 200%; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 16px; font: 13px var(--mono); color: var(--muted); }
.meta strong { color: var(--ink); font-weight: 500; }
.side { display: grid; gap: 12px; min-width: 0; }
.loupe-box { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
.loupe-box canvas { display: block; width: 100%; height: auto; border-radius: 4px; background: #000; }
.label { font: 600 12px var(--display); text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 0 0 6px; }
.actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
button { font: 500 14px var(--body); color: var(--ink); background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 9px 10px; cursor: pointer; text-align: left; }
button:hover { border-color: var(--muted); }
button:focus-visible { outline: 2px solid var(--pitch); outline-offset: 2px; }
button.primary { background: var(--pitch); border-color: var(--pitch); color: #fff; }
.strip { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 6px; }
.thumb { position: relative; border: 2px solid transparent; border-radius: 5px; overflow: hidden; cursor: pointer; background: #000; padding: 0; }
.thumb canvas { display: block; width: 100%; height: auto; }
.thumb.cur { border-color: var(--ball); }
.thumb .tag { position: absolute; left: 3px; bottom: 3px; font: 500 11px var(--mono); padding: 0 4px; border-radius: 3px; background: rgba(0,0,0,.65); color: #fff; }
.tag.ball { background: var(--ok); } .tag.none { background: var(--none); } .tag.unsure { background: var(--warn); }
.notice { border-left: 3px solid var(--warn); background: var(--panel); padding: 8px 12px; border-radius: 4px; }
.saved { font: 13px var(--mono); color: var(--muted); min-height: 1.5em; }
footer { color: var(--muted); font-size: 12px; }
@media (prefers-reduced-motion: reduce) { .bar > i { transition: none; } }
</style>

<div class="wrap">
  <header>
    <h1>Balle en vol</h1>
    <span class="count" id="count">0 / 0 images</span>
  </header>
  <div class="bar" aria-hidden="true"><i id="bar"></i></div>
  <p class="help">Cliquez au <b>centre du ballon de match</b>. Si on ne le voit pas (caché, flou introuvable, hors cadre) : <kbd>N</kbd>. Si vous hésitez : <kbd>U</kbd>. Navigation : <kbd>←</kbd> <kbd>→</kbd>, zoom ×2 : <kbd>Z</kbd>. Le cercle pointillé rappelle votre clic de l'image précédente. Chaque réponse est enregistrée aussitôt et passe à l'image suivante.</p>
  <p class="notice" id="nodb" hidden>Enregistrement indisponible ici : ouvrez la page depuis claude.ai en étant connecté pour que vos clics soient sauvegardés.</p>
  <div class="desk">
    <div style="display:grid;gap:8px;min-width:0">
      <div class="stage" id="stage"><canvas id="view" width="800" height="450" aria-label="Image du tir à annoter"></canvas></div>
      <div class="meta"><span>Tir <strong id="shotno">–</strong></span><span>Image <strong id="frameno">–</strong> / 6</span><span id="when">–</span><span id="status">–</span></div>
      <div class="strip" id="strip"></div>
    </div>
    <div class="side">
      <div class="loupe-box"><p class="label">Loupe ×4</p><canvas id="loupe" width="220" height="220"></canvas></div>
      <div class="actions">
        <button id="b-none" type="button">Pas visible <kbd>N</kbd></button>
        <button id="b-unsure" type="button">Incertain <kbd>U</kbd></button>
        <button id="b-prev" type="button">← Précédente</button>
        <button id="b-next" type="button">Suivante →</button>
        <button id="b-zoom" type="button">Zoom ×2 <kbd>Z</kbd></button>
        <button id="b-todo" type="button" class="primary">Prochaine à faire</button>
      </div>
      <div class="saved" id="saved"></div>
    </div>
  </div>
  <footer>Images : SoccerTrack v2 (Atom Scott et al.), licence CC BY 4.0. Recadrages 800 × 450 px de la vidéo d'origine.</footer>
</div>

<script>
const SHOTS = __SHOTS__;
const W = 800, H = 450, K = 6, OFF = [-4, 4, 10, 16, 24, 34];
const frames = [];
SHOTS.forEach((s, si) => { for (let k = 0; k < K; k++) frames.push({ si, k, id: s.id + "_" + k }); });
const clicks = new Map();        // frame id -> {x, y, status}
const sprites = new Map();       // shot id -> Image
let cur = 0, db = null, col = null, mouse = null;

const $ = (id) => document.getElementById(id);
const view = $("view"), ctx = view.getContext("2d"), loupe = $("loupe"), lctx = loupe.getContext("2d");

function sprite(si) {
  const s = SHOTS[si];
  if (!sprites.has(s.id)) {
    const im = new Image();
    im.onload = () => { if (frames[cur].si === si) draw(); drawStrip(); };
    im.src = "out/" + s.id + ".jpg";
    sprites.set(s.id, im);
  }
  return sprites.get(s.id);
}
function ready(im) { return im && im.complete && im.naturalWidth > 0; }

function ring(c, x, y, r, color, dashed) {
  c.save(); c.strokeStyle = color; c.lineWidth = 2; if (dashed) c.setLineDash([4, 4]);
  c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.stroke(); c.restore();
}

function draw() {
  const f = frames[cur], im = sprite(f.si);
  ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
  if (ready(im)) ctx.drawImage(im, 0, f.k * H, W, H, 0, 0, W, H);
  const ball = getComputedStyle(document.documentElement).getPropertyValue("--ball").trim() || "#e8590c";
  if (f.k > 0) {
    const p = clicks.get(frames[cur - 1].id);
    if (p && p.status === "ball") ring(ctx, p.x, p.y, 14, "#ffffff", true);
  }
  const c = clicks.get(f.id);
  if (c && (c.status === "ball" || c.status === "unsure") && c.x != null) ring(ctx, c.x, c.y, 10, ball, false);
  const s = SHOTS[f.si];
  $("shotno").textContent = (f.si + 1) + " / " + SHOTS.length;
  $("frameno").textContent = f.k + 1;
  const dt = OFF[f.k] / 25;
  $("when").textContent = (dt >= 0 ? "+" : "") + dt.toFixed(2) + " s après le tir";
  $("status").textContent = c ? ({ ball: "ballon placé", none: "pas visible", unsure: "incertain" })[c.status] : "à faire";
  drawLoupe();
  drawStrip();
}

function drawLoupe() {
  lctx.fillStyle = "#000"; lctx.fillRect(0, 0, 220, 220);
  const f = frames[cur], im = sprite(f.si);
  if (!mouse || !ready(im)) return;
  const half = 220 / 4 / 2;
  lctx.imageSmoothingEnabled = false;
  lctx.drawImage(im, mouse.x - half, f.k * H + mouse.y - half, half * 2, half * 2, 0, 0, 220, 220);
  lctx.strokeStyle = "rgba(255,255,255,.8)"; lctx.lineWidth = 1;
  lctx.beginPath(); lctx.moveTo(110, 96); lctx.lineTo(110, 124); lctx.moveTo(96, 110); lctx.lineTo(124, 110); lctx.stroke();
}

function drawStrip() {
  const f = frames[cur], s = SHOTS[f.si], strip = $("strip"), im = sprite(f.si);
  if (strip.dataset.shot !== s.id) {
    strip.innerHTML = ""; strip.dataset.shot = s.id;
    for (let k = 0; k < K; k++) {
      const b = document.createElement("button"); b.type = "button"; b.className = "thumb";
      b.setAttribute("aria-label", "Image " + (k + 1));
      const c = document.createElement("canvas"); c.width = 200; c.height = 112;
      const t = document.createElement("span"); t.className = "tag";
      b.append(c, t); b.onclick = () => go(frames.findIndex(x => x.si === f.si && x.k === k)); strip.append(b);
    }
  }
  [...strip.children].forEach((b, k) => {
    const c = b.querySelector("canvas").getContext("2d");
    if (ready(im)) c.drawImage(im, 0, k * H, W, H, 0, 0, 200, 112);
    const st = clicks.get(s.id + "_" + k), t = b.querySelector(".tag");
    t.className = "tag " + (st ? st.status : ""); t.textContent = (OFF[k] >= 0 ? "+" : "") + (OFF[k] / 25).toFixed(2) + "s";
    b.classList.toggle("cur", k === f.k);
  });
}

function progress() {
  const n = frames.filter(f => clicks.has(f.id)).length;
  $("count").textContent = n + " / " + frames.length + " images";
  $("bar").style.width = (100 * n / frames.length) + "%";
}
function go(i) { cur = Math.max(0, Math.min(frames.length - 1, i)); mouse = null; draw(); sprite(Math.min(SHOTS.length - 1, frames[cur].si + 1)); }
function nextTodo() { const i = frames.findIndex((f, j) => j > cur && !clicks.has(f.id)); go(i >= 0 ? i : frames.findIndex(f => !clicks.has(f.id))); }

let pending = Promise.resolve();
function save(status, x, y) {
  const f = frames[cur], s = SHOTS[f.si];
  const body = { shot: s.id, k: f.k, frame: s.frames[f.k], status, x: x ?? null, y: y ?? null, at: Date.now() };
  clicks.set(f.id, body); progress();
  if (cur < frames.length - 1) go(cur + 1); else draw();
  if (!col) { $("saved").textContent = "Non enregistré (stockage indisponible)."; return; }
  const ref = col.doc(f.id);
  pending = pending.then(() => ref.set(body)).then(
    () => { $("saved").textContent = "Enregistré."; },
    () => { $("saved").textContent = "Échec de l'enregistrement : réessayez cette image."; clicks.delete(f.id); progress(); });
}

function pos(ev) {
  const r = view.getBoundingClientRect();
  return { x: (ev.clientX - r.left) * W / r.width, y: (ev.clientY - r.top) * H / r.height };
}
view.addEventListener("mousemove", ev => { mouse = pos(ev); drawLoupe(); });
view.addEventListener("mouseleave", () => { mouse = null; drawLoupe(); });
view.addEventListener("click", ev => { const p = pos(ev); save("ball", Math.round(p.x * 10) / 10, Math.round(p.y * 10) / 10); });
$("b-none").onclick = () => save("none");
$("b-unsure").onclick = () => save("unsure", mouse?.x, mouse?.y);
$("b-prev").onclick = () => go(cur - 1);
$("b-next").onclick = () => go(cur + 1);
$("b-zoom").onclick = () => $("stage").classList.toggle("zoom");
$("b-todo").onclick = nextTodo;
document.addEventListener("keydown", ev => {
  if (ev.target.closest && ev.target.closest("input,textarea")) return;
  const k = ev.key.toLowerCase();
  if (k === "n") save("none");
  else if (k === "u") save("unsure", mouse?.x, mouse?.y);
  else if (k === "arrowleft") go(cur - 1);
  else if (k === "arrowright") go(cur + 1);
  else if (k === "z") $("stage").classList.toggle("zoom");
  else return;
  ev.preventDefault();
});

go(0); progress();
(async () => {
  db = window.claude ? await window.claude.use("db") : null;
  if (!db) { $("nodb").hidden = false; return; }
  col = db.collection("clicks");
  let first = true;
  col.onSnapshot(snap => {
    snap.docs.forEach(d => { if (d.exists) clicks.set(d.id, d.data()); });
    progress(); draw();
    if (first) { first = false; nextTodo(); }
  }, () => { $("saved").textContent = "Lecture des clics interrompue : rechargez la page."; });
})();
</script>
"""


def main():
    man = json.loads((HERE / "manifest.json").read_text())
    shots = [{"id": s["id"], "frames": s["frames"]} for s in man]
    (HERE / "page.html").write_text(PAGE.replace("__SHOTS__", json.dumps(shots)))
    print(len(shots), "shots,", 6 * len(shots), "frames")


if __name__ == "__main__":
    main()
