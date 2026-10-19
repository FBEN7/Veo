export const meta = {
  name: 'ball-v3-unknown-check',
  description: 'Blind check of 83 ball tiles (v3 picks on frames without a verified ball, with controls): 2 judges, a third on disagreements',
  phases: [{ title: 'Judge', detail: 'two blind judges' }, { title: 'Adjudicate', detail: 'third judge on B / not-B disagreements' }],
}
const DIR = '/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5/v3unk'
const ZOOM = '/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5/zoom_chunk.py'
const JUDG = `${DIR}_judgements`
const LABELS = {
  type: 'object',
  properties: {
    labels: { type: 'array', items: { type: 'object', properties: {
      tile: { type: 'integer' }, label: { type: 'string', enum: ['B', 'B*', 'S', 'N'] } },
      required: ['tile', 'label'] } },
    unsure_tiles: { type: 'array', items: { type: 'integer' } },
    notes: { type: 'string' },
  },
  required: ['labels', 'unsure_tiles', 'notes'],
}
const RULES = `Each tile is a crop of a fixed panoramic football camera (youth football) centred on a candidate ball position; a 24 px
box is marked by four red L-shaped corner marks (the corner VERTEX of each L is the box corner; the L's lines point outwards).
Label with exactly one of:
- B: the centre of the MATCH BALL IN PLAY is inside the box (within the four corner vertices).
- B*: a ball resting in or right by a goal (e.g. inside the goal behind the goal line).
- S: a ball that is off the pitch or not in play (beyond the touchline by the fence/stands, held by an outfield player in a
  stoppage, a ball placed for a set piece and not moving, spare balls by the side).
- N: anything else: no ball, boots, gloves, hands, heads, shirts, line marks, linesman flags, the penalty spot (a small flat pale mark between
  penalty-area lines), a ball whose centre is outside the box.
Unsure whether it is the ball, or whether its centre is inside the box: N (or S/B* if it is surely a ball but not in play).
A goalkeeper holding the ball counts as B. A ball at a player's feet counts as B when it is clearly a ball. A motion-blurred
streak is a ball only if it is clearly the ball (bright, ball-sized, no line or boot explains it).`

const LAYOUT = `LAYOUT: ${DIR}/ holds sheet_K.jpg (5 rows of 6 tiles; blank cells are empty; sheet 3 has only 4 rows, its last with 5 tiles) and view_K_P.png
(1.3x enlargements: P = 1 is rows 1-3 of sheet K, P = 2 rows 4-5). There are 83 tiles, numbered 1-83
in their headers. EVERY TILE IS INDEPENDENT: each comes from a different frame (often a different match); rows and neighbours carry
no meaning, so judge each tile on its own content.`

const judgePrompt = i => `You are blind judge #${i} labelling ball candidates. ${RULES}

${LAYOUT}

STRICT BLINDING: read ONLY files inside ${DIR}/ and run only
  python ${ZOOM} ${DIR} ${JUDG}/z_j${i}.png T1 T2 ...
(3x zoom of the listed tiles; then read that png). Do NOT open, list or grep anything else (no key directory, no other judge's
file, no repository code). Do not try to infer which detector produced a tile.

Work through all views in order (view_1_1, view_1_2, view_2_1, view_2_2, view_3_1, view_3_2), zooming whenever in doubt (borderline box position,
faint blobs, boots vs ball); zooming every tile in groups of 6 is fine. Append your labels as you go, one "TILE LABEL" line per
tile, to ${JUDG}/j${i}.txt (create it; if it already has lines, continue after the last labelled tile). Label all 83 tiles.
Return all labels, the tiles you were unsure about, and short notes on recurring hard cases.`

const adjPrompt = tiles => `You are the deciding blind judge for ${tiles.length} ball-candidate tiles on which two judges
disagreed. ${RULES}

The tiles are in ${DIR}/ (sheet_K.jpg, view_K_P.png, layout.json); every tile is independent (a different frame), rows carry no
meaning. STRICT BLINDING: read only that directory and run only
  python ${ZOOM} ${DIR} ${JUDG}/z_adj.png T1 T2 ...
Do not open anything else (no key directory, no judge files). Tiles to decide: ${tiles.join(', ')}.
Zoom every one of them. Write "TILE LABEL" lines to ${JUDG}/adj.txt and return the labels.`

const isB = l => l === 'B'
const asMap = r => Object.fromEntries(((r && r.labels) || []).map(x => [x.tile, x.label]))

phase('Judge')
const pair = await parallel([1, 2].map(i => () =>
  agent(judgePrompt(i), { label: `judge:j${i}`, phase: 'Judge', schema: LABELS })))
const [a, b] = pair.map(asMap)
const all = Array.from({ length: 83 }, (_, k) => k + 1)
const missing = all.filter(t => !(t in a) || !(t in b))
if (missing.length) log(`tiles missing from a judge: ${missing.join(', ')}`)
const dis = all.filter(t => !(t in a) || !(t in b) || isB(a[t]) !== isB(b[t]))
log(`${dis.length} B / not-B disagreements`)
let adj = {}
if (dis.length) {
  phase('Adjudicate')
  const r = await agent(adjPrompt(dis), { label: 'adjudicate', phase: 'Adjudicate', schema: LABELS })
  adj = asMap(r)
}
return { j1: a, j2: b, adj, disagreements: dis, unsure: pair.map(p => p && p.unsure_tiles), notes: pair.map(p => p && p.notes) }
