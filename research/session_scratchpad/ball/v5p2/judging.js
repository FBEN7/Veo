export const meta = {
  name: 'v5p2-judging',
  description: 'Protocol v5 part 2: blind judging of the M7 / M9 tiles (one judge, a targeted second, a third on disagreements)',
  phases: [{ title: 'Judge', detail: 'first judge, every tile of a chunk' }, { title: 'Recheck', detail: 'second judge on tiles not labelled N, unsure ones and 1 in 7 of the rest' }, { title: 'Adjudicate', detail: 'third judge on B / not-B disagreements' }],
}
// args: {base: ".../.cache/soccertrack_v2/ball_verified/v5p2/tiles", zoom: ".../zoom_chunk.py", chunks: ["chunk_01", ...]}
const BASE = args.base
const ZOOM = args.zoom
const JUDG = `${BASE}_judgements`
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
- N: anything else: no ball, boots, gloves, hands, heads, shirts, line marks, linesman flags, the penalty spot (a small flat pale
  mark between penalty-area lines), a ball whose centre is outside the box.
Unsure whether it is the ball, or whether its centre is inside the box: N (or S/B* if it is surely a ball but not in play).
A goalkeeper holding the ball counts as B. A ball at a player's feet counts as B when it is clearly a ball. A motion-blurred
streak is a ball only if it is clearly the ball (bright, ball-sized, no line or boot explains it).`
const layout = chunk => `LAYOUT: ${BASE}/${chunk}/ holds sheet_K.jpg (5 ROWS of up to 6 tiles; blank cells are empty) and view_K_P.png
(1.3x enlargements: P = 1 is rows 1-3 of sheet K, P = 2 rows 4-5). EACH ROW IS ONE VIDEO FRAME: the tiles of a row are different
candidate positions in that same frame, in random order. A row may contain no ball, or the ball in one tile; two tiles of a row can
both contain the ball centre only when their boxes overlap. Use the other tiles of the row as context. Every tile has its number in
its header.`
const blind = (chunk, tag) => `STRICT BLINDING: read ONLY files inside ${BASE}/${chunk}/ and run only
  python ${ZOOM} ${BASE}/${chunk} ${JUDG}/z_${chunk}_${tag}.png T1 T2 ...
(3x zoom of the listed tiles; then read that png). Do NOT open, list or grep anything else (no key directory, no other judge's
file, no other chunk, no repository code). Do not try to infer which detector produced a tile.`
const j1Prompt = chunk => `You are the first blind judge labelling ball candidates. ${RULES}

${layout(chunk)}

${blind(chunk, 'j1')}

Work through all views in order, zooming whenever in doubt (borderline box position, faint blobs, boots vs ball). Append your
labels as you go, one "TILE LABEL" line per tile, to ${JUDG}/${chunk}_j1.txt (create it; if it already has lines, continue after the
last labelled tile). Label every tile number from 1 to the largest. Return all labels, the tiles you were unsure about, and short
notes (at most 5 lines).`
const j2Prompt = (chunk, tiles) => `You are an independent blind judge labelling ${tiles.length} ball candidates. ${RULES}

${layout(chunk)}

${blind(chunk, 'j2')}

Label ONLY these tiles: ${tiles.join(', ')}. Zoom every one of them (in groups of up to 6); use their rows as context. Append one
"TILE LABEL" line per tile to ${JUDG}/${chunk}_j2.txt (create it; if it already has lines, continue with the listed tiles not yet
labelled). Return the labels, the tiles you were unsure about, and short notes (at most 5 lines).`
const adjPrompt = (chunk, tiles) => `You are the deciding blind judge for ${tiles.length} ball-candidate tiles on which two judges
disagreed. ${RULES}

${layout(chunk)}

${blind(chunk, 'adj')}

Tiles to decide: ${tiles.join(', ')}. Zoom every one of them. Write "TILE LABEL" lines to ${JUDG}/${chunk}_adj.txt and return the
labels (notes at most 5 lines).`

const isB = l => l === 'B'
const asMap = r => Object.fromEntries(((r && r.labels) || []).map(x => [x.tile, x.label]))

const results = await pipeline(
  args.chunks,
  chunk => agent(j1Prompt(chunk), { label: `j1:${chunk}`, phase: 'Judge', schema: LABELS }),
  async (r1, chunk) => {
    const a = asMap(r1)
    const tiles = Object.keys(a).map(Number).sort((x, y) => x - y)
    const unsure = new Set((r1 && r1.unsure_tiles) || [])
    // Recheck: every tile not labelled N, every unsure tile, and 1 in 7 of the rest (fixed by tile number).
    const random = tiles.filter(t => a[t] === 'N' && !unsure.has(t) && t % 7 === 0)
    const re = tiles.filter(t => a[t] !== 'N' || unsure.has(t) || random.includes(t))
    const r2 = await agent(j2Prompt(chunk, re), { label: `j2:${chunk}`, phase: 'Recheck', schema: LABELS })
    const b = asMap(r2)
    const dis = re.filter(t => (t in b) && isB(a[t]) !== isB(b[t]))
    let adj = {}
    if (dis.length) {
      const r3 = await agent(adjPrompt(chunk, dis), { label: `adj:${chunk}`, phase: 'Adjudicate', schema: LABELS })
      adj = asMap(r3)
    }
    log(`${chunk}: ${tiles.length} tiles, rechecked ${re.length} (random ${random.length}), disagreements ${dis.length}`)
    return { chunk, j1: a, j2: b, adj, random, disagreements: dis }
  },
)
return results
