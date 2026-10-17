export const meta = {
  name: 'ball-recheck-train-B',
  description: 'Second blind judge on ~250 ball tiles, third judge on B / not-B disagreements',
  phases: [{ title: 'Recheck', detail: 'second blind judge per chunk' }, { title: 'Adjudicate', detail: 'third judge on B / not-B disagreements' }],
}
const BASE = '/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5/train_B'
const ZOOM = '/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5/zoom_chunk.py'
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

const layout = chunk => `LAYOUT: ${BASE}/${chunk}/ holds sheet_K.jpg (5 rows of 6 tiles; sheet 4 has only 2 rows), view_K_P.png
(1.3x enlargements: P = 1 is rows 1-3 of sheet K, P = 2 rows 4-5) and layout.json. Tiles are numbered 1-100 in their headers.
EVERY TILE IS INDEPENDENT: each comes from a different frame (often a different match); rows and neighbours carry no meaning.`

const blind = (chunk, tag) => `STRICT BLINDING: read ONLY files inside ${BASE}/${chunk}/ and run only
  python ${ZOOM} ${BASE}/${chunk} ${JUDG}/z_${chunk}_${tag}.png T1 T2 ...
(3x zoom of the listed tiles; then read that png). Do NOT open, list or grep anything else (no key directory, no other judge's
file such as *_j1.txt, no other chunk, no repository code). Do not try to infer which detector produced a tile.`

const j2Prompt = (chunk, tiles) => `You are an independent blind judge labelling ${tiles.length} ball candidates. ${RULES}

${layout(chunk)}

${blind(chunk, 'j2')}

Label ONLY these tiles: ${tiles.join(', ')}. Zoom every one of them (in groups of up to 6). Append one "TILE LABEL" line per tile
to ${JUDG}/${chunk}_j2.txt (create it; if it already has lines, continue with the listed tiles not yet labelled). Return the
labels, the tiles you were unsure about, and short notes (at most 5 lines).`

const adjPrompt = (chunk, tiles) => `You are the deciding blind judge for ${tiles.length} ball-candidate tiles on which two judges
disagreed. ${RULES}

${layout(chunk)}

${blind(chunk, 'adj')}

Tiles to decide: ${tiles.join(', ')}. Zoom every one of them. Write "TILE LABEL" lines to ${JUDG}/${chunk}_adj.txt and return the
labels (notes at most 5 lines).`

const isB = l => l === 'B'
const asMap = r => Object.fromEntries(((r && r.labels) || []).map(x => [x.tile, x.label]))
const CHUNKS = Object.keys(args)

const results = await pipeline(
  CHUNKS,
  chunk => agent(j2Prompt(chunk, args[chunk].recheck), { label: `j2:${chunk}`, phase: 'Recheck', schema: LABELS }),
  async (r2, chunk) => {
    const a = args[chunk].j1
    const b = asMap(r2)
    const re = args[chunk].recheck
    const missing = re.filter(t => !(t in b))
    const dis = re.filter(t => (t in b) && isB(a[t]) !== isB(b[t]))
    let adj = {}
    if (dis.length) {
      const r3 = await agent(adjPrompt(chunk, dis), { label: `adj:${chunk}`, phase: 'Adjudicate', schema: LABELS })
      adj = asMap(r3)
    }
    log(`${chunk}: rechecked ${re.length}, j2 missing ${missing.length}, disagreements ${dis.length}`)
    return { chunk, j2: b, adj, missing, disagreements: dis, unsure: r2 && r2.unsure_tiles, notes: r2 && r2.notes }
  },
)
return results
