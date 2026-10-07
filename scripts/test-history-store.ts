// R018 — undo/redo HISTORY store (src/history/store.ts). Runs the REAL
// GraphStore + the REAL history store in-process. Node-only (no DOM/localStorage).
//
// The history store subscribes to GraphStore and pushes a full snapshot of the
// PREVIOUS state whenever the *structural* shape changes (invariant 6: pure
// position drags never enter history). This harness proves the documented
// contract, checks the "no aliasing / no phantom step / valid after every step"
// invariants, and is written so it can go RED: see the mutation proofs in the
// task (aliasing, phantom step, position-in-history).
//
// Run: npm run test:history-store   (needs a shell with node + tsx)
//
// Each case's comment names the bug it would catch.

import { useGraphStore, captureStructuralSnapshot, type GraphSnapshot } from '../src/canvas/GraphStore'
import { validateGraphState } from '../src/canvas/invariants'
import { useHistoryStore } from '../src/history/store'

let passed = 0
let failed = 0

function check(name: string, cond: boolean, detail = ''): void {
  if (cond) { passed++; process.stdout.write(`  ✓ ${name}\n`) }
  else { failed++; process.stdout.write(`  ✗ ${name}${detail ? '  ' + detail : ''}\n`) }
}

const g = () => useGraphStore.getState()
const h = () => useHistoryStore.getState()

/** Canonical structural shape (ids, layer types, params, edges — no positions). */
function structural(): string {
  return JSON.stringify(captureStructuralSnapshot(g()))
}

/** Reset the graph AND the history baseline so the next mutation is the first
 *  pushed step. resetGraph fires the subscriber (baseline), then clear() drops
 *  the pushed entry and the past. */
function fresh(): void {
  g().resetGraph()
  h().clear()
}

function nodeParams(id: string): Record<string, unknown> {
  return g().nodes.find((n) => n.id === id)?.data.params ?? {}
}

// ── 1. undo/redo round-trips restore the exact STRUCTURAL snapshot ───────────
// Catches: a snapshot that isn't captured/restored byte-for-byte (lost params,
// lost edges, position sneaking into the structural compare).
console.log('undo/redo round-trips by operation kind')
{
  // add node
  fresh()
  const before = structural()
  const a = g().addLayer('Linear', { x: 0, y: 0 }, { params: { in_features: 3, out_features: 8 } })
  const afterAdd = structural()
  check('add: structural changed', afterAdd !== before)
  h().undo()
  check('add: undo restores exact structural snapshot', structural() === before, structural())
  check('add: node gone after undo', !g().nodes.some((n) => n.id === a))
  h().redo()
  check('add: redo restores exact post-add snapshot', structural() === afterAdd)

  // connect
  fresh()
  const b = g().addLayer('ReLU', { x: 0, y: 0 })
  h().clear()
  const preConnect = structural()
  g().connectNodes('input', b)
  const postConnect = structural()
  check('connect: edge present', g().edges.some((e) => e.source === 'input' && e.target === b))
  h().undo()
  check('connect: undo restores exact structural snapshot', structural() === preConnect, structural())
  check('connect: edge gone after undo', !g().edges.some((e) => e.source === 'input' && e.target === b))
  h().redo()
  check('connect: redo restores the edge', structural() === postConnect)

  // update params
  fresh()
  const c = g().addLayer('Linear', { x: 0, y: 0 }, { params: { in_features: 3, out_features: 8 } })
  h().clear()
  const preUpdate = structural()
  g().updateNodeParams(c, { out_features: 64 })
  const postUpdate = structural()
  check('update: post state has new param', nodeParams(c).out_features === 64)
  h().undo()
  check('update: undo restores exact structural snapshot', structural() === preUpdate)
  check('update: param rolled back', nodeParams(c).out_features === 8)
  h().redo()
  check('update: redo reapplies the param', structural() === postUpdate)

  // delete node
  fresh()
  const d = g().addLayer('ReLU', { x: 0, y: 0 })
  g().connectNodes('input', d)
  h().clear()
  const preDelete = structural()
  g().deleteNode(d)
  const postDelete = structural()
  check('delete: node gone', !g().nodes.some((n) => n.id === d))
  h().undo()
  check('delete: undo restores node+edge exact snapshot', structural() === preDelete)
  h().redo()
  check('delete: redo removes it again', structural() === postDelete)

  // delete edge (onEdgesChange)
  fresh()
  const e = g().addLayer('ReLU', { x: 0, y: 0 })
  g().connectNodes('input', e)
  h().clear()
  const preEdgeDelete = structural()
  const edgeId = g().edges.find((x) => x.target === e)!.id
  g().onEdgesChange([{ type: 'remove', id: edgeId }])
  check('delete-edge: edge removed', !g().edges.some((x) => x.id === edgeId))
  h().undo()
  check('delete-edge: undo restores exact structural snapshot', structural() === preEdgeDelete)
  h().redo()
  check('delete-edge: redo removes it again', !g().edges.some((x) => x.id === edgeId))

  // auto-port sync: connect an outer node → a Subgraph node creates a read-only
  // proxy input INSIDE the subgraph (subgraphPorts.reconcileSubgraphPorts).
  fresh()
  const sg = g().addLayer('Subgraph', { x: 0, y: 0 })
  g().addLayer('Input', { x: 0, y: 0 }, { params: { name: 'x', shape: [1, 4] } })
  h().clear()
  const preProxy = structural()
  g().connectNodes('input', sg)
  const sgNodeAfter = g().nodes.find((n) => n.id === sg)!
  const inner = sgNodeAfter.data.params.subgraph as GraphSnapshot
  check('proxy: an inner input proxy was created', !!inner && inner.nodes.some((n) => n.params._proxyOf === 'input'))
  h().undo()
  check('proxy: undo restores exact pre-connect snapshot', structural() === preProxy)
  // after undo the proxy must be released/removed again
  const sgAfterUndo = g().nodes.find((n) => n.id === sg)!
  const innerAfterUndo = sgAfterUndo.data.params.subgraph as GraphSnapshot
  check('proxy: undo removes the auto-created proxy', !innerAfterUndo.nodes.some((n) => n.params._proxyOf === 'input'))
  h().redo()
  check('proxy: redo recreates the proxy', (() => {
    const s = g().nodes.find((n) => n.id === sg)!.data.params.subgraph as GraphSnapshot
    return s.nodes.some((n) => n.params._proxyOf === 'input')
  })())

  // load: loadSnapshot replaces the whole graph
  fresh()
  g().addLayer('ReLU', { x: 0, y: 0 })
  h().clear()
  const preLoad = structural()
  g().loadSnapshot({
    nodes: [
      { id: 'i', layerType: 'Input', params: { name: 'x', shape: [1, 5] } },
      { id: 'l', layerType: 'Linear', params: { in_features: 5, out_features: 2 } },
    ],
    edges: [{ source: 'i', target: 'l' }],
  })
  check('load: graph replaced', g().nodes.some((n) => n.id === 'l'))
  const loadedShape = structural()
  h().undo()
  // Opening a document is NOT an undoable edit (section 6): undo must not bring back the graph that
  // belonged to the previously open document.
  check('load: undo does not restore the pre-load (other document) snapshot', structural() === loadedShape && structural() !== preLoad)
}

// ── 2. redo stack is cleared by a NEW mutation after an undo ─────────────────
// Catches: stale redo entries surviving a divergent edit (time-travel corruption).
console.log('redo is cleared by a new mutation after undo')
{
  fresh()
  const a = g().addLayer('Linear', { x: 0, y: 0 })
  g().addLayer('ReLU', { x: 0, y: 0 })
  h().undo()
  check('redo available after undo', h().future.length > 0 && h().canRedo)
  g().addLayer('GELU', { x: 0, y: 0 })
  check('new mutation empties the redo stack', h().future.length === 0 && !h().canRedo)
  check('the new mutation is live', g().nodes.some((n) => n.data.layerType === 'GELU'))
  // the pre-undo add is still undoable (past retains it)
  check('undo stack still has history', h().past.length > 0 && h().canUndo)
  void a
}

// ── 3. MAX_HISTORY (50) cap — 51st mutation drops the oldest ─────────────────
// Catches: unbounded growth (memory) or an out-of-range slice on undo/redo.
console.log('MAX_HISTORY cap')
{
  fresh()
  const first = structural()
  for (let i = 0; i < 51; i++) g().addLayer('ReLU', { x: 0, y: i })
  check('past length capped at 50', h().past.length === 50, `len=${h().past.length}`)
  check('oldest entry dropped: past[0] is NOT the pre-first state', JSON.stringify(h().past[0]) !== first)
  check('canUndo true at cap', h().canUndo)

  // undo 60× must never throw and never go below 0
  let threw = false
  try { for (let i = 0; i < 60; i++) h().undo() } catch { threw = true }
  check('undo ×60 never throws', !threw)
  check('past never goes below 0', h().past.length === 0)
  check('future capped at 50', h().future.length <= 50, `len=${h().future.length}`)
  check('canUndo false at empty past', !h().canUndo)

  // redo 60× likewise
  threw = false
  try { for (let i = 0; i < 60; i++) h().redo() } catch { threw = true }
  check('redo ×60 never throws', !threw)
  check('past capped at 50 after redo', h().past.length <= 50, `len=${h().past.length}`)
  check('future empty after full redo', h().future.length === 0, `len=${h().future.length}`)
}

// ── 4. position-only changes never enter history (invariant 6) ───────────────
// Catches: a drag polluting the undo stack; autoLayout registering as an edit.
console.log('invariant 6: position-only changes are not history')
{
  fresh()
  const a = g().addLayer('Linear', { x: 0, y: 0 })
  h().clear()
  const pastBefore = h().past.length

  // node drag (react-flow onNodesChange position change)
  g().onNodesChange([{ id: a, type: 'position', position: { x: 111, y: 222 } }])
  check('drag: position applied', g().nodes.find((n) => n.id === a)?.position.x === 111)
  check('drag: no history entry added', h().past.length === pastBefore)

  // autoLayout
  g().autoLayout()
  check('autoLayout: no history entry added', h().past.length === pastBefore)

  // a structural change AFTER a move keeps the moved positions on undo
  const movedX = g().nodes.find((n) => n.id === a)!.position.x
  g().addLayer('ReLU', { x: 0, y: 0 })
  check('move+struct: structural change pushed one entry', h().past.length === pastBefore + 1)
  h().undo()
  check('move+struct: undo keeps the moved position', g().nodes.find((n) => n.id === a)?.position.x === movedX)
}

// ── 5. rejected mutations are not phantom steps ──────────────────────────────
// Catches: an invalid connect (no state change) leaving a bogus undo entry, or a
// sanitised param update being recorded with the RAW (pre-coerce) value.
console.log('rejected / sanitised mutations')
{
  fresh()
  g().addLayer('Linear', { x: 0, y: 0 })
  h().clear()
  const before = h().past.length

  check('invalid connect (unknown source) rejected', g().connectNodes('ghost', 'input') === false)
  check('invalid connect (self-loop) rejected', g().connectNodes('input', 'input') === false)
  check('no phantom history entry from a rejected mutation', h().past.length === before)

  // junk params: coerceParams rewrites them; the committed state IS a real
  // structural change, so it is recorded — but as the SANITISED state.
  const lin = g().nodes.find((n) => n.data.layerType === 'Linear')!
  g().updateNodeParams(lin.id, { in_features: 'not-a-number', out_features: 0.9 })
  check('sanitised update recorded exactly once', h().past.length === before + 1)
  const coerced = nodeParams(lin.id)
  check('recorded state uses coerced values (no raw junk)',
    Number.isInteger(coerced.in_features) && Number.isInteger(coerced.out_features),
    JSON.stringify(coerced))
}

// ── 6. a document swap starts a fresh history ────────────────────────────────
// Catches (found by this suite): opening file B pushed file A's graph onto the undo stack, so the
// FIRST undo after opening B restored A's graph into B's canvas — and autosave then wrote it into
// B's file (silent data corruption across files). loadSnapshot / resetGraph are document swaps,
// not edits: history is cleared, the loaded graph becomes the baseline.
console.log('document swap (loadSnapshot / resetGraph) starts a fresh history')
{
  fresh()
  g().addLayer('ReLU', { x: 0, y: 0 })
  g().addLayer('GELU', { x: 0, y: 0 })
  check('precondition: file A has undoable edits', h().canUndo && h().past.length === 2)
  const loadedOk = g().loadSnapshot({
    nodes: [{ id: 'i', layerType: 'Input', params: { name: 'x', shape: [1, 2] } }],
    edges: [],
  })
  check('loadSnapshot of "file B" succeeds', loadedOk)
  const loaded = structural()
  check('after opening B: nothing to undo', !h().canUndo && h().past.length === 0 && h().future.length === 0)
  h().undo()
  check('undo right after opening B does NOT restore A (B unchanged)', structural() === loaded)
  g().addLayer('Linear', { x: 0, y: 0 })
  check('an edit inside B is undoable again', h().canUndo && h().past.length === 1)
  h().undo()
  check('undo after an edit in B returns to the LOADED graph', structural() === loaded)
  h().redo()
  check('redo works inside B', structural() !== loaded)
  // File → New is a document swap too.
  g().addLayer('Tanh', { x: 0, y: 0 })
  g().resetGraph()
  check('after File → New: nothing to undo / redo', !h().canUndo && !h().canRedo)
  const fresh0 = structural()
  h().undo()
  check('undo after File → New does not resurrect the previous document', structural() === fresh0)
  // A REJECTED load (invalid graph) is not a swap: the old document and its history stay intact.
  fresh()
  g().addLayer('ReLU', { x: 0, y: 0 })
  const before = structural()
  const rejected = g().loadSnapshot({ nodes: [{ id: 'x', layerType: 'Input', params: {} }], edges: [{ source: 'x', target: 'ghost' }] })
  check('an invalid snapshot is rejected', rejected === false)
  check('a rejected load leaves graph and history untouched', structural() === before && h().canUndo && h().past.length === 1)
}

// ── 7. revision monotonicity across undo/redo ────────────────────────────────
// Catches: an undo that does not bump revision → a stale in-flight inference for
// the old graph would be accepted (see verify-graph-revision.ts).
console.log('revision monotonicity')
{
  fresh()
  const r0 = g().revision
  g().addLayer('Linear', { x: 0, y: 0 })
  const r1 = g().revision
  check('add bumps revision', r1 > r0)
  h().undo()
  const r2 = g().revision
  check('undo bumps revision (stale inference discarded)', r2 > r1)
  h().redo()
  const r3 = g().revision
  check('redo bumps revision', r3 > r2)
  h().undo(); h().redo(); h().undo()
  check('revision never decreases across undo/redo', g().revision >= r3)
}

// ── 8. history snapshots share no mutable state with the live graph ──────────
// Catches: aliasing — captureSnapshot stores `n.data.params` BY REFERENCE, so an
// in-place mutation of a live node's params object would rewrite history.
console.log('no aliasing between history and live graph')
{
  fresh()
  const a = g().addLayer('Linear', { x: 0, y: 0 }, { params: { in_features: 3, out_features: 8 } })
  // Adding an UNRELATED node makes the subscriber push a snapshot of the
  // current state, which still references node A's live `params` object.
  g().addLayer('ReLU', { x: 0, y: 0 })
  const pastIdx = h().past.length - 1
  const stored = h().past[pastIdx].nodes.find((n) => n.id === a)
  check('past snapshot contains the node', !!stored)
  const storedParamBefore = stored?.params.out_features
  // mutate the live params object IN PLACE (no store write)
  const live = g().nodes.find((n) => n.id === a)!
  live.data.params.out_features = 999
  check('live in-place mutation applied', nodeParams(a).out_features === 999)
  check('past snapshot unchanged (deep-copied, not aliased)',
    h().past[pastIdx].nodes.find((n) => n.id === a)?.params.out_features === storedParamBefore,
    JSON.stringify(h().past[pastIdx].nodes.find((n) => n.id === a)?.params))
}

// ── 9. randomized sequence: undo/redo NEVER yields an invalid graph ──────────
// Catches: an undo/redo path that commits an invalid graph (dangling edge, cycle,
// self-loop). Seeded RNG — on failure the seed + the step list are printed.
console.log('randomized: graph stays valid through 300 mixed steps')
{
  // mulberry32 — deterministic, seeded, no Math.random.
  const seed = 0x5eed1234
  let s = seed >>> 0
  const rng = (): number => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const pick = <T,>(arr: T[]): T => arr[Math.floor(rng() * arr.length)]

  fresh()
  const steps: string[] = []
  let invalidAt = -1
  let invalidIssues = ''

  for (let i = 0; i < 300; i++) {
    const op = pick(['add', 'connect', 'update', 'deleteNode', 'deleteEdge', 'undo', 'redo'] as const)
    const ids = g().nodes.map((n) => n.id)
    switch (op) {
      case 'add': {
        const lt = pick(['Linear', 'ReLU', 'Concat', 'Subgraph', 'GELU'])
        const pos = { x: rng() * 500, y: rng() * 500 }
        const id = g().addLayer(lt, pos)
        steps.push(`add ${lt} -> ${id}`)
        break
      }
      case 'connect': {
        if (ids.length < 2) { steps.push('connect(skip)'); break }
        const src = pick(ids)
        const tgt = pick(ids)
        const ok = g().connectNodes(src, tgt)
        steps.push(`connect ${src}->${tgt} = ${ok}`)
        break
      }
      case 'update': {
        if (ids.length === 0) { steps.push('update(skip)'); break }
        const id = pick(ids)
        g().updateNodeParams(id, { out_features: Math.floor(rng() * 20) + 1, inplace: rng() > 0.5 })
        steps.push(`update ${id}`)
        break
      }
      case 'deleteNode': {
        if (ids.length === 0) { steps.push('deleteNode(skip)'); break }
        const id = pick(ids.filter((x) => x !== 'input'))
        if (!id) { steps.push('deleteNode(skip)'); break }
        g().deleteNode(id)
        steps.push(`deleteNode ${id}`)
        break
      }
      case 'deleteEdge': {
        const edges = g().edges
        if (edges.length === 0) { steps.push('deleteEdge(skip)'); break }
        const ed = pick(edges)
        g().onEdgesChange([{ type: 'remove', id: ed.id! }])
        steps.push(`deleteEdge ${ed.source}->${ed.target}`)
        break
      }
      case 'undo': steps.push('undo'); h().undo(); break
      case 'redo': steps.push('redo'); h().redo(); break
    }

    const v = validateGraphState(g().nodes, g().edges)
    if (!v.ok) {
      invalidAt = i
      invalidIssues = JSON.stringify(v.issues.filter((x) => x.severity === 'error'))
      break
    }
  }

  check('graph valid after every step of the randomized sequence', invalidAt === -1,
    invalidAt >= 0 ? `seed=${seed} step=${invalidAt} ${invalidIssues}\nsteps:\n${steps.join('\n')}` : '')
  check('randomized sequence actually exercised undo/redo', steps.includes('undo') && steps.includes('redo'))
  check('randomized sequence actually mutated the graph', g().nodes.length >= 1)
}

console.log(`\n${failed === 0 ? '✓ all history-store checks passed' : `✗ ${failed} check(s) failed`} (${passed} passed)`)
process.exit(failed === 0 ? 0 : 1)
