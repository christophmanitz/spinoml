// Phase 3 — GraphStore correctness (TODO §4). Exercises every mutation listed
// in §4.2 (add/remove/update node, add/remove/rewire edge, change params/
// input/output) plus the invalid cases (§4.2) and the validate-before-commit
// guard (§4.3). Runs the REAL zustand store in-process (no DOM needed).
//
// Run: npm run test:graphstore   (needs a shell with node + tsx, no python)

import { useGraphStore } from '../src/canvas/GraphStore'
import { validateGraphState, wouldCreateCycle } from '../src/canvas/invariants'
import { defaultParamsFor } from '../src/layers/registry'
import type { LayerNode, GraphSnapshot } from '../src/canvas/GraphStore'
import type { Edge } from '@xyflow/react'

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) {
    console.log(`  ✓ ${name}`)
  } else {
    failures++
    console.log(`  ✗ ${name} ${detail}`)
  }
}
function hasErr(issues: { code: string; severity: string }[], code?: string) {
  return issues.some((i) => i.severity === 'error' && (!code || i.code === code))
}

// ── helpers ────────────────────────────────────────────────────────────────
function layerNode(id: string, layerType: string, params: Record<string, unknown> = {}, extra: Record<string, unknown> = {}): LayerNode {
  return {
    id, type: 'layer', position: { x: 0, y: 0 },
    data: { layerType, params: { ...defaultParamsFor(layerType), ...params } },
    ...extra,
  } as LayerNode
}
function graphEdge(id: string, source: string, target: string, extra: Record<string, unknown> = {}): Edge {
  return { id, source, target, ...extra } as Edge
}
function snap(nodes: LayerNode[], edges: Edge[]): GraphSnapshot {
  return {
    nodes: nodes.map((n) => ({ id: n.id, layerType: n.data.layerType, params: n.data.params, position: n.position })),
    edges: edges.map((e) => ({ source: e.source, target: e.target })),
  }
}
function reset() {
  useGraphStore.getState().resetGraph()
}
function ids() {
  return useGraphStore.getState().nodes.map((n) => n.id)
}

// ── 1. validator: healthy DAG ──────────────────────────────────────────────
console.log('validator: healthy DAG')
{
  const nodes = [
    layerNode('input', 'Input', { name: 'x', shape: [1, 3] }),
    layerNode('a', 'Linear', { in_features: 3, out_features: 32 }),
    layerNode('b', 'ReLU', { inplace: false }),
    layerNode('out', 'Output', { name: 'out' }),
  ]
  const edges = [graphEdge('e1', 'input', 'a'), graphEdge('e2', 'a', 'b'), graphEdge('e3', 'b', 'out')]
  const v = validateGraphState(nodes, edges)
  check('ok', v.ok)
  check('no issues', v.issues.length === 0, JSON.stringify(v.issues))
  check('wouldCreateCycle(after-edge unaffected)', !wouldCreateCycle(nodes, edges, 'a', 'b'))
  check('wouldCreateCycle(input→out) at bottom', !wouldCreateCycle(nodes, edges, 'input', 'out'))
}

// ── 2. validator: invalid cases ────────────────────────────────────────────
console.log('validator: invalid cases')
{
  const nodes = [layerNode('a', 'Linear', { in_features: 1, out_features: 1 })]
  check('dup-node', hasErr(validateGraphState([...nodes, layerNode('a', 'Linear')], []).issues, 'dup-node'))
  check('unknown-layer', hasErr(validateGraphState([layerNode('a', 'Nope')], []).issues, 'unknown-layer'))
  check('dangling source', hasErr(validateGraphState(nodes, [graphEdge('e1', 'ghost', 'a')]).issues, 'unknown-node'))
  check('dangling target', hasErr(validateGraphState(nodes, [graphEdge('e1', 'a', 'ghost')]).issues, 'unknown-node'))
  check('self-loop', hasErr(validateGraphState(nodes, [graphEdge('e1', 'a', 'a')]).issues, 'self-loop'))
  check('dup-edge-id', hasErr(validateGraphState(nodes, [graphEdge('e1', 'a', 'a'), graphEdge('e1', 'a', 'a')]).issues, 'dup-edge'))
  // single-input fan-in → WARN but loadable (ok stays true)
  const fan = validateGraphState(
    [layerNode('in1', 'Input', { name: 'a' }), layerNode('in2', 'Input', { name: 'b' }), layerNode('tgt', 'Linear', { in_features: 1, out_features: 1 })],
    [graphEdge('e1', 'in1', 'tgt'), graphEdge('e2', 'in2', 'tgt')],
  )
  check('single-input fan-in → warn, not error', fan.issues.some((i) => i.code === 'single-input' && i.severity === 'warn'))
  check('fan-in still ok(true)', fan.ok === true)
  // invalid handle: empty-string handle is an error
  check('invalid-handle (empty)', hasErr(validateGraphState(nodes, [graphEdge('e1', 'a', 'a', { sourceHandle: '' })]).issues, 'invalid-handle'))
  // invalid params: out-of-range int survives coerce → validator catches
  const badParam = validateGraphState([layerNode('c', 'Conv2d', { in_channels: 0, out_channels: 0 })], [])
  check('param-range (out_channels 0 < min 1)', hasErr(badParam.issues, 'param-range'))
  check('param-type (string in_features)', hasErr(validateGraphState([layerNode('d', 'Linear', { in_features: 'abc' })], []).issues, 'param-type'))
  check('select-out-of-range', hasErr(validateGraphState([layerNode('e', 'GraphConv', { aggr: 'bogus' })], []).issues, 'select-out-of-range'))
  // cycle detection on a real cycle
  const cyc = validateGraphState(
    [layerNode('a', 'Linear'), layerNode('b', 'ReLU')],
    [graphEdge('e1', 'a', 'b'), graphEdge('e2', 'b', 'a')],
  )
  check('cycle → error', hasErr(cyc.issues, 'cycle'))
}

// ── 3. store mutations: valid flow ─────────────────────────────────────────
console.log('store: valid mutation flow')
reset()
{
  const g = () => useGraphStore.getState()
  const a = g().addLayer('Linear', { x: 0, y: 0 })
  const b = g().addLayer('ReLU', { x: 0, y: 0 })
  const m = g().addLayer('Concat', { x: 0, y: 0 })
  check('add node a+b+m exist', ids().includes(a) && ids().includes(b) && ids().includes(m))

  check('connect input→a', g().connectNodes('input', a) === true)
  check('connect a→b', g().connectNodes(a, b) === true)
  check('single-input target rewired: connect input→b replaces a→b',
    g().connectNodes('input', b) === true &&
    !g().edges.some((e) => e.source === a && e.target === b) &&
    g().edges.some((e) => e.source === 'input' && e.target === b))

  // merge (Concat) accepts ≥2 inputs — fan-in allowed
  check('connect input→m', g().connectNodes('input', m) === true)
  check('connect a→m (merge fan-in)', g().connectNodes(a, m) === true)
  check('no error-level issues', g().graphIssues().filter((i) => i.severity === 'error').length === 0, JSON.stringify(g().graphIssues()))

  // update params (change params)
  g().updateNodeParams(a, { out_features: 128 })
  check('params updated', (g().nodes.find((n) => n.id === a)?.data.params.out_features) === 128)

  // change layer (swap type in place)
  g().replaceNodeLayer(a, 'GELU')
  check('layer swapped to GELU', g().nodes.find((n) => n.id === a)?.data.layerType === 'GELU')

  // remove edge via onEdgesChange
  const toRemove = g().edges.find((e) => e.source === 'input' && e.target === b)?.id
  g().onEdgesChange([{ type: 'remove', id: toRemove! }])
  check('edge removed', !g().edges.some((e) => e.id === toRemove))
  check('graph clean after edge removal', g().graphIssues().filter((i) => i.severity === 'error').length === 0)

  // delete node → incident edges removed with it
  g().deleteNode(b)
  check('node deleted', !ids().includes(b))
  check('no dangling edges after delete', !g().edges.some((e) => e.source === b || e.target === b))
  check('clean after delete', g().graphIssues().filter((i) => i.severity === 'error').length === 0, JSON.stringify(g().graphIssues()))
}

// ── 4. store mutations: invalid cases rejected ─────────────────────────────
console.log('store: invalid mutations rejected')
reset()
{
  const g = () => useGraphStore.getState()
  const a = g().addLayer('Linear', { x: 0, y: 0 })
  const before = g().edges.length

  check('unknown source → false', g().connectNodes('ghost', a) === false)
  check('unknown target → false', g().connectNodes('input', 'ghost') === false)
  check('self-loop input→input → false', g().connectNodes('input', 'input') === false)
  check('nothing was added', g().edges.length === before)

  // cycle: input → a → b → (back to a) rejected
  const b = g().addLayer('ReLU', { x: 0, y: 0 })
  g().connectNodes('input', a)
  g().connectNodes(a, b)
  check('cycle-closing edge b→a → false', g().connectNodes(b, a) === false)
  check('cycle-closing edge a→input → false', g().connectNodes(a, 'input') === false)
  check('still acyclic', g().graphIssues().reduce((n, i) => n + (isErr(i) ? 1 : 0), 0) === 0, JSON.stringify(g().graphIssues()))
  function isErr(i: { severity: string }) { return i.severity === 'error' }

  // unknown layer → refused without committing
  const beforeNodes = g().nodes.length
  const ret = g().addLayer('BogusLayer', { x: 0, y: 0 })
  check('addLayer(BogusLayer) → empty id', ret === '')
  check('node not added', g().nodes.length === beforeNodes)

  // replaceNodeLayer to an unknown type → no-op
  g().replaceNodeLayer(a, 'Nope')
  check('replaceNodeLayer(Nope) no-op', g().nodes.find((n) => n.id === a)?.data.layerType === 'Linear')

  // invalid param value passes coerce (no clamp) → flagged by validator
  g().updateNodeParams(a, { out_features: 0 })
  check('out-of-range param flagged', g().graphIssues().some((i) => i.code === 'param-range'))
}

// ── 5. loadSnapshot: no invalid state may become authoritative (§4.3) ──────
console.log('loadSnapshot: validate before commit')
reset()
{
  const g = () => useGraphStore.getState()

  check('valid snapshot loads', g().loadSnapshot(snap(
    [layerNode('i', 'Input'), layerNode('o', 'Output', { name: 'y' })],
    [graphEdge('e1', 'i', 'o')],
  )) === true)
  check('state replaced', ids().includes('i') && ids().includes('o'))

  check('dangling-edge snapshot rejected', g().loadSnapshot(snap(
    [layerNode('i', 'Input')],
    [graphEdge('e1', 'i', 'ghost')],
  )) === false)
  check('state preserved after rejection', ids().includes('o') && !ids().includes('ghost'))

  check('cycle snapshot rejected', g().loadSnapshot(snap(
    [layerNode('a', 'Linear'), layerNode('b', 'ReLU')],
    [graphEdge('e1', 'a', 'b'), graphEdge('e2', 'b', 'a')],
  )) === false)
  check('self-loop snapshot rejected', g().loadSnapshot(snap(
    [layerNode('a', 'Linear')],
    [graphEdge('e1', 'a', 'a')],
  )) === false)
  check('unknown-layer snapshot rejected', g().loadSnapshot(snap(
    [layerNode('a', 'Nope')],
    [],
  )) === false)

  // warn-level (single-input fan-in) still loads — legacy files stay openable
  check('warn-only snapshot loads', g().loadSnapshot(snap(
    [layerNode('in1', 'Input', { name: 'a' }), layerNode('in2', 'Input', { name: 'b' }),
     layerNode('lin', 'Linear', { in_features: 1, out_features: 1 })],
    [graphEdge('e1', 'in1', 'lin'), graphEdge('e2', 'in2', 'lin')],
  )) === true)

  // loadSnapshot regenerates edge ids (`e1…`) — a snapshot with repeated
  // source/target pairs still yields globally unique edge ids, so it loads.
  check('repeated-pair snapshot loads with unique ids', g().loadSnapshot(snap(
    [layerNode('i1', 'Input', { name: 'x' }), layerNode('o', 'Concat', { dim: 1 })],
    [graphEdge('x', 'i1', 'o'), graphEdge('y', 'i1', 'o')],
  )) === true
    && new Set(g().edges.map((e) => e.id)).size === g().edges.length)
}

// ── 6. connectNodes edge-id generation is collision-safe ───────────────────
console.log('store: edge id generation')
reset()
{
  const g = () => useGraphStore.getState()
  const a = g().addLayer('Linear', { x: 0, y: 0 })
  const b = g().addLayer('ReLU', { x: 0, y: 0 })
  g().connectNodes('input', a)
  g().connectNodes(a, b)
  // delete the middle edge, then reconnect → must NOT reuse a colliding id
  const mid = g().edges.find((e) => e.source === a && e.target === b)!.id
  g().onEdgesChange([{ type: 'remove', id: mid }])
  g().connectNodes(a, b)
  const ids2 = new Set(g().edges.map((e) => e.id))
  check('edge ids unique after delete+reconnect', ids2.size === g().edges.length, JSON.stringify([...ids2]))
  check('validator agrees', !hasErr(g().graphIssues(), 'dup-edge'))
}

console.log(failures === 0 ? '\n✓ all graph-store checks passed' : `\n✗ ${failures} check(s) failed`)
process.exit(failures === 0 ? 0 : 1)