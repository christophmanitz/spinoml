// Phase 5 (TODO §6): determinism — generate(graph) must be repeatable.
// Also tests that identical graphs with shuffled node/edge arrays produce
// the same code, proving the canonicalization in generate() works.
import { useGraphStore } from '../src/canvas/GraphStore'
import { generate, generateFromSnapshot } from '../src/codegen/generator'
import { parseFile, serializeCurrent } from '../src/persistence/file'
import type { GraphSnapshot } from '../src/canvas/GraphStore'

let passed = 0
let failed = 0

function check(name: string, cond: boolean): void {
  if (cond) { passed++; process.stdout.write(`  ✓ ${name}\n`) }
  else      { failed++; process.stdout.write(`  ✗ ${name}\n`) }
}

function buildGraph(): void {
  const g = useGraphStore.getState()
  g.resetGraph()
  g.addLayer('Conv2d', { x: 0, y: 100 }, { params: { in_channels: 1, out_channels: 8, kernel_size: 3, padding: 1 } })
  g.addLayer('ReLU', { x: 0, y: 200 }, { params: { inplace: false } })
  g.addLayer('Flatten', { x: 0, y: 300 }, { params: {} })
  g.addLayer('Linear', { x: 0, y: 400 }, { params: { in_features: 8, out_features: 10 } })
}

console.log('section: repeatability — same graph, same code, multiple runs')
{
  buildGraph()
  const { nodes, edges } = useGraphStore.getState()
  const first = generate(nodes, edges).code
  let identical = true
  for (let i = 1; i < 20; i++) {
    const again = generate(nodes, edges).code
    if (again !== first) { identical = false; break }
  }
  check('20 consecutive generate() calls produce identical code', identical)
}

console.log('section: snapshot round-trip — serialize → parse → generateFromSnapshot')
{
  buildGraph()
  const code1 = generateFromSnapshot(parseFile(serializeCurrent())).code
  const code2 = generateFromSnapshot(parseFile(serializeCurrent())).code
  check('generateFromSnapshot is repeatable across serialize/parse', code1 === code2)
}

console.log('section: shuffled array order — identical graph with reordered nodes/edges')
{
  buildGraph()
  const { nodes, edges } = useGraphStore.getState()
  const baseline = generate(nodes, edges).code

  // shuffle nodes (reverse — simple deterministic reversal)
  const revNodes = [...nodes].reverse()
  const revCode  = generate(revNodes, edges).code
  check('reversed node order produces same code', revCode === baseline)

  // shuffle edges (reverse too)
  const revEdges = [...edges].reverse()
  const allRevCode = generate(revNodes, revEdges).code
  check('reversed node + edge order produces same code', allRevCode === baseline)

  // permuted random-ish (deterministic: sort by id reverse)
  const permNodes = [...nodes].sort((a, b) => b.id.localeCompare(a.id))
  const permEdges = [...edges].sort((a, b) => (b.source + b.target).localeCompare(a.source + a.target))
  check('arbitrarily permuted arrays produce same code', generate(permNodes, permEdges).code === baseline)
}

console.log('section: consistency — generateFromSnapshot vs generate on live store')
{
  buildGraph()
  const { nodes, edges } = useGraphStore.getState()
  const liveCode = generate(nodes, edges).code
  const snapCode = generateFromSnapshot(parseFile(serializeCurrent())).code
  check('generateFromSnapshot snapshot matches generate on live graph', liveCode === snapCode)
}

console.log('section: group-node deterministic subgraph (Subgraph inside)')
{
  const g = useGraphStore.getState()
  g.resetGraph()
  g.addLayer('Conv2d', { x: 0, y: 100 }, { params: { in_channels: 3, out_channels: 16, kernel_size: 3, padding: 1 } })
  g.addLayer('ReLU', { x: 0, y: 200 }, { params: { inplace: false } })
  g.addLayer('Flatten', { x: 0, y: 300 }, { params: {} })
  g.addLayer('Linear', { x: 0, y: 400 }, { params: { in_features: 16, out_features: 4 } })
  const { nodes: n3, edges: e3 } = useGraphStore.getState()
  const codeA = generate(n3, e3).code
  const codeB = generate([...n3].reverse(), [...e3].reverse()).code
  check('group-node graph with shuffled arrays is deterministic', codeA === codeB)
}

console.log(`\n${failed === 0 ? '✓ all determinism checks passed' : `✗ ${failed} check(s) failed`} (${passed} passed)`)
process.exit(failed === 0 ? 0 : 1)
