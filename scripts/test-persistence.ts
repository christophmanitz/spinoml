// Phase 4 (TODO §5): graph persistence — save → close → reload → compare, and
// malformed-file fail-safety. Node-only harness (no localStorage/autosave here).
import { useGraphStore } from '../src/canvas/GraphStore'
import { parseFile, serializeCurrent, FORMAT_VERSION } from '../src/persistence/file'
import type { GraphSnapshot } from '../src/canvas/GraphStore'

let passed = 0
let failed = 0

function check(name: string, cond: boolean): void {
  if (cond) {
    passed++
    process.stdout.write(`  ✓ ${name}\n`)
  } else {
    failed++
    process.stdout.write(`  ✗ ${name}\n`)
  }
}

function snap(nodes: Record<string, unknown>[], edges: Record<string, unknown>[] = []): string {
  return JSON.stringify({ format: 'spinoml', version: FORMAT_VERSION, savedAt: 'x', graph: { nodes, edges } })
}

// build a small but non-trivial model in the store
function buildModel(): GraphSnapshot {
  const g = useGraphStore.getState()
  g.resetGraph()
  g.addLayer('Conv2d', { x: 0, y: 100 }, { params: { in_channels: 1, out_channels: 8, kernel_size: 3, padding: 1 } })
  g.addLayer('ReLU', { x: 0, y: 200 }, { params: { inplace: false } })
  g.addLayer('Flatten', { x: 0, y: 300 }, { params: {} })
  g.addLayer('Linear', { x: 0, y: 400 }, { params: { in_features: 8, out_features: 10 } })
  return serializeCurrentJson()
}

function serializeCurrentJson(): GraphSnapshot {
  return JSON.parse(serializeCurrent()).graph as GraphSnapshot
}

function skipL(g: unknown): string { return JSON.stringify(g) }

console.log('section: save → reload → compare round trip')
{
  const built = buildModel()
  const txt = serializeCurrent()
  const parsed = parseFile(txt)
  check('parseFile(serializeCurrent()) deep-equals stored graph', skipL(parsed) === skipL(built))

  // store untouched by parse; then load from disk text → identical snapshot
  useGraphStore.getState().resetGraph()
  check('loadSnapshot(true) for freshly serialized model', useGraphStore.getState().loadSnapshot(parsed) === true)
  const rel = serializeCurrentJson()
  check('store graph after load equals the on-disk graph', skipL(rel.nodes) === skipL(built.nodes) && skipL(rel.edges) === skipL(built.edges))
  check('graph has the layers we added (not empty)', rel.nodes.length >= 5)
}

console.log('section: malformed files — parseFile must throw, never return junk')
{
  const cases: Array<[string, string]> = [
    ['empty file', ''],
    ['whitespace only', '   \n\t  '],
    ['invalid JSON', 'not json at all'],
    ['unbalanced braces', '{'],
    ['truncated JSON', '{"format":"spinoml"'],
    ['array root', '[1,2,3]'],
    ['plain object root', '{}'],
    ['missing format', '{"version":1}'],
    ['wrong format', '{"format":"other","version":1,"graph":{"nodes":[],"edges":[]}}'],
    ['missing version', '{"format":"spinoml"}'],
    ['version as string', '{"format":"spinoml","version":"1"}'],
    ['version too new', '{"format":"spinoml","version":999,"graph":{"nodes":[],"edges":[]}}'],
    ['version fraction', '{"format":"spinoml","version":0.5,"graph":{"nodes":[],"edges":[]}}'],
    ['version zero', '{"format":"spinoml","version":0,"graph":{"nodes":[],"edges":[]}}'],
    ['version negative', '{"format":"spinoml","version":-1,"graph":{"nodes":[],"edges":[]}}'],
    ['graph missing', '{"format":"spinoml","version":1}'],
    ['graph wrong type', '{"format":"spinoml","version":1,"graph":5}'],
    ['graph missing nodes', '{"format":"spinoml","version":1,"graph":{"edges":[]}}'],
    ['graph missing edges', '{"format":"spinoml","version":1,"graph":{"nodes":[]}}'],
    ['nodes wrong type', '{"format":"spinoml","version":1,"graph":{"nodes":{},"edges":[]}}'],
    ['edges wrong type', '{"format":"spinoml","version":1,"graph":{"nodes":[],"edges":"x"}}'],
  ]
  for (const [name, text] of cases) {
    let threw = false
    try { parseFile(text) } catch { threw = true }
    check(`reject — ${name}`, threw)
  }

  // unknown fields are tolerated (forward-compat within the version gate)
  const extraOk = JSON.parse(snap([{ id: 'a', layerType: 'Flatten', params: {} }]))
  extraOk.metadata = { author: 'anon' }
  extraOk.graph.meta = { appVersion: 'x' }
  let ok = false
  try { const s = parseFile(JSON.stringify(extraOk)); ok = s.nodes.length === 1 } catch { /* ignore */ }
  check('tolerate unknown top-level/extra fields', ok)

  // fail-safety: parseFile errors must not have mutated the store
  const before = serializeCurrentJson()
  try { parseFile('{') } catch { /* expected */ }
  check('failed parse leaves store untouched', skipL(serializeCurrentJson()) === skipL(before))

  // old-schema v0 (never emitted, no migration path) → explicit too-old error
  let tooOld = false
  try { parseFile('{"format":"spinoml","version":0,"graph":{"nodes":[],"edges":[]}}') } catch (e) { tooOld = String((e as Error).message).includes('too old') }
  check('v0 → explicit "too old" (no silent fallback)', tooOld)
}

console.log('section: structurally invalid but valid-JSON files — fail closed')
{
  const invalid: Array<[string, string, string]> = [
    ['directed cycle', snap([{ id: 'a', layerType: 'Linear', params: { in_features: 1, out_features: 1 } }, { id: 'b', layerType: 'ReLU', params: { inplace: false } }], [
      { source: 'a', target: 'b' }, { source: 'b', target: 'a' },
    ]), 'cycle'],
    ['duplicate node ids', snap([{ id: 'a', layerType: 'Flatten', params: {} }, { id: 'a', layerType: 'Flatten', params: {} }]), 'duplicate'],
    ['unknown layer', snap([{ id: 'a', layerType: 'Nope', params: {} }]), 'unknown layer'],
    ['dangling edge source', snap([{ id: 'a', layerType: 'Flatten', params: {} }], [{ source: 'ghost', target: 'a' }]), 'dangling'],
    ['self-loop', snap([{ id: 'a', layerType: 'Flatten', params: {} }], [{ source: 'a', target: 'a' }]), 'self-loop'],
    ['missing layerType on a node', snap([{ id: 'a', params: {} }]), 'missing layerType'],
    ['non-object node entry', '{"format":"spinoml","version":1,"graph":{"nodes":[5],"edges":[]}}', 'non-object node'],
    ['non-object edge entry', '{"format":"spinoml","version":1,"graph":{"nodes":[{"id":"a","layerType":"Flatten","params":{}}],"edges":[5]}}', 'non-object edge'],
  ]

  for (const [name, text, expect] of invalid) {
    // re-baseline per case so a legitimately accepted load can never poison the next check
    useGraphStore.getState().resetGraph()
    const baseline = serializeCurrentJson()

    let parsed: GraphSnapshot | null = null
    try { parsed = parseFile(text) } catch { parsed = null }
    check(`parseFile accepts ${name} (JSON-level ok)`, parsed !== null)

    // loadSnapshot must reject cleanly (return false, NEVER throw) and leave the store untouched
    let loaded = true
    let err: unknown = null
    try { loaded = useGraphStore.getState().loadSnapshot(parsed!) } catch (e) { err = e }
    check(`loadSnapshot rejects ${name} (${expect})`, loaded === false && err === null)
    check(`store unchanged after rejected load of ${name}`, skipL(serializeCurrentJson()) === skipL(baseline))
  }

  // missing params is NOT invalid — defaults are applied, exactly like addLayer
  useGraphStore.getState().resetGraph()
  const parsedDefaults = parseFile(snap([{ id: 'a', layerType: 'Flatten' }]))
  check('missing params → defaults applied, loads', useGraphStore.getState().loadSnapshot(parsedDefaults) === true)
  const afterDefaults = useGraphStore.getState().nodes
  check('Flatten defaults filled for missing params', afterDefaults.length === 1 && Object.keys(afterDefaults[0].data.params).length > 0)
}

console.log('section: schema versioning')
{
  const now = serializeCurrent()
  const obj = JSON.parse(now)
  check(`serialized file carries format:${obj.format} version:${obj.version}`, obj.format === 'spinoml' && obj.version === FORMAT_VERSION)
  check('FORMAT_VERSION is an integer', Number.isInteger(FORMAT_VERSION))
  check('v1 file parses (current)', (() => { try { parseFile(now); return true } catch { return false } })())
}

console.log('section: never silently replace corrupt graph with empty graph')
{
  // reset to a non-trivial model, then try to open garbage — the graph must survive
  const full = buildModel()
  for (const [name, text] of [['empty', ''], ['bad JSON', '{{{'], ['structurally invalid', snap([{ id: 'x', layerType: 'Nope', params: {} }])]] as Array<[string, string]>) {
    let rejected = false
    try {
      const s = parseFile(text)
      rejected = !useGraphStore.getState().loadSnapshot(s)
    } catch { rejected = true }
    check(`corrupt input "${name}" rejected`, rejected)
    check(`non-empty graph preserved after "${name}"`, skipL(serializeCurrentJson().nodes) === skipL(full.nodes))
  }
}

console.log(`\n${failed === 0 ? '✓ all persistence checks passed' : `✗ ${failed} check(s) failed`} (${passed} passed)`)
process.exit(failed === 0 ? 0 : 1)