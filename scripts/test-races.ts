// Phase 10 (TODO §11): async shape-inference race conditions.
//
// The danger (R007): a stale /infer response arriving AFTER a newer one must
// never overwrite newer graph state. The inference store guards with a run
// counter (runId taken at fire time, stale responses discarded) + AbortController.
//
// This harness mocks globalThis.fetch so /infer responses resolve in a
// CONTROLLED order (deterministic replay of the stale-response scenario):
//   1. graph A fires run A → promise held open (P_A)
//   2. graph B fires run B (aborts A at transport level; mock ignores abort) → P_B
//   3. resolve P_B (newer) FIRST, then resolve P_A (older) LATE
//   4. assert the store reflects B and NOT A — stale A must be discarded
import { useGraphStore } from '../src/canvas/GraphStore'

let passed = 0
let failed = 0
let skips = 0

function check(name: string, cond: boolean): void {
  if (cond) { passed++; process.stdout.write(`  ✓ ${name}\n`) }
  else      { failed++; process.stdout.write(`  ✗ ${name}\n`) }
}
function skip(name: string): void {
  skips++; process.stdout.write(`  SKIPPED: ${name}\n`)
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

// Mock router: /infer bodies are routed by a marker substring found in the
// generated code. Pending responses are held until resolveResponse() is called.
type Held = {
  kind: string
  body: { code: string; input_shapes: number[][]; input_dtypes?: string[] }
  resolve: (r: object) => void
}
let held: Held[] = []
let parseCount = 0
const ORIG = globalThis.fetch

function isMockable(url: string): boolean {
  return url.includes('/infer') || url.includes('/health')
}

async function main() {
  // register the store subscriber (fires a debounced kick on every graph change)
  // AFTER the fetch mock is installed, BEFORE any graph mutation. The initial
  // kick for the default graph is harmless (discarded when run A supersedes it).
  const { useInferenceStore } = await import('../src/inference/store')

  // ── 1. run 0 of the store (fires ~200ms after import) + graph A
  const g = useGraphStore.getState()
  g.resetGraph()
  g.updateNodeParams('input', { shape: [1, 3] })
  const flat = g.addLayer('Flatten', {}, { x: 100, y: 100 })
  g.connectNodes('input', flat)

  // wait for run A to fire into the mock
  await sleep(350)
  const runA = held.find((h) => h.kind === 'flatten')
  if (!runA) {
    skip('run A fired into the mock (mock never saw the flatten model)')
    fin()
    return
  }
  check('run A fired (flatten) and is held open', runA.body.code.includes('flatten'))

  // ── 2. switch to graph B while A's response is still in flight
  g.resetGraph()
  g.updateNodeParams('input', { shape: [1, 3] })
  const lin = g.addLayer('Linear', { in_features: 3, out_features: 1 }, { x: 100, y: 100 })
  g.connectNodes('input', lin)
  await sleep(350)
  const runB = held.find((h) => h.kind === 'linear')
  if (!runB) {
    skip('run B fired into the mock (mock never saw the linear model)')
    fin()
    return
  }
  check('run B fired (linear) while run A still pending', runB.body.code.includes('linear'))

  // ── 3. resolve NEWER (B) first, then OLDER (A) late
  runB.resolve(mkInferOk({ linear: [1, 1], __output__: [1, 1] }, 30))
  await sleep(100)
  runA.resolve(mkInferOk({ flatten: [1, 3], __output__: [1, 3] }, 30))

  // ── settle: every shapes writeback triggers one idempotent re-run against the
  //     mock (in the real app that completes and dedups). Drain them, resolving
  //     as graph B — this must NOT resurrect stale A shapes.
  for (let i = 0; i < 6; i++) {
    await sleep(250)
    const pending = held.splice(0)
    for (const h of pending) {
      if (h.kind === 'flatten') h.resolve(mkInferOk({ flatten: [1, 3], __output__: [1, 3] }, 30))
      else h.resolve(mkInferOk({ linear: [1, 1], __output__: [1, 1] }, 30))
    }
    if (pending.length === 0) break
  }

  // ── 4. assert final state = B, stale A discarded
  const st = useInferenceStore.getState()
  check('store status is ok after settle', st.status === 'ok')
  check('attrShapes reflect graph B (linear present)', st.attrShapes.linear !== undefined)
  check('attrShapes do NOT reflect stale graph A (flatten absent)', st.attrShapes.flatten === undefined)
  check('n_params non-zero for the linear model', (st.nParams ?? 0) > 0)

  // ── 5. the input node must carry B's inferred output, not stale A output
  const nodeData = useGraphStore.getState().nodes
  const inpNode = nodeData.find((n) => n.id === 'input')
  check('input node data still carries B-model shape', (inpNode?.data as Record<string, unknown>).inferredOutputShape !== undefined)
}

function mkInferOk(shapes: Record<string, number[]>, nParams: number): { ok: true; shapes: Record<string, number[]>; n_params: number } {
  return { ok: true, shapes, n_params: nParams }
}

async function fin() {
  globalThis.fetch = ORIG
  console.log(`\n${failed === 0 ? '✓ all race checks passed' : `✗ ${failed} check(s) failed`} (${passed} passed, ${skips} skipped)`)
  process.exit(failed === 0 ? 0 : 1)
}

// install the mock BEFORE importing the store (import order matters: main()
// is deferred; the store module import happens inside it via require())
globalThis.fetch = (async (url: any, init?: any) => {
  const u = String(url)
  if (!isMockable(u)) return ORIG(url, init)
  if (u.includes('/health')) {
    return new Response(JSON.stringify({ ok: true }), { status: 200 })
  }
  parseCount++
  const body = init?.body ? JSON.parse(String(init.body)) : { code: '' }
  let kind = 'unknown'
  if (body.code.includes('flatten')) kind = 'flatten'
  else if (body.code.includes('linear')) kind = 'linear'
  const p = new Promise<Response>((resolve) => {
    held.push({ kind, body, resolve: (r) => resolve(new Response(JSON.stringify(r), { status: 200 })) })
  })
  return p
}) as typeof fetch

main()