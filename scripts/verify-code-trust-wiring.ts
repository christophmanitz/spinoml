#!/usr/bin/env tsx
// Phase 43 — code-trust wiring. Proves that every store that ships generated
// model code to a sidecar/executor refuses to do so unless every Custom/DataOp
// blob's hash is in the local trust store. Functional where a store can be
// driven with a mocked fetch / live GraphStore (inference, viz, smoke) and with
// direct calls (guard, snapshot, verifier); static source-order checks where a
// store cannot be driven without Tauri (training launch).

import { readdirSync, readFileSync } from 'node:fs'
import { join, relative } from 'node:path'
import { useGraphStore } from '../src/canvas/GraphStore'
import { defaultParamsFor } from '../src/layers/registry'
import { trust } from '../src/trust/trustStore'
import { hashBlob, type ArchNodeInput } from '../src/trust/codeBlobs'
import { listUntrusted, assertTrusted, UntrustedCodeError } from '../src/trust/guard'
import { UNTRUSTED_MESSAGE } from '../src/trust/gate'
import { buildRunSnapshot } from '../src/training/snapshot'
import { verifyModelForTraining } from '../src/inference/verifier'
import { defaultTrainingConfig } from '../src/training/types'

let failures = 0
function check(name: string, cond: boolean, detail = ''): void {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name}${detail ? '  ' + detail : ''}`) }
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

// ── fetch mock (installed before the stores are imported) ───────────────────
type Held = { resolve: (r: object) => void }
const held: Held[] = []
let holdMode = false
let inferBodies: string[] = []
let activationCalls = 0
let smokeCalls = 0
const ORIG = globalThis.fetch

function mkOk(code: string, nParams: number): object {
  const shapes: Record<string, number[]> = {}
  for (const m of code.matchAll(/self\.([A-Za-z_][A-Za-z0-9_]*)\s*=/g)) shapes[m[1]] = [1, 3]
  shapes['__output__'] = [1, 3]
  return { ok: true, shapes, n_params: nParams }
}

globalThis.fetch = (async (url: unknown, init?: RequestInit) => {
  const u = String(url)
  if (u.includes('/infer')) {
    const body = init?.body ? (JSON.parse(String(init.body)) as { code: string }) : { code: '' }
    inferBodies.push(body.code)
    if (holdMode) {
      return await new Promise<Response>((resolve) => {
        held.push({ resolve: (r) => resolve(new Response(JSON.stringify(r), { status: 200 })) })
      })
    }
    return new Response(JSON.stringify(mkOk(body.code, 2222)), { status: 200 })
  }
  if (u.includes('/activations')) {
    activationCalls++
    return new Response(JSON.stringify({ ok: false, error: 'unexpected activations call' }), { status: 200 })
  }
  if (u.includes('/dataset/smoke')) {
    smokeCalls++
    return new Response(JSON.stringify({ ok: false, error: 'unexpected smoke call' }), { status: 200 })
  }
  if (u.includes('/health')) return new Response(JSON.stringify({ ok: true }), { status: 200 })
  return ORIG(url as string, init)
}) as typeof fetch

function spinoml(nodes: ArchNodeInput[]): string {
  return JSON.stringify({
    format: 'spinoml', version: 1, savedAt: '2026-01-01T00:00:00Z',
    graph: {
      nodes: nodes.map((n) => ({ id: n.id, layerType: n.data?.layerType, params: n.data?.params ?? {} })),
      edges: [],
    },
  })
}

async function main(): Promise<void> {
  console.log('phase 43: code-trust wiring')

  const src = String(defaultParamsFor('Custom').source ?? '')
  const srcHash = await hashBlob('custom-layer', src)
  trust.approve(srcHash, 'user-approval') // human trust, pre-seeded (TESTS ONLY)

  const { useInferenceStore } = await import('../src/inference/store')
  const g = useGraphStore.getState()

  // ── 1. Inference: trusted Custom → ok (one /infer call) ───────────────────
  console.log('\n1. inference: trusted Custom')
  g.resetGraph()
  g.updateNodeParams('input', { shape: [1, 3] })
  const cid = g.addLayer('Custom', { x: 120, y: 80 }, { params: { source: src } })
  g.connectNodes('input', cid)
  await sleep(500)
  check('trusted Custom: /infer called once', inferBodies.length === 1, `calls=${inferBodies.length}`)
  check('trusted Custom: status ok', useInferenceStore.getState().status === 'ok')
  check('trusted Custom: node has inferred output shape',
    !!useGraphStore.getState().nodes.find((n) => n.id === cid)?.data.inferredOutputShape)
  check('trusted Custom: no untrusted blobs', useInferenceStore.getState().untrusted.length === 0)

  // ── 2. Editing to an unapproved source → untrusted, no /infer ────────────
  console.log('\n2. inference: edited (untrusted) source fail-closed')
  const evilSrc = 'class Evil(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.LazyLinear(3)\n\n    def forward(self, x):\n        return self.fc(x) + 1\n'
  const evilHash = await hashBlob('custom-layer', evilSrc)
  g.updateNodeParams(cid, { source: evilSrc })
  await sleep(500)
  check('edited source: /infer NEVER called', inferBodies.length === 1, `calls=${inferBodies.length}`)
  check('edited source: status untrusted', useInferenceStore.getState().status === 'untrusted')
  check('edited source: untrusted.length === 1', useInferenceStore.getState().untrusted.length === 1)
  check('edited source: untrusted sha is the blob hash', useInferenceStore.getState().untrusted[0]?.sha256 === evilHash)
  check('edited source: error is UNTRUSTED_MESSAGE(1)', useInferenceStore.getState().error === UNTRUSTED_MESSAGE(1))
  check('edited source: node shapes cleared',
    !useGraphStore.getState().nodes.find((n) => n.id === cid)?.data.inferredOutputShape)

  // ── 3. Approving the current source re-runs inference → ok ───────────────
  console.log('\n3. inference: approval re-runs')
  trust.approve(evilHash, 'user-approval') // TESTS ONLY
  await sleep(550)
  check('after approve: /infer re-ran once', inferBodies.length === 2, `calls=${inferBodies.length}`)
  check('after approve: status ok', useInferenceStore.getState().status === 'ok')
  check('after approve: untrusted cleared', useInferenceStore.getState().untrusted.length === 0)
  check('after approve: shapes reapplied',
    !!useGraphStore.getState().nodes.find((n) => n.id === cid)?.data.inferredOutputShape)

  // ── 4. One-character edit of the approved source → untrusted again ───────
  console.log('\n4. inference: one-character edit')
  g.updateNodeParams(cid, { source: evilSrc.replace('+ 1', '+ 2') })
  await sleep(500)
  check('one-char edit: /infer NEVER called', inferBodies.length === 2, `calls=${inferBodies.length}`)
  check('one-char edit: status untrusted', useInferenceStore.getState().status === 'untrusted')
  check('one-char edit: shapes cleared',
    !useGraphStore.getState().nodes.find((n) => n.id === cid)?.data.inferredOutputShape)

  // ── 5. LLM-style addLayer of an untrusted Custom → untrusted, no /infer ──
  console.log('\n5. inference: LLM-added untrusted Custom')
  g.resetGraph()
  g.updateNodeParams('input', { shape: [1, 3] })
  const evil2 = 'class Evil2(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(3, 3)\n\n    def forward(self, x):\n        return self.fc(x)\n'
  const eid = g.addLayer('Custom', { x: 120, y: 80 }, { params: { source: evil2 } })
  g.connectNodes('input', eid)
  await sleep(500)
  check('LLM addLayer: /infer NEVER called', inferBodies.length === 2, `calls=${inferBodies.length}`)
  check('LLM addLayer: status untrusted', useInferenceStore.getState().status === 'untrusted')
  check('LLM addLayer: untrusted.length === 1', useInferenceStore.getState().untrusted.length === 1)

  // ── 6. Stale-response guard survives the (async) trust hash ──────────────
  console.log('\n6. inference: stale response dropped after graph edit')
  inferBodies = []
  g.resetGraph()
  g.updateNodeParams('input', { shape: [1, 3] })
  const flat = g.addLayer('Flatten', { x: 120, y: 80 }, {})
  g.connectNodes('input', flat)
  await sleep(500)
  check('stale setup: built-in graph reaches ok', useInferenceStore.getState().status === 'ok')
  check('stale setup: exactly one /infer', inferBodies.length === 1, `calls=${inferBodies.length}`)

  holdMode = true
  g.updateNodeParams(flat, { start_dim: 1 })
  await sleep(400)
  check('stale setup: one request held open', held.length === 1, `held=${held.length}`)
  holdMode = false
  const lin = g.addLayer('Linear', { x: 200, y: 80 }, {})
  g.connectNodes(flat, lin)
  await sleep(400)
  check('stale setup: newer response applied (nParams=2222)', useInferenceStore.getState().nParams === 2222)
  held[0].resolve({ ok: true, shapes: { __output__: [9, 9] }, n_params: 1111 })
  await sleep(200)
  check('stale response dropped (nParams stays 2222)', useInferenceStore.getState().nParams === 2222)
  check('stale response dropped (status stays ok)', useInferenceStore.getState().status === 'ok')

  // ── 7. Guard primitives (functional) ─────────────────────────────────────
  console.log('\n7. guard primitives')
  const fresh = 'class Fresh(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(3, 3)\n\n    def forward(self, x):\n        return self.fc(x)\n'
  const untrustedNodes: ArchNodeInput[] = [{ id: 'guard-x', data: { layerType: 'Custom', params: { source: fresh } } }]
  const lu = await listUntrusted(untrustedNodes)
  check('listUntrusted finds the untrusted blob', lu.length === 1 && lu[0].kind === 'custom-layer' && lu[0].nodeId === 'guard-x')
  let caught: unknown = null
  try { await assertTrusted(untrustedNodes) } catch (e) { caught = e }
  check('assertTrusted throws UntrustedCodeError', caught instanceof UntrustedCodeError)
  check('UntrustedCodeError carries the blobs', caught instanceof UntrustedCodeError && caught.blobs.length === 1)
  check('UntrustedCodeError.message is explicit', caught instanceof UntrustedCodeError && caught.message === UNTRUSTED_MESSAGE(1))
  check('UntrustedCodeError.name is set', caught instanceof UntrustedCodeError && caught.name === 'UntrustedCodeError')
  let noThrow = true
  try { await assertTrusted([{ id: 'guard-y', data: { layerType: 'Linear', params: { in_features: 3, out_features: 3 } } }]) }
  catch { noThrow = false }
  check('assertTrusted passes a blob-free graph', noThrow)

  // ── 8. Training verifier fail-closed on untrusted ────────────────────────
  console.log('\n8. training verifier (NewRunModal path)')
  const verifierSpin = spinoml([
    { id: 'in', data: { layerType: 'Input', params: { shape: [1, 3] } } },
    { id: 'vc', data: { layerType: 'Custom', params: { source: fresh } } },
  ])
  const v = await verifyModelForTraining(verifierSpin)
  check('verifier returns unknown for untrusted code', v.status === 'unknown')
  check('verifier names the untrusted reason', v.status === 'unknown' && v.reason.includes('not approved'))
  const trustedSpin = spinoml([
    { id: 'in', data: { layerType: 'Input', params: { shape: [1, 3] } } },
    { id: 'tc', data: { layerType: 'Custom', params: { source: src } } },
  ])
  const v2 = await verifyModelForTraining(trustedSpin) // trusted → proceeds to /infer mock → valid
  check('verifier proceeds for trusted code (mock /infer → valid)', v2.status === 'valid')

  // ── 9. buildRunSnapshot carries code_trust ───────────────────────────────
  console.log('\n9. run.json snapshot code_trust')
  const snap = await buildRunSnapshot(trustedSpin, 'class Model(nn.Module):\n    pass\n')
  check('snapshot has code_trust array', Array.isArray(snap.code_trust))
  check('snapshot code_trust has one entry', snap.code_trust.length === 1)
  check('snapshot entry node/path/kind', snap.code_trust[0]?.node === 'tc' && snap.code_trust[0]?.path === 'tc' && snap.code_trust[0]?.kind === 'custom-layer')
  check('snapshot entry sha256 is lowercase hex', /^[0-9a-f]{64}$/.test(snap.code_trust[0]?.sha256 ?? ''))
  check('snapshot entry origin user-approval', snap.code_trust[0]?.origin === 'user-approval')
  check('snapshot entry approved_at is a string', typeof snap.code_trust[0]?.approved_at === 'string')
  check('snapshot entry sha matches hashBlob', snap.code_trust[0]?.sha256 === (await hashBlob('custom-layer', src)))
  check('old snapshot field graph_sha256 unchanged (hex)', /^[0-9a-f]{64}$/.test(snap.graph_sha256))
  check('old snapshot field model_py_sha256 unchanged (hex)', /^[0-9a-f]{64}$/.test(snap.model_py_sha256))
  check('old snapshot field preprocessing unchanged (array)', Array.isArray(snap.preprocessing))
  check('snapshot version unchanged (1)', snap.version === 1)
  const snapUntrusted = await buildRunSnapshot(spinoml([
    { id: 'in', data: { layerType: 'Input', params: { shape: [1, 3] } } },
    { id: 'uc', data: { layerType: 'Custom', params: { source: fresh } } },
  ]), 'x\n')
  check('untrusted blob recorded as origin unrecorded / approved_at null',
    snapUntrusted.code_trust[0]?.origin === 'unrecorded' && snapUntrusted.code_trust[0]?.approved_at === null)
  const snapDataOp = await buildRunSnapshot(spinoml([
    { id: 'do', data: { layerType: 'DataOp', params: { script: 'print(1)\n' } } },
  ]), 'x\n')
  check('DataOp script is recorded as dataop-script',
    snapDataOp.code_trust.length === 1 && snapDataOp.code_trust[0]?.kind === 'dataop-script')

  // ── 10. Visualization store functional: untrusted → no /activations ─────
  console.log('\n10. visualization store gate')
  const { useVizStore } = await import('../src/visualization/store')
  g.resetGraph()
  g.updateNodeParams('input', { shape: [1, 3] })
  const vid = g.addLayer('Custom', { x: 120, y: 80 }, { params: { source: fresh } })
  g.connectNodes('input', vid)
  activationCalls = 0
  await useVizStore.getState().run()
  check('viz: no /activations call for untrusted code', activationCalls === 0, `calls=${activationCalls}`)
  check('viz: error state set with explicit message', (useVizStore.getState().error ?? '').includes('not approved'))
  check('viz: running flag cleared', useVizStore.getState().running === false)

  // ── 11. Datasets smoke store functional: untrusted → no /dataset/smoke ──
  console.log('\n11. datasets smoke store gate')
  const { useDatasetsStore } = await import('../src/datasets/store')
  useDatasetsStore.setState({
    entries: [{ name: 'd.csv', relpath: 'd.csv', abspath: '/tmp/d.csv', is_dir: false, size_bytes: 0 }],
  })
  smokeCalls = 0
  await useDatasetsStore.getState().runSmoke('d.csv')
  check('smoke: no /dataset/smoke call for untrusted code', smokeCalls === 0, `calls=${smokeCalls}`)
  check('smoke: error state set with explicit message',
    (useDatasetsStore.getState().smoke['d.csv']?.error ?? '').includes('not approved'))

  // ── 12. Training startRun functional (fake Tauri executor) ───────────────
  console.log('\n12. training startRun gate + run.json code_trust')
  const invokeCalls: { cmd: string; args: Record<string, unknown> }[] = []
  let fakeModelContent = ''
  ;(globalThis as unknown as { window: unknown }).window = globalThis
  ;(globalThis as unknown as { __TAURI_INTERNALS__: unknown }).__TAURI_INTERNALS__ = {
    invoke: async (cmd: string, args?: Record<string, unknown>) => {
      invokeCalls.push({ cmd, args: args ?? {} })
      if (cmd === 'read_workspace_file') return fakeModelContent
      if (cmd === 'list_training_runs') return []
      if (cmd === 'start_training_run') return undefined
      throw new Error(`fake invoke: unexpected ${cmd}`)
    },
  }
  const { useTrainingStore } = await import('../src/training/store')
  const runInput = {
    label: 'trust-test',
    modelRelpath: 'm.spinoml',
    datasetRelpath: 'd.csv',
    datasetAbspath: '/tmp/d.csv',
    targetColumn: 'y',
    featureColumns: null,
    training: defaultTrainingConfig(),
  }

  // untrusted → reject, nothing written, executor never called
  fakeModelContent = spinoml([
    { id: 'in', data: { layerType: 'Input', params: { shape: [1, 3] } } },
    { id: 'uc', data: { layerType: 'Custom', params: { source: fresh } } },
  ])
  invokeCalls.length = 0
  let startErr: unknown = null
  try { await useTrainingStore.getState().startRun(runInput) } catch (e) { startErr = e }
  check('startRun rejects untrusted with UntrustedCodeError', startErr instanceof UntrustedCodeError)
  check('startRun untrusted: executor never launched', !invokeCalls.some((c) => c.cmd === 'start_training_run'))
  check('startRun untrusted: nothing written to disk',
    !invokeCalls.some((c) => c.cmd.includes('write') || c.cmd.includes('start') || c.cmd.includes('mkdir')))

  // trusted → launched, run.json snapshot carries code_trust
  fakeModelContent = spinoml([
    { id: 'in', data: { layerType: 'Input', params: { shape: [1, 3] } } },
    { id: 'tc', data: { layerType: 'Custom', params: { source: src } } },
  ])
  invokeCalls.length = 0
  const launchedId = await useTrainingStore.getState().startRun(runInput)
  check('startRun trusted: returns a run id', typeof launchedId === 'string' && launchedId.length > 0)
  const startCall = invokeCalls.find((c) => c.cmd === 'start_training_run')
  check('startRun trusted: executor launched once', !!startCall)
  const writtenRunJson = startCall ? (JSON.parse(String(startCall.args.runJson)) as {
    snapshot: { code_trust: { node: string; origin: string; sha256: string }[]; graph_sha256: string; model_py_sha256: string; preprocessing: unknown[] }
  }) : null
  check('written run.json has snapshot.code_trust', writtenRunJson?.snapshot.code_trust.length === 1)
  check('written run.json code_trust entry names the node', writtenRunJson?.snapshot.code_trust[0]?.node === 'tc')
  check('written run.json code_trust entry is user-approved', writtenRunJson?.snapshot.code_trust[0]?.origin === 'user-approval')
  check('written run.json old snapshot fields unchanged',
    typeof writtenRunJson?.snapshot.graph_sha256 === 'string' && typeof writtenRunJson?.snapshot.model_py_sha256 === 'string' && Array.isArray(writtenRunJson?.snapshot.preprocessing))

  // ── 13. Static wiring invariants ─────────────────────────────────────────
  console.log('\n13. static wiring')
  const srcDir = join(process.cwd(), 'src')
  const files = walkTs(srcDir)
  const offenders: string[] = []
  const CLIENT_FNS = /\b(inferShapes|runActivationsReq|smokeDataset|smokeDatasetMulti)\b/
  for (const f of files) {
    const text = readFileSync(f, 'utf8')
    const importsClient = /import\s*\{[^}]*\b(inferShapes|runActivationsReq|smokeDataset|smokeDatasetMulti)\b[^}]*\}\s*from\s*['"][^'"]*client['"]/.test(text)
    if (!importsClient) continue
    if (!/from\s+['"][^'"]*trust\/guard['"]/.test(text)) offenders.push(relPosix(f))
    void CLIENT_FNS
  }
  check('every executor-client importer also imports trust/guard', offenders.length === 0, `offenders=${offenders.join(',')}`)

  const infSrc = readFileSync(join(srcDir, 'inference', 'store.ts'), 'utf8')
  check('Status union contains untrusted', /type Status =[^\n]*'untrusted'/.test(infSrc))
  check('inference store has untrusted state field', infSrc.includes('untrusted: UntrustedBlob[]'))
  check('inference store subscribes to trust changes', infSrc.includes('trust.subscribe('))
  check('inference store gates with listUntrusted + UNTRUSTED_MESSAGE',
    infSrc.includes('listUntrusted') && infSrc.includes('UNTRUSTED_MESSAGE'))

  const trSrc = readFileSync(join(srcDir, 'training', 'store.ts'), 'utf8')
  const runIdx = trSrc.indexOf('startRun:')
  const evalIdx = trSrc.indexOf('startEvalRun:')
  const runAssert = trSrc.indexOf('assertTrusted', runIdx)
  const runStart = trSrc.indexOf('training.start', runIdx)
  const evalAssert = trSrc.indexOf('assertTrusted', evalIdx)
  const evalStart = trSrc.indexOf('training.start', evalIdx)
  check('startRun asserts trust before training.start', runAssert !== -1 && runStart !== -1 && runAssert < runStart)
  check('startRun asserts on the frozen snapshot, not the live canvas',
    trSrc.includes('parseFile(modelContent)') && trSrc.slice(runIdx, runStart).includes('assertTrusted'))
  check('startEvalRun asserts trust before training.start', evalAssert !== -1 && evalStart !== -1 && evalAssert < evalStart)
  check('startRun reads frozen bytes then generates from the parsed snapshot',
    trSrc.includes('fs.read') && trSrc.includes('generateFromSnapshot'))

  const vizSrc = readFileSync(join(srcDir, 'visualization', 'store.ts'), 'utf8')
  check('viz store imports + calls assertTrusted before runActivationsReq',
    /trust\/guard/.test(vizSrc) && vizSrc.indexOf('assertTrusted') !== -1 && vizSrc.indexOf('assertTrusted') < vizSrc.indexOf('await runActivationsReq'))
  check('viz store catches UntrustedCodeError into error state', vizSrc.includes('UntrustedCodeError'))

  const dsSrc = readFileSync(join(srcDir, 'datasets', 'store.ts'), 'utf8')
  const dsAssert = dsSrc.indexOf('assertTrusted')
  const dsMulti = dsSrc.indexOf('await smokeDatasetMulti(')
  const dsSingle = dsSrc.indexOf('await smokeDataset(')
  check('datasets store imports + calls assertTrusted before smoke client',
    /trust\/guard/.test(dsSrc) && dsAssert !== -1 && dsMulti !== -1 && dsSingle !== -1 && dsAssert < dsMulti && dsAssert < dsSingle)
  check('datasets store catches UntrustedCodeError into error state', dsSrc.includes('UntrustedCodeError'))

  const verSrc = readFileSync(join(srcDir, 'inference', 'verifier.ts'), 'utf8')
  check('verifier imports guard and maps untrusted to unknown',
    /trust\/guard/.test(verSrc) && verSrc.includes('assertTrusted') && verSrc.includes("'unknown'"))
  const modal = readFileSync(join(srcDir, 'training', 'NewRunModal.tsx'), 'utf8')
  check('NewRunModal blocks unknown verification', modal.includes("verify.status === 'unknown'"))

  if (failures > 0) {
    console.error(`\n${failures} check(s) failed`)
    process.exit(1)
  }
  console.log('\n✓ all code-trust wiring checks passed')
  process.exit(0)
}

function walkTs(dir: string): string[] {
  const out: string[] = []
  for (const ent of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, ent.name)
    if (ent.isDirectory()) out.push(...walkTs(p))
    else if (ent.isFile() && (p.endsWith('.ts') || p.endsWith('.tsx'))) out.push(p)
  }
  return out
}

function relPosix(p: string): string {
  return relative(process.cwd(), p).split('\\').join('/')
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
