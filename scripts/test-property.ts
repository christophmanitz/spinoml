// Property-based tests over RANDOM VALID graphs.  Run: `npm run test:property`.
//
// Every graph is built correct-by-construction by `scripts/lib/graph-gen.ts`,
// which also carries an INDEPENDENT shape / parameter-count oracle. For each
// graph we prove:
//   (1) validateGraphState reports ok with no error issues,
//   (2) loadSnapshot accepts it and generate() returns issues: [],
//   (3) generation is deterministic and survives a serialize→parse round trip,
//   (4) ONE python process execs every generated model: forward (batch 3) is
//       finite, output shape + parameter count equal the oracle, and backward
//       yields finite gradients for every grad-requiring parameter,
//   (5) for a sample of 40 graphs the real torch sidecar `/infer` agrees with
//       the oracle AND with independent forward-hook shapes from python.
//
// Failure output prints the seed, index, family and the graph snapshot (truncated)
// so it can be replayed with PROPERTY_SEED=… PROPERTY_ONLY=<index>.
//
// SECURITY: this script executes ONLY code produced by the app's own generator
// from graphs the script itself built — no external input, no network.

import { spawn, spawnSync, type ChildProcess } from 'node:child_process'
import { setTimeout as wait } from 'node:timers/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { GraphSnapshot } from '../src/canvas/GraphStore'
import { useGraphStore } from '../src/canvas/GraphStore'
import { validateGraphState } from '../src/canvas/invariants'
import { generate, generateFromSnapshot, type CodegenResult } from '../src/codegen/generator'
import { parseFile, serializeCurrent } from '../src/persistence/file'
import { inferShapes, type InferResult } from '../src/inference/client'
import {
  generatePropertyGraphs,
  graphAt,
  resolveN,
  resolveOnly,
  resolveSeed,
  snapshotToEdges,
  snapshotToLayerNodes,
  type GeneratedGraph,
} from './lib/graph-gen'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = join(HERE, '..')
const PY_SCRIPT = join(HERE, 'lib', 'property_run.py')
// Interpreter that can import torch: the runner exports PYTHON (resolved from
// SPINOML_CONDA_ENV / a plain pip env); direct runs fall back to `python`.
const PYTHON = process.env.PYTHON ?? 'python'
const PYTHON_BATCH = 3
const SIDECAR_PORT = Number(process.env.SPINOML_TORCH_PORT ?? '7421')
const SIDECAR_URL = `http://127.0.0.1:${SIDECAR_PORT}`
const SIDECAR_CALLS = Number(process.env.PROPERTY_SIDECAR_CALLS ?? '40')

const SEED = resolveSeed()
const N = resolveN()
const ONLY = resolveOnly()

// ─────────────────────────────────────────────────────────────────────────────
// Reporting helpers
// ─────────────────────────────────────────────────────────────────────────────

let failures = 0
const failureDetails: string[] = []

function fail(msg: string): void {
  failures++
  failureDetails.push(msg)
  console.log(`  ✗ ${msg}`)
}

function replay(g: GeneratedGraph): string {
  const snap = JSON.stringify(g.snapshot)
  const short = snap.length > 600 ? `${snap.slice(0, 600)}…(${snap.length} bytes)` : snap
  return `replay: PROPERTY_SEED=${SEED} PROPERTY_ONLY=${g.index}   family=${g.family}\n    snapshot=${short}`
}

type FamilyStat = { count: number; pyPass: number; pyFail: number; scPass: number; scFail: number }

function newStats(): Map<string, FamilyStat> {
  const m = new Map<string, FamilyStat>()
  for (const f of ['mlp', 'cnn', 'residual', 'branch', 'multi-input', 'sequence']) {
    m.set(f, { count: 0, pyPass: 0, pyFail: 0, scPass: 0, scFail: 0 })
  }
  return m
}

function printTable(stats: Map<string, FamilyStat>): void {
  const headers = ['family', 'n', 'python+', 'python-', 'sidecar+', 'sidecar-']
  const rows: string[][] = []
  let total = 0
  let pyFail = 0
  let scFail = 0
  for (const [family, s] of stats) {
    if (s.count === 0) continue
    total += s.count
    pyFail += s.pyFail
    scFail += s.scFail
    rows.push([family, String(s.count), String(s.pyPass), String(s.pyFail), String(s.scPass), String(s.scFail)])
  }
  rows.push(['TOTAL', String(total), '', String(pyFail), '', String(scFail)])
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => (r[i] ?? '').length)))
  const line = (vals: string[]) => vals.map((v, i) => v.padEnd(widths[i])).join(' | ')
  console.log('')
  console.log(line(headers))
  console.log(widths.map((w) => '-'.repeat(w)).join('-+-'))
  for (const r of rows) console.log(line(r))
  console.log('')
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidecar lifecycle (owns it only when it started it)
// ─────────────────────────────────────────────────────────────────────────────

async function sidecarUp(): Promise<boolean> {
  try {
    const r = await fetch(`${SIDECAR_URL}/health`)
    return r.ok
  } catch {
    return false
  }
}

type Sidecar = { child: ChildProcess | null; owned: boolean }

async function startSidecar(): Promise<Sidecar> {
  if (await sidecarUp()) return { child: null, owned: false }
  const child = spawn('python', ['sidecar-torch/main.py'], {
    cwd: REPO_ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, SPINOML_TORCH_PORT: String(SIDECAR_PORT) },
  })
  child.stdout?.on('data', () => undefined)
  child.stderr?.on('data', () => undefined)
  const deadline = Date.now() + 15_000
  while (Date.now() < deadline) {
    if (await sidecarUp()) return { child, owned: true }
    if (child.exitCode !== null) break
    await wait(200)
  }
  child.kill('SIGKILL')
  throw new Error('torch sidecar did not become healthy within 15s')
}

function stopSidecar(s: Sidecar): void {
  if (s.owned && s.child) s.child.kill('SIGTERM')
}

// ─────────────────────────────────────────────────────────────────────────────
// Python result types
// ─────────────────────────────────────────────────────────────────────────────

type PyCaseResult = {
  ok: boolean
  error: string | null
  output_shape: number[] | null
  n_params: number | null
  grads_ok: boolean
  grads_detail: string
  finite: boolean
  hook_shapes: Record<string, number[]>
  output_shape_b1: number[] | null
}

function parsePy(raw: string): Record<string, PyCaseResult> {
  const obj: unknown = JSON.parse(raw)
  if (typeof obj !== 'object' || obj === null || !('results' in obj)) {
    throw new Error('python output missing "results"')
  }
  const results = (obj as { results: unknown }).results
  const out: Record<string, PyCaseResult> = {}
  for (const [k, v] of Object.entries(results as Record<string, unknown>)) {
    const r = v as Record<string, unknown>
    out[k] = {
      ok: r.ok === true,
      error: typeof r.error === 'string' ? r.error : null,
      output_shape: Array.isArray(r.output_shape) ? (r.output_shape as number[]) : null,
      n_params: typeof r.n_params === 'number' ? r.n_params : null,
      grads_ok: r.grads_ok === true,
      grads_detail: typeof r.grads_detail === 'string' ? r.grads_detail : '',
      finite: r.finite === true,
      hook_shapes: (typeof r.hook_shapes === 'object' && r.hook_shapes !== null
        ? (r.hook_shapes as Record<string, number[]>)
        : {}),
      output_shape_b1: Array.isArray(r.output_shape_b1) ? (r.output_shape_b1 as number[]) : null,
    }
  }
  return out
}

const eqShape = (a: readonly number[] | null | undefined, b: readonly number[]): boolean =>
  Array.isArray(a) && a.length === b.length && a.every((v, i) => v === b[i])

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

async function main(): Promise<number> {
  console.log(`property test — seed=${SEED} n=${N}${ONLY !== null ? ` only=${ONLY}` : ''}`)
  const graphs = ONLY !== null ? [graphAt(ONLY, SEED)] : generatePropertyGraphs(SEED, N)
  const stats = newStats()

  // ── (1)-(3) TypeScript-side structural / determinism / persistence checks ──
  const codeById = new Map<number, CodegenResult>()
  const pyCases: Record<string, unknown>[] = []
  for (const g of graphs) {
    const st = stats.get(g.family)!
    st.count++
    const label = `#${g.index} ${g.family}`
    const nodes = snapshotToLayerNodes(g.snapshot)
    const edges = snapshotToEdges(g.snapshot)

    // (1) validator
    const v = validateGraphState(nodes, edges)
    if (!v.ok || v.issues.some((i) => i.severity === 'error')) {
      fail(`${label}: validateGraphState rejected a valid graph: ${JSON.stringify(v.issues)}`)
      console.log(replay(g))
      st.pyFail++
      continue
    }

    // (2) loadSnapshot + generate
    if (!useGraphStore.getState().loadSnapshot(g.snapshot)) {
      fail(`${label}: loadSnapshot rejected a valid graph`)
      console.log(replay(g))
      st.pyFail++
      continue
    }
    const res = generate(nodes, edges)
    if (res.issues.length > 0) {
      fail(`${label}: generate() reported issues: ${JSON.stringify(res.issues)}`)
      console.log(replay(g))
      st.pyFail++
      continue
    }
    codeById.set(g.index, res)

    // (3) determinism + persistence round-trip
    const c1 = generate(nodes, edges).code
    const c2 = generate(nodes, edges).code
    const parsed: GraphSnapshot = parseFile(serializeCurrent())
    const round = generateFromSnapshot(parsed).code
    if (c1 !== c2 || res.code !== c1 || round !== c1) {
      fail(`${label}: generation is not deterministic / round-trip unstable`)
      console.log(replay(g))
      st.pyFail++
      continue
    }
    if (!useGraphStore.getState().loadSnapshot(parsed)) {
      fail(`${label}: loadSnapshot rejected the parsed round-trip snapshot`)
      st.pyFail++
      continue
    }
    if (JSON.stringify(graphAt(g.index, SEED).snapshot) !== JSON.stringify(g.snapshot)) {
      fail(`${label}: generator is not deterministic for the same seed`)
      st.pyFail++
      continue
    }

    pyCases.push({
      index: g.index,
      family: g.family,
      code: res.code,
      inputs: g.inputs,
      expected_output_shape: g.expectedOutputShape,
      expected_params: g.expectedParams,
    })
  }

  // ── (4) ONE python process runs every generated model ─────────────────────
  console.log(`running ${pyCases.length} generated models in one python process…`)
  const job = JSON.stringify({ batch: PYTHON_BATCH, cases: pyCases })
  const proc = spawnSync(
    PYTHON,
    [PY_SCRIPT],
    { input: job, encoding: 'utf-8', maxBuffer: 512 * 1024 * 1024, cwd: REPO_ROOT },
  )
  let pyResults: Record<string, PyCaseResult> = {}
  if (proc.error) {
    fail(`failed to run python: ${proc.error.message}`)
  } else if (proc.status !== 0) {
    fail(`property_run.py exited ${proc.status}: ${(proc.stderr ?? '').trim().slice(0, 800)}`)
  } else {
    try {
      pyResults = parsePy((proc.stdout ?? '').trim())
    } catch (e: unknown) {
      fail(e instanceof Error ? e.message : String(e))
    }
  }

  for (const g of graphs) {
    if (!codeById.has(g.index)) continue
    const st = stats.get(g.family)!
    const r = pyResults[String(g.index)]
    const label = `#${g.index} ${g.family}`
    const wantShape = [PYTHON_BATCH, ...g.expectedOutputShape]
    const wantB1 = [1, ...g.expectedOutputShape]
    if (!r) {
      fail(`${label}: no python result returned`)
      st.pyFail++
      continue
    }
    if (!r.ok) {
      fail(`${label}: python error: ${r.error ?? 'unknown'}`)
      console.log(replay(g))
      st.pyFail++
      continue
    }
    if (!r.finite) {
      fail(`${label}: forward output is not finite`)
      st.pyFail++
      continue
    }
    if (!eqShape(r.output_shape, wantShape)) {
      fail(`${label}: output shape ${JSON.stringify(r.output_shape)} != oracle ${JSON.stringify(wantShape)}`)
      console.log(replay(g))
      st.pyFail++
      continue
    }
    if (r.n_params !== g.expectedParams) {
      fail(`${label}: param count ${r.n_params} != oracle ${g.expectedParams}`)
      console.log(replay(g))
      st.pyFail++
      continue
    }
    if (!r.grads_ok) {
      fail(`${label}: backward check failed: ${r.grads_detail}`)
      console.log(replay(g))
      st.pyFail++
      continue
    }
    if (!eqShape(r.output_shape_b1, wantB1)) {
      fail(`${label}: batch-1 output shape ${JSON.stringify(r.output_shape_b1)} != ${JSON.stringify(wantB1)}`)
      st.pyFail++
      continue
    }
    if (Object.keys(r.hook_shapes).length === 0) {
      fail(`${label}: python captured no forward-hook shapes`)
      st.pyFail++
      continue
    }
    st.pyPass++
  }

  // ── (5) sidecar /infer on a spread sample ─────────────────────────────────
  const sample = selectSidecarSample(graphs, SIDECAR_CALLS)
  let sidecar: Sidecar = { child: null, owned: false }
  try {
    sidecar = await startSidecar()
    console.log(`sidecar ${sidecar.owned ? 'started' : 'reused'} on ${SIDECAR_URL}; checking ${sample.length} graphs`)
    for (const g of sample) {
      const st = stats.get(g.family)!
      const res = codeById.get(g.index)
      const py = pyResults[String(g.index)]
      const label = `#${g.index} ${g.family}`
      const inputs = g.inputs
      if (!res || !py) {
        st.scFail++
        continue
      }
      const ac = new AbortController()
      const timer = setTimeout(() => ac.abort(), 15_000)
      let result: InferResult | { ok: false; offline: true; error: string; shapes: Record<string, number[]> }
      try {
        result = await inferShapes(res.code, inputs.map((i) => i.shape), inputs.map((i) => i.dtype), ac.signal)
      } finally {
        clearTimeout(timer)
      }
      if ('offline' in result) {
        fail(`${label}: sidecar offline / HTTP error: ${result.error}`)
        st.scFail++
        continue
      }
      if (!result.ok) {
        const detail = result.stage ? `${result.stage}: ${result.error}` : result.error
        fail(`${label}: sidecar /infer failed: ${detail}`)
        st.scFail++
        continue
      }
      let bad = false
      if (result.n_params !== g.expectedParams) {
        fail(`${label}: sidecar n_params ${result.n_params} != oracle ${g.expectedParams}`)
        bad = true
      }
      const out = result.shapes['__output__']
      if (!eqShape(Array.isArray(out) ? out : null, [1, ...g.expectedOutputShape])) {
        fail(`${label}: sidecar __output__ ${JSON.stringify(out)} != [1,${g.expectedOutputShape}]`)
        bad = true
      }
      for (const [nodeId, attr] of Object.entries(res.attrMap)) {
        const oracle = g.expectedModuleShapes[nodeId]
        if (!oracle) continue
        const sc = result.shapes[attr]
        const hook = py.hook_shapes[attr]
        if (!eqShape(Array.isArray(sc) ? sc : null, oracle)) {
          fail(`${label}: sidecar shape ${attr} ${JSON.stringify(sc)} != oracle ${JSON.stringify(oracle)}`)
          bad = true
        }
        if (!eqShape(hook ?? null, oracle)) {
          fail(`${label}: python hook shape ${attr} ${JSON.stringify(hook)} != oracle ${JSON.stringify(oracle)}`)
          bad = true
        }
      }
      if (!bad) st.scPass++
    }
  } finally {
    stopSidecar(sidecar)
  }

  printTable(stats)
  console.log(`seed = ${SEED}`)
  if (failures > 0) {
    console.log(`\n✗ ${failures} property failure(s)`)
    return 1
  }
  console.log('\n✓ all property checks passed')
  return 0
}

function selectSidecarSample(graphs: GeneratedGraph[], max: number): GeneratedGraph[] {
  const byFamily = new Map<string, GeneratedGraph[]>()
  for (const g of graphs) {
    const list = byFamily.get(g.family) ?? []
    list.push(g)
    byFamily.set(g.family, list)
  }
  const lists = [...byFamily.values()]
  const out: GeneratedGraph[] = []
  let round = 0
  while (out.length < max) {
    let added = false
    for (const list of lists) {
      if (round < list.length && out.length < max) {
        out.push(list[round])
        added = true
      }
    }
    if (!added) break
    round++
  }
  return out
}

main()
  .then((code) => process.exit(code))
  .catch((e: unknown) => {
    console.error(e)
    process.exit(1)
  })
