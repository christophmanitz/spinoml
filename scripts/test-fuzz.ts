// Fuzz test over deliberately INVALID graphs.  Run: `npm run test:fuzz`.
//
// Starting from valid graphs produced by `scripts/lib/graph-gen.ts`, mutation
// operators produce mutants that are tagged with their EXPECTED outcome (set by
// reading the invariants / coercion / sidecar, not by observing the app):
//
//   structural  → validateGraphState reports an error AND loadSnapshot returns
//                 false leaving the store byte-for-byte unchanged,
//   coerced     → the raw validator rejects, but coerceParams heals the value;
//                 loadSnapshot accepts and the resulting store is valid,
//   load-reject → validator only warns, loadSnapshot still rejects cleanly,
//   semantic    → structurally valid but shape-incompatible: loadSnapshot
//                 accepts, then generate + the real /infer must answer with a
//                 STRUCTURED ok:false error that verificationFromInferResult
//                 classifies `invalid` (never `valid`, never `unknown`).
//
// For every mutant we assert no unhandled exception escapes, and that a mutant
// expected to be rejected is never silently accepted. The suite also stress-
// tests a deep valid chain.
//
// SECURITY: executes only code produced by the app's generator from graphs the
// script itself built — no external input, no network beyond 127.0.0.1.

import { spawn, type ChildProcess } from 'node:child_process'
import { setTimeout as wait } from 'node:timers/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { useGraphStore } from '../src/canvas/GraphStore'
import { validateGraphState } from '../src/canvas/invariants'
import { generateFromSnapshot } from '../src/codegen/generator'
import { inferShapes } from '../src/inference/client'
import { verificationFromInferResult } from '../src/inference/verifier'
import {
  chainGraph,
  graphAt,
  mutantsFor,
  rawStructuralMutants,
  resolveSeed,
  snapshotToEdges,
  snapshotToLayerNodes,
  type Mutant,
} from './lib/graph-gen'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = join(HERE, '..')
const SEED = resolveSeed()
const BASE = (() => {
  const raw = process.env.FUZZ_BASE
  const n = raw ? Number(raw) : 40
  return Number.isFinite(n) && n > 0 ? Math.trunc(n) : 40
})()
const SIDECAR_PORT = Number(process.env.SPINOML_TORCH_PORT ?? '7421')
const SIDECAR_URL = `http://127.0.0.1:${SIDECAR_PORT}`

// ─────────────────────────────────────────────────────────────────────────────
// Reporting
// ─────────────────────────────────────────────────────────────────────────────

type OpStat = { operator: string; kind: string; count: number; pass: number; fail: number }
const opStats = new Map<string, OpStat>()
let failures = 0

function stat(operator: string, kind: string): OpStat {
  let s = opStats.get(operator)
  if (!s) {
    s = { operator, kind, count: 0, pass: 0, fail: 0 }
    opStats.set(operator, s)
  }
  return s
}

function record(m: Mutant, index: number, message: string): void {
  failures++
  const s = stat(m.operator, m.kind)
  s.fail++
  console.log(`  ✗ [${m.operator}] graph #${index} ${m.note}: ${message}`)
  console.log(`    replay: FUZZ_BASE=${BASE} PROPERTY_SEED=${SEED} (base index ${index}) operator=${m.operator}`)
}

function fingerprint(): string {
  const s = useGraphStore.getState()
  return JSON.stringify({
    nodes: s.nodes.map((n) => ({ id: n.id, t: n.data.layerType, p: n.data.params })),
    edges: s.edges.map((e) => ({ id: e.id, s: e.source, t: e.target })),
  })
}

function storeIsValid(): boolean {
  const s = useGraphStore.getState()
  return validateGraphState(s.nodes, s.edges).ok
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidecar lifecycle
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
  const deadline = Date.now() + 20_000
  while (Date.now() < deadline) {
    if (await sidecarUp()) return { child, owned: true }
    if (child.exitCode !== null) break
    await wait(200)
  }
  child.kill('SIGKILL')
  throw new Error('torch sidecar did not become healthy within 20s')
}

function stopSidecar(s: Sidecar): void {
  if (s.owned && s.child) s.child.kill('SIGTERM')
}

// ─────────────────────────────────────────────────────────────────────────────
// Mutant execution
// ─────────────────────────────────────────────────────────────────────────────

async function checkSemantic(m: Mutant, index: number, baseInputs: { shape: number[]; dtype: string }[]): Promise<void> {
  let code: string
  try {
    code = generateFromSnapshot(m.snapshot).code
  } catch (e: unknown) {
    record(m, index, `generate threw: ${e instanceof Error ? e.message : String(e)}`)
    return
  }
  const ac = new AbortController()
  const timer = setTimeout(() => ac.abort(), 20_000)
  let result
  try {
    result = await inferShapes(
      code,
      baseInputs.map((i) => i.shape),
      baseInputs.map((i) => i.dtype),
      ac.signal,
    )
  } catch (e: unknown) {
    record(m, index, `inferShapes threw: ${e instanceof Error ? e.message : String(e)}`)
    return
  } finally {
    clearTimeout(timer)
  }
  if ('offline' in result) {
    record(m, index, `sidecar offline / HTTP error: ${result.error}`)
    return
  }
  if (result.ok) {
    record(m, index, 'silently accepted by the sidecar (expected a structured error)')
    return
  }
  if (!result.stage && !result.error_code) {
    record(m, index, `structured error missing stage/error_code: ${JSON.stringify(result)}`)
    return
  }
  const verdict = verificationFromInferResult(result)
  if (verdict.status !== 'invalid') {
    record(m, index, `classified '${verdict.status}' (expected 'invalid')`)
  }
}

async function runMutant(m: Mutant, index: number, baseInputs: { shape: number[]; dtype: string }[]): Promise<void> {
  const s = stat(m.operator, m.kind)
  s.count++
  try {
    // (a) validator — must reject when expected, never throw.
    let hasError = false
    try {
      const v = validateGraphState(snapshotToLayerNodes(m.snapshot), snapshotToEdges(m.snapshot))
      hasError = v.issues.some((i) => i.severity === 'error')
    } catch (e: unknown) {
      record(m, index, `validateGraphState threw: ${e instanceof Error ? e.message : String(e)}`)
      return
    }
    if (m.expectValidatorError && !hasError) {
      record(m, index, 'silently accepted by validateGraphState')
      return
    }

    // (b) loadSnapshot — reject + leave state untouched, or accept + stay valid.
    const before = fingerprint()
    let accepted = false
    try {
      accepted = useGraphStore.getState().loadSnapshot(m.snapshot)
    } catch (e: unknown) {
      record(m, index, `loadSnapshot threw: ${e instanceof Error ? e.message : String(e)}`)
      return
    }
    if (m.expectLoadReject) {
      if (accepted) {
        record(m, index, 'silently accepted by loadSnapshot')
        return
      }
      if (fingerprint() !== before) {
        record(m, index, 'store was mutated during a rejected loadSnapshot')
        return
      }
    } else if (m.kind === 'coerced' || m.kind === 'semantic') {
      if (!accepted) {
        record(m, index, 'loadSnapshot unexpectedly rejected')
        return
      }
      if (!storeIsValid()) {
        record(m, index, 'loadSnapshot accepted but left error-level issues in the store')
        return
      }
    }

    // (c) semantic — the real sidecar must reject with a structured error.
    if (m.kind === 'semantic') {
      await checkSemantic(m, index, baseInputs)
      if (s.fail > 0 && s.pass + s.fail === s.count) {
        // checkSemantic already recorded; don't double count pass
        return
      }
      // fall through to count pass only if no failure was recorded for this run
      if (s.pass + s.fail < s.count) s.pass++
      return
    }

    s.pass++
  } catch (e: unknown) {
    record(m, index, `unhandled exception: ${e instanceof Error ? e.message : String(e)}`)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10k-node chain stress test
// ─────────────────────────────────────────────────────────────────────────────

function runChainStress(): void {
  const DEPTH = 10_000
  const operator = 'deep-valid-chain'
  const s = stat(operator, 'stress')
  s.count++
  const chain = chainGraph(DEPTH)
  const t0 = Date.now()
  try {
    const v = validateGraphState(snapshotToLayerNodes(chain), snapshotToEdges(chain))
    if (!v.ok) {
      s.fail++
      failures++
      console.log(`  ✗ [${operator}] a valid ${DEPTH}-deep chain was rejected: ${JSON.stringify(v.issues.slice(0, 3))}`)
      return
    }
    const accepted = useGraphStore.getState().loadSnapshot(chain)
    if (!accepted) {
      s.fail++
      failures++
      console.log(`  ✗ [${operator}] loadSnapshot rejected a valid ${DEPTH}-deep chain`)
      return
    }
    const ms = Date.now() - t0
    if (ms > 10_000) {
      s.fail++
      failures++
      console.log(`  ✗ [${operator}] took ${ms}ms (> 10s budget)`)
      return
    }
    s.pass++
  } catch (e: unknown) {
    s.fail++
    failures++
    console.log(`  ✗ [${operator}] FINDING: crashed on a valid ${DEPTH}-deep chain: ${e instanceof Error ? e.message : String(e)}`)
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

async function main(): Promise<number> {
  console.log(`fuzz test — seed=${SEED} base graphs=${BASE}`)
  let sidecar: Sidecar = { child: null, owned: false }
  try {
    sidecar = await startSidecar()
    console.log(`sidecar ${sidecar.owned ? 'started' : 'reused'} on ${SIDECAR_URL}`)

    for (let i = 0; i < BASE; i++) {
      const g = graphAt(i, SEED)
      for (const m of mutantsFor(g)) {
        await runMutant(m, i, g.inputs)
      }
      // Validator-only mutants that a GraphSnapshot cannot express.
      for (const r of rawStructuralMutants()) {
        const asMutant: Mutant = {
          operator: r.operator,
          kind: 'structural',
          snapshot: { nodes: [], edges: [] },
          expectValidatorError: r.expectValidatorError,
          expectLoadReject: false,
          expectSidecarError: false,
          note: r.note,
        }
        const s = stat(r.operator, 'raw-structural')
        s.count++
        try {
          const v = validateGraphState(r.nodes, r.edges)
          const hasError = v.issues.some((x) => x.severity === 'error')
          if (r.expectValidatorError && !hasError) {
            record(asMutant, i, 'silently accepted by validateGraphState')
            continue
          }
          s.pass++
        } catch (e: unknown) {
          record(asMutant, i, `validateGraphState threw: ${e instanceof Error ? e.message : String(e)}`)
        }
      }
    }

    runChainStress()

    // Sidecar must still be healthy after every malformed request.
    if (!(await sidecarUp())) {
      failures++
      console.log('  ✗ sidecar unhealthy after the fuzz run')
    }
  } finally {
    stopSidecar(sidecar)
  }

  // ── per-operator table ───────────────────────────────────────────────────
  const headers = ['operator', 'kind', 'count', 'pass', 'fail']
  const rows = [...opStats.values()].map((s) => [s.operator, s.kind, String(s.count), String(s.pass), String(s.fail)])
  const total = [...opStats.values()].reduce((a, s) => a + s.count, 0)
  rows.push(['TOTAL', '', String(total), '', String(failures)])
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => (r[i] ?? '').length)))
  const line = (vals: string[]) => vals.map((v, i) => v.padEnd(widths[i])).join(' | ')
  console.log('')
  console.log(line(headers))
  console.log(widths.map((w) => '-'.repeat(w)).join('-+-'))
  for (const r of rows) console.log(line(r))
  console.log('')

  if (total < 400) {
    console.log(`✗ only ${total} mutants (< 400 required)`)
    return 1
  }
  if (failures > 0) {
    console.log(`✗ ${failures} fuzz failure(s) across ${total} mutants`)
    return 1
  }
  console.log(`✓ all ${total} mutants behaved as expected (seed=${SEED})`)
  return 0
}

main()
  .then((code) => process.exit(code))
  .catch((e: unknown) => {
    console.error(e)
    process.exit(1)
  })
