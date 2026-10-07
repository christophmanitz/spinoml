import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { generateFromSnapshot } from '../src/codegen/generator'
import { parseFile, serializeCurrent } from '../src/persistence/file'
import { useGraphStore, type GraphSnapshot } from '../src/canvas/GraphStore'
import {
  REFERENCE_NAMES,
  cnnSnapshot,
  mlpSnapshot,
  multiInputSnapshot,
  type ReferenceName,
} from './lib/reference-graphs'

/**
 * Proves, numerically, that the PyTorch model SpinoML GENERATES from a graph is
 * the same model as an equivalent hand-written module — for three small
 * reference graphs (MLP, CNN, multi-input). Model level only; no training here.
 *
 * Usage:
 *   npm run verify:reference -- --write-fixtures   # (re)write committed fixtures
 *   npm run verify:reference                        # assert fixtures + run checks
 */

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = join(HERE, '..')
// Interpreter that can import torch: the runner exports PYTHON (resolved from
// SPINOML_CONDA_ENV / a plain pip env); direct runs fall back to `python`.
const PYTHON = process.env.PYTHON ?? 'python'
const FIXTURES_DIR = join(REPO_ROOT, 'examples', 'reference-experiments')
const COMPARE_SCRIPT = join(HERE, 'lib', 'reference_compare.py')
const FIXED_SAVED_AT = '1970-01-01T00:00:00.000Z'
const WRITE_FIXTURES = process.argv.includes('--write-fixtures')

const SNAPSHOT_BUILDERS: Record<ReferenceName, () => GraphSnapshot> = {
  mlp: mlpSnapshot,
  cnn: cnnSnapshot,
  'multi-input': multiInputSnapshot,
}

const EXPECTED_PARAMS: Record<ReferenceName, number> = {
  mlp: 210,
  cnn: 170,
  'multi-input': 130,
}

const EXPECTED_DESCRIPTION: Record<ReferenceName, string> = {
  mlp: 'Input x [1,10] float32 → Linear(10,16) → ReLU → Linear(16,2) → Output',
  cnn: 'Input x [1,64] float32 → Reshape [1,8,8] → Conv2d(1,4,k3,p1) → ReLU → MaxPool2d(2) → Flatten → Linear(64,2) → Output',
  'multi-input':
    'Input a [1,6] + Input b [1,4] → Linear(6,8)+ReLU and Linear(4,8)+ReLU → Concat(dim=1) → Linear(16,2) → Output',
}

const REFERENCE_CLASS: Record<ReferenceName, string> = {
  mlp: 'RefMLP',
  cnn: 'RefCNN',
  'multi-input': 'RefMultiInput',
}

// ─── Serialization (the app's own save path) ─────────────────────────────────
//
// `serializeCurrent()` is exactly what File → Save writes: the versioned
// `.spinoml` envelope around `captureRootSnapshot()`. The only non-deterministic
// field is `savedAt`, so we pin it to a fixed value after the fact; everything
// else (graph shape, key order, indentation) is produced by the app's function.
function serializeSnapshot(snapshot: GraphSnapshot): string {
  const loaded = useGraphStore.getState().loadSnapshot(snapshot)
  if (!loaded) throw new Error('loadSnapshot rejected the reference graph')
  const raw = serializeCurrent()
  const pattern = /"savedAt": "[^"]*"/
  if (!pattern.test(raw)) throw new Error('serializeCurrent() output has no savedAt field')
  return raw.replace(pattern, `"savedAt": "${FIXED_SAVED_AT}"`)
}

function readmeFor(name: ReferenceName): string {
  return [
    `# ${name} reference experiment`,
    '',
    `Hand-written reference module: \`${REFERENCE_CLASS[name]}\` in`,
    '`scripts/lib/reference_models.py`. Graph:',
    `\`${name}Snapshot()\` in \`scripts/lib/reference-graphs.ts\`.`,
    '',
    `- ${EXPECTED_DESCRIPTION[name]}`,
    `- Expected parameters: ${EXPECTED_PARAMS[name]} (trainable ${EXPECTED_PARAMS[name]})`,
    '',
    '`model.spinoml` is the serialized graph. `scripts/verify-reference.ts`',
    'regenerates the PyTorch model from it and numerically compares the generated',
    '`Model` against the hand-written reference (parameters, forward passes, loss,',
    'gradients, and a negative control).',
    '',
  ].join('\n')
}

// ─── Python result parsing ────────────────────────────────────────────────────

type PyCheck = { check: string; ok: boolean; detail: string; skipped: boolean }

function parseResponse(raw: string): Record<string, PyCheck[]> {
  let obj: unknown
  try {
    obj = JSON.parse(raw)
  } catch (e: unknown) {
    throw new Error(
      `python did not print JSON (${e instanceof Error ? e.message : String(e)})`,
      { cause: e },
    )
  }
  if (typeof obj !== 'object' || obj === null || !('results' in obj)) {
    throw new Error('python result missing "results"')
  }
  const results = (obj as { results: unknown }).results
  if (typeof results !== 'object' || results === null) {
    throw new Error('python result "results" is not an object')
  }
  const out: Record<string, PyCheck[]> = {}
  for (const [name, value] of Object.entries(results as Record<string, unknown>)) {
    if (!Array.isArray(value)) throw new Error(`results["${name}"] is not an array`)
    out[name] = value.map((entry): PyCheck => {
      if (typeof entry !== 'object' || entry === null) {
        throw new Error(`results["${name}"] contains a non-object check`)
      }
      const rec = entry as Record<string, unknown>
      if (typeof rec.check !== 'string' || typeof rec.ok !== 'boolean') {
        throw new Error(`results["${name}"] check missing {check, ok}`)
      }
      return {
        check: rec.check,
        ok: rec.ok,
        detail: typeof rec.detail === 'string' ? rec.detail : '',
        skipped: rec.skipped === true,
      }
    })
  }
  return out
}

// ─── Summary table ────────────────────────────────────────────────────────────

type Column = { label: string; checks: string[] }
const COLUMNS: Column[] = [
  { label: 'params', checks: ['param_count', 'param_shapes', 'copy_weights'] },
  { label: 'forward f32', checks: ['forward_f32'] },
  { label: 'forward f64', checks: ['forward_f64'] },
  { label: 'loss', checks: ['loss_f64'] },
  { label: 'grads', checks: ['grads_f64'] },
  { label: 'negative control', checks: ['negative_control'] },
  { label: 'determinism', checks: ['determinism'] },
]

const REQUIRED_CHECKS = [
  'generated_load',
  'generated_construct',
  'param_count',
  'param_shapes',
  'copy_weights',
  'forward_f32',
  'forward_f64',
  'loss_f64',
  'grads_f64',
  'negative_control',
  'cuda_forward_f32',
]

type Status = 'PASS' | 'FAIL' | 'SKIP'

function statusOf(checks: Map<string, PyCheck>, column: Column): Status {
  let skipped = false
  for (const key of column.checks) {
    const c = checks.get(key)
    if (!c || !c.ok) return 'FAIL'
    if (c.skipped) skipped = true
  }
  return skipped ? 'SKIP' : 'PASS'
}

function printTable(rows: { name: ReferenceName; checks: Map<string, PyCheck> }[]): void {
  const headers = ['experiment', ...COLUMNS.map((c) => c.label)]
  const cells: string[][] = rows.map((row) => [row.name, ...COLUMNS.map((c) => statusOf(row.checks, c))])
  const widths = headers.map((h, i) => Math.max(h.length, ...cells.map((r) => r[i].length)))
  const line = (vals: string[]) => vals.map((v, i) => v.padEnd(widths[i])).join(' | ')

  console.log('')
  console.log(line(headers))
  console.log(widths.map((w) => '-'.repeat(w)).join('-+-'))
  for (const row of cells) console.log(line(row))
  console.log('')
}

// ─── Main ─────────────────────────────────────────────────────────────────────

type Prepared = { name: ReferenceName; modelPy: string }

function main(): number {
  const tmp = mkdtempSync(join(tmpdir(), 'spinoml-reference-'))
  const failures: string[] = []
  const prepared: Prepared[] = []
  const determinism: Record<string, { ok: boolean; detail: string }> = {}

  try {
    for (const name of REFERENCE_NAMES) {
      const snapshot = SNAPSHOT_BUILDERS[name]()
      const text = serializeSnapshot(snapshot)
      const fixture = join(FIXTURES_DIR, name, 'model.spinoml')

      // Determinism: three independent parses → generate must yield identical code.
      const codes: string[] = []
      let issues: string[] = []
      for (let i = 0; i < 3; i++) {
        const result = generateFromSnapshot(parseFile(text))
        codes.push(result.code)
        issues = result.issues
      }
      if (issues.length > 0) {
        failures.push(`${name}: generator reported issues: ${JSON.stringify(issues)}`)
      }
      const deterministic = codes[0] === codes[1] && codes[1] === codes[2]
      determinism[name] = {
        ok: deterministic && issues.length === 0,
        detail: deterministic ? '3 generate() runs produced identical code' : 'generation is not deterministic',
      }
      if (!deterministic) failures.push(`${name}: generateFromSnapshot is not deterministic`)

      const modelPy = join(tmp, `${name}.py`)
      writeFileSync(modelPy, codes[0])
      prepared.push({ name, modelPy })

      if (WRITE_FIXTURES) {
        mkdirSync(join(FIXTURES_DIR, name), { recursive: true })
        writeFileSync(fixture, text)
        writeFileSync(join(FIXTURES_DIR, name, 'README.md'), readmeFor(name))
        console.log(`wrote ${fixture}`)
      } else {
        if (!existsSync(fixture)) {
          failures.push(`${name}: committed fixture missing at ${fixture} (run with --write-fixtures)`)
        } else {
          const committed = readFileSync(fixture, 'utf-8')
          if (committed !== text) {
            failures.push(`${name}: committed fixture differs from freshly serialized snapshot`)
          }
        }
      }
    }

    // ONE python process for all experiments.
    const job = JSON.stringify({
      experiments: prepared.map((p) => ({ name: p.name, model_py: p.modelPy })),
    })
    const proc = spawnSync(
      PYTHON,
      [COMPARE_SCRIPT],
      { input: job, encoding: 'utf-8', maxBuffer: 128 * 1024 * 1024 },
    )
    if (proc.error) {
      failures.push(`failed to run python: ${proc.error.message}`)
    } else if (proc.status !== 0) {
      failures.push(`reference_compare.py exited ${proc.status}: ${(proc.stderr ?? '').trim()}`)
    } else {
      let results: Record<string, PyCheck[]>
      try {
        results = parseResponse((proc.stdout ?? '').trim())
      } catch (e: unknown) {
        failures.push(e instanceof Error ? e.message : String(e))
        results = {}
      }

      const rows: { name: ReferenceName; checks: Map<string, PyCheck> }[] = []
      for (const name of REFERENCE_NAMES) {
        const checks = new Map<string, PyCheck>()
        for (const c of results[name] ?? []) checks.set(c.check, c)
        // Synthesize the TS-side determinism check so it appears in the table.
        checks.set('determinism', {
          check: 'determinism',
          ok: determinism[name].ok,
          detail: determinism[name].detail,
          skipped: false,
        })
        rows.push({ name, checks })
      }

      // Validate required checks + collect failure details.
      let cudaLine: string | null = null
      for (const row of rows) {
        if (!(row.name in results)) {
          failures.push(`${row.name}: no result returned by python`)
          continue
        }
        for (const key of REQUIRED_CHECKS) {
          if (!row.checks.has(key)) failures.push(`${row.name}: missing check "${key}"`)
        }
        for (const c of row.checks.values()) {
          if (!c.ok) failures.push(`${row.name}: ${c.check} FAILED — ${c.detail}`)
          if (c.check === 'cuda_forward_f32' && c.skipped) cudaLine = c.detail
        }
      }
      if (rows.length > 0) printTable(rows)
      if (cudaLine) console.log(cudaLine)
    }
  } finally {
    rmSync(tmp, { recursive: true, force: true })
  }

  for (const f of failures) console.log(`  ✗ ${f}`)
  const ok = failures.length === 0
  console.log('')
  console.log(ok ? '✓ all reference equivalence checks passed' : `✗ ${failures.length} failure(s)`)
  return ok ? 0 : 1
}

process.exit(main())
