// Parity test between the SIDECAR's tool-argument validation and the FRONTEND's
// coercion + graph validation. Run: `npm run test:llm-validation-parity`.
//
// The provider is untrusted input, and the frontend's `coerceParams` silently
// REPLACES junk values with defaults — so a tool call the sidecar accepts can
// become a DIFFERENT graph than the model believes. Therefore the sidecar must
// never accept a value the frontend would reject or silently change.
//
// For every node type in the three registries and every field we feed a corpus
// of values through:
//   - the Node verdict: `validateNodeParams(registry, type, params).ok`
//   - the frontend verdict: no `validateGraphState` error on the coerced params
//     (layers) AND `coerceParams` left the supplied field value unchanged after
//     normalising numeric strings, boolean strings and integer-valued floats.
//
// A "dangerous mismatch" — the sidecar ACCEPTS a value the frontend would reject
// or change — is a failure (the mismatch table must stay empty). The sidecar is
// intentionally STRICTER in a few documented ways (magnitude > 1e9, unknown /
// prototype keys); those are reported separately and are not failures, because
// rejecting more than the frontend can never make the graph diverge. To prove
// the test can fail, loosen one sidecar rule (e.g. let an int accept 6.5) and it
// goes red on the 6.5 corpus entry.

import { readFileSync } from 'node:fs'
import type { Edge } from '@xyflow/react'
import { LAYERS, coerceParams } from '../src/layers/registry'
import { validateGraphState } from '../src/canvas/invariants'
import type { LayerNode } from '../src/canvas/GraphStore'
import { TRAINING_NODES, coerceTrainingParams } from '../src/training/graph/registry'
import { DATA_NODES, coerceDataParams } from '../src/data/graph/registry'
import { validateNodeParams } from '../sidecar-llm/tool-validation.mjs'
import { generateCatalogJson } from './gen-layer-catalog.ts'

type RegistryKey = 'layers' | 'training' | 'data'
type Field = { name: string; type: string; min?: number; max?: number; arity?: number; default?: unknown; options?: string[] }

interface Registry {
  key: RegistryKey
  nodes: Record<string, { fields: Field[] }>
  coerce: (type: string, raw: Record<string, unknown>) => Record<string, unknown>
  validate: (type: string, params: Record<string, unknown>) => boolean
}

const FRONTEND_LAYERS: Registry = {
  key: 'layers',
  nodes: LAYERS as unknown as Registry['nodes'],
  coerce: (type, raw) => coerceParams(type, raw),
  validate: (type, params) => {
    const node = { id: 'n', type: 'layer', position: { x: 0, y: 0 }, data: { layerType: type, params } } as LayerNode
    return validateGraphState([node], [] as Edge[]).ok
  },
}
const FRONTEND_TRAINING: Registry = {
  key: 'training',
  nodes: TRAINING_NODES as unknown as Registry['nodes'],
  coerce: (type, raw) => coerceTrainingParams(type, raw),
  // The training/data stores only coerce; they have no `validateParams`.
  validate: () => true,
}
const FRONTEND_DATA: Registry = {
  key: 'data',
  nodes: DATA_NODES as unknown as Registry['nodes'],
  coerce: (type, raw) => coerceDataParams(type, raw),
  validate: () => true,
}

const REGISTRIES: Registry[] = [FRONTEND_LAYERS, FRONTEND_TRAINING, FRONTEND_DATA]

// ── corpus ──────────────────────────────────────────────────────────────────
const BASE_CORPUS: unknown[] = [
  1, 0, 2, 512, -1, 1e9, 1e9 + 1, 1e12, Number.NaN, Number.POSITIVE_INFINITY,
  '64', '6.5', '6.0', '1e-3', '', ' ', 'NaN', 'true', 'false', 'relu', 'gelu', 'x', 'a,b',
  true, false,
  null, undefined,
  [], [1], [1, 2, 3], [1, -1], ['a'], ['a', 'b'],
  {}, { a: 1 },
  'a'.repeat(5000),
]

function corpusFor(field: Field): unknown[] {
  const values = [...BASE_CORPUS]
  if (field.min !== undefined) values.push(field.min, field.min - 1)
  if (field.max !== undefined) values.push(field.max, field.max + 1)
  if (field.arity !== undefined) values.push(Array(field.arity).fill(1))
  return values
}

// ── normalisation of "the frontend stored the same value" ───────────────────
function numericString(v: unknown): boolean {
  return typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v))
}

function unchanged(supplied: unknown, stored: unknown, field: Field): boolean {
  switch (field.type) {
    case 'int':
    case 'float': {
      if (numericString(supplied) && Number(supplied) === stored) return true
      // integer-valued floats are truncated by the frontend's int coercion
      if (typeof supplied === 'number' && Number.isFinite(supplied) && Math.trunc(supplied) === stored) return true
      if (typeof supplied === 'number' && Number.isFinite(supplied) && supplied === stored) return true
      return Object.is(supplied, stored)
    }
    case 'bool': {
      if (supplied === 'true' && stored === true) return true
      if (supplied === 'false' && stored === false) return true
      return Object.is(supplied, stored)
    }
    case 'tuple-int': {
      if (typeof supplied === 'number' && Array.isArray(stored) && stored.every((x) => x === supplied)) return true
      // the frontend splits a numeric string and broadcasts a 1-element array
      if (numericString(supplied) && Array.isArray(stored) && stored.every((x) => x === Number(supplied))) return true
      if (Array.isArray(supplied) && Array.isArray(stored)) {
        if (supplied.length !== stored.length) return false
        return supplied.every((v, i) => (numericString(v) ? Number(v) === stored[i] : v === stored[i]))
      }
      return false
    }
    default:
      return JSON.stringify(supplied) === JSON.stringify(stored)
  }
}

interface Mismatch {
  registry: RegistryKey
  type: string
  field: string
  fate: string
  value: string
  node: boolean
  frontend: boolean
}

const kit = (v: unknown): string => {
  try {
    return JSON.stringify(v) ?? String(v)
  } catch {
    return String(v)
  }
}

const dangerous: Mismatch[] = []
const stricter: Mismatch[] = []
let cases = 0

for (const reg of REGISTRIES) {
  for (const [type, spec] of Object.entries(reg.nodes)) {
    for (const field of spec.fields) {
      for (const value of corpusFor(field)) {
        cases++
        const params: Record<string, unknown> = { [field.name]: value }
        const nodeOk = validateNodeParams(reg.key, type, params).ok
        let frontendOk: boolean
        let fate: string
        try {
          const coerced = reg.coerce(type, params)
          frontendOk = reg.validate(type, coerced) && unchanged(value, coerced[field.name], field)
          fate = frontendOk ? 'store-unchanged' : 'change-or-reject'
        } catch {
          frontendOk = false
          fate = 'coerce-threw'
        }
        if (nodeOk && !frontendOk) dangerous.push({ registry: reg.key, type, field: field.name, fate, value: kit(value), node: nodeOk, frontend: frontendOk })
        if (!nodeOk && frontendOk) stricter.push({ registry: reg.key, type, field: field.name, fate, value: kit(value), node: nodeOk, frontend: frontendOk })
      }
    }
  }
}

// ── prototype/forbidden keys must be rejected by the sidecar ────────────────
let forbiddenViolations = 0
for (const reg of REGISTRIES) {
  for (const type of Object.keys(reg.nodes)) {
    const parsed = JSON.parse('{"__proto__":{"in_features":7},"constructor":1,"prototype":2}') as Record<string, unknown>
    if (validateNodeParams(reg.key, type, parsed).ok) {
      forbiddenViolations++
      console.log(`  ✗ ${reg.key}.${type} accepted forbidden prototype keys`)
    }
  }
}
// a valid known key smuggled in a proto-carrying object must be rejected too
for (const reg of REGISTRIES) {
  for (const [type, spec] of Object.entries(reg.nodes)) {
    const first = spec.fields[0]
    if (!first) continue
    const parsed = JSON.parse(`{"${first.name}":null,"__proto__":{"x":1}}`) as Record<string, unknown>
    if (validateNodeParams(reg.key, type, parsed).ok) forbiddenViolations++
  }
}

// ── catalog drift ───────────────────────────────────────────────────────────
const committed = readFileSync(new URL('../sidecar-llm/layer-catalog.generated.json', import.meta.url), 'utf8')
const fresh = generateCatalogJson()
const catalogInSync = committed === fresh

// ── handler wiring: every add_*/update_* tool validates via the SAME function ─
const mainSrc = readFileSync(new URL('../sidecar-llm/main.mjs', import.meta.url), 'utf8')
function toolBody(name: string): string {
  const start = mainSrc.indexOf(`'${name}',`)
  if (start === -1) return ''
  const next = mainSrc.indexOf('\n    tool(', start)
  return mainSrc.slice(start, next === -1 ? mainSrc.length : next)
}
const WIRED_TOOLS = [
  'add_layer', 'update_params',
  'add_training_node', 'update_training_params',
  'add_data_node', 'update_data_params',
]
const unwired = WIRED_TOOLS.filter((name) => !toolBody(name).includes('validateNodeParams('))

// ── report ──────────────────────────────────────────────────────────────────
console.log(`cases checked: ${cases}`)
console.log(`dangerous mismatches: ${dangerous.length}`)
for (const m of dangerous.slice(0, 40)) console.log(`  ✗ ${m.registry}.${m.type}.${m.field} = ${m.value} (frontend ${m.fate})`)
console.log(`sidecar stricter than frontend (allowed hardening): ${stricter.length}`)
for (const m of stricter.slice(0, 8)) console.log(`  · ${m.registry}.${m.type}.${m.field} = ${m.value}`)
console.log(`forbidden-key violations: ${forbiddenViolations}`)
console.log(`catalog in sync: ${catalogInSync}`)
console.log(`handlers wired to validateNodeParams: ${unwired.length === 0 ? 'yes' : `no (${unwired.join(', ')})`}`)

if (!catalogInSync) {
  console.error('\nsidecar-llm/layer-catalog.generated.json is OUT OF DATE — run `npm run gen:layer-catalog`')
}
const failed = dangerous.length > 0 || forbiddenViolations > 0 || !catalogInSync || unwired.length > 0
console.log(`\n${failed ? 'FAIL' : 'PASS'}: parity check`)
process.exitCode = failed ? 1 : 0
